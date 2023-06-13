import copy
import re
import time
from typing import Union, List, Tuple, Optional, Dict

import torch
from accelerate import init_empty_weights
from accelerate import load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download
from transformers.modeling_outputs import BaseModelOutputWithPast

from models.configuration_moss import MossConfig
from models.modeling_moss import MossForCausalLM
from models.tokenization_moss import MossTokenizer


class MossStreamInference:
    def __init__(
            self,
            meta_instruction: str,
            model_name: str = "fnlp/moss-moon-003-sft-int4",
            ai_name: str = "MOSS"
    ) -> None:
        self.ai_name = ai_name
        self._num_layers, self._heads, self._hidden, self._vocab_size = 34, 24, 256, 107008
        self._model = MossForCausalLM.from_pretrained(model_name).half().cuda()
        self._tokenizer = MossTokenizer.from_pretrained(model_name)
        self._meta_instruction = meta_instruction
        self._stopwords = torch.LongTensor([self._tokenizer.convert_tokens_to_ids("<eom>")])
        self._last_token_len = 0

    @staticmethod
    def _top_k_top_p_filtering(logits, top_k, top_p, filter_value=-float("Inf"), min_tokens_to_keep=1, ):
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value

        return logits

    def _infer(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            past_key_values: Optional[Tuple[torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Inference method that computes logits and past key values.

        Args:
            input_ids (torch.Tensor): The input IDs tensor.
            attention_mask (torch.Tensor): The attention mask tensor.
            past_key_values (Optional[Tuple[torch.Tensor]]): The past key values tuple.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor]]: A tuple containing the logits and past key values.
        """
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }
        with torch.no_grad():
            outputs: BaseModelOutputWithPast = self._model(**inputs)

        return outputs.logits, outputs.past_key_values

    def _stream_sample(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            temperature: float = 0.7,
            repetition_penalty: float = 1.02,
            top_k: int = 0,
            top_p: float = 0.92,
            max_iterations: int = 1024,
            regulation_start: int = 512,
            length_penalty: float = 1.0,
            max_time: int = 60
    ) -> torch.Tensor:
        """
        Performs a streaming top-k search using the given parameters.

        Args:
            input_ids (torch.Tensor): The input IDs tensor.
            attention_mask (torch.Tensor): The attention mask tensor.
            temperature (float, optional): The temperature for logits. Defaults to 0.7.
            repetition_penalty (float, optional): The repetition penalty factor. Defaults to 1.02.
            top_k (int, optional): The top-k value for filtering. Defaults to 0.
            top_p (float, optional): The top-p value for filtering. Defaults to 0.92.
            max_iterations (int, optional): The maximum number of iterations. Defaults to 1024.
            regulation_start (int, optional): The number of iterations after which regulation starts. Defaults to 512.
            length_penalty (float, optional): The length penalty factor. Defaults to 1.
            max_time (int, optional): The maximum allowed time in seconds. Defaults to 60.

        Returns:
            torch.Tensor: The generated output IDs tensor.
        """
        assert input_ids.dtype == torch.int64 and attention_mask.dtype == torch.int64

        self.bsz, self.seqlen = input_ids.shape

        input_ids, attention_mask = input_ids.to('cuda'), attention_mask.to('cuda')
        last_token_indices = attention_mask.sum(1) - 1

        moss_stopwords = self._stopwords.to(input_ids.device)
        queue_for_moss_stopwords = torch.empty(size=(self.bsz, len(self._stopwords)), device=input_ids.device,
                                               dtype=input_ids.dtype)
        all_shall_stop = torch.tensor([False] * self.bsz, device=input_ids.device)
        moss_stop = torch.tensor([False] * self.bsz, device=input_ids.device)

        generations, start_time = torch.ones(self.bsz, 1, dtype=torch.int64), time.time()

        past_key_values = None
        new_generated_id = None
        for i in range(int(max_iterations)):
            logits, past_key_values = self._infer(
                input_ids if i == 0 else new_generated_id, attention_mask, past_key_values
            )

            if i == 0:
                logits = logits.gather(
                    1,
                    last_token_indices.view(self.bsz, 1, 1).repeat(1, 1, self._vocab_size)
                ).squeeze(1)
            else:
                logits = logits[:, -1, :]

            if repetition_penalty > 1:
                score = logits.gather(1, input_ids)
                # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
                # just gather the histroy token from input_ids, preprocess then scatter back
                # here we apply extra work to exclude special token

                score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)

                logits.scatter_(1, input_ids, score)

            logits = logits / temperature

            filtered_logits = self._top_k_top_p_filtering(logits, top_k, top_p)
            probabilities = torch.softmax(filtered_logits, dim=-1)

            cur_len = i
            if cur_len > int(regulation_start):
                for i in self._stopwords:
                    probabilities[:, i] = probabilities[:, i] * pow(length_penalty, cur_len - regulation_start)

            new_generated_id = torch.multinomial(probabilities, 1)

            input_ids, attention_mask = torch.cat([input_ids, new_generated_id], dim=1), torch.cat(
                [attention_mask, torch.ones((self.bsz, 1), device=attention_mask.device, dtype=attention_mask.dtype)],
                dim=1)

            generations = torch.cat([generations, new_generated_id.cpu()], dim=1)

            # stop words components
            queue_for_moss_stopwords = torch.cat([queue_for_moss_stopwords[:, 1:], new_generated_id], dim=1)

            moss_stop |= (queue_for_moss_stopwords == moss_stopwords).all(1)

            all_shall_stop |= moss_stop

            yield input_ids

            if all_shall_stop.all().item():
                break

            elif max_time > 0 and time.time() - start_time > max_time:
                break

    def _preprocess(self, text: str) -> str:
        return self._meta_instruction + text

    def _tokenize(self, text: str):
        tokens = self._tokenizer.batch_encode_plus([text], return_tensors="pt")
        input_ids, attention_mask = tokens['input_ids'], tokens['attention_mask']
        return input_ids, attention_mask

    @staticmethod
    def _cleanup_query(query):
        # avoid user manually input special MOSS tokens
        clean_query = query
        while "<|" in clean_query or "|>" in clean_query:
            clean_query = query.replace('<|', '{').replace('|>', '}')
        return clean_query

    @property
    def last_token_len(self):
        return self._last_token_len

    @torch.no_grad()
    def stream_chat(
            self,
            query: str,
            temperature: float = 0.7,
            repetition_penalty: float = 1.02,
            top_k: int = 40,
            top_p: float = 0.8,
            max_iterations: int = 2048,
            regulation_start: int = 512,
            length_penalty: float = 1.0,
            max_time: int = 60,
            history: str = ""
    ) -> List[str]:
        clean_query = self._cleanup_query(query)
        history += f"<|Human|>:{clean_query}<eoh>\n<|{self.ai_name}|>:"
        input_ids, attention_mask = self._tokenize(
            self._preprocess(history)
        )
        output_start_index = input_ids.shape[1]
        for output_ids in self._stream_sample(
                input_ids,
                attention_mask,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                top_k=top_k,
                top_p=top_p,
                max_iterations=max_iterations,
                regulation_start=regulation_start,
                length_penalty=length_penalty,
                max_time=max_time
        ):
            stream_output = self._tokenizer.decode(output_ids[0][output_start_index:], skip_special_tokens=True)
            new_history = history + self._tokenizer.decode(output_ids[0][output_start_index:])
            self._last_token_len = len(output_ids[0])
            yield stream_output, new_history
