# Moss Stream Chat

English | [中文指南](https://github.com/GitEventhandler/MOSS-stream-chat/blob/master/README_zh.md)

## About

This project is a [MOSS](https://github.com/OpenLMLab/MOSS) frontend that supports streaming conversations. You may need to use wsl if you run this app on windows (package triton only support linux currently).  
![](https://github.com/GitEventhandler/MOSS-stream-chat/blob/master/screenshot/moss_stream_chat.gif)

## Requirements

There are no new dependencies required compared to MOSS. You can use MOSS's python environment directly.

### Install conda

You can find the installation tutorial on [Anaconda's official website](https://www.anaconda.com/download/). Remember to
init conda before using it.

```shell
conda init
```

### Create Virtual Environment

```shell
conda create --name moss python=3.8
conda activate moss
```

### Install pytorch

#### Linux and windows

```shell
# CUDA 11.6
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
# CUDA 11.7
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
# CPU Only
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch
```

#### OSX

```shell
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch
```

### Install other python packages

```shell
pip install -r requirements.txt
```

## Usage

First, create fnlp folder, and put your model files under fnlp folder.
```shell
mkdir fnlp
```

To run with default config.

```shell
streamlit run web_demo.py
```

If you want to pass in arguments, be careful to use command "web_demo.py **--** [params]".

```
streamlit run web_demo.py -- [-h] [--model_name {fnlp/moss-moon-003-sft,fnlp/moss-moon-003-sft-int8,fnlp/moss-moon-003-sft-int4}] [--ai_name AI_NAME] [--gpu GPU]

optional arguments:
  -h, --help            show this help message and exit
  --model_name {fnlp/moss-moon-003-sft,fnlp/moss-moon-003-sft-int8,fnlp/moss-moon-003-sft-int4}
  --ai_name AI_NAME     AI's name.
  --gpu GPU             GPU mask.
```
