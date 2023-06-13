# Moss Stream Chat

[English](https://github.com/GitEventhandler/MOSS-stream-chat/blob/master/README.md) | 中文指南

## 关于

本项目是[MOSS](https://github.com/OpenLMLab/MOSS)的一个streamlit前端，支持流式返回对话。如果想在Windows上运行本项目，可能需要使用wsl（目前triton包只支持linux）。  
![](https://github.com/GitEventhandler/MOSS-stream-chat/blob/master/screenshot/moss_stream_chat.gif)

## 环境要求

相比MOSS，本项目没有引入其它依赖。因此，如果运行过MOSS，使用它的环境可以直接运行本项目。

### 安装conda

具体安装过程请查阅[Anaconda的官方网站](https://www.anaconda.com/download/)。安装后应先调用conda init。

```shell
conda init
```

### 创建虚拟环境

```shell
conda create --name moss python=3.8
conda activate moss
```

### 安装PyTorch

#### Linux和windows

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

### 安装其他的Python包

```shell
pip install -r requirements.txt
```

## 使用方法

首先创建一个fnlp文件夹，然后将MOSS的模型文件放到fnlp文件夹下。
```shell
mkdir fnlp
```

直接用默认环境运行，请执行以下命令。

```shell
streamlit run web_demo.py
```

若想设置参数，请调用"web_demo.py **--** [params]".

```
streamlit run web_demo.py -- [-h] [--model_name {fnlp/moss-moon-003-sft,fnlp/moss-moon-003-sft-int8,fnlp/moss-moon-003-sft-int4}] [--ai_name AI_NAME] [--gpu GPU]

optional arguments:
  -h, --help            show this help message and exit
  --model_name {fnlp/moss-moon-003-sft,fnlp/moss-moon-003-sft-int8,fnlp/moss-moon-003-sft-int4}
  --ai_name AI_NAME     AI's name.
  --gpu GPU             GPU mask.
```
