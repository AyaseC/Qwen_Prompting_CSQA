# Qwen_Prompting_CSQA

This repository contains the scripts for evaluating Commonsense_qa dataset using Qwen Model.

The dataset is available [here](https://huggingface.co/datasets/tau/commonsense_qa)

The model is available [here](https://huggingface.co/Qwen/Qwen-7B)


## Setup & Use

Follow the instruction for setting up Qwen2, then call the following command in terminal:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --prompt cot --dataset "commonsense_qa" --model "Qwen/Qwen-7B" --batch_size 8 --icl_sample "commonsense_qa"
```

For running the evaluation.

Change cot for other types of prompting such as simple, icl or icl_reasoning for evaluation of other prompting method. For example:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --prompt simple --dataset "commonsense_qa" --model "Qwen/Qwen-7B" --batch_size 8 --icl_sample "commonsense_qa"
```
