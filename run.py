import argparse
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

# Import your evaluation functions
from Qwen import get_demo, load_models_tokenizer, evaluate_simple_prompt, evaluate_cot_prompt, evaluate_icl_prompt, evaluate_icl_reasoning_prompt

# Your model loading and tokenizer initialization here
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Evaluate model on CommonsenseQA")
    parser.add_argument('--prompt', choices=['simple', 'cot', 'icl', 'icl_reasoning'], required=True, 
                        help="Specify the type of prompt to use: 'simple', 'cot', 'icl', 'icl_reasoning'")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for evaluation")
    parser.add_argument('--dataset', type=str, required=True, help="Path to your dataset CSV")
    parser.add_argument('--model', type=str, required=True, help="Name or path of the model & tokenizer")
    parser.add_argument('--icl_sample', type=str, help="Path to sample for ICL prompts (if needed)")

    args = parser.parse_args()

    # Load the dataset
    dataset = load_dataset(args.dataset,split="validation").to_pandas()
    train = load_dataset(args.icl_sample,split="train").to_pandas()

    # Load the model and tokenizer
    model,tokenizer=load_models_tokenizer(args.model)

    # Call the appropriate function based on the command-line argument
    if args.prompt == 'simple':
        accuracy = evaluate_simple_prompt(dataset, model, tokenizer, args.batch_size)
    elif args.prompt == 'cot':
        accuracy = evaluate_cot_prompt(dataset, model, tokenizer, args.batch_size)
    elif args.prompt == 'icl':
        if not args.icl_sample:
            raise ValueError("You must provide --icl_sample for ICL prompt evaluation")

        accuracy = evaluate_icl_prompt(dataset, train, model, tokenizer, args.batch_size)
    else:
        accuracy = evaluate_icl_reasoning_prompt(dataset, model, tokenizer, args.batch_size)

    print(f"Final Accuracy for {args.prompt} prompt: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
