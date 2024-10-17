import os
from typing import List
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers.trainer_utils import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from datasets import load_dataset
from torch.amp import autocast
from collections import Counter


def load_models_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        pad_token='<|extra_0|>',
        eos_token='<|endoftext|>',
        padding_side='left',
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        path,
        pad_token_id=tokenizer.pad_token_id,
        trust_remote_code=True
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        path,
        pad_token_id=tokenizer.pad_token_id,
        trust_remote_code=True
    )
    return model, tokenizer

model,tokenizer=load_models_tokenizer("Qwen/Qwen-7B")

dataset = load_dataset("commonsense_qa",split="validation").to_pandas()
sample = load_dataset("commonsense_qa",split="train").to_pandas()

def simple_prompt(question, concept, choices):
    labels = choices["label"]
    texts = choices["text"]
    choice_strings = [f"{label}: {text}" for label, text in zip(labels, texts)]
    return f"Instruction: This is a multiple choice question and you are required to answer with the letter corresponding to the correct choice.\nQuestion: {question}\nConcept: {concept}\nChoices: {', '.join(choice_strings)}\nYour Answer: "

def cot_prompt(question, concept, choices):
    labels = choices["label"]
    texts = choices["text"]
    choice_strings = [f"{label}: {text}" for label, text in zip(labels, texts)]
    return (
        "Instruction: This is a multiple choice question and you are required to answer with the letter corresponding to the correct choice.\n"
        f"Question: {question}\nConcept: {concept}\nChoices: {', '.join(choice_strings)}\n"
        "Let's think step by step. Please provide your answer followed by your thinking steps."
        "Your Answer:  "
    )

def icl_prompt(examples, concept, question, choices):
    # Prepare a string for examples
    example_strings = []
    for example in examples:
        q = example['question']
        c = example['choices']
        ct = example['question_concept']
        answer = example['answerKey']
        choice_strings = [f"{label}: {text}" for label, text in zip(c['label'], c['text'])]
        example_strings.append(f"Example Question: {q}\nConcept: {ct}\nChoices: {', '.join(choice_strings)}\nAnswer: {answer}")

    # Format the ICL prompt
    example_prompt = "\n\n".join(example_strings)  # Separate examples with a double newline
    labels = choices["label"]
    texts = choices["text"]
    choice_strings = [f"{label}: {text}" for label, text in zip(labels, texts)]

    # Include the new question at the end
    return f"Instruction: This is a multiple choice question and you are required to answer with the letter corresponding to the correct choice.\nThere are a few examples of Questions:\n{example_prompt}\n\nNow, let's solve this one:\nQuestion: {question}\nConcept: {concept}\nChoices: {', '.join(choice_strings)}\nYour Answer: "



def icl_reasoning_prompt(question, concept, choices):
    labels = choices["label"]
    texts = choices["text"]
    choice_strings = [f"{label}: {text}" for label, text in zip(labels, texts)]
    
    return (
        "Instruction: This is a multiple choice question. Please answer with the letter corresponding to the correct choice.\n\n"
        
        "Here are some examples of questions with their correct answers and reasoning:\n\n"
        
        "{\n"
        "Example Question 1: The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?\n"
        "Concept: punishing\n"
        "Choices: { A: Ignore, B: Enforce, C: Authoritarian, D: Yell at, E: Avoid }\n" 
        "Answer: A\n"
        "Reasoning:\n"
        "- The phrase 'punishing blow' suggests that the sanctions had a negative and harsh impact on the school. "
        "- The word 'seemed to' implies that the sanctions disregarded or did not acknowledge the school's efforts to change. "
        "- 'A: Ignore' is correct as it means to disregard, aligning with the context of the sanctions.\n"
        "- 'B: Enforce' is incorrect because 'enforce' implies supporting or applying something rigorously, which does not align with the idea of ignoring efforts. \n"
        "- 'C: Authoritarian' refers to a strict and controlling government style, which doesn't fit the context of the school's efforts being ignored.\n"
        "- 'D: Yell' at suggests verbal aggression, which is not related to the broader impact of sanctions.\n"
        "- 'E: Avoid' is wrong because avoiding is about evading something, not disregarding efforts.\n"
        "}\n\n"
        
        "{\n"
        "Example Question 2: Sammy wanted to go to where the people were. Where might he go?\n"
        "Concept: people\n"
        "Choices: { A: Race track, B: Populated areas, C: The desert, D: Apartment, E: Roadblock }\n" 
        "Answer: B\n"
        "Reasoning:\n"
        "- The phrase 'where the people were' indicates Sammy wants to go to a location that has many people. "
        "- 'B: Populated areas' refers to places that have a high density of people living or gathering, which directly aligns with Sammy’s intent.\n"
        "- 'A: Race track' may have a crowd, but only during events. It is not consistently populated.\n"
        "- 'C: The desert' is incorrect because it is a sparsely populated area by nature.\n"
        "- 'D: Apartment' refers to an individual living space, which may only have a few residents.\n"
        "- 'E: Roadblock' typically refers to a barrier on a road, which is not associated with large gatherings of people.\n"
        "}\n\n" 

        "{\n"
        "Example Question 3: Google Maps and other highway and street GPS services have replaced what?\n"
        "Concept: highway\n"
        "Choices: { A: United States, B: Mexico, C: Countryside, D: Atlas, E: Oceans }\n" 
        "Answer: D\n"
        "Reasoning:\n"
        "- Google Maps and other GPS services are used for navigating highways and streets.\n"
        "- 'Atlas' refers to a collection of physical maps, which were used for navigation before digital GPS systems.\n"
        "- 'D: Atlas' is the correct answer because GPS services have replaced the need for traditional map books and atlases.\n"
        "- 'A: United States' and 'B: Mexico' are countries, not tools for navigation.\n"
        "- 'C: Countryside' is a type of rural area and is unrelated to navigation systems.\n"
        "- 'E: Oceans' are large bodies of water, not related to tools that provide maps or navigation.\n"
        "}\n\n"
        
        f"Now, let's solve this question:\n"
        f"Question: {question}\n"
        f"Concept: {concept}\n"
        f"Choices: {', '.join(choice_strings)}\n"
        "Please provide your answer followed by your reasoning as above.\n"
        "Your Answer: "
    )




model.eval()


def get_answer_letter(decoded_output):
    # Find the position of "Your Answer:" in the generated output
    start_pos = decoded_output.lower().find("your answer:")

    # Check if "Your Answer:" exists in the output
    if start_pos != -1:
        # Extract the letter right after "Your Answer:"
        answer_part = decoded_output[start_pos + len("Your Answer:"):].strip()  # Take the first non-space character
        for char in answer_part:
            if char.upper() in ['A', 'B', 'C', 'D', 'E']:
                return char.upper()
            
    return "Answer not found"

def get_demo(dataset):
    shuffled_dataset = dataset.sample(frac=1)  # Shuffle
    examples = shuffled_dataset.head(3).to_dict(orient='records')
    return examples




def evaluate_model(dataset, sample, model, tokenizer, batch_size=8):
    """Evaluate the model using different prompting strategies with batching and autocast."""
    results = []
    examples = get_demo(sample) # Get examples for ICL outside the loop
    # device = next(model.parameters()).device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy_results = { "simple": 0, "cot": 0, "icl": 0, "icl_reasoning": 0 }
    total_counts = { "simple": 0, "cot": 0, "icl": 0, "icl_reasoning": 0 }

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for evaluation.")
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    # Create a progress bar with total dataset size
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        # Get batch data
        batch_data = dataset.iloc[i:i + batch_size]
        
        
        # Prepare results for each prompt type in the batch
        batch_results = []

        # Process each row in the batch
        for index, data in batch_data.iterrows():
            question = data['question']
            concept = data ['question_concept']
            choices = data['choices']
            answer_key = data['answerKey']

            # Prepare and process each prompt individually
            for prompt_key in ["simple", "cot", "icl", "icl_reasoning"]:
                if prompt_key == "simple":
                    input_text = simple_prompt(question, concept, choices)
                elif prompt_key == "cot":
                    input_text = cot_prompt(question, concept, choices)
                elif prompt_key == "icl":
                    input_text = icl_prompt(examples, question, concept, choices)
                else:
                    input_text = icl_reasoning_prompt(question, concept, choices)

                # Tokenize the input
                encoded_input = tokenizer(
                    input_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=3000
                ).to(device)

                # Generate output with autocast
                with torch.amp.autocast('cuda'):
                    output = model.module.generate(
                        encoded_input.input_ids,
                        attention_mask=encoded_input.attention_mask,
                        max_new_tokens=150,
                        temperature=0.2,
                        do_sample=False
                    )

                # Decode and process the output
                print(f"Full_Answer {prompt_key.capitalize()}: {(tokenizer.decode(output[0], skip_special_tokens=True))}")
                answer = get_answer_letter(tokenizer.decode(output[0], skip_special_tokens=True))
                correct = answer == answer_key

                # Store results for the current prompt
                batch_results.append({
                    'question': question,
                    'answer_key': answer_key,
                    f'{prompt_key}_answer': answer,
                    f'{prompt_key}_correct': correct
                })

                # Update accuracy counts
                total_counts[prompt_key] += 1
                if correct:
                    accuracy_results[prompt_key] += 1

                # Real-time output
                print(f"Question: {question}")
                print(f"Choices: {choices['text']}")
                print(f"Answer Key: {answer_key}")
                print(f"{prompt_key.capitalize()} Answer: {answer} (Correct: {correct})")
                print("-" * 50)

        results.extend(batch_results)

        # Calculate and print real-time accuracy after the batch
        for prompt_key in ["simple", "cot", "icl", "icl_reasoning"]:
            batch_accuracy = (accuracy_results[prompt_key] / total_counts[prompt_key]) * 100 if total_counts[prompt_key] > 0 else 0
            print(f"Real-Time {prompt_key.capitalize()} Accuracy: {batch_accuracy:.2f}%")
    
    

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Calculate and print final accuracy
    for prompt_key in ["simple", "cot", "icl", "icl_reasoning"]:
        final_accuracy = results_df[f'{prompt_key}_correct'].mean()
        accuracy_results[prompt_key] = final_accuracy * 100
        print(f"{prompt_key.capitalize()} Prompt Accuracy: {final_accuracy * 100:.2f}%")

    # Save results to CSV
    results_df.to_csv("evaluation_results_update.csv", index=False)
    print("Results saved to 'evaluation_results.csv'.")

    # Save accuracy results to a separate CSV file
    accuracy_df = pd.DataFrame(list(accuracy_results.items()), columns=['Prompt Type', 'Accuracy'])
    accuracy_df.to_csv("accuracy_numbers_update.csv", index=False)
    print("Accuracy results saved to 'accuracy_numbers.csv'.")
# Usage


def evaluate_model_majority(dataset, model, tokenizer, batch_size=8, num_samples=5):
    """Evaluate the model using different prompting strategies with batching and autocast, incorporating majority voting."""
    results = []
    examples = get_demo(dataset)  # Get examples for ICL outside the loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy_results = {"simple": 0, "cot": 0, "icl": 0, "icl_reasoning": 0}
    total_counts = {"simple": 0, "cot": 0, "icl": 0, "icl_reasoning": 0}

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for evaluation.")
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    # Create a progress bar with total dataset size
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        # Get batch data
        batch_data = dataset.iloc[i:i + batch_size]
        
        # Prepare results for each prompt type in the batch
        batch_results = []

        # Process each row in the batch
        for index, data in batch_data.iterrows():
            question = data['question']
            concept = data['question_concept']
            choices = data['choices']
            answer_key = data['answerKey']

            # Prepare and process each prompt individually
            for prompt_key in ["simple", "cot", "icl", "icl_reasoning"]:
                # Store all generated answers for majority voting
                all_answers = []

                for _ in range(num_samples):
                    if prompt_key == "simple":
                        input_text = simple_prompt(question, concept, choices)
                    elif prompt_key == "cot":
                        input_text = cot_prompt(question, concept, choices)
                    elif prompt_key == "icl":
                        input_text = icl_prompt(examples, question, concept, choices)
                    else:
                        input_text = icl_reasoning_prompt(question, concept, choices)

                    # Tokenize the input
                    encoded_input = tokenizer(
                        input_text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=3000
                    ).to(device)

                    # Generate output with autocast
                    with torch.amp.autocast('cuda'):
                        output = model.module.generate(
                            encoded_input.input_ids,
                            attention_mask=encoded_input.attention_mask,
                            max_new_tokens=50,
                            temperature=0.5,
                            do_sample=True,
                            top_k=50
                        )

                    # Decode and store the output
                    answer = get_answer_letter(tokenizer.decode(output[0], skip_special_tokens=True))
                    all_answers.append(answer)

                # Apply majority voting to determine the final answer
                most_common_answer, count = Counter(all_answers).most_common(1)[0]
                correct = most_common_answer == answer_key

                # Store results for the current prompt
                batch_results.append({
                    'question': question,
                    'answer_key': answer_key,
                    f'{prompt_key}_answer': most_common_answer,
                    f'{prompt_key}_correct': correct
                })

                # Update accuracy counts
                total_counts[prompt_key] += 1
                if correct:
                    accuracy_results[prompt_key] += 1

                # Real-time output
                print(f"Question: {question}")
                print(f"Choices: {choices['text']}")
                print(f"Answer Key: {answer_key}")
                print(f"{prompt_key.capitalize()} Answer: {most_common_answer} (Correct: {correct})")
                print("-" * 50)

        results.extend(batch_results)

        # Calculate and print real-time accuracy after the batch
        for prompt_key in ["simple", "cot", "icl", "icl_reasoning"]:
            batch_accuracy = (accuracy_results[prompt_key] / total_counts[prompt_key]) * 100 if total_counts[prompt_key] > 0 else 0
            print(f"Real-Time {prompt_key.capitalize()} Accuracy: {batch_accuracy:.2f}%")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Calculate and print final accuracy
    for prompt_key in ["simple", "cot", "icl", "icl_reasoning"]:
        final_accuracy = results_df[f'{prompt_key}_correct'].mean()
        accuracy_results[prompt_key] = final_accuracy * 100
        print(f"{prompt_key.capitalize()} Prompt Accuracy: {final_accuracy * 100:.2f}%")

    # Save results to CSV
    results_df.to_csv("evaluation_results_major.csv", index=False)
    print("Results saved to 'evaluation_results_major.csv'.")

    # Save accuracy results to a separate CSV file
    accuracy_df = pd.DataFrame(list(accuracy_results.items()), columns=['Prompt Type', 'Accuracy'])
    accuracy_df.to_csv("accuracy_numbers_major.csv", index=False)
    print("Accuracy results saved to 'accuracy_numbers_major.csv'.")


# Usage
# evaluate_model(dataset, sample, model, tokenizer)



def evaluate_simple_prompt(dataset, model, tokenizer, batch_size=8):
    """Evaluate the model using the simple prompt."""
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy = 0
    total = 0
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for evaluation.")
        model = torch.nn.DataParallel(model)

    model = model.to(device)
    model.eval()

    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating Simple Prompt"):
        batch_data = dataset.iloc[i:i + batch_size]

        for index, data in batch_data.iterrows():
            question = data['question']
            concept = data['question_concept']
            choices = data['choices']
            answer_key = data['answerKey']

            input_text = simple_prompt(question, concept, choices)
            
            encoded_input = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=3000
            ).to(device)

            with torch.amp.autocast('cuda'):
                output = model.module.generate(
                    encoded_input.input_ids,
                    attention_mask=encoded_input.attention_mask,
                    max_new_tokens=150,
                    temperature=0.2,
                    do_sample=False
                )

            answer = get_answer_letter(tokenizer.decode(output[0], skip_special_tokens=True))
            correct = answer == answer_key
            results.append({
                'question': question,
                'answer_key': answer_key,
                'simple_answer': answer,
                'simple_correct': correct
            })
            
            total += 1
            if correct:
                accuracy += 1

        print(f"Real-Time Simple Accuracy: {(accuracy / total) * 100:.2f}%")

    results_df = pd.DataFrame(results)
    final_accuracy = accuracy / total * 100
    print(f"Final Simple Prompt Accuracy: {final_accuracy:.2f}%")
    results_df.to_csv("simple_prompt_results.csv", index=False)
    return final_accuracy


def evaluate_cot_prompt(dataset, model, tokenizer, batch_size=8):
    """Evaluate the model using the chain-of-thought prompt."""
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy = 0
    total = 0

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for evaluation.")
        model = torch.nn.DataParallel(model)


    model = model.to(device)
    model.eval()

    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating CoT Prompt"):
        batch_data = dataset.iloc[i:i + batch_size]

        for index, data in batch_data.iterrows():
            question = data['question']
            concept = data['question_concept']
            choices = data['choices']
            answer_key = data['answerKey']

            input_text = cot_prompt(question, concept, choices)

            encoded_input = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=3000
            ).to(device)

            with torch.amp.autocast('cuda'):
                output = model.module.generate(
                    encoded_input.input_ids,
                    attention_mask=encoded_input.attention_mask,
                    max_new_tokens=150,
                    temperature=0.2,
                    do_sample=False
                )

            answer = get_answer_letter(tokenizer.decode(output[0], skip_special_tokens=True))
            correct = answer == answer_key
            results.append({
                'question': question,
                'answer_key': answer_key,
                'cot_answer': answer,
                'cot_correct': correct
            })
            
            total += 1
            if correct:
                accuracy += 1

        print(f"Real-Time CoT Accuracy: {(accuracy / total) * 100:.2f}%")

    results_df = pd.DataFrame(results)
    final_accuracy = accuracy / total * 100
    print(f"Final CoT Prompt Accuracy: {final_accuracy:.2f}%")
    results_df.to_csv("cot_prompt_results.csv", index=False)
    return final_accuracy

def evaluate_icl_prompt(dataset, sample, model, tokenizer, batch_size=8):
    """Evaluate the model using the in-context learning prompt."""
    results = []
    examples = get_demo(sample)  # Get examples for ICL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy = 0
    total = 0

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for evaluation.")
        model = torch.nn.DataParallel(model)


    model = model.to(device)
    model.eval()

    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating ICL Prompt"):
        batch_data = dataset.iloc[i:i + batch_size]

        for index, data in batch_data.iterrows():
            question = data['question']
            concept = data['question_concept']
            choices = data['choices']
            answer_key = data['answerKey']

            input_text = icl_prompt(examples, question, concept, choices)

            encoded_input = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=3000
            ).to(device)

            with torch.amp.autocast('cuda'):
                output = model.module.generate(
                    encoded_input.input_ids,
                    attention_mask=encoded_input.attention_mask,
                    max_new_tokens=150,
                    temperature=0.2,
                    do_sample=False
                )

            answer = get_answer_letter(tokenizer.decode(output[0], skip_special_tokens=True))
            correct = answer == answer_key
            results.append({
                'question': question,
                'answer_key': answer_key,
                'icl_answer': answer,
                'icl_correct': correct
            })
            
            total += 1
            if correct:
                accuracy += 1

        print(f"Real-Time ICL Accuracy: {(accuracy / total) * 100:.2f}%")

    results_df = pd.DataFrame(results)
    final_accuracy = accuracy / total * 100
    print(f"Final ICL Prompt Accuracy: {final_accuracy:.2f}%")
    results_df.to_csv("icl_prompt_results.csv", index=False)
    return final_accuracy



def evaluate_icl_reasoning_prompt(dataset, model, tokenizer, batch_size=8):
    """Evaluate the model using the in-context learning prompt."""
    results = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy = 0
    total = 0
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for evaluation.")
        model = torch.nn.DataParallel(model)


    model = model.to(device)
    model.eval()

    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating ICL Prompt"):
        batch_data = dataset.iloc[i:i + batch_size]

        for index, data in batch_data.iterrows():
            question = data['question']
            concept = data['question_concept']
            choices = data['choices']
            answer_key = data['answerKey']

            input_text = icl_reasoning_prompt(question, concept, choices)

            encoded_input = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=3000
            ).to(device)

            with torch.amp.autocast('cuda'):
                output = model.module.generate(
                    encoded_input.input_ids,
                    attention_mask=encoded_input.attention_mask,
                    max_new_tokens=150,
                    temperature=0.2,
                    do_sample=False
                )

            answer = get_answer_letter(tokenizer.decode(output[0], skip_special_tokens=True))
            correct = answer == answer_key
            results.append({
                'question': question,
                'answer_key': answer_key,
                'icl_answer': answer,
                'icl_correct': correct
            })
            
            total += 1
            if correct:
                accuracy += 1

        print(f"Real-Time ICL Accuracy: {(accuracy / total) * 100:.2f}%")

    results_df = pd.DataFrame(results)
    final_accuracy = accuracy / total * 100
    print(f"Final ICL Reasoning Prompt Accuracy: {final_accuracy:.2f}%")
    results_df.to_csv("icl_reasoning_prompt_results1.csv", index=False)
    return final_accuracy

