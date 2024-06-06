import os
import csv
import torch
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
from datasets import Dataset


basepath = ... # set accordingly

# open csv file
with open(basepath, 'r') as file:
    reader = csv.reader(file)
    data = list(reader)[1:]

tokenizer = AutoTokenizer.from_pretrained(f"facebook/opt-350m")

num_few_shot_examples = 5

few_shot_prompt = 'This is a bot that correctly answers questions. \n'
num_few_shot_examples -= 1
for sample in [data[61]] + data[:num_few_shot_examples]:
    question = sample[2].strip()
    answer = sample[3].strip()
    if answer[-1] != ".":
        answer += "."
    few_shot_prompt += 'Q: ' + question + ' A: ' + answer + ' '

dataset = []
for sample_id, sample in tqdm(enumerate(data[num_few_shot_examples:]), total=len(data[num_few_shot_examples:])):

    if sample_id == 61:
        continue  # "I have no comment" answer in context

    # sample: ['\ufeffType', 'Category', 'Question', 'Best Answer', 'Correct Answers', 'Incorrect Answers', 'Source']
    question = sample[2].strip()
    answer = sample[3].strip() + "." if sample[3].strip()[-1] != "." else sample[3].strip()
    
    additional_answers = sample[4].strip().strip(';')
    additional_answers = [s.strip() + '.' if s.strip()[-1] != '.' else s.strip() for s in additional_answers.split(';')]

    incorrect_answers = sample[5].strip().strip(';')
    incorrect_answers = [s.strip() + '.' if s.strip()[-1] != '.' else s.strip() for s in incorrect_answers.split(';')]

    inputs = tokenizer(few_shot_prompt + "Q: " + question + " A:", padding=False, truncation=False)

    sample = {"question": question,
              "input_ids": inputs.input_ids,
              "attention_mask": inputs.attention_mask,
              "answer": answer,
              "additional_answers": additional_answers,
              "incorrect_answers": incorrect_answers,
              }

    dataset.append(sample)

dataset = pd.DataFrame.from_dict(dataset)
dataset = Dataset.from_pandas(dataset)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)
dataset.save_to_disk(f'truthful_qa_dataset')
