import os
import json
import pandas as pd
from datasets import Dataset
from tqdm import tqdm


basepath = ... # set accordingly

with open(basepath, 'r') as infile:
    data = json.load(infile)['data']

dataset = {'story': [],
        'question': [],
        'answer': [],
        'additional_answers': [],
        'id': []}

for sample_id, sample in tqdm(enumerate(data), total=len(data)):
    story = sample['story']
    questions = sample['questions']
    answers = sample['answers']
    additional_answers = sample['additional_answers']

    for question_index, question in enumerate(questions):
        dataset['story'].append(story)
        dataset['question'].append(question['input_text'])
        dataset['answer'].append({
            'text': answers[question_index]['input_text'],
            'answer_start': answers[question_index]['span_start']
        })
        dataset['id'].append(sample['id'] + '_' + str(question_index))
        additional_answers_list = []

        for i in range(3):
            additional_answers_list.append(additional_answers[str(i)][question_index]['input_text'])

        dataset['additional_answers'].append(additional_answers_list)
        story = story + ' Q: ' + question['input_text'] + ' A: ' + answers[question_index]['input_text']
        if not story[-1] == '.':
            story = story + '.'
        all_answers = [answers[question_index]['input_text']] + additional_answers_list

dataset_df = pd.DataFrame.from_dict(dataset)

dataset = Dataset.from_pandas(dataset_df)

dataset.save_to_disk(f'coqa_dataset')
