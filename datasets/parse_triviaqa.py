import os
os.environ["HF_HOME"] = ...                     # set accordingly
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # set accordingly

import pathlib
import datasets
import torch
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained(f"facebook/opt-350m")

if not pathlib.Path(f'trivia_qa_dataset').exists():

    print('Preprocessing dataset')
    val_data = datasets.load_dataset("trivia_qa", "rc.nocontext", split="validation")
    train_data = datasets.load_dataset("trivia_qa", "rc.nocontext", split="train")
    data_for_few_shot_prompt = train_data.select(range(0, 10))

    few_shot_prompt = 'This is a bot that correctly answers questions. \n'
    for sample in data_for_few_shot_prompt:
        few_shot_prompt += 'Q: ' + sample['question'] + ' A: ' + sample['answer']['value'] + ' '

    batch_size = 1

    def process_data_to_model_inputs(batch):
        # tokenize the inputs and labels
        answers = [answer["value"] for answer in batch["answer"]]

        batch_with_prompt = [few_shot_prompt + "Q: " + question + " A:" for question in batch["question"]]
        inputs = tokenizer(batch_with_prompt, padding=False, truncation=False)
        outputs = tokenizer(answers, padding=False, truncation=False)

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["decoder_input_ids"] = outputs.input_ids
        batch["decoder_attention_mask"] = outputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()
        batch['answer'] = answers

        # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
        # We have to make sure that the PAD token is ignored
        batch["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
        ]

        return batch

    val_data = val_data.map(process_data_to_model_inputs,
                            batched=True,
                            batch_size=batch_size,
                            remove_columns=["search_results", "question_source", "entity_pages"])
    val_data.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
        output_all_columns=True)

    val_data.save_to_disk(f'trivia_qa_dataset')
else:
    val_data = datasets.load_from_disk(f'trivia_qa_dataset')
