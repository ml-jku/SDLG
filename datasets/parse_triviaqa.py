import argparse
from pathlib import Path

from transformers import AutoTokenizer

import datasets

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "trivia_qa_dataset"
NUM_FEW_SHOT_EXAMPLES = 10
TRIVIA_QA_REVISION = "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f"


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare TriviaQA for SDLG.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation"),
        default="validation",
        help="TriviaQA split to prepare (default: validation)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optionally retain only the first N examples.",
    )
    parser.add_argument(
        "--tokenizer",
        default="facebook/opt-350m",
        help="Tokenizer identifier (default: facebook/opt-350m)",
    )
    return parser.parse_args()


def build_few_shot_prompt(train_data, num_examples=NUM_FEW_SHOT_EXAMPLES):
    prompt = "This is a bot that correctly answers questions. \n"
    for sample in train_data.select(range(num_examples)):
        prompt += "Q: " + sample["question"] + " A: " + sample["answer"]["value"] + " "
    return prompt


def prepare_dataset(split, tokenizer, max_examples=None):
    data = datasets.load_dataset(
        "trivia_qa",
        "rc.nocontext",
        split=split,
        revision=TRIVIA_QA_REVISION,
    )
    train_data = datasets.load_dataset(
        "trivia_qa",
        "rc.nocontext",
        split="train",
        revision=TRIVIA_QA_REVISION,
    )
    few_shot_prompt = build_few_shot_prompt(train_data)

    # Do not evaluate the examples included in a training-split prompt.
    if split == "train":
        data = data.select(range(NUM_FEW_SHOT_EXAMPLES, len(data)))

    if max_examples is not None:
        if max_examples < 1:
            raise ValueError("max_examples must be positive")
        data = data.select(range(min(max_examples, len(data))))

    def process_data_to_model_inputs(batch):
        answers = [answer["value"] for answer in batch["answer"]]
        prompted_questions = [
            few_shot_prompt + "Q: " + question + " A:" for question in batch["question"]
        ]
        inputs = tokenizer(prompted_questions, padding=False, truncation=False)
        outputs = tokenizer(answers, padding=False, truncation=False)

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["decoder_input_ids"] = outputs.input_ids
        batch["decoder_attention_mask"] = outputs.attention_mask
        batch["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in output_ids]
            for output_ids in outputs.input_ids
        ]
        batch["answer"] = answers
        return batch

    data = data.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=1,
        remove_columns=["search_results", "question_source", "entity_pages"],
    )
    data.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
            "decoder_input_ids",
            "decoder_attention_mask",
            "labels",
        ],
        output_all_columns=True,
    )
    return data


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    dataset = prepare_dataset(args.split, tokenizer, args.max_examples)
    dataset.save_to_disk(str(args.output))


if __name__ == "__main__":
    main()
