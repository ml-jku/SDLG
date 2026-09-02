import argparse
import csv
import hashlib
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from datasets import Dataset, DownloadManager

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "truthful_qa_dataset"
FEW_SHOT_INDICES = (61, 0, 1, 2, 3)
EXPECTED_NUM_EXAMPLES = 817
TRUTHFUL_QA_REVISION = "fdd8ad1c0d00a478cf8b0bb41a3ad8378c16293b"
TRUTHFUL_QA_URL = (
    "https://raw.githubusercontent.com/sylinrl/TruthfulQA/"
    f"{TRUTHFUL_QA_REVISION}/TruthfulQA.csv"
)
TRUTHFUL_QA_SHA256 = "f9bd9e859cc102cb1f647f1064da7e009be752c416845cf9fa56e6eaae403a7d"


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare TruthfulQA for SDLG.")
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Path to TruthfulQA.csv (the paper revision is downloaded when omitted)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--tokenizer",
        default="facebook/opt-350m",
        help="Tokenizer identifier (default: facebook/opt-350m)",
    )
    return parser.parse_args()


def resolve_input(input_path):
    if input_path is None:
        input_path = Path(DownloadManager().download(TRUTHFUL_QA_URL))

    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if digest != TRUTHFUL_QA_SHA256:
        raise ValueError(
            f"Unexpected SHA-256 checksum for {input_path}. Expected the "
            f"TruthfulQA revision {TRUTHFUL_QA_REVISION}."
        )
    return input_path


def add_period(text):
    text = text.strip()
    return text if text.endswith(".") else text + "."


def split_answers(value):
    return [add_period(answer) for answer in value.strip().strip(";").split(";")]


def build_few_shot_prompt(data):
    prompt = "This is a bot that correctly answers questions. \n"
    for index in FEW_SHOT_INDICES:
        sample = data[index]
        prompt += "Q: " + sample[2].strip() + " A: " + add_period(sample[3]) + " "
    return prompt


def evaluation_samples(data):
    few_shot_indices = set(FEW_SHOT_INDICES)
    return (
        (sample_id, sample)
        for sample_id, sample in enumerate(data)
        if sample_id not in few_shot_indices
    )


def prepare_dataset(input_path, tokenizer):
    with input_path.open(encoding="utf-8") as file:
        data = list(csv.reader(file))[1:]
    if len(data) != EXPECTED_NUM_EXAMPLES:
        raise ValueError(
            f"Expected the {EXPECTED_NUM_EXAMPLES}-question pre-2025 TruthfulQA "
            f"revision, but found {len(data)} rows."
        )

    few_shot_prompt = build_few_shot_prompt(data)
    prepared = []

    for _, sample in tqdm(
        evaluation_samples(data), total=len(data) - len(FEW_SHOT_INDICES)
    ):
        question = sample[2].strip()
        inputs = tokenizer(
            few_shot_prompt + "Q: " + question + " A:",
            padding=False,
            truncation=False,
        )
        prepared.append(
            {
                "question": question,
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "answer": add_period(sample[3]),
                "additional_answers": split_answers(sample[4]),
                "incorrect_answers": split_answers(sample[5]),
            }
        )

    dataset = Dataset.from_pandas(pd.DataFrame.from_dict(prepared))
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"],
        output_all_columns=True,
    )
    return dataset


def main():
    args = parse_args()
    input_path = resolve_input(args.input)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    prepare_dataset(input_path, tokenizer).save_to_disk(str(args.output))


if __name__ == "__main__":
    main()
