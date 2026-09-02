import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from datasets import Dataset, DownloadManager

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "coqa_dataset"
COQA_URL = "https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-dev-v1.0.json"
COQA_SHA256 = "dfa367a9733ce53222918d0231d9b3bedc2b8ee831a2845f62dfc70701f2540a"


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare the CoQA development split.")
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Path to coqa-dev-v1.0.json (downloaded when omitted)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def resolve_input(input_path):
    if input_path is None:
        input_path = Path(DownloadManager().download(COQA_URL))

    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if digest != COQA_SHA256:
        raise ValueError(
            f"Unexpected SHA-256 checksum for {input_path}. Expected the official "
            "CoQA development file."
        )
    return input_path


def build_dataset(input_path):
    with input_path.open(encoding="utf-8") as infile:
        data = json.load(infile)["data"]

    prepared = {
        "story": [],
        "question": [],
        "answer": [],
        "additional_answers": [],
        "id": [],
    }

    for sample in tqdm(data):
        story = sample["story"]
        questions = sample["questions"]
        answers = sample["answers"]
        additional_answers = sample["additional_answers"]

        for question_index, question in enumerate(questions):
            prepared["story"].append(story)
            prepared["question"].append(question["input_text"])
            prepared["answer"].append(
                {
                    "text": answers[question_index]["input_text"],
                    "answer_start": answers[question_index]["span_start"],
                }
            )
            prepared["id"].append(f"{sample['id']}_{question_index}")
            prepared["additional_answers"].append(
                [
                    additional_answers[str(index)][question_index]["input_text"]
                    for index in range(3)
                ]
            )

            story += (
                " Q: "
                + question["input_text"]
                + " A: "
                + answers[question_index]["input_text"]
            )
            if not story.endswith("."):
                story += "."

    return Dataset.from_pandas(pd.DataFrame.from_dict(prepared))


def main():
    args = parse_args()
    input_path = resolve_input(args.input)
    build_dataset(input_path).save_to_disk(str(args.output))


if __name__ == "__main__":
    main()
