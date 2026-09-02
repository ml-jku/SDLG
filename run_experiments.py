import os

import evaluate
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
from args import Args
from sdlg import generate_semantically_diverse_output_sequences
from utils import (
    atomic_pickle_dump,
    compute_correctness,
    compute_likelihood,
    compute_semantic_pairs_for_results,
    discard_likelihood_logits,
    generate_text,
    get_models_and_tokenizers,
    seed_everything,
)

CUDA_ID_LLM = int(os.environ.get("SDLG_CUDA_ID_LLM", "0"))
CUDA_ID_DEBERTA = int(os.environ.get("SDLG_CUDA_ID_DEBERTA", str(CUDA_ID_LLM)))
DATASET_DIRECTORIES = {
    "coqa": os.path.join("datasets", "coqa_dataset"),
    "trivia_qa": os.path.join("datasets", "trivia_qa_dataset"),
    "truthful_qa": os.path.join("datasets", "truthful_qa_dataset"),
}
DATASET_PREPARATION_SCRIPTS = {
    "coqa": "python datasets/parse_coqa.py",
    "trivia_qa": "python datasets/parse_triviaqa.py",
    "truthful_qa": "python datasets/parse_truthful_qa.py",
}


class ConfigLoader(yaml.SafeLoader):
    """Safe YAML loader that accepts tuple values written by the initial release."""


ConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: loader.construct_sequence(node),
)


def encode(examples, tokenizer):
    return tokenizer(
        examples["story"] + " Q: " + examples["question"] + " A:",
        truncation=False,
        padding=False,
    )


def encode_and_format_dataset(dataset, tokenizer):
    dataset = dataset.map(
        encode,
        batched=False,
        fn_kwargs={"tokenizer": tokenizer},
        load_from_cache_file=False,
    )
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"],
        output_all_columns=True,
    )
    return dataset


def get_results(
    args,
    base_path,
    llm_model,
    tokenizer,
    device_llm,
    deberta_model,
    deberta_tokenizer,
    device_deberta,
    dataset,
):
    rouge = evaluate.load("rouge")
    exact_match_metric = evaluate.load("exact_match")
    bleurt = evaluate.load("bleurt")

    deberta_embeddings = deberta_model.deberta.embeddings.word_embeddings(
        torch.arange(deberta_tokenizer.vocab_size, device=device_deberta)
    ).detach()

    if args.dataset == "coqa":
        id_to_question_mapping = dict(zip(dataset["id"], dataset["question"]))

    dataloader = DataLoader(dataset, batch_size=1)
    for batch_index, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        result_path = os.path.join(base_path, f"results_dict_{batch_index}.pkl")
        if os.path.exists(result_path) and not args.overwrite_existing:
            continue

        prompt = batch["input_ids"][0].to("cpu")

        if args.dataset == "coqa":
            question = id_to_question_mapping[batch["id"][0]]
        else:
            question = batch["question"][0]

        results_dict = {
            "input_ids": batch["input_ids"],
            "question": question,
            "correctness_dict": {},
            "sdlg": {"generations": [], "likelihoods": []},
            "baseline": {"generations": [], "likelihoods": []},
        }

        # (1) Most-likely output sequence
        most_likely_generation = generate_text(
            args=args,
            model=llm_model,
            tokenizer=tokenizer,
            input_ids=batch["input_ids"],
            len_prompt=len(prompt),
            decoding_method="most_likely",
            device=device_llm,
        )

        if args.dataset == "coqa":
            reference_answers = batch["answer"]["text"] + [
                answer[0] for answer in batch["additional_answers"]
            ]
            incorrect_answers = []
        elif args.dataset == "trivia_qa":
            reference_answers = batch["answer"]
            incorrect_answers = []
        elif args.dataset == "truthful_qa":
            reference_answers = batch["answer"] + [
                answer[0] if answer[0][-1] == "." else answer[0] + "."
                for answer in batch["additional_answers"]
            ]
            if "I have no comment." not in reference_answers:
                reference_answers.append("I have no comment.")
            incorrect_answers = [
                answer[0] if answer[0][-1] == "." else answer[0] + "."
                for answer in batch["incorrect_answers"]
            ]
        else:
            raise ValueError(f"dataset {args.dataset!r} not implemented")

        results_dict["correctness_dict"] = compute_correctness(
            args=args,
            reference_answers=reference_answers,
            incorrect_answers=incorrect_answers,
            most_likely_generation_text=most_likely_generation["generation_text"][0],
            exact_match_metric=exact_match_metric,
            rouge=rouge,
            bleurt=bleurt,
        )

        # SDLG requires the initial sequence logits while ranking substitutions.
        most_likely_likelihood = compute_likelihood(
            prompt=prompt,
            generation=most_likely_generation,
            model=llm_model,
            device=device_llm,
            compute_cleaned=args.compute_cleaned,
            store_logits=True,
        )

        # (2.1) SDLG
        results_dict["sdlg"]["generations"].append(most_likely_generation)
        results_dict["sdlg"]["likelihoods"].append(most_likely_likelihood)
        results_dict = generate_semantically_diverse_output_sequences(
            results_dict=results_dict,
            deberta_model=deberta_model,
            deberta_tokenizer=deberta_tokenizer,
            device_deberta=device_deberta,
            deberta_embeddings=deberta_embeddings,
            model=llm_model,
            tokenizer=tokenizer,
            device_llm=device_llm,
            input_ids=batch["input_ids"],
            prompt=prompt,
            question=question,
            initial_generation=most_likely_generation,
            initial_likelihood=most_likely_likelihood,
            args=args,
        )

        stored_most_likely_likelihood = most_likely_likelihood
        if not args.store_logits:
            stored_most_likely_likelihood = discard_likelihood_logits(
                most_likely_likelihood
            )
            if results_dict["sdlg"]["likelihoods"]:
                results_dict["sdlg"]["likelihoods"][0] = stored_most_likely_likelihood

        # (2.2) Multinomial sampling by default; the existing beam-search
        # parameters in Args can instead configure diverse beam search.
        results_dict["baseline"]["generations"].append(most_likely_generation)
        results_dict["baseline"]["likelihoods"].append(stored_most_likely_likelihood)

        num_baseline_calls = (
            args.num_total_generations - 1
        ) // args.num_return_sequences_baseline
        for _ in range(num_baseline_calls):
            baseline_generation = generate_text(
                args=args,
                model=llm_model,
                tokenizer=tokenizer,
                input_ids=batch["input_ids"],
                len_prompt=len(prompt),
                decoding_method="baseline",
                device=device_llm,
            )

            results_dict["baseline"]["generations"].append(baseline_generation)
            results_dict["baseline"]["likelihoods"].append(
                compute_likelihood(
                    prompt=prompt,
                    generation=baseline_generation,
                    model=llm_model,
                    device=device_llm,
                    compute_cleaned=args.compute_cleaned,
                    store_logits=args.store_logits,
                )
            )

        atomic_pickle_dump(results_dict, result_path)


def select_device(cuda_id):
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return f"cuda:{cuda_id}"
    return "cpu"


def get_dataset_directory(dataset_name):
    try:
        dataset_directory = DATASET_DIRECTORIES[dataset_name]
    except KeyError as error:
        raise ValueError(f"dataset {dataset_name!r} not implemented") from error

    if not os.path.isdir(dataset_directory):
        command = DATASET_PREPARATION_SCRIPTS[dataset_name]
        raise FileNotFoundError(
            f"Dataset {dataset_name!r} has not been prepared. Run `{command}` "
            "from the repository root."
        )
    return dataset_directory


def main():
    args = Args()
    args.validate()
    dataset_directory = get_dataset_directory(args.dataset)

    base_path = os.path.join("results", args.run_id)
    os.makedirs(base_path, exist_ok=True)

    config_path = os.path.join(base_path, "config.yaml")
    current_config = args.experiment_config()
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as file:
            existing_args = yaml.load(file, Loader=ConfigLoader) or {}

        if existing_args != current_config:
            differing_keys = sorted(set(existing_args) | set(current_config))
            differing_keys = [
                key
                for key in differing_keys
                if existing_args.get(key) != current_config.get(key)
            ]
            raise SystemExit(
                "The configuration for this run_id differs in: "
                + ", ".join(differing_keys)
                + ". Choose a new run_id or restore the saved configuration."
            )
        print("continuing existing run ...")
    else:
        print("starting new run ...")

    args.args_to_yaml(base_path)
    print("run_id", args.run_id)

    seed_everything(seed=args.seed_value)

    device_llm = select_device(CUDA_ID_LLM)
    device_deberta = select_device(CUDA_ID_DEBERTA)
    print("device_llm:", device_llm)
    print("device_deberta:", device_deberta)

    llm_model, tokenizer, deberta_model, deberta_tokenizer = get_models_and_tokenizers(
        model_type_llm=args.llm_model,
        device_llm=device_llm,
        model_type_deberta=args.deberta_model,
        device_deberta=device_deberta,
        use_flash_attention=args.use_flash_attention,
    )

    if args.dataset == "coqa":
        dataset = datasets.load_from_disk(dataset_directory)
        dataset = encode_and_format_dataset(dataset, tokenizer)
    elif args.dataset == "trivia_qa":
        dataset = datasets.load_from_disk(dataset_directory)
    elif args.dataset == "truthful_qa":
        dataset = datasets.load_from_disk(dataset_directory)
    print("# dataset:", len(dataset))

    get_results(
        args=args,
        base_path=base_path,
        llm_model=llm_model,
        tokenizer=tokenizer,
        device_llm=device_llm,
        deberta_model=deberta_model,
        deberta_tokenizer=deberta_tokenizer,
        device_deberta=device_deberta,
        dataset=dataset,
    )

    compute_semantic_pairs_for_results(
        base_path=base_path,
        model_type=args.deberta_model,
        deberta_tokenizer=deberta_tokenizer,
        deberta_model=deberta_model,
        num_instances=len(dataset),
        device=device_deberta,
    )


if __name__ == "__main__":
    main()
