import importlib.util
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import run_experiments
import sdlg
import utils
from args import Args


def test_default_configuration_and_generation_count_validation():
    args = Args()
    assert args.do_sample_baseline is True
    assert args.store_logits is False
    assert "overwrite_existing" not in args.experiment_config()
    args.validate()

    legacy_config = yaml.load(
        "alphas: !!python/tuple [0.2, 0.3, 0.5]",
        Loader=run_experiments.ConfigLoader,
    )
    assert legacy_config == {"alphas": [0.2, 0.3, 0.5]}

    args.num_return_sequences_baseline = 2
    with pytest.raises(ValueError, match="divisible"):
        args.validate()

    args = Args()
    args.num_beams_baseline = 9
    args.num_return_sequences_baseline = 9
    args.num_beam_groups_baseline = 9
    args.diversity_penalty_baseline = 0.5
    args.do_sample_baseline = False
    args.validate()

    args.do_sample_baseline = True
    with pytest.raises(ValueError, match="requires do_sample_baseline=False"):
        args.validate()


def test_missing_dataset_reports_preparation_command(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match=r"python datasets/parse_coqa\.py"):
        run_experiments.get_dataset_directory("coqa")


def test_ranking_cleaning_and_invalid_token_removal():
    assert sdlg.rank_tensor([0.4, 0.2, 0.4, 0.1]).tolist() == [1, 2, 1, 3]
    generation = torch.tensor([5, 9, 2, 7])
    assert utils.remove_invalid_ids(generation, [2, 9]).tolist() == [5]
    assert utils.clean_generation("An answer. Q: another question") == "An answer."


def test_likelihoods_can_omit_full_vocabulary_logits():
    class Model:
        def __call__(self, input_ids, labels):
            _, sequence_length = input_ids.shape
            return {
                "loss": torch.tensor(0.5),
                "logits": torch.zeros((1, sequence_length, 7)),
            }

    generation = {
        "generation_ids": [torch.tensor([3, 4])],
        "generation_text": ["answer"],
        "cleaned_generation_ids": [torch.tensor([3, 4])],
        "cleaned_generation_text": ["answer"],
    }
    likelihood = utils.compute_likelihood(
        prompt=torch.tensor([1, 2]),
        generation=generation,
        model=Model(),
        device="cpu",
    )

    assert likelihood["average_neg_log_likelihood"] == [0.5]
    assert likelihood["neg_log_likelihood"] == [1.0]
    assert likelihood["generation_logits"] == []
    assert likelihood["store_logits"] is False
    assert len(utils.prepare_likelihood(**likelihood)) == 1


def test_semantic_clusters_require_bidirectional_entailment():
    generations = [
        {"generation_text": ["a"]},
        {"generation_text": ["b"]},
        {"generation_text": ["c"]},
    ]
    semantic_pairs = np.array(
        [
            [True, True, False],
            [True, True, True],
            [False, True, True],
        ]
    )

    clusters = utils.compute_semantic_clusters(
        generations,
        semantic_pairs,
        cleaned_semantic_pairs=[],
    )
    assert clusters["semantic_clusters"].tolist() == [0, 0, 0]


def test_atomic_pickle_dump(tmp_path):
    output_path = tmp_path / "result.pkl"
    utils.atomic_pickle_dump({"value": 3}, output_path)
    with output_path.open("rb") as file:
        assert pickle.load(file) == {"value": 3}
    assert list(tmp_path.iterdir()) == [output_path]


def test_llm_tokenizer_can_be_loaded_without_a_deberta_device(monkeypatch):
    tokenizer = object()
    monkeypatch.setattr(
        utils.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: tokenizer,
    )

    model, loaded_tokenizer, deberta_model, deberta_tokenizer = (
        utils.get_models_and_tokenizers(
            model_type_llm="opt-125m",
            get_tokenizer_only_llm=True,
        )
    )
    assert (model, loaded_tokenizer, deberta_model, deberta_tokenizer) == (
        None,
        tokenizer,
        None,
        None,
    )

    with pytest.raises(ValueError, match="not supported"):
        utils.get_models_and_tokenizers(
            model_type_deberta="unknown",
            get_tokenizer_only_deberta=True,
        )


def _generation_args(num_total_generations):
    return SimpleNamespace(
        alphas=[1 / 3, 1 / 3, 1 / 3],
        compute_cleaned=False,
        eos_token_ids=4,
        invalid_ids=[],
        num_total_generations=num_total_generations,
        store_logits=False,
        token_prob_threshold=None,
    )


def _generation(ids, text):
    return {
        "generation_ids": [torch.tensor(ids)],
        "generation_text": [text],
        "cleaned_generation_ids": [torch.tensor(ids)],
        "cleaned_generation_text": [text],
    }


def _results(initial_generation, initial_likelihood):
    return {
        "sdlg": {
            "generations": [initial_generation],
            "likelihoods": [initial_likelihood],
        }
    }


def test_sdlg_generation_limit_includes_the_initial_sequence(monkeypatch):
    initial_generation = _generation([1], "one")
    initial_likelihood = {"generation_logits": [torch.zeros((1, 5))]}
    alternative = _generation([2], "two")
    monkeypatch.setattr(sdlg, "generate_text", lambda **_kwargs: alternative)
    monkeypatch.setattr(
        sdlg,
        "compute_likelihood",
        lambda *_args, **_kwargs: {"neg_log_likelihood": [1.0]},
    )

    results = sdlg.generate_semantically_diverse_output_sequences(
        results_dict=_results(initial_generation, initial_likelihood),
        deberta_model=None,
        deberta_tokenizer=None,
        device_deberta="cpu",
        deberta_embeddings=None,
        model=None,
        tokenizer=SimpleNamespace(decode=lambda *_args, **_kwargs: "token"),
        device_llm="cpu",
        input_ids=torch.tensor([[99]]),
        prompt=torch.tensor([99]),
        question="question",
        initial_generation=initial_generation,
        initial_likelihood=initial_likelihood,
        args=_generation_args(num_total_generations=3),
    )
    assert len(results["sdlg"]["generations"]) == 3


def test_eos_substitution_retains_the_complete_prefix(monkeypatch):
    initial_generation = _generation([10, 11, 12], "one two")
    initial_likelihood = {"generation_logits": [torch.zeros((3, 20))]}
    token_info = {(1, 2, 4): (1.0, 1.0, 0.2)}
    monkeypatch.setattr(
        sdlg,
        "compute_token_score_ranking",
        lambda *_args, **_kwargs: (torch.tensor([0]), token_info),
    )
    monkeypatch.setattr(
        sdlg,
        "compute_likelihood",
        lambda *_args, **_kwargs: {"neg_log_likelihood": [1.0]},
    )
    tokenizer = SimpleNamespace(
        decode=lambda *_args, **_kwargs: "one.",
        encode=lambda *_args, **_kwargs: torch.tensor([[10, 11, 4]]),
    )

    results = sdlg.generate_semantically_diverse_output_sequences(
        results_dict=_results(initial_generation, initial_likelihood),
        deberta_model=None,
        deberta_tokenizer=None,
        device_deberta="cpu",
        deberta_embeddings=None,
        model=None,
        tokenizer=tokenizer,
        device_llm="cpu",
        input_ids=torch.tensor([[99]]),
        prompt=torch.tensor([99]),
        question="question",
        initial_generation=initial_generation,
        initial_likelihood=initial_likelihood,
        args=_generation_args(num_total_generations=2),
    )
    assert results["sdlg"]["generations"][1]["generation_ids"][0].tolist() == [
        10,
        11,
        4,
    ]


def test_truthfulqa_prompt_examples_are_excluded_from_evaluation():
    parser_path = (
        Path(__file__).resolve().parents[1] / "datasets" / "parse_truthful_qa.py"
    )
    spec = importlib.util.spec_from_file_location("parse_truthful_qa", parser_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    data = [[str(index)] for index in range(70)]
    evaluation_indices = {index for index, _sample in module.evaluation_samples(data)}
    assert set(module.FEW_SHOT_INDICES).isdisjoint(evaluation_indices)
    assert len(evaluation_indices) == len(data) - len(module.FEW_SHOT_INDICES)
