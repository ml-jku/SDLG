import os

import yaml


class Args:
    def __init__(self):
        # (0) dataset & models
        self.run_id = "coqa_opt-2.7b"
        self.dataset = ["coqa", "trivia_qa", "truthful_qa"][0]
        self.llm_model = [
            "opt-125m",
            "opt-350m",
            "opt-1.3b",
            "opt-2.7b",
            "opt-6.7b",
            "opt-13b",
            "opt-30b",
            "opt-66b",
        ][3]
        self.deberta_model = [
            "deberta-base-mnli",
            "deberta-large-mnli",
            "deberta-xlarge-mnli",
            "deberta-v2-xlarge-mnli",
            "deberta-v2-xxlarge-mnli",
        ][1]

        # (1.1) general
        self.seed_value = 42
        self.num_total_generations = 10
        self.max_length_of_generated_sequence = 256
        self.eos_token_ids = 4  # "." (period)
        self.invalid_ids = [2, 50118, 1209, 1864]  # </s>, line break, " Q", "Q"
        self.compute_cleaned = False
        self.store_logits = False
        self.overwrite_existing = False
        self.use_flash_attention = False

        # (1.2) most likely generation
        self.num_beams_most_likely = 5
        self.num_return_sequences_most_likely = 1
        self.do_sample_most_likely = False
        self.temperature_most_likely = 1.0
        self.top_p_most_likely = 1

        # (2.1) SDLG
        self.num_beams_sdlg = 5
        self.num_return_sequences_sdlg = 1
        self.do_sample_sdlg = False
        self.temperature_sdlg = 1.0
        self.top_p_sdlg = 1
        self.token_prob_threshold = 0.001
        # Weights for attribution, substitution, and importance scores.
        self.alphas = [1 / 3, 1 / 3, 1 / 3]

        # (2.2) MS
        self.num_beams_baseline = 1
        self.num_return_sequences_baseline = 1  # diverse beam search: set > 1
        self.num_beam_groups_baseline = 1  # diverse beam search: set > 1
        self.diversity_penalty_baseline = 0.0  # diverse beam search: set > 0
        self.do_sample_baseline = True
        self.temperature_baseline = 1
        self.top_p_baseline = 1

    def validate(self):
        if self.num_total_generations < 2:
            raise ValueError("num_total_generations must be at least 2")
        if self.num_return_sequences_most_likely != 1:
            raise ValueError("SDLG requires exactly one most-likely generation")
        if self.num_return_sequences_sdlg != 1:
            raise ValueError(
                "SDLG currently supports one returned sequence per substitution"
            )
        if self.num_return_sequences_baseline < 1:
            raise ValueError("num_return_sequences_baseline must be positive")
        if self.num_beams_baseline < 1:
            raise ValueError("num_beams_baseline must be positive")
        if self.num_beam_groups_baseline < 1:
            raise ValueError("num_beam_groups_baseline must be positive")
        if self.num_beam_groups_baseline > self.num_beams_baseline:
            raise ValueError(
                "num_beam_groups_baseline cannot exceed num_beams_baseline"
            )
        if self.num_beams_baseline % self.num_beam_groups_baseline != 0:
            raise ValueError(
                "num_beams_baseline must be divisible by num_beam_groups_baseline"
            )
        if (
            not self.do_sample_baseline
            and self.num_return_sequences_baseline > self.num_beams_baseline
        ):
            raise ValueError(
                "num_return_sequences_baseline cannot exceed "
                "num_beams_baseline when sampling is disabled"
            )
        if self.num_beam_groups_baseline > 1:
            if self.do_sample_baseline:
                raise ValueError(
                    "diverse beam search requires do_sample_baseline=False"
                )
            if self.diversity_penalty_baseline <= 0:
                raise ValueError(
                    "diverse beam search requires a positive diversity penalty"
                )

        num_additional_baseline_generations = self.num_total_generations - 1
        if (
            num_additional_baseline_generations % self.num_return_sequences_baseline
            != 0
        ):
            raise ValueError(
                "num_total_generations - 1 must be divisible by "
                "num_return_sequences_baseline"
            )
        if len(self.alphas) != 3:
            raise ValueError("alphas must contain three weights")

    def args_to_yaml(self, base_path):
        os.makedirs(base_path, exist_ok=True)
        with open(
            os.path.join(base_path, "config.yaml"), "w", encoding="utf-8"
        ) as file:
            yaml.safe_dump(self.experiment_config(), file, sort_keys=False)

    def experiment_config(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != "overwrite_existing"
        }
