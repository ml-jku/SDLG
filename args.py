import os
import yaml


class Args:
    def __init__(self):

        # (0) dataset & models
        self.run_id = 'coqa_opt-2.7b'
        self.dataset = ['coqa', 'trivia_qa', 'truthful_qa'][0]
        self.llm_model = ['opt-125m', 'opt-350m', 'opt-1.3b', 'opt-2.7b', 'opt-6.7b', 'opt-13b', 'opt-30b', 'opt-66b'][3]
        self.deberta_model = ["deberta-base-mnli", "deberta-large-mnli", "deberta-xlarge-mnli", "deberta-v2-xlarge-mnli", "deberta-v2-xxlarge-mnli"][1]

        # (1.1) general
        self.seed_value = 42
        self.num_total_generations = 10
        self.max_length_of_generated_sequence = 256
        self.eos_token_ids = 4                     # 4: "." (period)
        self.invalid_ids = [2, 50118, 1209, 1864]  # 2: <\s>, 50118: "line break", 1209: " Q", 1864: "Q"
        self.compute_cleaned = False
        self.store_logits = True

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
        self.alphas = (1/3, 1/3, 1/3)  # weighting of attribution, substitution, and importance scores

        # (2.2) MS
        self.num_beams_baseline = 1
        self.num_return_sequences_baseline = 1   # for diverse beam search (ms: 1)
        self.num_beam_groups_baseline = 1        # for diverse beam search (ms: 1)
        self.diversity_penalty_baseline = 0.0    # for diverse beam search (ms: 0.0)
        self.do_sample_baseline = False
        self.temperature_baseline = 1
        self.top_p_baseline = 1
        

    # save args
    def args_to_yaml(self, base_path):
        os.makedirs(base_path, exist_ok=True)
        serializable_attrs = {k: v for k, v in self.__dict__.items()}
        with open(os.path.join(base_path, f'config.yaml'), 'w') as file:
            yaml.dump(serializable_attrs, file, sort_keys=False)
