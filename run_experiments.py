import os
os.environ["HF_HOME"] = ...                     # set accordingly
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # set accordingly

import pickle
import yaml

import numpy as np
import datasets
import evaluate
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset

from args import Args
from utils import seed_everything, get_models_and_tokenizers, compute_correctness, compute_semantic_paris
from utils import generate_text, prepare_generated_text, compute_likelihood, prepare_likelihood
from sdlg import generate_semantically_diverse_output_sequences


CUDA_ID_LLM = 0                 # set accordingly
CUDA_ID_DEBERTA = CUDA_ID_LLM   # set accordingly


def encode(examples):
    return tokenizer(examples['story'] + ' Q: ' + examples['question'] + ' A:', truncation=False, padding=False)


def encode_and_format_dataset(dataset):
    dataset = dataset.map(encode, batched=False, load_from_cache_file=False)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)
    return dataset


def get_results(args, base_path, llm_model, tokenizer, device_llm, deberta_model, deberta_tokenizer, device_deberta, dataset):

    squad_metric = evaluate.load("squad")
    rouge = evaluate.load('rouge')
    exact_match_metric = evaluate.load("exact_match")
    bleurt = evaluate.load("bleurt")

    deberta_embeddings = deberta_model.deberta.embeddings.word_embeddings(
        torch.tensor([list(range(0, deberta_tokenizer.vocab_size))]).to(device_deberta)
    ).squeeze().detach()

    if args.dataset == 'coqa':
        id_to_question_mapping = dict(zip(dataset['id'], dataset['question']))

    dataloader = DataLoader(dataset, batch_size=1)
    for b, batch in tqdm(enumerate(dataloader)):

        prompt = batch['input_ids'][0].to('cpu')

        if args.dataset == 'coqa':
            question = id_to_question_mapping[batch['id'][0]]  
        else:
            question = batch["question"][0]

        results_dict = {'input_ids': batch['input_ids'],
                        'question': question,
                        'correctness_dict': {},
                        'sdlg': {'generations': [],      # list of dicts
                                'likelihoods': []},      # list of dicts
                        'baseline': {'generations': [],  # list of dicts
                                     'likelihoods': []}  # list of dicts
                        }
        
        ### (1) most likely output sequence
        most_likely_generation = generate_text(args=args, 
                                               model=llm_model, 
                                               tokenizer=tokenizer, 
                                               input_ids=batch['input_ids'], 
                                               len_prompt=len(prompt), 
                                               decoding_method='most_likely', 
                                               device=device_llm)
        
        # compute correctness score
        if args.dataset == 'coqa':
            reference_answers = batch['answer']['text'] + [x[0] for x in batch['additional_answers']]
            incorrect_answers = []
        elif args.dataset == 'trivia_qa':
            reference_answers = batch['answer']
            incorrect_answers = []
        elif args.dataset == 'truthful_qa':
            reference_answers = batch['answer'] + [x[0] if x[0][-1] == "." else x[0] + "." for x in batch['additional_answers']]
            if "I have no comment." not in reference_answers:
                reference_answers.append("I have no comment.")
            incorrect_answers = [x[0] if x[0][-1] == "." else x[0] + "." for x in batch['incorrect_answers']]

        correctness_dict = compute_correctness(args=args, 
                                               reference_answers=reference_answers, 
                                               incorrect_answers=incorrect_answers, 
                                               most_likely_generation_text=most_likely_generation['generation_text'][0], 
                                               exact_match_metric=exact_match_metric, 
                                               rouge=rouge, 
                                               bleurt=bleurt)
        
        results_dict['correctness_dict'] = correctness_dict

        # compute likelihood
        most_likely_generation_likelihoods = compute_likelihood(prompt=prompt, 
                                                                generation=most_likely_generation, 
                                                                model=llm_model, 
                                                                device=device_llm, 
                                                                compute_cleaned=args.compute_cleaned, 
                                                                store_logits=args.store_logits)
        
        ### (2) sample addtional output sequences

        # (2.1) SDLG
        results_dict['sdlg']['generations'].append(most_likely_generation)
        results_dict['sdlg']['likelihoods'].append(most_likely_generation_likelihoods)
        
        results_dict = generate_semantically_diverse_output_sequences(results_dict=results_dict, 
                                                                      deberta_model=deberta_model, 
                                                                      deberta_tokenizer=deberta_tokenizer, 
                                                                      device_deberta=device_deberta,
                                                                      deberta_embeddings=deberta_embeddings,
                                                                      model=llm_model, 
                                                                      tokenizer=tokenizer, 
                                                                      device_llm=device_llm,
                                                                      input_ids=batch['input_ids'], 
                                                                      prompt=prompt,
                                                                      question=question, 
                                                                      initial_generation=most_likely_generation,
                                                                      initial_likelihood=most_likely_generation_likelihoods,
                                                                      args=args)      

        # (2.2) MS
        assert args.num_total_generations % args.num_return_sequences_baseline == 0
        results_dict['baseline']['generations'].append(most_likely_generation)
        results_dict['baseline']['likelihoods'].append(most_likely_generation_likelihoods)

        for i in range(int(args.num_total_generations / args.num_return_sequences_baseline)):
            baseline_generation = generate_text(args=args, 
                                                model=llm_model, 
                                                tokenizer=tokenizer, 
                                                input_ids=batch['input_ids'], 
                                                len_prompt=len(prompt), 
                                                decoding_method='baseline', 
                                                device=device_llm)

            results_dict['baseline']['generations'].append(baseline_generation)
            results_dict['baseline']['likelihoods'].append(compute_likelihood(prompt=prompt, 
                                                                              generation=baseline_generation, 
                                                                              model=llm_model, 
                                                                              device=device_llm, 
                                                                              compute_cleaned=args.compute_cleaned, 
                                                                              store_logits=args.store_logits))

        with open(os.path.join(base_path, f'results_dict_{b}.pkl'), 'wb') as outfile:
            pickle.dump(results_dict, outfile)


if __name__ == '__main__':

    args = Args()

    base_path = os.path.join('results', args.run_id)
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
    
    if os.path.exists(os.path.join(base_path, f'config.yaml')):
        with open(os.path.join(base_path, f'config.yaml'), 'r') as file:
            existing_args = yaml.load(file, Loader=yaml.FullLoader)
        changes = False

        for k, v in existing_args.items():
            if k not in args.__dict__:
                print(f"new arg: {k}")
                changes = True
            elif v != args.__dict__[k]:
                print(f"arg {k} changed from {v} to {args.__dict__[k]}")
                changes = True
        if changes:
            exit()
        print("continuing existing run ...")
    else:
        print("starting new run ...")

    # save args
    args.args_to_yaml(base_path)

    print("run_id", args.run_id)

    seed_everything(seed=args.seed_value)

    # prepare model & tokenizer
    device_llm = "mps" if torch.backends.mps.is_built() else f"cuda:{CUDA_ID_LLM}" if torch.cuda.is_available() else "cpu"
    print("device_llm: ", device_llm)
    device_deberta = "mps" if torch.backends.mps.is_built() else f"cuda:{CUDA_ID_DEBERTA}" if torch.cuda.is_available() else "cpu"
    print("device_deberta: ", device_deberta)

    llm_model, tokenizer, deberta_model, deberta_tokenizer = get_models_and_tokenizers(model_type_llm=args.llm_model, 
                                                                                       device_llm=device_llm, 
                                                                                       model_type_deberta=args.deberta_model, 
                                                                                       device_deberta=device_deberta)
    
    # prepare data
    if args.dataset == 'coqa':
        dataset = datasets.load_from_disk(os.path.join("datasets", f'coqa_dataset'))
        dataset = encode_and_format_dataset(dataset)
    elif args.dataset == 'trivia_qa':
        dataset = datasets.load_from_disk(os.path.join("datasets", f'trivia_qa_dataset'))
    elif args.dataset == "truthful_qa":
        dataset = datasets.load_from_disk(os.path.join("datasets", f'truthful_qa_dataset'))
    else:
        raise ValueError(f"dataset {args.dataset} not implemented")
    print("# dataset:", len(dataset))

    get_results(args=args,
                base_path=base_path, 
                llm_model=llm_model, 
                tokenizer=tokenizer, 
                device_llm=device_llm, 
                deberta_model=deberta_model, 
                deberta_tokenizer=deberta_tokenizer, 
                device_deberta=device_deberta, 
                dataset=dataset)

    compute_semantic_paris(base_path=base_path, 
                           model_type=args.deberta_model, 
                           deberta_tokenizer=deberta_tokenizer, 
                           deberta_model=deberta_model, 
                           num_instances=len(dataset),
                           device=device_deberta)
    