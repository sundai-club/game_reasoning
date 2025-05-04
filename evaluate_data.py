import os
import time
import sys
import numpy as np
import pandas as pd
import json
from optparse import OptionParser

from vllm import LLM, SamplingParams
from datasets import load_dataset
import countdown_verifier

def parse_args():
    parser = OptionParser()
    
    parser.add_option("--idx_start", type="int", dest="idx_start")
    parser.add_option("--idx_end", type="int", dest="idx_end")
    
    (options, args) = parser.parse_args()

    return options

def load_prompts_and_targets(parquet_file):
    dataset = load_dataset("parquet", data_files=parquet_file)
    
    prompts = []
    targets = []
    for item in dataset['train']:
        prompt_text = item['prompt'][0]['content']
        ground_truth = item['reward_model']['ground_truth']
        
        prompts.append(prompt_text)
        targets.append(ground_truth)
    
    return prompts, targets

def save_append_jsonl(file_path, data):
    with open(file_path, "a") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")

def parse_and_save_responses(responses, file_path):
    all_responses = [
        [{'role': 'user', 'content': output.prompt}] +
        [{'role': 'assistant', 'content': output.outputs[i].text} for i in range(len(output.outputs))]
        for output in responses
    ]

    save_append_jsonl(file_path, all_responses)
    return all_responses

def main(results_dir='./results'):

    options = parse_args()
    print(options)
    
    idx_start = options.idx_start
    idx_end = options.idx_end

    prompts, targets = load_prompts_and_targets("test.parquet")
    prompts = prompts[idx_start:idx_end]
    targets = targets[idx_start:idx_end]
    
    model_name = 'Qwen/Qwen2.5-1.5B'

    llm = LLM(model=model_name, gpu_memory_utilization=0.95, tensor_parallel_size=1, enable_prefix_caching=True)

    sampling_params = SamplingParams(
        max_tokens=2000,
        n=256,  # 256 samples per prompt
        temperature=0.7
    )

    # Generate responses
    gens = llm.generate(prompts, sampling_params, use_tqdm=True)

    # Optional: Save raw generations
    parse_and_save_responses(gens, f'{results_dir}/{idx_start}_{idx_end}.jsonl')

    # Now, build a matrix (n prompts x 256 generations) of scores
    n_prompts = len(prompts)
    n_samples = 256
    scores_matrix = np.zeros((n_prompts, n_samples))

    for i, (output, ground_truth) in enumerate(zip(gens, targets)):
        for j, single_output in enumerate(output.outputs):
            response_text = single_output.text
            score = countdown_verifier.compute_score(response_text, ground_truth)
            scores_matrix[i, j] = score if score == 1 else 0

    # Save the scores matrix
    scores_df = pd.DataFrame(scores_matrix)
    scores_df.to_csv(f'{results_dir}/scores_{idx_start}_{idx_end}.csv', index=False)

if __name__ == "__main__":
    main()
