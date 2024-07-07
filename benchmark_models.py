import os
import json
import time
import torch
import random
from common import *
from vllm import LLM, SamplingParams
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment


def load_dataset(dataset_path: str)-> list:
    
    print(f"Loading datasetfrom {dataset_path}...")
    with open(dataset_path) as f:
        dataset = json.load(f)
        
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    random.shuffle(dataset)
    
    prompts_dataset = [data[0] for data in dataset]
    
    print(f"Done loading dataset.")
    return prompts_dataset

def create_filtered_dataset(dataset: list, tokenzier:str) -> list:
    tokenzier = AutoTokenizer.from_pretrained(tokenzier, trust_remote_code=LLM_ARGS['trust_remote_code'])

    filtered_dataset = list()
    
    for data in dataset:
        prompt_token_ids = tokenzier(data).input_ids
        if len(prompt_token_ids) < 4 or len(prompt_token_ids) > 1024:
            continue
        filtered_dataset.append(data)
        if len(filtered_dataset) > 500:
            break
    
    return filtered_dataset


def load_prompts(dataset: list, batch_size:int) -> list:
    return random.sample(dataset, batch_size)
    

def run_iterative_benchmark(session_output_path: str, dataset: list) -> str:
    
    benchmark_results = list()
    
    for tensor_parallel_size in TENSOR_PARALLEL_SIZES:
        for model in MODELS:
            try:
                llm = LLM(model=model, tokenizer=model,
                          tensor_parallel_size=tensor_parallel_size, distributed_executor_backend='mp' if tensor_parallel_size > 1 else None,
                          **LLM_ARGS)
                
                print(f"Loaded model {model}")
            except Exception as e:
                print(f"Cannot load model, because of {e}")
                continue
            
            filtered_dataset = create_filtered_dataset(dataset, tokenzier=model)
            
            for batch_size in BATCH_SIZES:
                prompts = load_prompts(filtered_dataset, batch_size)
                print(f"Loaded prompts with batch size {batch_size}")
                
                for output_len in OUTPUT_TOKEN_LENGTHS:
                    
                    sampling_params = SamplingParams(max_tokens=output_len, **SAMPLING_ARGS)
                    
                    try:
                        print(f"Warming up....")
                        for i in range(WARMUP_ITERS):
                            llm.generate(prompts, sampling_params)
                        print(f"Warmup done...")
                        
                        print(f"Measuring inference time taken..")
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        
                        for i in range(NUM_ITERS):
                            result = llm.generate(prompts, sampling_params)
                        
                        torch.cuda.synchronize()
                        elapsed_time = time.perf_counter() - start
                        
                        print(f"[GPUS:{tensor_parallel_size}][{model}][{batch_size}][{output_len}] Inference done")
                        
                        mean_time = elapsed_time / NUM_ITERS
                        
                        num_tokens = 0
                        for output in result:
                            num_tokens += len(output.outputs[0].token_ids)
                                            
                        benchmark_result = {
                            'num_gpus': tensor_parallel_size,
                            'model': model,
                            'batch_size': batch_size,
                            'output_len': output_len,
                            'latency': mean_time,
                            'num_output_tokens': num_tokens,
                            'tokens_per_sec': num_tokens / mean_time,
                            'requests_per_sec': len(prompts) / mean_time
                        }
                        benchmark_results.append(benchmark_result)
                        
                        # Not the best way, but I don't want to lose my results
                        with open(os.path.join(session_output_path, BENCHMARK_RESULTS_JSON), 'w') as f:
                            json.dump(benchmark_results, f, indent=4)

                    except Exception as e:
                        print(f"Cannot run inference, because of {e}")
                        continue
            try:
                destroy_model_parallel()
                destroy_distributed_environment()
                del llm
                torch.cuda.synchronize() 
            except Exception as e:
                print(f"{e}")
                continue

    

def init_profile_folders(output_folder: str) -> str:
    
    session_output_path = os.path.join(output_folder, TIMESTAMP)
    os.makedirs(session_output_path, exist_ok=True)
    
    return session_output_path


if __name__ == '__main__':
    
    output_folder = '/home/mcw/thrisha/benchmark_1_gpu'
    dataset_path = '/home/mcw/thrisha/data/ShareGPT_V3_unfiltered_cleaned_split.json'
    
    random.seed(13)
    dataset = load_dataset(dataset_path=dataset_path)
    session_output_path = init_profile_folders(output_folder)
    run_iterative_benchmark(session_output_path, dataset)