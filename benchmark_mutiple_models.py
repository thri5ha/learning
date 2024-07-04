import time
import numpy as np
from benchmark_utils import *

def warmup(n: int, use_beam_search: bool,batch_size: int= 8, input_len: int = 32, output_len: int = 128):
    
    sampling_params = SamplingParams(
        n=n,
        temperature=0.0 if use_beam_search else 1.0,
        top_p=1.0,
        use_beam_search=use_beam_search,
        ignore_eos=True,
        max_tokens=output_len,
    )

    dummy_prompt_token_ids = np.random.randint(10000,
                                               size=(batch_size,
                                                     input_len))
    dummy_inputs: List[PromptStrictInputs] = [{
        "prompt_token_ids": batch
    } for batch in dummy_prompt_token_ids.tolist()]

    print("Warming up...")
    for _ in tqdm(range(WARMUP_ITERS), desc="Warmup iterations"):
        llm.generate(dummy_inputs,
                        sampling_params=sampling_params,
                        use_tqdm=False)
    print("Warm up done...")


def get_prompts(requests: List[Tuple[str, int, int]]):
    prompts: List[str] = []
    sampling_params: List[SamplingParams] = []
    for prompt, _, output_len in requests:
        prompts.append(prompt)
        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=0.0 if use_beam_search else 1.0,
                top_p=1.0,
                use_beam_search=use_beam_search,
                ignore_eos=True,
                max_tokens=output_len,
            ))
    
    return prompts, sampling_params


def benchmark_throughput(prompts, sampling_params) -> float:
    
    start = time.perf_counter()
    llm.generate(prompts, sampling_params, use_tqdm=True)
    end = time.perf_counter()

    return end - start

def benchmark_latency(prompts, sampling_params) -> [float]:
    
    latencies = list()
    for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
        start_time = time.perf_counter()
        llm.generate(dummy_inputs,
                        sampling_params=sampling_params,
                        use_tqdm=False)
        end_time = time.perf_counter()
        latency = end_time - start_time

        latencies.append(latency)

    return latencies


def main():
    args = get_args()
    
    random.seed(args.seed)

    requests = load_requests(args)
    
    llm = load_model(args.model, args.tokenizer,
    args.tensor_parallel_size, args.seed,
    args.trust_remote_code, args.dtype, args.max_model_len,
    args.enforce_eager, args.kv_cache_dtype,
    args.quantization_param_path, args.device,
    args.enable_prefix_caching, args.enable_chunked_prefill,
    args.max_num_batched_tokens, args.distributed_executor_backend,
    args.gpu_memory_utilization, args.download_dir, args.load_format)
    
    prompts, sampling_params = get_prompts(requests)
    warmup(args.n, args.use_beam_search)
    elapsed_time = benchmark_throughput(prompts, sampling_params)
    list_of_elapsed_time = benchmark_latency(prompts, sampling_params)
    
    total_num_tokens = sum(prompt_len + output_len
                           for _, prompt_len, output_len in requests)
    
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} tokens/s")
    
    latencies = np.array(latencies)
    percentages = [10, 25, 50, 75, 90, 99]
    percentiles = np.percentile(latencies, percentages)
    print(f'Avg latency: {np.mean(latencies)} seconds')
    for percentage, percentile in zip(percentages, percentiles):
        print(f'{percentage}% percentile latency: {percentile} seconds')



if __name__ == '__main__':
    main()