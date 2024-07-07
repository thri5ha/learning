from datetime import datetime

MODELS_1B = ['facebook/opt-1.3b']
MODELS_3B = ['databricks/dolly-v2-3b', 'openlm-research/open_llama_3b_v2']
MODELS_7B = []

MODELS = MODELS_3B

BATCH_SIZES = [1, 10, 20, 30, 50, 100, 150, 180, 240, 300]
OUTPUT_TOKEN_LENGTHS = [20, 64, 100, 200, 300]

NUM_ITERS = 5

WARMUP_ITERS = 5

TENSOR_PARALLEL_SIZES = [1]

TIMESTAMP = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

BENCHMARK_RESULTS_JSON = "benchmark_results.json"
BENCHMARK_RESULTS_XLSX = "benchmark_results.csv"

LLM_ARGS = {
    'quantization': None,
    'seed': 13,
    'trust_remote_code': False,
    'dtype': 'half',
    'max_model_len': None,
    'gpu_memory_utilization': 0.8,
    'enforce_eager': False,
    'kv_cache_dtype': 'auto',
    'quantization_param_path': None,
    'device': 'auto',
    'enable_prefix_caching': False,
    'download_dir': None,
    'enable_chunked_prefill': False,
    'max_num_batched_tokens': None,
    'load_format': 'auto',
}

SAMPLING_ARGS = {
    'n': 1,
    'use_beam_search': False,
    'top_p': 1.0,
    'ignore_eos': True
}
SAMPLING_ARGS['temperature'] = 0.0 if SAMPLING_ARGS['use_beam_search'] else 1.0