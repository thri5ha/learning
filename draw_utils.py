import os
import json
from common import BENCHMARK_RESULTS_JSON, BENCHMARK_RESULTS_XLSX
import pandas as pd
import matplotlib.pyplot as plt


def draw_stuff(output_folder, df):
    draw_graphs_per_model(output_folder, df)
    
def draw_graphs_per_model(output_folder, df):
    
    grouped = df.groupby('model')

    for model, group in grouped:
        
        model_output_folder = os.path.join(output_folder, f'{model.replace("/", "_").replace(".", "_")}')
        os.makedirs(model_output_folder, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        output_lens = group['output_len'].unique()
        for output_len in output_lens:
            df_batch = group[(group['output_len'] == output_len) & (group['batch_size'] == 180)]
            plt.plot(df_batch['tokens_per_sec'], df_batch['latency'], marker='o', linestyle='-', label=f'output_len {output_len}')
        # plt.plot(group['tokens_per_sec'], group['latency'], marker='o', linestyle='-')
        plt.title(f'{model}: Throughput vs Latency')
        plt.xlabel('Throughput (Tokens per Second)')
        plt.ylabel('Latency (seconds)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(model_output_folder, 'throughput_vs_latency.png'))
        plt.close()

        batch_sizes = group['batch_size'].unique()
        # Create a figure for latency vs output_len
        plt.figure(figsize=(10, 6))
        for batch_size in batch_sizes:
            df_batch = group[group['batch_size'] == batch_size]
            plt.plot(df_batch['output_len'], df_batch['latency'], marker='o', linestyle='-', label=f'Batch Size {batch_size}')
        plt.title(f'{model}: Latency vs Output Length')
        plt.xlabel('Output Length')
        plt.ylabel('Latency (seconds)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(model_output_folder, 'latency_vs_num_output.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        for batch_size in batch_sizes:
            df_batch = group[group['batch_size'] == batch_size]
            plt.plot(df_batch['output_len'], df_batch['tokens_per_sec'], marker='o', linestyle='-', label=f'Batch Size {batch_size}')
        plt.title(f'{model}: Throughput vs Output Length')
        plt.xlabel('Output Length')
        plt.ylabel('Throughput (tokens / seconds)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(model_output_folder, 'throughput_vs_num_output.png'))
        plt.close()


def parse_benchmark_results(output_folder):
    with open(os.path.join(benchmark_folder, BENCHMARK_RESULTS_JSON), 'r') as f:
        benchmark_results = json.load(f)
        
    df_benchmark_results = pd.DataFrame(benchmark_results)
    df_benchmark_results.to_csv(os.path.join(benchmark_folder, BENCHMARK_RESULTS_XLSX), index=False)

    draw_stuff(output_folder, df_benchmark_results)
    
    print(f"Drew all stuff at {output_folder}.")


if __name__ == '__main__':
    
    benchmark_folder = "/home/mcw/thrisha/benchmark_1_gpu/2024_07_07_22_58_19"

    parse_benchmark_results(benchmark_folder)
    