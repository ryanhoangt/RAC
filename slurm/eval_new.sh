#!/bin/bash
#SBATCH --job-name=rac-trace-gen
#SBATCH --output=slurm/logs/log_%j.out
#SBATCH --error=slurm/logs/log_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac-new

cd open-r1-main

##### MAIN SCRIPT #####
set -euo pipefail

export VLLM_WORKER_MULTIPROC_METHOD=spawn

NUM_GPUS=1
MODEL_DIR=models/DeepSeek-R1-Distill-Qwen-7B_pruned_40_all_tokens1000000_prunemethod_SparseGPT_thirds_1_2_3__dataset_DeepSeek-R1-Distill-Qwen-7B_trace_OpenR1-Math-220k
# MODEL_DIR=models/DeepSeek-R1-Distill-Qwen-14B_pruned_40_all_tokens1000000_prunemethod_SparseGPT_thirds_1_2_3__dataset_DeepSeek-R1-Distill-Qwen-14B_trace_OpenR1-Math-220k
MODEL_TAG=$(basename "$MODEL_DIR")

MODEL_ARGS="model_name=${MODEL_DIR},\
dtype=bfloat16,\
trust_remote_code=true,\
max_model_length=32768,\
gpu_memory_utilization=0.8,\
data_parallel_size=${NUM_GPUS},\
generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

OUTPUT_DIR=eval_outputs/new_eval_env_w_flash_attn

echo "Start at: $(date)"
echo "Running on node: $(hostname)"

lighteval vllm "$MODEL_ARGS" "lighteval|math_500|0" \
  --output-dir "$OUTPUT_DIR" \
  --save-details
