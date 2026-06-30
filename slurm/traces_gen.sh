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
conda activate rac

cd open-r1-main

echo "Start at: $(date)"
echo "Running on node: $(hostname)"

# python src/open_r1/grpo.py --config recipes/DeepSeek-R1-Distill-Qwen-7B/grpo/config_demo.yaml --model_name_or_path /mnt/data/vhoangth2/pretrained-weights/DeepSeek-R1-Distill-Qwen-7B --dataset_name open-r1/OpenR1-Math-220k --dataset_prompt_column problem --save_dir traces/ --num_generations 2 --beta 0 --report_to none --use_vllm False --max_completion_length 8192 --do_train False --trace_only --trace_tokens 1_000_000 
python src/open_r1/grpo.py --config recipes/DeepSeek-R1-Distill-Qwen-14B/grpo/config_demo.yaml --model_name_or_path /mnt/data/vhoangth2/pretrained-weights/DeepSeek-R1-Distill-Qwen-14B --dataset_name open-r1/OpenR1-Math-220k --dataset_prompt_column problem --save_dir traces/ --num_generations 2 --beta 0 --report_to none --use_vllm False --max_completion_length 8192 --do_train False --trace_only --trace_tokens 1_000_000 