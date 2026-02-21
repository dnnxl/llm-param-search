#!/bin/bash
#SBATCH --job-name=02_Optimization
#SBATCH --output=/data/dxie/llm-optimization/02_optimization_llama.txt
#SBATCH --partition=nukwa-long
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --nodelist=nukwa-10.cnca

python /data/dxie/llm-optimization/01_optimization.py \
  --experiment_id experiment_02 \
  --seed 42 \
  --device cuda \
  --model_name unsloth/mistral-7b-instruct-v0.3-bnb-4bit \
  --max_seq_length 8192 \
  --data_path /data/dxie/llm-optimization/dataset/FEINA_test_split_train.csv \
  --optimization_algorithm qNEHVI \
  --n_batch 150 \
  --n_init 50 \
  --mc_samples 128 \
  --ref_sari 20.0 \
  --ref_bert 0.75 \
  --noise_sari 0.5 \
  --noise_bert 0.01 \
  --temp_min 0.0 \
  --temp_max 2.0 \
  --top_p_min 0.6 \
  --top_p_max 1.0 \
  --top_k_min 20 \
  --top_k_max 640 \
  --rep_pen_min 1.0 \
  --rep_pen_max 1.4 \
  --max_tokens_min 64 \
  --max_tokens_max 512 \
  --num_restarts 10 \
  --raw_samples 256 \
  --candidates_generation 1