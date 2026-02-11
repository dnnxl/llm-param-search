#!/bin/bash
#SBATCH --job-name=02_Hyper
#SBATCH --output=/data/dxie/llm-optimization/02_experiments_sensity.txt
#SBATCH --partition=nukwa-long
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --nodelist=nukwa-08.cnca

python /data/dxie/llm-optimization/02_experiments_sensity.py