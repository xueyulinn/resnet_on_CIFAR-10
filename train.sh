#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=resnet_cifar-10
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

cd /scratch/$USER/resnet_on_CIFAR-10
mkdir -p logs
for i in 0 1 2 3; do        
    uv run --no-sync python scripts/run_ablations.py $i >> logs/cfg$i.log 2>&1 &
done
wait
for i in 4 5 6 7; do          
    uv run --no-sync python scripts/run_ablations.py $i >> logs/cfg$i.log 2>&1 &
done
wait