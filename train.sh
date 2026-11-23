#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --job-name=alexnet
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

cd /scratch/$USER/resnet_on_CIFAR-10
uv run --no-sync python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
uv run --no-sync python scripts/run_ablations.pyexport UV_CACHE_DIR=/scratch/$USER/.uv-cache