#!/bin/bash
#SBATCH --job-name=check_cuda
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

echo "Checking CUDA version..."
nvcc --version

echo "Checking NVIDIA driver and GPU status..."
nvidia-smi

