#!/bin/bash
#
#SBATCH --job-name=DualStain_Train_SR # Job name
#SBATCH --partition=gpu # Partition for the job
#SBATCH --array=0-0 # Array jobs (adjust for parallelism if needed)
#SBATCH --cpus-per-task=4 # Number of CPU cores per task
#SBATCH --time=5-00:00:00 # Runtime (HH:MM:SS)
#SBATCH --output=/hpc/dhl_ec/VirtualSlides/DualStain/logs/%a.%j.%N.train_SR.log # Log file
#SBATCH --error=/hpc/dhl_ec/VirtualSlides/DualStain/logs/%a.%j.%N.train_SR.err # Error file
#SBATCH --mem=200000M # Memory limit
#SBATCH --mail-type=ALL # Email notifications (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ys2af@virginia.edu # Email address
#SBATCH --gres=gpu:1 # Number of GPUs

echo '----------------------------'
echo ' JOB ID: '$SLURM_ARRAY_JOB_ID
echo ' CURRENT TASK ID: '$SLURM_JOB_ID
echo ' CURRENT TASK NUMBER: '$SLURM_ARRAY_TASK_ID
echo '----------------------------'
echo ' MIN TASK ID: '$SLURM_ARRAY_TASK_MIN
echo ' MAX TASK ID: '$SLURM_ARRAY_TASK_MAX
echo ' TOTAL NUMBER OF TASKS: '$SLURM_ARRAY_TASK_COUNT
echo '----------------------------'

# Load conda environment
eval "$(conda shell.bash hook)"
conda activate /hpc/local/Rocky8/dhl_ec/software/mambaforge3/envs/convocals

# Define input/output directories and paths
HE_DATA_DIR="/hpc/dhl_ec/VirtualSlides/HE"
EVG_DATA_DIR="/hpc/dhl_ec/VirtualSlides/SR"
CSV_DATA="/hpc/dhl_ec/VirtualSlides/Projects/CONVOCALS/AtheroExpressCLAM/merged_dataset_SR.csv"
CSV_DATA_BAL="/hpc/dhl_ec/VirtualSlides/Projects/CONVOCALS/AtheroExpressCLAM/merged_dataset_balanced_SR.csv"
SPLIT_DIR=" /hpc/dhl_ec/VirtualSlides/Projects/CONVOCALS/AtheroExpressCLAM/wsi_classification_binary_dual_stains_balanced_dual_stains_SR_100_k10"
RESULTS_DIR="/hpc/dhl_ec/VirtualSlides/DualStain/results"
CKPT_PATH="/hpc/dhl_ec/VirtualSlides/Projects/CONVOCALS/AtheroExpressCLAM/results/HE_IPH_classification_sb_noinst_k10_new_s1/s_9_checkpoint.pt"

# === Train the model === #
# Ensure that the logs and dataset_csv directories are created before running this script.

CUDA_VISIBLE_DEVICES=0 python main_dual_SR.py \
--he_data_dir $HE_DATA_DIR \
--evg_data_dir $EVG_DATA_DIR \
--csv_dataset $CSV_DATA_BAL \
--results_dir $RESULTS_DIR \
--split_dir $SPLIT_DIR \
--drop_out \
--early_stopping \
--lr 1e-4 \
--label_frac 1.0 \
--exp_code SR_IPH_classification_sb_noinst_k10_transfer_dual_eq \
--model_size dino_version \
--bag_loss ce \
--model_type clam_sb \
--log_data \
--subtyping \
--weighted_sample \
--no_inst_cluster \
--B 8 \
--bag_weight 0.7 \
--n_classes 2