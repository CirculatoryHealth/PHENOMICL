#!/bin/bash
#
#SBATCH --job-name=EVG_noinstnoaug # set the name of the job
#SBATCH --partition=gpu # set the partition for the job
#SBATCH --array=0-0 # set the number of array jobs to be executed; when there are 2 workers and 1000 slides, 2 slides will be processed per array job, so --array=0-499
#SBATCH --cpus-per-task=2 # set the number of cores
#SBATCH --time=300:00:00 # set the total run time limit (HH:MM:SS)
#SBATCH --output=/hpc/dhl_ec/VirtualSlides/EVG/logs_EVG_train/%a.%j.%N.vit.log # make sure to create the logs_evg_segm folder first
#SBATCH --error=/hpc/dhl_ec/VirtualSlides/EVG/logs_EVG_train/%a.%j.%N.vit.errors # make sure to create the logs_evg_segm folder first
#SBATCH --mem=32G # set the memory limit
#SBATCH --mail-type=ALL # select which email types will be sent (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ys2af@virginia.edu # set destination email
# --gres=gpu:1

echo '----------------------------'
echo ' JOB ID: '$SLURM_ARRAY_JOB_ID
echo ' CURRENT TASK ID: '$SLURM_JOB_ID
echo ' CURRENT TASK NUMBER: '$SLURM_ARRAY_TASK_ID
echo '----------------------------'
echo ' MIN TASK ID: '$SLURM_ARRAY_TASK_MIN
echo ' MAX TASK ID: '$SLURM_ARRAY_TASK_MAX
echo ' TOTAL NUMBER OF TASKS: '$SLURM_ARRAY_TASK_COUNT
echo '----------------------------'

eval "$(conda shell.bash hook)"
# conda  activate /hpc/dhl_ec/VirtualSlides/cglastonbury/wsi
conda activate /hpc/local/Rocky8/dhl_ec/software/mambaforge3/envs/convocals
# conda activate convocals

CSV_INPUT_DATA="/hpc/dhl_ec/VirtualSlides/CD34/20231004.CONVOCALS.samplelist.withSMAslides.csv"
CSV_DATA="/hpc/dhl_ec/VirtualSlides/EVG/dataset_csv/AtheroExpress_EVG_WSI_dataset_binary_IPH.csv"
CSV_DATA_BAL="/hpc/dhl_ec/VirtualSlides/EVG/dataset_csv/AtheroExpress_EVG_WSI_dataset_binary_IPH_eq.csv"
H5_DIR="/hpc/dhl_ec/VirtualSlides/EVG/PROCESSED/features/h5_files"
CLASSIFIER="IPH.bin"

DATA_DIR="/hpc/dhl_ec/VirtualSlides/EVG/PROCESSED/features"
SPLIT_DIR="/hpc/dhl_ec/VirtualSlides/EVG/dataset_csv/wsi_classification_binary_100_k10"

# === Create CSV dataset === #
# we create the csv dataset for the classification task

# python generate_label_updated.py --csv_input $CSV_INPUT_DATA \
# --csv_output $CSV_DATA \
# --csv_output_bal $CSV_DATA_BAL \
# --h5_dir $H5_DIR \
# --classifier $CLASSIFIER

# === Create splits === #
# We create splits for the classification task
# so the validation, training and test sets

#python create_splits_seq.py --task wsi_classification_binary --csv_dataset $CSV_DATA --k 10

# === Train the model === #
# here we train the model

 CUDA_VISIBLE_DEVICES=0,1 python main.py \
--drop_out \
--early_stopping \
--lr 1e-4 \
--csv_dataset $CSV_DATA \
--k 10 \
--label_frac 1.0 \
--exp_code EVG_IPH_classification_sb_noinst_k10_new \
--model_size dino_version \
--bag_loss ce \
--task wsi_classification_binary \
--model_type clam_sb \
--log_data \
--subtyping \
--data_root_dir $DATA_DIR \
--split_dir $SPLIT_DIR \
--weighted_sample \
--no_inst_cluster \
--B 8 \
--bag_weight 0.7
