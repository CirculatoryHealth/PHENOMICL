#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function

print(
    "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
)
print("                                         Main Modeling")
print("")
print("* Version          : v2.0.0")
print("")
print("* Last update      : 2025-01-01")
print("* Written by       : Yipei (Petra) Song")
print(
    "* Edited by         : Craig Glastonbury | Sander W. van der Laan | Clint L. Miller | Yipei Song | Francesco Cisternino."
)
print("")
print("* Description      : Main modeling with dual-stain support.")
print("")
print(
    "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
)

import argparse
import os
import numpy as np
import pandas as pd

# Internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils_dual import *
from utils.core_utils_trans_dual import train
from datasets.dataset_generic_dual import Generic_WSI_Classification_Dataset

# PyTorch imports
import torch

from utils.eval_utils import initiate_model
from models.model_clam_sum import CLAM_SB, CLAM_MB


def load_pretrained_model(args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    model.train()
    model.cuda()
    return model

def get_model(args, checkpoint=None):
    """
    Load or initialize the model based on arguments. 
    If no checkpoint is provided, return None.
    """
    if checkpoint is not None:
        model = load_pretrained_model(args, checkpoint)
    else:
        print("No checkpoint provided. Returning None for model initialization.")
        model = None
    return model


def main(args):
    """
    Main training and evaluation loop with k-fold cross-validation.
    """
    # Create results directory if not exists
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    # Define folds for cross-validation
    start = args.k_start if args.k_start != -1 else 0
    end = args.k_end if args.k_end != -1 else args.k

    # Accuracy and AUC tracking
    all_test_auc, all_val_auc, all_test_acc, all_val_acc = [], [], [], []

    folds = np.arange(start, end)
    for i in folds:
        print(f"\nTraining Fold {i}!", flush=True)
        model = get_model(args, args.ckpt_path)
        seed_torch(args.seed)

        # Load train, val, and test splits
        train_dataset, val_dataset, test_dataset = dataset.return_splits(
            from_id=False, csv_path=f"{args.split_dir}/splits_{i}.csv"
        )
        datasets = (train_dataset, val_dataset, test_dataset)

        # Train with the current split
        results, test_auc, val_auc, test_acc, val_acc = train(
            datasets, i, args, model
        )
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)

        # Save results
        save_pkl(os.path.join(args.results_dir, f"split_{i}_results.pkl"), results)

    # Summarize results
    final_df = pd.DataFrame(
        {
            "folds": folds,
            "test_auc": all_test_auc,
            "val_auc": all_val_auc,
            "test_acc": all_test_acc,
            "val_acc": all_val_acc,
        }
    )
    summary_name = (
        f"summary_partial_{start}_{end}.csv" if len(folds) != args.k else "summary.csv"
    )
    final_df.to_csv(os.path.join(args.results_dir, summary_name))


# Argument parser
parser = argparse.ArgumentParser(description="Configurations for Dual-Stain WSI Training")

# Directories
parser.add_argument("--he_data_dir", type=str, required=True, help="Directory containing HE h5 files")
parser.add_argument("--evg_data_dir", type=str, required=True, help="Directory containing EVG h5 files")
parser.add_argument("--csv_dataset", type=str, required=True, help="Path to the dataset CSV")
parser.add_argument("--results_dir", type=str, default="./results", help="Results directory")
parser.add_argument("--split_dir", type=str, default=None, help="Path to split directory")

# Training settings
parser.add_argument("--max_epochs", type=int, default=200, help="Maximum number of epochs, default 200")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--reg", type=float, default=1e-3, help="Weight decay")
parser.add_argument("--label_frac", type=float, default=1.0, help="Fraction of training labels")
parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
parser.add_argument("--k", type=int, default=10, help="Number of folds for cross-validation")
parser.add_argument("--k_start", type=int, default=-1, help="Start fold for cross-validation")
parser.add_argument("--k_end", type=int, default=-1, help="End fold for cross-validation")
parser.add_argument("--early_stopping", action="store_true", default=False, help="enable early stopping")
parser.add_argument("--opt", type=str, choices=["adam", "sgd"], default="adam")
parser.add_argument("--testing", action="store_true", default=False, help="debugging tool")

# Model and loss settings
parser.add_argument("--n_classes", type=int, required=True, help="Number of classes")
parser.add_argument("--model_type", type=str, default="clam_sb", choices=["clam_sb", "clam_mb"], help="Model type")
parser.add_argument("--model_size", type=str, default=None, choices=["small", "big", "dino_version"], help="Model size")
parser.add_argument("--bag_loss", type=str, default="ce", choices=["svm", "ce", "focal"], help="Bag-level loss")
parser.add_argument("--drop_out", action="store_true", help="Enable dropout")
parser.add_argument("--subtyping", action="store_true", help="Subtyping problem")
parser.add_argument("--ckpt_path", type=str, default=None, help="Pretrained checkpoint path")
parser.add_argument("--apply_bag_augmentation", action="store_true", help="Enable bag augmentation")
parser.add_argument("--weighted_sample", action="store_true", help="Enable weighted sampling")
parser.add_argument("--log_data", action="store_true", help="Enable TensorBoard logging")

# CLAM-specific settings
parser.add_argument("--exp_code", type=str, help="experiment code for saving results")
parser.add_argument("--bag_weight", type=float, default=1.0, help="Weight coefficient for bag-level loss")
parser.add_argument("--no_inst_cluster", action="store_true", default=False, help="Disable instance-level clustering")
parser.add_argument("--inst_loss", type=str, choices=["svm", "ce", None], default=None, help="Instance-level clustering loss")
parser.add_argument("--B", type=int, default=96, help="Number of positive/negative patches to sample for CLAM")
args = parser.parse_args()

def seed_torch(seed=7):
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

# Dataset configuration
stain_data_dirs = {
    "HE": os.path.join(args.he_data_dir, "PROCESSED/features_imagenet"),
    "EVG": os.path.join(args.evg_data_dir, "PROCESSED/features_512_imagenet"),
}
dataset = Generic_WSI_Classification_Dataset(
    csv_path=args.csv_dataset,
    stain_data_dirs=stain_data_dirs,
    shuffle=False,
    seed=args.seed,
    print_info=True,
    label_dict={"no": 0, "yes": 1},
    patient_strat=True,
    apply_bag_augmentation=args.apply_bag_augmentation,
)

if args.model_type in ["clam_sb", "clam_mb"]:
    assert args.subtyping

else:
    raise NotImplementedError

# Update results and split directories
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)
args.results_dir = os.path.join(args.results_dir, f"{args.exp_code}_s{args.seed}")
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)
print("args.results_dir:", args.results_dir)

if args.split_dir is None:
    args.split_dir = os.path.join("splits", f"{args.task}_{int(args.label_frac * 100)}")
else:
    args.split_dir = os.path.join("splits", args.split_dir)
assert os.path.isdir(args.split_dir)

# Save settings for reproducibility
settings = vars(args)
with open(os.path.join(args.results_dir, f"experiment_{args.exp_code}.txt"), "w") as f:
    for key, value in settings.items():
        f.write(f"{key}: {value}\n")

print("################# Settings ###################", flush=True)
for key, val in settings.items():
    print("{}:  {}".format(key, val))

if __name__ == "__main__":
    main(args)
    print("Training complete!")
