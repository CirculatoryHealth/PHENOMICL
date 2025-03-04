#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function

print(
    "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
)
print("                                         Main Modeling")
print("")
print("* Version          : v1.0.0")
print("")
print("* Last update      : 2023-10-03")
print("* Written by       : Francesco Cisternino & Yipei (Petra) Song")
print(
    "* Edite by         : Craig Glastonbury | Sander W. van der Laan | Clint L. Miller | Yipei Song | Francesco Cisternino."
)
print("")
print("* Description      : Main modeling.")
print("")
print("                     [1] https://github.com/MaryamHaghighat/PathProfiler")
print("")
print(
    "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
)

import argparse
import pdb
import os
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils_trans import train
from datasets.dataset_generic import (
    Generic_WSI_Classification_Dataset,
    Generic_MIL_Dataset,
)

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

from utils.eval_utils import initiate_model
from models.model_clam_sum import CLAM_SB, CLAM_MB


def load_pretrained_model(args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    model.train()
    model.cuda()
    return model


def create_model(args):
   if args.model_type == 'clam_sb':
       model = CLAM_SB(dropout=args.drop_out, size_arg=args.model_size, n_classes=args.n_classes)
   elif args.model_type == 'clam_mb':
       model = CLAM_MB(dropout=args.drop_out, size_arg=args.model_size, n_classes=args.n_classes)
  # elif args.model_type == 'mil':
       # Assuming MIL also takes similar arguments
      # model = MIL_fc(dropout=args.drop_out, n_classes=args.n_classes)
   else:
       raise ValueError("Unsupported model type")

   model.cuda()  # Move model to CUDA if available
   return model


def get_model(args, checkpoint=None):
    if checkpoint is not None:
        model = load_pretrained_model(args, checkpoint)
    else:
        model = create_model(args)
        # model = None
    model.cuda()
    return model


# def main(args, preloaded_model=None):
def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    # k_start and k_end for the k-fold Cross Validation
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    # Accuracy and AUC lists
    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    # folds: list of folds (e.g. [0,1,2,3] for 4 folds)
    folds = np.arange(start, end)

    for i in folds:
        model = get_model(args, args.ckpt_path)
        # seed for reproducibility
        seed_torch(args.seed)
        # Return splits from the csv file (the split dir is passed as argument, the number after the underscore
        # is equal to the number of the current fold (i)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(
            from_id=False, csv_path="{}/splits_{}.csv".format(args.split_dir, i)
        )

        datasets = (train_dataset, val_dataset, test_dataset)
        # Train with the current split
        results, test_auc, val_auc, test_acc, val_acc = train(datasets, i, args, model)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        # write results to pkl
        filename = os.path.join(args.results_dir, "split_{}_results.pkl".format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame(
        {
            "folds": folds,
            "test_auc": all_test_auc,
            "val_auc": all_val_auc,
            "test_acc": all_test_acc,
            "val_acc": all_val_acc,
        }
    )

    if len(folds) != args.k:
        save_name = "summary_partial_{}_{}.csv".format(start, end)
    else:
        save_name = "summary.csv"
    final_df.to_csv(os.path.join(args.results_dir, save_name))


# Generic training settings
parser = argparse.ArgumentParser(description="Configurations for WSI Training")
# data_root_dir => see files organization in README.md
parser.add_argument(
    "--data_root_dir", type=str, default="../GTEX_FEATURES_DINO", help="data directory"
)
parser.add_argument(
    "--max_epochs",
    type=int,
    default=200,
    help="maximum number of epochs to train (default: 200)",
)
parser.add_argument(
    "--lr", type=float, default=1e-4, help="learning rate (default: 0.0001)"
)
parser.add_argument(
    "--label_frac",
    type=float,
    default=1.0,
    help="fraction of training labels (default: 1.0)",
)
parser.add_argument(
    "--reg", type=float, default=1e-3, help="weight decay (default: 1e-5)"
)
parser.add_argument(
    "--csv_dataset",
    type=str,
    default="/hpc/dhl_ec/VirtualSlides/Projects/CONVOCALS/AtheroexpressCLAM/dataset_csv/WSI_gtex.csv",
    help="path to file containing the csv with labels; this can be balanced or unbalanced depending on --task",
)
## Specific settings
parser.add_argument(
    "--seed",
    type=int,
    default=1,
    help="random seed for reproducible experiment (default: 1)",
)
parser.add_argument("--k", type=int, default=10, help="number of folds (default: 10)")
parser.add_argument(
    "--k_start", type=int, default=-1, help="start fold (default: -1, last fold)"
)
parser.add_argument(
    "--k_end", type=int, default=-1, help="end fold (default: -1, first fold)"
)
parser.add_argument(
    "--results_dir", default="./results", help="results directory (default: ./results)"
)
parser.add_argument(
    "--split_dir",
    type=str,
    default=None,
    help="manually specify the set of splits to use, "
    + "instead of infering from the task and label_frac argument (default: None)",
)
parser.add_argument(
    "--log_data", action="store_true", default=False, help="log data using tensorboard"
)
parser.add_argument(
    "--testing", action="store_true", default=False, help="debugging tool"
)
parser.add_argument(
    "--early_stopping", action="store_true", default=False, help="enable early stopping"
)
parser.add_argument("--opt", type=str, choices=["adam", "sgd"], default="adam")
parser.add_argument(
    "--drop_out", action="store_true", default=False, help="enabel dropout (p=0.25)"
)
parser.add_argument(
    "--bag_loss",
    type=str,
    choices=["svm", "ce", "focal"],
    default="ce",
    help="slide-level classification loss function (default: ce)",
)
parser.add_argument(
    "--model_type",
    type=str,
    choices=["clam_sb", "clam_mb", "mil"],
    default="clam_sb",
    help="type of model (default: clam_sb, clam w/ single attention branch)",
)
parser.add_argument("--exp_code", type=str, help="experiment code for saving results")
parser.add_argument(
    "--weighted_sample",
    action="store_true",
    default=False,
    help="enable weighted sampling",
)
parser.add_argument(
    "--apply_bag_augmentation",
    action="store_true",
    default=False,
    help="enable weighted sampling",
)
parser.add_argument(
    "--model_size",
    type=str,
    choices=["small", "big", "dino_version", "imagenet"],
    default="dino_version",
    help="size of model, does not affect mil",
)
parser.add_argument(
    "--task",
    type=str,
    choices=[
        "task_1_tumor_vs_normal",
        "task_2_tumor_subtyping",
        "wsi_classification",
        "wsi_classification_binary",
        "wsi_classification_binary_eq",
        "wsi_classification_binary_gender",
    ],
    default="wsi_classification",
)

### CLAM specific options

parser.add_argument(
    "--ckpt_path",
    type=str,
    default=None,
    help="Path to the pretrained model checkpoint. Set to None for training from scratch.",
)

parser.add_argument(
    "--no_inst_cluster",
    action="store_true",
    default=False,
    help="disable instance-level clustering",
)
parser.add_argument(
    "--inst_loss",
    type=str,
    choices=["svm", "ce", None],
    default=None,
    help="instance-level clustering loss function (default: None)",
)
parser.add_argument(
    "--subtyping", action="store_true", default=True, help="subtyping problem"
)
parser.add_argument(
    "--bag_weight",
    type=float,
    default=1.0,
    help="clam: weight coefficient for bag-level loss (default: 0.7)",
)
parser.add_argument(
    "--B",
    type=int,
    default=96,
    help="number of positive/negative patches to sample for clam",
)


# for transfer learning
parser.add_argument(
    "--n_classes", type=int, required=True, help="Number of classes for the model."
)

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# for transfer learning
args.n_classes = 2


# for transfer learning
def create_model(args):
    print("\nInit Model...", end=" ")
    model_dict = {"dropout": args.drop_out, "n_classes": args.n_classes}
    # Model type
    if args.model_type == "clam" and args.subtyping:
        model_dict.update({"subtyping": True})
    # Model size
    if args.model_size is not None and args.model_type != "mil":
        model_dict.update({"size_arg": args.model_size})
    # Model type 2: single branch, multiple branch
    if args.model_type in ["clam_sb", "clam_mb"]:
        if args.subtyping:
            model_dict.update({"subtyping": True})
        # B parameter for clustering
        if args.B > 0:
            model_dict.update({"k_sample": args.B})

        ## CREATION OF THE MODEL
        if args.model_type == "clam_sb":
            model = CLAM_SB(**model_dict)
        elif args.model_type == "clam_mb":
            model = CLAM_MB(**model_dict)
        else:
            raise NotImplementedError

    model.relocate()
    return model


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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoding_size = 384
settings = {
    "num_splits": args.k,
    "k_start": args.k_start,
    "k_end": args.k_end,
    "task": args.task,
    "max_epochs": args.max_epochs,
    "results_dir": args.results_dir,
    "lr": args.lr,
    "experiment": args.exp_code,
    "reg": args.reg,
    "label_frac": args.label_frac,
    "bag_loss": args.bag_loss,
    "seed": args.seed,
    "model_type": args.model_type,
    "model_size": args.model_size,
    "use_drop_out": args.drop_out,
    "weighted_sample": args.weighted_sample,
    "apply_bag_augmentation": args.apply_bag_augmentation,
    "opt": args.opt,
}

if args.model_type in ["clam_sb", "clam_mb"]:
    settings.update(
        {"bag_weight": args.bag_weight, "inst_loss": args.inst_loss, "B": args.B}
    )

print("\nLoad Dataset")

if args.task == "task_1_tumor_vs_normal":
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(
        csv_path="dataset_csv/tumor_vs_normal_dummy_clean.csv",
        data_dir=os.path.join(args.data_root_dir, "tumor_vs_normal_resnet_features"),
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict={"normal_tissue": 0, "tumor_tissue": 1},
        patient_strat=False,
        ignore=[],
    )

elif args.task == "task_2_tumor_subtyping":
    args.n_classes = 3
    dataset = Generic_MIL_Dataset(
        csv_path="dataset_csv/tumor_subtyping_dummy_clean.csv",
        data_dir=os.path.join(args.data_root_dir, "tumor_subtyping_resnet_features"),
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict={"subtype_1": 0, "subtype_2": 1, "subtype_3": 2},
        patient_strat=False,
        ignore=[],
    )

elif args.task == "wsi_classification":
    args.n_classes = 6
    dataset = Generic_MIL_Dataset(
        csv_path=args.csv_dataset,  # './dataset_csv/AtheroExpress_WSI_dataset_binary.csv',
        data_dir=os.path.join(args.data_root_dir, ""),
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict={
            "TIA": 0,
            "Asymptomatic": 1,
            "Stroke": 2,
            "Retinal infarction": 3,
            "Ocular": 4,
            "Other": 5,
        },
        patient_strat=False,
        ignore=[],
    )

elif args.task == "wsi_classification_binary":
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(
        csv_path=args.csv_dataset,  # './dataset_csv/AtheroExpress_FSMA_WSI_dataset_binary_IPH.csv',
        data_dir=os.path.join(args.data_root_dir, ""),
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict={"no": 0, "yes": 1},
        patient_strat=False,
        apply_bag_augmentation=args.apply_bag_augmentation,
        ignore=[],
    )

elif args.task == "wsi_classification_binary_gender":
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(
        csv_path=args.csv_dataset,
        data_dir=os.path.join(args.data_root_dir, ""),
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict={"male": 0, "female": 1},
        patient_strat=False,
        apply_bag_augmentation=args.apply_bag_augmentation,
        ignore=[],
    )

elif args.task == "wsi_classification_binary_eq":
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(
        csv_path=args.csv_dataset,  # './dataset_csv/AtheroExpress_SMA_WSI_dataset_binary_IPH_eq.csv',
        data_dir=os.path.join(args.data_root_dir, ""),
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict={"no": 0, "yes": 1},
        patient_strat=False,
        apply_bag_augmentation=args.apply_bag_augmentation,
        ignore=[],
    )


if args.model_type in ["clam_sb", "clam_mb"]:
    assert args.subtyping

else:
    raise NotImplementedError

if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(
    args.results_dir, str(args.exp_code) + "_s{}".format(args.seed)
)
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)
print("args.results_dir:", args.results_dir)

if args.split_dir is None:
    args.split_dir = os.path.join(
        "splits", args.task + "_{}".format(int(args.label_frac * 100))
    )
else:
    args.split_dir = os.path.join("splits", args.split_dir)

print("split_dir: ", args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({"split_dir": args.split_dir})


with open(args.results_dir + "/experiment_{}.txt".format(args.exp_code), "w") as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################", flush=True)
for key, val in settings.items():
    print("{}:  {}".format(key, val))

if __name__ == "__main__":
    #    args = parser.parse_args()

    # if args.ckpt_path:
    #     preloaded_model = load_pretrained_model(args, args.ckpt_path)
    # else:
    #     print("Checkpoint path must be provided for training.")
    #     preloaded_model = None
    #     # sys.exit(1)  # Exit the script if no checkpoint path is provided

    # results = main(args, preloaded_model)
    results = main(args)
    print("Training complete!")
    print("finished!")
    print("end script")

print(
    "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
)
print(
    "+ The MIT License (MIT)                                                                                               +"
)
print(
    "+ Copyright (c) 2023 Francesco Cisternino | Craig Glastonbury | Sander W. van der Laan | Clint L. Miller | Yipei Song +"
)
print(
    "+                                                                                                                     +"
)
print(
    "+ Permission is hereby granted, free of charge, to any person obtaining a copy of this software and                   +"
)
print(
    '+ associated documentation files (the "Software"), to deal in the Software without restriction, including           +'
)
print(
    "+ without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell             +"
)
print(
    "+ copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the            +"
)
print(
    "+ following conditions:                                                                                               +"
)
print(
    "+                                                                                                                     +"
)
print(
    "+ The above copyright notice and this permission notice shall be included in all copies or substantial                +"
)
print(
    "+ portions of the Software.                                                                                           +"
)
print(
    "+                                                                                                                     +"
)
print(
    '+ THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT             +'
)
print(
    "+ LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO           +"
)
print(
    "+ EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER           +"
)
print(
    "+ IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR             +"
)
print(
    "+ THE USE OR OTHER DEALINGS IN THE SOFTWARE.                                                                          +"
)
print(
    "+                                                                                                                     +"
)
print(
    "+ Reference: http://opensource.org.                                                                                   +"
)
print(
    "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
)
