#!/usr/bin/env python3
# -*- coding: utf-8 -*-
print(
    "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
)
print("                                                 Create Splits")
print("")
print("* Version          : v1.0.0")
print("")
print("* Last update      : 2023-10-03")
print("* Written by       : Yipei (Petra) Song")
print(
    "* Edite by         : Craig Glastonbury | Sander W. van der Laan | Clint L. Miller | Yipei Song | Francesco Cisternino."
)
print("")
print(
    "* Description      : Create training, validation, and test datasets for the modeling task."
)
print("")
print("                     [1] https://github.com/MaryamHaghighat/PathProfiler")
print("")
print(
    "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
)

import pdb
import os
import pandas as pd
from datasets.dataset_generic import (
    Generic_WSI_Classification_Dataset,
    Generic_MIL_Dataset,
    save_splits,
)
import argparse
import numpy as np

parser = argparse.ArgumentParser(
    description="Creating splits for whole slide classification"
)
parser.add_argument(
    "--label_frac", type=float, default=1.0, help="fraction of labels (default: 1)"
)
parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
parser.add_argument("--k", type=int, default=10, help="number of splits (default: 10)")
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
)

# DATA DIRECTORY
parser.add_argument(
    "--csv_dataset",
    type=str,
    default="/hpc/dhl_ec/VirtualSlides/Projects/CONVOCALS/AtheroexpressCLAM/dataset_csv/WSI_gtex.csv",
    help="path to file containing the csv with labels",
)
parser.add_argument(
    "--val_frac",
    type=float,
    default=0.1,
    help="fraction of labels for validation (default: 0.1)",
)
parser.add_argument(
    "--test_frac",
    type=float,
    default=0.1,
    help="fraction of labels for test (default: 0.1)",
)

args = parser.parse_args()

if args.task == "task_1_tumor_vs_normal":
    args.n_classes = 2
    dataset = Generic_WSI_Classification_Dataset(
        csv_path=args.csv_dataset,
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict={"normal_tissue": 0, "tumor_tissue": 1},
        patient_strat=True,
        ignore=[],
    )

elif args.task == "task_2_tumor_subtyping":
    args.n_classes = 3
    dataset = Generic_WSI_Classification_Dataset(
        csv_path=args.csv_dataset,
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict={"subtype_1": 0, "subtype_2": 1, "subtype_3": 2},
        patient_strat=True,
        patient_voting="maj",
        ignore=[],
    )

elif args.task == "wsi_classification":
    args.n_classes = 6
    dataset = Generic_WSI_Classification_Dataset(
        csv_path=args.csv_dataset,
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
        patient_strat=True,
        patient_voting="maj",
        ignore=[],
    )

elif args.task == "wsi_classification_binary":
    args.n_classes = 2
    dataset = Generic_WSI_Classification_Dataset(
        csv_path=args.csv_dataset,
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict={"no": 0, "yes": 1},
        patient_strat=True,
        patient_voting="maj",
        ignore=[],
    )

elif args.task == "wsi_classification_binary_gender":
    args.n_classes = 2
    dataset = Generic_WSI_Classification_Dataset(
        csv_path=args.csv_dataset,
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict={"male": 0, "female": 1},
        patient_strat=True,
        patient_voting="maj",
        ignore=[],
    )

elif args.task == "wsi_classification_binary_eq":
    args.n_classes = 2
    dataset = Generic_WSI_Classification_Dataset(
        csv_path=args.csv_dataset,
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict={"no": 0, "yes": 1},
        patient_strat=True,
        patient_voting="maj",
        ignore=[],
    )

else:
    raise NotImplementedError

num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
print(f"num_slides_cls: {num_slides_cls}; total: {sum(num_slides_cls )}")
val_num = np.round(num_slides_cls * args.val_frac).astype(int)
print(f"val_num: {val_num}")
test_num = np.round(num_slides_cls * args.test_frac).astype(int)
print(f"test_num: {test_num}")

if __name__ == "__main__":
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]

    for lf in label_fracs:
        # split_dir = 'splits_SMA_IPH/'+ str(args.task) + '_{}'.format(int(lf * 100))+'_k'+str(args.k)
        split_dir = (
            os.path.dirname(args.csv_dataset)
            + "/"
            + str(args.task)
            + "_transfer_HE_{}".format(int(lf * 100))
            + "_k"
            + str(args.k)
        )
        os.makedirs(split_dir, exist_ok=True)
        dataset.create_splits(
            k=args.k, val_num=val_num, test_num=test_num, label_frac=lf
        )
        for i in range(args.k):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            save_splits(
                splits,
                ["train", "val", "test"],
                os.path.join(split_dir, "splits_{}.csv".format(i)),
            )
            save_splits(
                splits,
                ["train", "val", "test"],
                os.path.join(split_dir, "splits_{}_bool.csv".format(i)),
                boolean_style=True,
            )
            descriptor_df.to_csv(
                os.path.join(split_dir, "splits_{}_descriptor.csv".format(i))
            )

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
