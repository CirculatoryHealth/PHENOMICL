#!/usr/bin/env python3
# -*- coding: utf-8 -*-
print(
    "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
)
print("                                                 Create Splits")
print("")
print("* Version          : v1.0.0")
print("")
print("* Last update      : 2024-12-31")
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
from datasets.dataset_generic_dual import (
    Generic_WSI_Classification_Dataset,
    Generic_Split,
    save_splits,
)
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Creating splits for dual-stain whole slide classification")

parser.add_argument("--label_frac", type=float, default=1.0, help="fraction of labels (default: 1)")
parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
parser.add_argument("--k", type=int, default=10, help="number of splits (default: 10)")
parser.add_argument(
    "--task",
    type=str,
    choices=[
        "wsi_classification_binary_dual_stains",
        "wsi_classification_binary_dual_stains_balanced"
    ],
    required=True,
    help="Specify task type. Choose between unbalanced or balanced dual-stain classification.",
)
parser.add_argument(
    "--val_frac", type=float, default=0.1, help="fraction of labels for validation (default: 0.1)"
)
parser.add_argument(
    "--test_frac", type=float, default=0.1, help="fraction of labels for test (default: 0.1)"
)
parser.add_argument("--csv_he", type=str, required=True, help="path to H&E dataset CSV")
parser.add_argument("--csv_evg", type=str, required=True, help="path to EVG dataset CSV")
parser.add_argument(
    "--data_dirs", type=str, nargs=2, required=True, help="Paths to H&E and EVG data directories"
)

args = parser.parse_args()

# Load and merge datasets
he_df = pd.read_csv(args.csv_he)
evg_df = pd.read_csv(args.csv_evg)

# Merge datasets on case_id
merged_df = pd.merge(
    he_df,
    evg_df,
    on="case_id",
    how="outer",
    suffixes=("_HE", "_EVG"),
)

# Save merged dataset for debugging purposes
merged_csv_path = os.path.join(os.path.dirname(args.csv_he), "merged_dataset_GLYCC.csv") #adjust for fibrin
merged_df.to_csv(merged_csv_path)
print(f"Merged dataset saved at {merged_csv_path}")

# Count unbalanced dataset slides
print("\nUnbalanced Dataset:")
print(f"Number of HE 'yes' slides: {len(merged_df[merged_df['label_HE'] == 'yes'])}")
print(f"Number of HE 'no' slides: {len(merged_df[merged_df['label_HE'] == 'no'])}")
print(f"Number of EVG 'yes' slides: {len(merged_df[merged_df['label_EVG'] == 'yes'])}")
print(f"Number of EVG 'no' slides: {len(merged_df[merged_df['label_EVG'] == 'no'])}")



# Balance the dataset based on slide_id_HE and slide_id_EVG
print("\nBalancing dataset...")
he_df = merged_df[["case_id", "slide_id_HE", "label_HE"]]
evg_df = merged_df[["case_id", "slide_id_EVG", "label_EVG"]]

# Separate based on labels
he_yes = he_df[he_df["label_HE"] == "yes"]
he_no = he_df[he_df["label_HE"] == "no"]
evg_yes = evg_df[evg_df["label_EVG"] == "yes"]
evg_no = evg_df[evg_df["label_EVG"] == "no"]

# Get counts
num_he_yes = len(he_yes)
num_he_no = len(he_no)
num_evg_yes = len(evg_yes)
num_evg_no = len(evg_no)

# Determine the balanced number of slides
balanced_he_count = min(num_he_yes, num_he_no)
balanced_evg_count = min(num_evg_yes, num_evg_no)
print('balanced he count:', balanced_he_count)
print('balanced evg count:', balanced_evg_count)

# Step 1: Select paired slides (same case_id)

paired_yes = merged_df[
    (merged_df["label_HE"] == "yes") & (merged_df["label_EVG"] == "yes")
]
paired_no = merged_df[
    (merged_df["label_HE"] == "no") & (merged_df["label_EVG"] == "no")
]
print('paired_no count:', len(paired_no))
print('paired_yes count:', len(paired_yes))
# Limit the paired samples to the balanced count
paired_yes = paired_yes.sample(
    min(len(paired_yes), balanced_he_count, balanced_evg_count), random_state=args.seed
)
paired_no = paired_no.sample(
    min(len(paired_no), balanced_he_count, balanced_evg_count), random_state=args.seed
)
print('paired_no count after sampling:', len(paired_no))
print('paired_yes count after sampling:', len(paired_yes))

# Step 2: Add remaining HE and EVG slides to balance
remaining_he_yes = he_yes[~he_yes["case_id"].isin(paired_yes["case_id"])]
remaining_evg_yes = evg_yes[~evg_yes["case_id"].isin(paired_yes["case_id"])]
remaining_he_no = he_no[~he_no["case_id"].isin(paired_no["case_id"])]
remaining_evg_no = evg_no[~evg_no["case_id"].isin(paired_no["case_id"])]

print('HE yes slides excluding pairs:', len(remaining_he_yes))
print('EVG yes slides excluding pairs:', len(remaining_evg_yes))
print('HE no slides excluding pairs:', len(remaining_he_no))
print('EVG no slides excluding pairs:', len(remaining_evg_no))



additional_he_yes = remaining_he_yes.sample(
    balanced_he_count - len(paired_yes), random_state=args.seed
)
additional_evg_yes = remaining_evg_yes.sample(
    balanced_evg_count - len(paired_yes), random_state=args.seed
)
additional_he_no = remaining_he_no.sample(
    balanced_he_count - len(paired_no), random_state=args.seed
)
additional_evg_no = remaining_evg_no.sample(
    balanced_evg_count - len(paired_no), random_state=args.seed
)
print('additional_he_yes',len(additional_he_yes))
print('additional_he_no',len(additional_he_no))
print('additional_evg_yes',len(additional_evg_yes))
print('additional_evg_no',len(additional_evg_no))


# Combine paired and additional slides
balanced_yes_pre = pd.merge(additional_he_yes, additional_evg_yes, on="case_id", how="outer")
print('balanced_yes_pre:',len(balanced_yes_pre))
print(balanced_yes_pre)

balanced_no_pre = pd.merge(additional_he_no, additional_evg_no, on="case_id", how="outer")
print('balanced_no_pre:',len(balanced_no_pre))
print(balanced_no_pre)

balanced_pre = pd.concat([balanced_yes_pre, balanced_no_pre]).reset_index(drop=True)
print('balanced_pre:',len(balanced_pre))
print(balanced_pre)

balanced_df = pd.concat([balanced_pre, paired_no, paired_yes]).reset_index(drop=True)

# Save balanced dataset
merged_balanced_csv_path = os.path.join(
    os.path.dirname(args.csv_he), "merged_dataset_balanced_GLYCC.csv" #adjust for fibrin
)
balanced_df.to_csv(merged_balanced_csv_path, index=False)
print(f"Balanced dataset saved at {merged_balanced_csv_path}")

# Count balanced dataset slides
print("\nBalanced Dataset:")
print(f"Number of HE 'yes' slides: {len(balanced_df[balanced_df['label_HE'] == 'yes'])}")
print(f"Number of HE 'no' slides: {len(balanced_df[balanced_df['label_HE'] == 'no'])}")
print(f"Number of EVG 'yes' slides: {len(balanced_df[balanced_df['label_EVG'] == 'yes'])}")
print(f"Number of EVG 'no' slides: {len(balanced_df[balanced_df['label_EVG'] == 'no'])}")

# Initialize dataset
if args.task == "wsi_classification_binary_dual_stains":
    args.n_classes = 2
    dataset = Generic_WSI_Classification_Dataset(
        csv_path=merged_csv_path,
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict={"no": 0, "yes": 1},
        patient_strat=True,
        stain_data_dirs={
            "HE": args.data_dirs[0],
            "EVG": args.data_dirs[1],
        },
    )
elif args.task == "wsi_classification_binary_dual_stains_balanced":
    args.n_classes = 2
    dataset = Generic_WSI_Classification_Dataset(
        csv_path=merged_balanced_csv_path,
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict={"no": 0, "yes": 1},
        patient_strat=True,
        stain_data_dirs={
            "HE": args.data_dirs[0],
            "EVG": args.data_dirs[1],
        },
    )
else:
    raise NotImplementedError


# Determine number of samples per class
num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
print(f"num_slides_cls: {num_slides_cls}; total: {sum(num_slides_cls)}")
val_num = np.round(num_slides_cls * args.val_frac).astype(int)
print(f"val_num: {val_num}")
test_num = np.round(num_slides_cls * args.test_frac).astype(int)
print(f"test_num: {test_num}")

# Create splits
if __name__ == "__main__":
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]

    for lf in label_fracs:
        split_dir = (
            os.path.dirname(args.csv_he)
            + "/"
            + str(args.task)
            + "_dual_stains_GLYCC_{}".format(int(lf * 100)) #adjust for fibrin
            + "_k"
            + str(args.k)
        )
        os.makedirs(split_dir, exist_ok=True)
        dataset.create_splits(
            k=args.k, val_num=val_num, test_num=test_num, label_frac=lf
        )
        for i in range(args.k):
            dataset.set_splits()
            descriptor_df, counts = dataset.test_split_gen(return_descriptor=True)
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

print("Splits created successfully!")
