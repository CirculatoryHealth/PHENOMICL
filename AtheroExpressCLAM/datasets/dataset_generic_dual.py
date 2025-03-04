from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
from scipy import stats
from augmentation import bag_augmentation
import sys
from torch.utils.data import Dataset
import h5py

from utils.utils_dual import generate_split, nth

# def save_splits(split_datasets, column_keys, filename, boolean_style=False):
#     splits = [split_datasets[i].slide_data['case_id'] for i in range(len(split_datasets))]
#     if not boolean_style:
#         df = pd.concat(splits, ignore_index=True, axis=1)
#         df.columns = column_keys
#     else:
#         df = pd.concat(splits, ignore_index=True, axis=0)
#         index = df.values.tolist()
#         one_hot = np.eye(len(split_datasets)).astype(bool)
#         bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
#         if len(index) != bool_array.shape[0]:
#             raise ValueError(
#                 f"Mismatch between index length ({len(index)}) and bool_array rows ({bool_array.shape[0]})"
#             )
#         df = pd.DataFrame(bool_array, index=index, columns=column_keys)

#     df.to_csv(filename)
#     print(f"Splits saved to {filename}")

def save_splits(split_datasets, column_keys, filename, boolean_style=False):
    """
    Save splits in CSV format with or without boolean-style representation.

    Args:
        split_datasets: List of datasets for each split (train, val, test).
        column_keys: List of column names for the CSV (e.g., ['train', 'val', 'test']).
        filename: Path to save the CSV file.
        boolean_style: If True, saves a boolean-style split representation.
    """
    splits = [split_datasets[i].slide_data['case_id'] for i in range(len(split_datasets))]
    if not boolean_style:
        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = column_keys
    else:
        # Concatenate case_ids and deduplicate
        df = pd.concat(splits, ignore_index=True, axis=0)
        index = df.drop_duplicates().values.tolist()
        bool_array = np.zeros((len(index), len(split_datasets)), dtype=bool)

        for split_idx, split in enumerate(splits):
            for case_id in split.values:
                bool_array[index.index(case_id), split_idx] = True

        df = pd.DataFrame(bool_array, index=index, columns=column_keys)

    df.to_csv(filename)
    print(f"Splits saved to {filename}")

    

class Generic_WSI_Classification_Dataset(Dataset):
    def __init__(self,
                 csv_path,
                 stain_data_dirs,
                 shuffle=False,
                 seed=7,
                 print_info=True,
                 label_dict={"no": 0, "yes": 1},
                 ignore=[],
                 patient_strat=True,
                 label_col_HE="label_HE",
                 label_col_EVG="label_EVG",
                 apply_bag_augmentation=None):
        """
        Args:
            csv_path (string): Path to the combined CSV file with annotations.
            stain_data_dirs (dict): Dictionary mapping stain types (e.g., 'HE', 'EVG') to their respective directories.
            shuffle (boolean): Whether to shuffle the data.
            seed (int): Random seed for shuffling the data.
            print_info (boolean): Whether to print a summary of the dataset.
            label_dict (dict): Dictionary for converting str labels to int.
            ignore (list): List containing class labels to ignore.
            patient_strat (boolean): Whether to stratify splits by patient.
        """
        self.label_dict = label_dict
        self.num_classes = len(set(self.label_dict.values()))
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.stain_data_dirs = stain_data_dirs
        self.apply_bag_augmentation = apply_bag_augmentation

        # Load and process the CSV
        slide_data = pd.read_csv(csv_path)
        slide_data = self.df_prep(slide_data, label_dict, ignore, label_col_HE, label_col_EVG)

        # Shuffle data
        if shuffle:
            np.random.seed(seed)
            slide_data = slide_data.sample(frac=1).reset_index(drop=True)

        self.slide_data = slide_data
        self.patient_data_prep()
        self.cls_ids_prep()  # Initialize patient_cls_ids

        if print_info:
            self.summarize()

    def df_prep(self, data, label_dict, ignore, label_col_HE, label_col_EVG):
        """
        Prepare the dataset by processing the CSV file.
        """
        for label_col in [label_col_HE, label_col_EVG]:
            if label_col in data.columns:
                data[label_col] = data[label_col].map(label_dict)

        data['label'] = data[[label_col_HE, label_col_EVG]].max(axis=1)
        mask = data['label'].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        return data

    def patient_data_prep(self):
        """
        Prepare patient-level data for stratified splitting.
        """
        print("patient level number:", len(self.slide_data))
        patients = self.slide_data["case_id"].unique()
        print("unique patient number:", len(patients))
        patient_labels = []

        for p in patients:
            locations = self.slide_data[self.slide_data["case_id"] == p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data.loc[locations, "label"].values.max()  # MIL convention
            patient_labels.append(label)

        self.patient_data = {"case_id": patients, "label": np.array(patient_labels)}

    def cls_ids_prep(self):
        """
        Store indices for each class at the patient and slide levels.
        """
        self.patient_cls_ids = [[] for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

        self.slide_cls_ids = [[] for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def create_splits(self, k=3, val_num=(25, 25), test_num=(40, 40), label_frac=1.0):
        """
        Generate splits at the patient level.
        """
        settings = {
            "n_splits": k,
            "val_num": val_num,
            "test_num": test_num,
            "label_frac": label_frac,
            "seed": self.seed,
            "cls_ids": self.patient_cls_ids,
            "samples": len(self.patient_data["case_id"]),
        }
        self.split_gen = generate_split(**settings)

    def set_splits(self, start_from=None):
        """
        Assign splits to training, validation, and test sets.
        """
        if start_from:
            ids = nth(self.split_gen, start_from)
        else:
            ids = next(self.split_gen)

        slide_ids = [[] for _ in range(len(ids))]
        for split in range(len(ids)):
            for idx in ids[split]:
                case_id = self.patient_data["case_id"][idx]
                slide_indices = self.slide_data[self.slide_data["case_id"] == case_id].index.tolist()
                slide_ids[split].extend(slide_indices)

        self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]

    def test_split_gen(self, return_descriptor=False):
        """
        Generate split descriptors for training, validation, and testing sets.
        """
        if return_descriptor:
            # Prepare descriptor for each split
            descriptor = []
            counts = {}

            for split_name, ids in zip(["train", "val", "test"], [self.train_ids, self.val_ids, self.test_ids]):
                count = len(ids)
                labels = self.getlabel(ids)
                unique, label_counts = np.unique(labels, return_counts=True)
                counts[split_name] = dict(zip(unique, label_counts))

                # Add to descriptor
                descriptor.append({
                    "split": split_name,
                    "total_samples": count,
                    "class_counts": dict(zip(unique, label_counts)),
                })

            # Convert descriptor to DataFrame
            descriptor_df = pd.DataFrame(descriptor)
            return descriptor_df, counts
        else:
            # If not returning descriptor, simply print the counts for debugging
            print("Train split: {} samples".format(len(self.train_ids)))
            print("Validation split: {} samples".format(len(self.val_ids)))
            print("Test split: {} samples".format(len(self.test_ids)))

    def return_splits(self, from_id=True, csv_path=None):
        """
        Return train, validation, and test splits.
        """
        if from_id:
            return (
                self._get_split(self.train_ids),
                self._get_split(self.val_ids),
                self._get_split(self.test_ids),
            )
        else:
            assert csv_path is not None, "CSV path must be provided when from_id is False."
            all_splits = pd.read_csv(csv_path)
            return (
                self._get_split_from_df(all_splits, "train"),
                self._get_split_from_df(all_splits, "val"),
                self._get_split_from_df(all_splits, "test"),
            )

    def _get_split(self, ids):
        """
        Internal method to create a split object.
        """
        data_slice = self.slide_data.loc[ids].reset_index(drop=True)
        patient_data = data_slice.groupby('case_id').first().reset_index() #Return patient-level splits by grouping slides by case_id.
        return Generic_Split(patient_data, self.stain_data_dirs, self.num_classes)
        #return Generic_Split(data_slice, self.stain_data_dirs, self.num_classes)
    
    def _get_split_from_df(self, all_splits, split_key):
        """Create a split from a pre-existing CSV file."""
        split = all_splits[split_key].dropna().reset_index(drop=True)
        mask = self.slide_data['case_id'].isin(split.tolist())
        data_slice = self.slide_data[mask].reset_index(drop=True)
        return Generic_Split(data_slice, self.stain_data_dirs, self.num_classes)

    def getlabel(self, ids):
        return self.slide_data.loc[ids, "label"].values

    def summarize(self):
        """
        Print a summary of the dataset.
        """
        # Count slides for H&E and EVG separately
        num_he_slides = self.slide_data['slide_id_HE'].notna().sum()
        num_evg_slides = self.slide_data['slide_id_EVG'].notna().sum()

        # Total slides
        total_slides = num_he_slides + num_evg_slides
        print("Dataset Summary:")
        print(f"Number of patients: {len(self.patient_data['case_id'])}")
        print(f"Number of slides (H&E): {num_he_slides}")
        print(f"Number of slides (EVG): {num_evg_slides}")
        print(f"Total number of slides: {total_slides}")
        print(f"Number of classes: {self.num_classes}")

    def __getitem__(self, idx):
        """
        Fetch the features and labels for the given index.
        Handles dual-stain data by determining the stain type for the slide.
        """
        row = self.slide_data.iloc[idx]
        
        # Determine the stain type and slide_id
        if pd.notna(row["slide_id_HE"]):  # If HE slide exists
            stain = "HE"
            slide_id = row["slide_id_HE"]
        elif pd.notna(row["slide_id_EVG"]):  # If EVG slide exists
            stain = "EVG"
            slide_id = row["slide_id_EVG"]
        else:
            raise ValueError(f"Missing both HE and EVG slides for case_id: {row['case_id']}")
        
        # Fetch label
        label = row["label"]

        # Fetch features and coordinates from the respective stain's h5 file
        data_dir = self.stain_data_dirs[stain]
        h5_path = os.path.join(data_dir, "h5_files", f"{slide_id}.h5")
        with h5py.File(h5_path, "r") as hdf5_file:
            features = hdf5_file["features"][:]
            coords = hdf5_file["coords"][:]

        # Convert features to PyTorch tensors for model compatibility
        features = torch.from_numpy(features)
        coords = torch.from_numpy(coords)

        return features, label, coords
    

class Generic_Split(Dataset):
    def __init__(self, slide_data, stain_data_dirs, num_classes, train_mode=False, apply_bag_augmentation=None):
        """
        Args:
            slide_data (DataFrame): Data slice for the split.
            stain_data_dirs (dict): Dictionary mapping stain types (e.g., 'HE', 'EVG') to directories.
            num_classes (int): Number of classes.
            train_mode (bool): If True, applies augmentations.
            apply_bag_augmentation (callable): Augmentation function to apply to bags.
        """
        self.slide_data = slide_data
        self.stain_data_dirs = stain_data_dirs
        self.num_classes = num_classes
        self.train_mode = train_mode
        self.apply_bag_augmentation = apply_bag_augmentation

        # Expand the dataset to handle both stains
        self.expanded_indices = []
        for idx in range(len(self.slide_data)):
            row = self.slide_data.iloc[idx]
            if pd.notna(row["slide_id_HE"]):
                self.expanded_indices.append((idx, "HE"))
            if pd.notna(row["slide_id_EVG"]):
                self.expanded_indices.append((idx, "EVG"))

        # Prepare class-wise slide indices based on expanded indices
        self.slide_cls_ids = [[] for _ in range(self.num_classes)]
        for expanded_idx, (slide_idx, stain) in enumerate(self.expanded_indices):
            label = self.slide_data.iloc[slide_idx]["label"]
            self.slide_cls_ids[int(label)].append(expanded_idx)


    def __len__(self):
        return len(self.expanded_indices)

    def __getitem__(self, idx):
        """Fetch features, label, and coordinates for the given index."""
        base_idx, stain = self.expanded_indices[idx]
        row = self.slide_data.iloc[base_idx]

        # Get slide_id and label
        slide_id = row[f"slide_id_{stain}"]
        label = row["label"]

        # Fetch features from the appropriate h5 file
        data_dir = self.stain_data_dirs[stain]
        h5_path = os.path.join(data_dir, "h5_files", f"{slide_id}.h5")
        try:
            with h5py.File(h5_path, "r") as hdf5_file:
                if "features" not in hdf5_file:
                    raise KeyError(f"'features' not found in {h5_path}")
                features = hdf5_file["features"][:]
                coords = hdf5_file["coords"][:]
        except (KeyError, FileNotFoundError) as e:
            # Log the error message to stderr
            print(f"Error accessing {h5_path}: {e}", file=sys.stderr)
            # Skip this slide by returning None
            return None

        # Convert to tensors
        features = torch.from_numpy(features)
        coords = torch.from_numpy(coords)

        # Apply augmentations if needed
        if self.train_mode and self.apply_bag_augmentation:
            features = self.apply_bag_augmentation(features)

        return features, label, coords, stain

