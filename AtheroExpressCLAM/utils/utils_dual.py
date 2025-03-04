import pickle
import torch
import numpy as np
import torch.nn as nn
import pdb

import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
import pdb
import torch.nn.functional as F
import math
from itertools import islice
import collections
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubsetSequentialSampler(Sampler):
	"""Samples elements sequentially from a given list of indices, without replacement.

	Arguments:
		indices (sequence): a sequence of indices
	"""
	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return iter(self.indices)

	def __len__(self):
		return len(self.indices)

def collate_dual_stain(batch):
    """
    Custom collate function for dual-stain datasets.
    Batches `features_list`, `coords_list`, and `stain_list` into lists of lists.
	Custom collate function to skip None values.
    """
    batch = [item for item in batch if item is not None]  # Filter out None
    features_batch = [item[0] for item in batch]  # List of features_list
    labels_batch = torch.LongTensor([item[1] for item in batch])  # Batch labels
    coords_batch = [item[2] for item in batch]  # List of coords_list
    stains_batch = [item[3] for item in batch]  # List of stain_list Stain type(HE/EVG)
    return features_batch, labels_batch, coords_batch, stains_batch

def collate_MIL(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	label = torch.LongTensor([item[1] for item in batch])
	return [img, label]

def collate_features(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	coords = np.vstack([item[1] for item in batch])
	return [img, coords]


def get_simple_loader(dataset, batch_size=1, num_workers=1):
	kwargs = {'num_workers': 4, 'pin_memory': False, 'num_workers': num_workers} if device.type == "cuda" else {}
	loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = collate_MIL, **kwargs)
	return loader 

def get_split_loader(split_dataset, training = False, testing = False, weighted = False):
	"""
		return either the validation loader or training loader 
	"""
	kwargs = {'num_workers': 1} if device.type == "cuda" else {}
	if not testing:
		if training:
			if weighted:
				weights = make_weights_for_balanced_classes_split(split_dataset)
				loader = DataLoader(split_dataset, batch_size=1, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate_dual_stain, **kwargs)	
			else:
				loader = DataLoader(split_dataset, batch_size=1, sampler = RandomSampler(split_dataset), collate_fn = collate_dual_stain, **kwargs)
		else:
			loader = DataLoader(split_dataset, batch_size=1, sampler = SequentialSampler(split_dataset), collate_fn = collate_dual_stain, **kwargs)

	else:
		ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
		loader = DataLoader(split_dataset, batch_size=1, sampler = SubsetSequentialSampler(ids), collate_fn = collate_dual_stain, **kwargs )
	return loader

def get_optim(model, args):
	if args.opt == "adam":
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
	elif args.opt == 'sgd':
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
	else:
		raise NotImplementedError
	return optimizer

def print_network(net):
	num_params = 0
	num_params_train = 0
	print(net)
	
	for param in net.parameters():
		n = param.numel()
		num_params += n
		if param.requires_grad:
			num_params_train += n
	
	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)


def generate_split(cls_ids, val_num, test_num, samples, n_splits=5, seed=7, label_frac=1.0):
    """
    Generate stratified splits by `case_id`, ensuring a consistent number of validation/test samples per class.
    """
    indices = np.arange(samples).astype(int)
    np.random.seed(seed)
    for _ in range(n_splits):
        all_val_ids = []
        all_test_ids = []
        sampled_train_ids = []

        for c in range(len(val_num)):
            possible_indices = cls_ids[c]
			
            # Check if there are enough samples for the requested splits
            if len(possible_indices) < val_num[c] + test_num[c]:
                raise ValueError(
                    f"Not enough samples in class {c} for the requested validation and test splits."
                )
            # Select validation samples
            val_ids = np.random.choice(possible_indices, val_num[c], replace=False)
            remaining_ids = np.setdiff1d(possible_indices, val_ids)
            all_val_ids.extend(val_ids)

            # Select test samples
            test_ids = np.random.choice(remaining_ids, test_num[c], replace=False)
            remaining_ids = np.setdiff1d(remaining_ids, test_ids)
            all_test_ids.extend(test_ids)

            # Select training samples
            if label_frac == 1.0:
                sampled_train_ids.extend(remaining_ids)
            else:
                sample_num = math.ceil(len(remaining_ids) * label_frac)
                sampled_train_ids.extend(
                    np.random.choice(remaining_ids, sample_num, replace=False)
                )

        yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
	if n is None:
		return collections.deque(iterator, maxlen=0)
	else:
		return next(islice(iterator,n, None), default)

def calculate_error(Y_hat, Y):
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

	return error

def make_weights_for_balanced_classes_split(dataset):
    """
    Compute weights for balanced classes in the dataset for dual-stain training.

    Args:
        dataset: A Generic_Split object with expanded_indices and slide_cls_ids.

    Returns:
        weights: A list of weights for each sample in the expanded dataset.
    """
    # Total number of samples in the expanded dataset
    N = float(len(dataset.expanded_indices))

    # Calculate weight per class using slide_cls_ids, which stores indices for expanded dataset
    weight_per_class = [
        N / (len(dataset.slide_cls_ids[c]) + 1e-6)  # Avoid division by zero
        for c in range(len(dataset.slide_cls_ids))
    ]
    print(f"Weight per class: {weight_per_class}")

    # Assign weights to each expanded sample
    weights = [0] * int(N)
    for expanded_idx, (slide_idx, stain) in enumerate(dataset.expanded_indices):
        label = int(dataset.slide_data.iloc[slide_idx]["label"])
        weights[expanded_idx] = weight_per_class[label]

    return weights	
	# N = float(len(dataset))                                           
	# weight_per_class = [N/(len(dataset.slide_cls_ids[c]) + 1e-6) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
	# print(f'weight per class: {weight_per_class}')
	# weight = [0] * int(N)                                           
	# for idx in range(len(dataset)):
	# 	base_idx, _ = dataset.expanded_indices[idx]
	# 	label = int(dataset.slide_data.iloc[base_idx]["label"])
	# 	weight[idx] = weight_per_class[label]
	# return torch.DoubleTensor(weight)  
		# y = dataset.getlabel(idx)                      
		# weight[idx] = weight_per_class[y]return torch.DoubleTensor(weight)

def initialize_weights(module):
	for m in module.modules():
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			m.bias.data.zero_()
		
		elif isinstance(m, nn.BatchNorm1d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)

