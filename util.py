import os
import sys
import errno
import pickle
import math
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics as skmetrics
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from torchmetrics import MetricCollection, Accuracy, Precision, Recall
import torch.nn.utils.prune as prune
import io
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict, Union
from torch.nn import functional as F
import gzip

import seaborn as sns
import matplotlib.pyplot as plt

from collections import defaultdict, Counter
import torch

from sklearn.preprocessing import normalize

class AddGaussianNoise(object):
	def __init__(self, mean=0., std=1.):
		self.std = std
		self.mean = mean
		
	def __call__(self, tensor):
		return tensor + torch.randn(tensor.size()) * self.std + self.mean
	
	def __repr__(self):
		return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def get_prune_params(model, name='weight') -> List[Tuple[nn.Parameter, str]]:
	# iterate over network layers
	params_to_prune = []
	for _, module in model.named_children():
		for name_, param in module.named_parameters():
			if name in name_:
				params_to_prune.append((module, name))
	return params_to_prune


def get_prune_summary(model, name='weight') -> Dict[str, Union[List[Union[str, float]], float]]:
	num_global_zeros, num_layer_zeros, num_layer_weights = 0, 0, 0
	num_global_weights = 0
	global_prune_percent, layer_prune_percent = 0, 0
	prune_stat = {'Layers': [],
				  'Weight Name': [],
				  'Percent Pruned': [],
				  'Total Pruned': []}
	params_pruned = get_prune_params(model, 'weight')

	for layer, weight_name in params_pruned:

		num_layer_zeros = torch.sum(
			getattr(layer, weight_name) == 0.0).item()
		num_global_zeros += num_layer_zeros
		num_layer_weights = torch.numel(getattr(layer, weight_name))
		num_global_weights += num_layer_weights
		layer_prune_percent = num_layer_zeros / num_layer_weights * 100
		prune_stat['Layers'].append(layer.__str__())
		prune_stat['Weight Name'].append(weight_name)
		prune_stat['Percent Pruned'].append(
			f'{num_layer_zeros} / {num_layer_weights} ({layer_prune_percent:.5f}%)')
		prune_stat['Total Pruned'].append(f'{num_layer_zeros}')

	global_prune_percent = num_global_zeros / num_global_weights

	prune_stat['global'] = global_prune_percent
	return prune_stat

def l1_prune(model, amount=0.00, name='weight', verbose=True):
	"""
		Prunes the model param by param by given amount
	"""
	params_to_prune = get_prune_params(model, name)
	
	for params, name in params_to_prune:
		prune.l1_unstructured(params, name, amount)
		
	if verbose:
		info = get_prune_summary(model, name)
		global_pruning = info['global']
		info.pop('global')
		print(tabulate(info, headers='keys', tablefmt='github'))
		print("Total Pruning: {}%".format(global_pruning * 100))

def produce_mask_from_model_in_place(model):
	# use prune with 0 amount to init mask for the model
	# create mask in-place on model
	if check_mask_object_from_model(model):
		return
	l1_prune(model=model,
				amount=0.00,
				name='weight',
				verbose=False)
	layer_to_masked_positions = {}
	for layer, module in model.named_children():
		for name, weight_params in module.named_parameters():
			if 'weight' in name:
				if weight_params.is_cuda:
					layer_to_masked_positions[layer] = list(zip(*np.where(weight_params.cpu() == 0)))
				else:
					layer_to_masked_positions[layer] = list(zip(*np.where(weight_params == 0)))
		
	for layer, module in model.named_children():
		for name, mask in module.named_buffers():
			if 'mask' in name:
				for pos in layer_to_masked_positions[layer]:
					mask[pos] = 0
					
@torch.no_grad()
def fed_avg(models: List[nn.Module], weight: float, device='cuda:0'):
	"""
		models: list of nn.modules(unpruned/pruning removed)
		weights: normalized weights for each model
		cls:  Class of original model
	"""
	aggr_model = models[0].__class__().to(device)
	model_params = []
	num_models = len(models)
	for model in models:
		model_params.append(dict(model.named_parameters()))

	for name, param in aggr_model.named_parameters():
		param.data.copy_(torch.zeros_like(param.data))
		for i in range(num_models):
			weighted_param = torch.mul(
				model_params[i][name].data, weight)
			param.data.copy_(param.data + weighted_param)
	return aggr_model

@torch.no_grad()
def weighted_fedavg(worker_to_weight, worker_to_model, device='cuda:0'):
	"""
		weights_to_model: dict of accuracy to model, with accuracy being weight
	"""
	workers = worker_to_weight.keys()
	weights = [worker_to_weight[w] for w in workers]
	models = [make_prune_permanent(worker_to_model[w]) for w in workers]

	aggr_model = models[0].__class__().to(device)
	model_params = []
	num_models = len(models)
	for model in models:
		model_params.append(dict(model.named_parameters()))

	for name, param in aggr_model.named_parameters():
		param.data.copy_(torch.zeros_like(param.data))
		for i in range(num_models):
			weighted_param = torch.mul(
				model_params[i][name].data, weights[i])
			param.data.copy_(param.data + weighted_param)
	return aggr_model

def apply_local_mask(model, mask):
	# apply mask in-place to model
	# direct multiplying instead of adding mask object
	if not mask:
		return
	for layer, module in model.named_children():
		for name, weight_params in module.named_parameters():
			if 'weight' in name:
				weight_params.data.copy_(torch.tensor(np.multiply(weight_params.data, mask[layer])))


def create_model_no_prune(cls, device='cuda:0') -> nn.Module:
	model = cls().to(device)
	return model

def create_init_model(cls, device='cuda:0') -> nn.Module:
	model = cls().to(device)
	return model

def create_model(cls, device='cuda:0') -> nn.Module:
	"""
		Returns new model pruned by 0.00 %. This is necessary to create buffer masks
	"""
	model = cls().to(device)
	l1_prune(model, amount=0.00, name='weight', verbose=False)
	return model

def copy_model(model: nn.Module, device='cuda:0'):
	"""
		Returns a copy of the input model.
		Note: the model should have been pruned for this method to work to create buffer masks and whatnot.
	"""
	produce_mask_from_model_in_place(model)
	new_model = create_model(model.__class__, device)
	source_params = dict(model.named_parameters())
	source_buffer = dict(model.named_buffers())
	for name, param in new_model.named_parameters():
		param.data.copy_(source_params[name].data)
	for name, buffer_ in new_model.named_buffers():
		buffer_.data.copy_(source_buffer[name].data)
	return new_model


metrics = MetricCollection([
	Accuracy('MULTICLASS', num_classes = 10),
	Precision('MULTICLASS', num_classes = 10),
	Recall('MULTICLASS', num_classes = 10),
	# F1(), torchmetrics.F1 cannot be imported
])


def train(
	model: nn.Module,
	train_dataloader: DataLoader,
	optimizer_type: str,
	lr: float = 1e-3,
	device: str = 'cuda:0',
	verbose=True
) -> Dict[str, torch.Tensor]:


	if optimizer_type == "Adam":
		optimizer = torch.optim.Adam(lr=lr, params=model.parameters(), weight_decay=1e-4)
	elif optimizer_type == "SGD":
		optimizer = torch.optim.SGD(lr=lr, params=model.parameters(), momentum=0.5)
	model.train(True)
	torch.set_grad_enabled(True)

	# progress_bar = tqdm(enumerate(train_dataloader),
	# 					total=num_batch,
	# 					disable=not verbose,
	# 					)

	# for batch_idx, batch in progress_bar:
	# 	x, y = batch
	# 	x = x.to(device)
	# 	y = y.to(device)
	# 	y_hat = model(x)
	# 	loss = F.cross_entropy(y_hat, y)
	# 	model.zero_grad()

	# 	loss.backward()
	# 	optimizer.step()

	# 	losses.append(loss.item())
	# 	output = metrics(y_hat, y)

	# 	progress_bar.set_postfix({'loss': loss.item(),
	# 							  'acc': output['MulticlassAccuracy'].item()})

	# Default criterion set to NLL loss function
	criterion = nn.NLLLoss().to(device)
	
	for batch_idx, (images, labels) in enumerate(train_dataloader):
		images, labels = images.to(device), labels.to(device)
		# print(labels)
		model.zero_grad()
		log_probs = model(images)
		loss = criterion(log_probs, labels)
		loss.backward()
		
		optimizer.step()

	gradients = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}

	torch.set_grad_enabled(False)
	return gradients

@ torch.no_grad()
def test_by_data_set(
	model: nn.Module,
	data_loader: DataLoader,
	device='cuda:0',
	verbose=True
) -> Dict[str, torch.Tensor]:

	num_batch = len(data_loader)
	model.eval()
	global metrics

	metrics = metrics.to(device)
	progress_bar = tqdm(enumerate(data_loader),
						total=num_batch,
						file=sys.stdout,
						disable=not verbose)
	for batch_idx, batch in progress_bar:
		x, y = batch
		x = x.to(device)
		y = y.to(device)
		y_hat = model(x)

		output = metrics(y_hat, y)

		progress_bar.set_postfix({'acc': output['MulticlassAccuracy'].item()})


	outputs = metrics.compute()
	metrics.reset()
	model.train(True)
	outputs = {k: [v.item()] for k, v in outputs.items()}

	if verbose:
		print(tabulate(outputs, headers='keys', tablefmt='github'))
	return outputs


def get_pruned_ratio_by_weights(model):
	if check_mask_object_from_model(model):
		sys.exit("\033[91m" + "Warning - get_pruned_ratio_by_weights() is called when the model has mask." + "\033[0m")
	total_params_count = get_num_total_model_params(model)
	total_0_count = 0
	total_nan_count = 0
	for layer, module in model.named_children():
		for name, weight_params in module.named_parameters():
			if 'weight' in name:
				if weight_params.is_cuda:
					total_0_count += len(list(zip(*np.where(weight_params.cpu() == 0))))
					total_nan_count += len(torch.nonzero(torch.isnan(weight_params.cpu().view(-1))))
				else:
					total_0_count += len(list(zip(*np.where(weight_params == 0))))
					total_nan_count += len(torch.nonzero(torch.isnan(weight_params.view(-1))))
	if total_nan_count > 0:
		sys.exit("nan bug")
	return total_0_count / total_params_count

def get_pruned_ratio(model):
	if check_mask_object_from_model(model):
		return get_pruned_ratio_by_mask(model)
	return get_pruned_ratio_by_weights(model)

def get_pruned_ratio_by_mask(model):
	if not check_mask_object_from_model(model):
		sys.exit("\033[91m" + "Warning - mask object not found." + "\033[0m")
	total_params_count = get_num_total_model_params(model)
	total_0_count = 0
	for layer, module in model.named_children():
		for name, mask in module.named_buffers():
			if 'mask' in name:
				if mask.is_cuda:
					total_0_count += len(list(zip(*np.where(mask.cpu() == 0))))
				else:
					total_0_count += len(list(zip(*np.where(mask == 0))))
	return total_0_count / total_params_count

def sum_over_model_params(model):
	layer_to_model_sig_row = {}
	layer_to_model_sig_col = {}

	row_dim, col_dim = -1, -1
	for layer, module in model.named_children():
		if 'conv' in layer:
			row_dim, col_dim = 2, 3
		elif 'fc' in layer:
			row_dim, col_dim = 1, 0
		for name, weight_params in module.named_parameters():
			if 'weight' in name:
				if weight_params.is_cuda:
					layer_to_model_sig_row[layer] = torch.sum(weight_params.cpu().detach(), dim=row_dim)
					layer_to_model_sig_col[layer] = torch.sum(weight_params.cpu().detach(), dim=col_dim)
				else:
					layer_to_model_sig_row[layer] = torch.sum(weight_params.detach(), dim=row_dim)
					layer_to_model_sig_col[layer] = torch.sum(weight_params.detach(), dim=col_dim)

	return layer_to_model_sig_row, layer_to_model_sig_col


def get_num_total_model_params(model):
	total_num_model_params = 0
	# not including bias
	for layer_name, params in model.named_parameters():
		if 'weight' in layer_name:
			total_num_model_params += params.numel()
	return total_num_model_params    

def get_model_sig_sparsity(model, model_sig):
	total_num_model_params = get_num_total_model_params(model)
	total_num_sig_non_0_params = 0
	for layer, layer_sig in model_sig.items():
		if layer_sig.is_cuda:
			total_num_sig_non_0_params += len(list(zip(*np.where(layer_sig.cpu()!=0))))
		else:
			total_num_sig_non_0_params += len(list(zip(*np.where(layer_sig!=0))))
	return total_num_sig_non_0_params / total_num_model_params

def generate_mask_from_0_weights(model):
	params_to_prune = get_prune_params(model)
	for param, name in params_to_prune:
		weights = getattr(param, name)
		mask_amount = torch.eq(weights.data, 0.00).sum().item()
		prune.l1_unstructured(param, name, amount=mask_amount)
		
def make_prune_permanent(model): # in place and also return the model
	if check_mask_object_from_model(model):
		params_pruned = get_prune_params(model, name='weight')
		for param, name in params_pruned:
			prune.remove(param, name)
	return model

def check_mask_object_from_model(model):
	for layer, module in model.named_children():
		for name, mask in module.named_buffers():
			if 'mask' in name:
				return True
	return False


def get_model_weights(model):
	"""
	Args:
		model (_torch model_): NN Model

	Returns:
		layer_to_param _dict_: you know!
	"""
	layer_to_param = {} 
	for layer_name, param in model.named_parameters():
		if 'weight' in layer_name:
			layer_to_param[layer_name.split('.')[0]] = param.cpu().detach().numpy()
	return layer_to_param

def flatten_model_weights(model):
	weights = []
	for layer_name, param in model.named_parameters():
		if 'weight' in layer_name:
			weights.append(param.cpu().detach().numpy().flatten())
	return np.concatenate(weights)

def get_local_model_flattened_gradients(local_model, global_model):
	local_model = flatten_model_weights(local_model)
	global_model = flatten_model_weights(global_model)
	return local_model - global_model


def calc_mask_from_model_with_mask_object(model):
	layer_to_mask = {}
	for layer, module in model.named_children():
		for name, mask in module.named_buffers():
			if 'mask' in name:
				layer_to_mask[layer] = mask
	return {layer: layer_to_mask[layer] for layer in sorted(layer_to_mask.keys())} # make sure layers are always in order

def calc_mask_from_model_without_mask_object(model):
	layer_to_mask = {}
	for layer, module in model.named_children():
		for name, weight_params in module.named_parameters():
			if 'weight' in name:
				layer_to_mask[layer] = np.ones_like(weight_params.cpu())
				layer_to_mask[layer][weight_params.cpu() == 0] = 0
	return {layer: layer_to_mask[layer] for layer in sorted(layer_to_mask.keys())} # make sure layers are always in order


def generate_2d_top_magnitude_mask(model_path, percent, check_whole = False, keep_sign = False):

	"""
		returns 2d top magnitude mask.
		1. keep_sign == True
			it keeps the sign of the original weight. Used in introduce noise. 
			returns mask with -1, 1, 0.
		2. keep_sign == False
			calculate absolute magitude mask. Used in calculating weight overlapping.
			returns binary mask with 1, 0.
	"""
	
	layer_to_mask = {}

	with open(model_path, 'rb') as f:
		nn_layer_to_weights = pickle.load(f)
			
	for layer, param in nn_layer_to_weights.items():
	
		# take abs as we show magnitude values
		abs_param = np.absolute(param)

		mask_2d = np.empty_like(abs_param)
		mask_2d[:] = 0 # initialize as 0

		base_size = abs_param.size if check_whole else abs_param.size - abs_param[abs_param == 0].size

		top_boundary = math.ceil(base_size * percent)
					
		percent_threshold = -np.sort(-abs_param.flatten())[top_boundary]

		# change top weights to 1
		mask_2d[np.where(abs_param > percent_threshold)] = 1

		# sanity check
		# one_counts = (mask_2d == 1).sum()
		# print(one_counts/param.size)

		layer_to_mask[layer] = mask_2d
		if keep_sign:
			layer_to_mask[layer] *= np.sign(param)

	# sanity check
	# for layer in layer_to_mask:
	#     print((layer_to_mask[layer] == 1).sum()/layer_to_mask[layer].size)

	return layer_to_mask

def calculate_overlapping_mask(model_paths, check_whole, percent, model_validation=False):
	layer_to_masks = []

	for model_path in model_paths:
		layer_to_masks.append(generate_2d_top_magnitude_mask(model_path, percent, check_whole))

	ref_layer_to_mask = layer_to_masks[0]

	for layer_to_mask_iter in range(len(layer_to_masks[1:])):
		layer_to_mask = layer_to_masks[1:][layer_to_mask_iter]
		for layer, mask in layer_to_mask.items():
			ref_layer_to_mask[layer] *= mask
			if check_whole:
				# for debug - when each local model has high overlapping with the last global model, why the overlapping ratio for all local models seems to be low?
				if model_validation: # called by model_validation()
					print(f"Worker {model_paths[-1].split('/')[-2]}, layer {layer} - overlapping ratio on top {percent:.2%} is {(ref_layer_to_mask[layer] == 1).sum()/ref_layer_to_mask[layer].size/percent:.2%}")
				else:
					print(f"iter {layer_to_mask_iter + 1}, layer {layer} - overlapping ratio on top {percent:.2%} is {(ref_layer_to_mask[layer] == 1).sum()/ref_layer_to_mask[layer].size/percent:.2%}")
		print()

	return ref_layer_to_mask

def what_samples(dataloader):
	'''
		To debug data loader and see if correct number of the specified samples are loaded.
		Sample Usage 1: what_samples(global_test_loader)
		Sample Usage 2: for device in devices_list: what_samples(device._train_loader)
	'''
	label_counts = defaultdict(int)
	for batch in dataloader:
		# Assuming the labels are in the second element of the batch tuple
		_, labels = batch
		for label in labels:
			label_counts[label.item()] += 1

	# Print the label counts
	total = 0
	for label, count in label_counts.items():
		print(f"Label {label}: {count} samples")
		total += count
	print("Total count", total)

def subtract_nested_dicts(dict1, dict2):
	# Get all unique keys from both outer dictionaries
	outer_keys = set(dict1.keys()).union(set(dict2.keys()))
	
	result = {}
	for outer_key in outer_keys:
		result[outer_key] = {}
		# Get all unique keys from both inner dictionaries
		inner_keys = set(dict1.get(outer_key, {}).keys()).union(set(dict2.get(outer_key, {}).keys()))
		for inner_key in inner_keys:
			result[outer_key][inner_key] = dict1.get(outer_key, {}).get(inner_key, 0) - dict2.get(outer_key, {}).get(inner_key, 0)
	
	return result

def calc_overlapping_mask_percent(latest_block_global_model, validator_model, worker_model):
    val_mask = calc_mask_from_model_with_mask_object(validator_model)
    worker_mask = calc_mask_from_model_with_mask_object(worker_model)
    if not val_mask or not worker_mask:
        return 0

    # Flatten the masks
    val_mask = np.concatenate([layer_mask.flatten() for layer_mask in val_mask.values()])
    worker_mask = np.concatenate([layer_mask.flatten() for layer_mask in worker_mask.values()])

    global_mask = calc_mask_from_model_without_mask_object(latest_block_global_model)
    global_mask = np.concatenate([layer_mask.flatten() for layer_mask in global_mask.values()])

    # Convert masks to integer type for NOR operation
    val_mask = val_mask.astype(np.int32)
    worker_mask = worker_mask.astype(np.int32)

    # Identify positions where the global mask is unpruned (1)
    global_unpruned_positions = global_mask == 1

    # Apply NOR logic - did not use XNOR because otherwise free 100% rewards for itself after model pruning ratio converges 
    nor_mask = ~(val_mask | worker_mask) + 2

    # Focus only on positions where the global mask is unpruned
    valid_positions = global_unpruned_positions & (nor_mask == 1)

    # Calculate the percentage of overlapping positions
    overlapping_mask_percent = np.sum(valid_positions) / np.sum(global_unpruned_positions)
    return overlapping_mask_percent

def calc_top_overlapping_gradient_magnitude_percent(latest_block_global_model, validator_model, worker_model, top_percent):
    global_mask = calc_mask_from_model_without_mask_object(latest_block_global_model)
    global_mask = np.concatenate([layer_mask.flatten() for layer_mask in global_mask.values()])
    global_mask_positions = set(zip(*np.where(global_mask == 0)))
    
    # On the unpruned area of the latest_block_global_model (i.e., the training area), find the top x% locations of gradients with largest magnitude, and then calculate overlapping percentage
    worker_gradients = abs(get_local_model_flattened_gradients(worker_model, latest_block_global_model))
    validator_gradients = abs(get_local_model_flattened_gradients(validator_model, latest_block_global_model))
    
    if global_mask_positions:
        worker_gradients = np.delete(worker_gradients, list(global_mask_positions))
        validator_gradients = np.delete(validator_gradients, list(global_mask_positions))
    
    # Find the positions of the top x% gradients
    num_top_elements = int(np.ceil(len(validator_gradients) * top_percent))
    w_threshold = np.partition(worker_gradients, -num_top_elements)[-num_top_elements]
    v_threshold = np.partition(validator_gradients, -num_top_elements)[-num_top_elements]
    
    # Use np.where to get the positions of the top x% gradients
    w_top_positions = set(zip(*np.where(worker_gradients >= w_threshold)))
    v_top_positions = set(zip(*np.where(validator_gradients >= v_threshold)))
    
    # Find the intersection of the positions
    same_positions = w_top_positions.intersection(v_top_positions)

    # Calculate the overlapping percentage
    overlapping_mask_percent = len(same_positions) / num_top_elements
    return overlapping_mask_percent

# def calc_updates_direction(w_grad, v_grad, latest_block_global_model_pruned_ratio):

# 	# Calculate the sign of each element in the arrays
# 	sign_w_grad = np.sign(w_grad)
# 	sign_v_grad = np.sign(v_grad)

# 	# Compare the signs and count the number of elements with the same sign
# 	same_sign_count = np.sum(sign_w_grad == sign_v_grad)
	
# 	# Calculate the percentage of elements with the same sign
# 	total_elements = w_grad.size
# 	percent_same_sign = same_sign_count / total_elements - latest_block_global_model_pruned_ratio
	
# 	return percent_same_sign

def calc_gradient_sign_alignment(worker_model, validator_model, latest_block_global_model):
	
	global_model_mask = calc_mask_from_model_without_mask_object(latest_block_global_model)
	# Flatten the masks
	global_model_mask = np.concatenate([tensor.flatten() for tensor in global_model_mask.values()])

	worker_model_gradients = get_local_model_flattened_gradients(worker_model, latest_block_global_model)
	validator_model_gradients = get_local_model_flattened_gradients(validator_model, latest_block_global_model)

	# Identify positions where global_model_mask is 1
	one_positions = np.nonzero(global_model_mask == 1)[0]

	# Extract elements at these positions
	worker_elements = worker_model_gradients[one_positions]
	validator_elements = validator_model_gradients[one_positions]

	sign_w_grad = np.sign(worker_elements)
	sign_v_grad = np.sign(validator_elements)
	same_sign_count = np.sum(sign_w_grad == sign_v_grad)

	# Calculate sign alignment
	total_positions = one_positions.size
	alignment_percentage = same_sign_count / total_positions

	return alignment_percentage


def plotpos_book(pos_book, log_dir, comm_round, plot_diff=True):
	'''
		Debug the rewarding function to see if it rewards more to the 
		legitimates and less to the maliciouses.
		If not plot_diff, plot the current pos_book.
	'''
	os.makedirs(f'{log_dir}/pos_heat_maps/', exist_ok=True)
	curpos_book = pos_book[comm_round]
	if plot_diff and comm_round > 1:
		# Get the previous pos_book
		prevpos_book = pos_book[comm_round - 1]
		# Calculate the difference between the current and previous pos_books
		curpos_book = subtract_nested_dicts(curpos_book, prevpos_book)
	# Convert the dictionary to a 2D array
	rows = sorted(curpos_book.keys())
	cols = sorted(curpos_book[rows[0]].keys())
	array = [[curpos_book[row][col] for col in cols] for row in rows]

	# Transpose the array to exchange rows and columns
	transposed_array = list(map(list, zip(*array)))

	# Generate the heat map
	plt.figure(figsize=(16, 12))
	sns.heatmap(transposed_array, annot=True, fmt=".4f", cmap='Reds', cbar=True, xticklabels=rows, yticklabels=cols)

	# Add labels and title
	plt.xlabel('Book Owner ID')
	plt.ylabel('Device ID')
	plt.title('POS Heat Map')

	plt.savefig(f'{log_dir}/pos_heat_maps/pos_heat_map_round_{comm_round}.png')

def check_converged(accuracies, threshold=0.05, window=3):
    """
    Check if the accuracies have converged using standard deviation.

    Args:
        accuracies (list): List of accuracy values.
        threshold (float): Maximum standard deviation to consider as converged.
        window (int): Number of recent values to consider.

    Returns:
        bool: True if converged, False otherwise.
    """
    if len(accuracies) < window:
        return False  # Not enough data to check convergence

    recent_values = accuracies[-window:]
    return np.std(recent_values) < threshold

def plot_device_class_distribution(dataset_name, user_labels, log_path):
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    os.makedirs(log_path, exist_ok=True)

    device_ids = sorted(user_labels.keys())
    num_devices = len(device_ids)
    num_classes = 10

    # Initialize matrices
    distribution_matrix = np.zeros((num_devices, num_classes))
    raw_counts = np.zeros((num_devices, num_classes), dtype=int)

    for i, dev in enumerate(device_ids):
        label_dist = user_labels[dev]
        for label in range(num_classes):
            count = label_dist.get(label, 0)
            raw_counts[i, label] = count
            distribution_matrix[i, label] = count

    # Normalize for proportions
    row_sums = distribution_matrix.sum(axis=1, keepdims=True)
    distribution_matrix = np.divide(distribution_matrix, row_sums, out=np.zeros_like(distribution_matrix), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(10, 8))
    left = np.zeros(num_devices)
    colors = plt.cm.tab10(np.arange(num_classes))

    for label in range(num_classes):
        bar = ax.barh(np.arange(num_devices), distribution_matrix[:, label], left=left,
                      color=colors[label], label=str(label), height=1.0)

        for i, rect in enumerate(bar):
            count = raw_counts[i, label]
            if count > 0:
                ax.text(left[i] + distribution_matrix[i, label] / 2,
                        i,
                        str(count),
                        va='center',
                        ha='center',
                        fontsize=8,
                        color='white' if distribution_matrix[i, label] > 0.15 else 'black')
        left += distribution_matrix[:, label]

    ax.set_yticks(np.arange(num_devices))
    ax.set_yticklabels([f"Device {i + 1}" for i in device_ids])
    ax.set_xticks([])
    ax.set_title(f"{dataset_name.upper()} Label Distribution")
    ax.set_ylabel("Devices")
    ax.set_xlabel("Proportion")
    ax.legend(title="Label", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{log_path}/device_label_distribution.png")
    plt.close()