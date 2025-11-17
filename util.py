import os
import sys
import errno
import pickle
import math
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from torchmetrics import MetricCollection, Accuracy, Precision, Recall
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict, Union, Any, Optional
from torch.nn import functional as F
import gzip

from contextlib import nullcontext


import seaborn as sns
import matplotlib.pyplot as plt

from collections import defaultdict
import torch

from sklearn.preprocessing import normalize


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

def build_optimizer(params, optimizer_type: str, lr: float, weight_decay: float = 0.0):
	optimizer_type = optimizer_type.lower()
	if optimizer_type == "adam":
		return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
	if optimizer_type == "adamw":
		return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
	if optimizer_type == "sgd":
		return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
	raise ValueError(f"Unknown optimizer_type: {optimizer_type}")

def train(
	model: nn.Module,
	train_dataloader: DataLoader,
	optimizer_type: str = "adamw",
	lr: float = 1e-3,
	weight_decay: float = 1e-4,
	device: str = "cuda",
	metrics: Optional[nn.Module] = None,   # e.g., torchmetrics.MetricCollection
	amp: bool = True,
	grad_clip: Optional[float] = 1.0,
	return_gradients: bool = False
) -> Dict[str, Any]:

	model.to(device)
	if metrics is not None:
		metrics.to(device)
		metrics.reset()

	optimizer = build_optimizer(model.parameters(), optimizer_type, lr, weight_decay)
	loss_fn = nn.CrossEntropyLoss()

	model.train()
	scaler = torch.cuda.amp.GradScaler(enabled=amp and torch.cuda.is_available())

	running_loss = 0.0
	num_batches = 0

	# AMP context manager
	autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16, enabled=amp and torch.cuda.is_available())

	for x, y in train_dataloader:
		x = x.to(device, non_blocking=True)
		y = y.to(device, non_blocking=True).long()

		optimizer.zero_grad(set_to_none=True)

		with autocast_ctx:
			logits = model(x)            # logits (no softmax in model)
			loss = loss_fn(logits, y)

		# backward
		scaler.scale(loss).backward()
		if grad_clip is not None:
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
		scaler.step(optimizer)
		scaler.update()

		running_loss += loss.detach().item()
		num_batches += 1

		if metrics is not None:
			# If using torchmetrics, either update() then compute() after epoch,
			# or pass predictions/targets if your metrics are callable.
			metrics.update(logits, y)

	epoch_loss = running_loss / max(1, num_batches)

	results = {"loss": epoch_loss}
	if metrics is not None:
		computed = metrics.compute()
		# Torchmetrics returns tensors; convert to Python numbers
		results.update({k: (v.item() if torch.is_tensor(v) else float(v)) for k, v in computed.items()})
		metrics.reset()

	if return_gradients:
		grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
		results["gradients"] = grads

	return results

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

	for layer, module in model.named_children():
		for name, w in module.named_parameters(recurse=False):
			if "weight" not in name:
				continue
			w = w.detach()

			if w.ndim == 4:        # Conv2d: (out_c, in_c, kH, kW)
				row_dim, col_dim = 2, 3
			elif w.ndim == 2:      # Linear: (out_f, in_f)
				row_dim, col_dim = 1, 0
			elif w.ndim == 1:      # BatchNorm: (C,)
				# define a sensible “row/col” signature for 1D (e.g., identity)
				layer_to_model_sig_row[layer] = w.clone()
				layer_to_model_sig_col[layer] = w.clone()
				continue
			else:
				# Skip unusual shapes
				sys.exit(f"Unexpected weight tensor shape {w.shape} in layer {layer}")

			layer_to_model_sig_row[layer] = torch.sum(w, dim=row_dim)
			layer_to_model_sig_col[layer] = torch.sum(w, dim=col_dim)

	return layer_to_model_sig_row, layer_to_model_sig_col


def get_num_total_model_params(model):
	total_num_model_params = 0
	# not including bias
	for layer_name, params in model.named_parameters():
		if 'weight' in layer_name:
			total_num_model_params += params.numel()
	return total_num_model_params    
	
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
	with torch.no_grad():
		for layer, module in model.named_children():
			w = getattr(module, "weight", None)
			if w is None:
				continue
			w_np = w.detach().cpu().numpy()
			mask = np.ones_like(w_np, dtype=np.int32)
			mask[w_np == 0] = 0
			layer_to_mask[layer] = mask
	return dict(sorted(layer_to_mask.items(), key=lambda kv: kv[0]))

def _as_numpy_flat(x):
	import numpy as np, torch
	if isinstance(x, np.ndarray):
		return x.reshape(-1)
	if torch.is_tensor(x):
		return x.detach().cpu().numpy().reshape(-1)
	return np.asarray(x).reshape(-1)

def calc_overlapping_mask_percent(latest_block_global_model, validator_model, worker_model):
	val_mask = calc_mask_from_model_with_mask_object(validator_model)
	worker_mask = calc_mask_from_model_with_mask_object(worker_model)
	if not val_mask or not worker_mask:
		return 0

	# Flatten the masks
	# val_mask = np.concatenate([layer_mask.cpu().flatten().numpy() for layer_mask in val_mask.values()])
	# worker_mask = np.concatenate([layer_mask.cpu().flatten().numpy() for layer_mask in worker_mask.values()])

	val_mask    = np.concatenate([_as_numpy_flat(m) for m in val_mask.values()])
	worker_mask = np.concatenate([_as_numpy_flat(m) for m in worker_mask.values()])

	global_mask = calc_mask_from_model_without_mask_object(latest_block_global_model)
	# global_mask = np.concatenate([layer_mask.cpu().flatten().numpy() for layer_mask in global_mask.values()])
	global_mask = np.concatenate([_as_numpy_flat(m) for m in global_mask.values()])

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