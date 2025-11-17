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
from typing import Any, Dict, Optional


@torch.no_grad()
def fed_avg(models: List[nn.Module], weights: torch.Tensor, device='cuda:0'):
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
        Note: the model should have been pruned for this method to work to create buffer masks and what not.
    """
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
    device: str = "cuda:0",
    fast_dev_run: bool = False,
    verbose: bool = True,
    metrics: Optional[nn.Module] = None,   # e.g., torchmetrics.MetricCollection
    amp: bool = True,
    grad_clip: Optional[float] = 1.0,
    return_gradients: bool = False,
) -> Dict[str, Any]:

    # Move model to device
    model.to(device)
    
    # Prepare metrics
    if metrics is not None:
        metrics.to(device)
        metrics.reset()

    # Optimizer and loss
    optimizer = build_optimizer(model.parameters(), optimizer_type, lr, weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    torch.set_grad_enabled(True)

    # AMP support
    scaler = torch.cuda.amp.GradScaler(enabled=amp and torch.cuda.is_available())
    autocast_ctx = torch.cuda.amp.autocast(
        dtype=torch.float16,
        enabled=amp and torch.cuda.is_available()
    )

    losses = []

    # Training loop (no progress bar)
    for x, y in train_dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx:
            logits = model(x)
            loss = loss_fn(logits, y)

        # backward
        scaler.scale(loss).backward()

        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.detach().item())

        if metrics is not None:
            metrics.update(logits, y)

        if fast_dev_run:
            break

    # After training loop
    results: Dict[str, Any] = {}

    # metrics
    if metrics is not None:
        computed = metrics.compute()
        metrics.reset()
        results.update({k: [v.item()] for k, v in computed.items()})

    # Loss
    avg_loss = sum(losses) / max(1, len(losses))
    results["Loss"] = [avg_loss]

    torch.set_grad_enabled(False)

    if verbose:
        from tabulate import tabulate
        print(tabulate(results, headers="keys", tablefmt="github"))

    # Optional gradient return
    if return_gradients:
        grads = {
            name: p.grad.detach().clone()
            for name, p in model.named_parameters()
            if p.grad is not None
        }
        results["gradients"] = grads

    return results


# def evaluate(model, data_loader, verbose=True):
#     # Swithicing off gradient calculation to save memory
#     torch.no_grad()
#     # Switch to eval mode so that layers like Dropout function correctly
#     model.eval()

#     metric_names = ['Loss',
#                     'MulticlassAccuracy',
#                     'Balanced Accuracy',
#                     'Precision Micro',
#                     'Recall Micro',
#                     'Precision Macro',
#                     'Recall Macro']

#     score = {name: [] for name in metric_names}

#     num_batch = len(data_loader)

#     progress_bar = tqdm(enumerate(data_loader),
#                         total=num_batch,
#                         file=sys.stdout)

#     for i, (x, ytrue) in progress_bar:

#         yraw = model(x)

#         _, ypred = torch.max(yraw, 1)

#         score = calculate_metrics(score, ytrue, yraw, ypred)

#         progress_bar.set_description('Evaluating')

#     for k, v in score.items():
#         score[k] = [sum(v) / len(v)]

#     if verbose:
#         print('Evaluation Score: ')
#         print(tabulate(score, headers='keys', tablefmt='github'), flush=True)
#     model.train()
#     torch.enable_grad()
#     return score


# @torch.no_grad()
# def fevaluate(model, data_loader, verbose=True):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # Switch to eval mode so that layers like Dropout function correctly
#     model.eval()
#     metric_names = ['Loss',
#                     'MulticlassAccuracy',
#                     'Balanced Accuracy',
#                     'Precision Micro',
#                     'Recall Micro',
#                     'Precision Macro',
#                     'Recall Macro']

#     score = {name: [] for name in metric_names}

#     num_batch = len(data_loader)

#     classtypes = set()
#     progress_bar = tqdm(enumerate(data_loader),
#                         total=num_batch,
#                         file=sys.stdout)

#     for i, (x, ytrue) in progress_bar:
#         classtypes.add(int(ytrue[0]))
#         x = x.to(device)
#         # ytrue = ytrue.to(device)
#         yraw = model(x)
#         # yraw = yraw.to(device)
#         _, ypred = torch.max(yraw, 1)
#         # ypred = ypred.to(device)

#         score = calculate_metrics(score, ytrue, yraw.cpu(), ypred.cpu())

#         progress_bar.set_description('Evaluating')

#     pclass2 = ''
#     class_name = ['airplane', 'car', 'bird', 'cat',
#                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#     for c in classtypes:
#         pclass2 += class_name[c]+' '
#     """
#     pclass = ''
#     for c in classtypes:
#         if c == 0:
#             pclass = pclass + 'airplane '
#         elif c == 1:
#             pclass = pclass + 'car '
#         elif c == 2:
#             pclass = pclass + 'bird '
#         elif c == 3:
#             pclass = pclass + 'cat '
#         elif c == 4:
#             pclass = pclass + 'deer '
#         elif c == 5:
#             pclass = pclass + 'dog '
#         elif c == 6:
#             pclass = pclass + 'frog '
#         elif c == 7:
#             pclass = pclass + 'horse '
#         elif c == 8:
#             pclass = pclass + 'ship '
#         elif c == 9:
#             pclass = pclass + 'truck '
#             """
#     for k, v in score.items():
#         score[k] = [sum(v) / len(v)]

#     print('Acc. for classes', classtypes, pclass2,
#           ": ", score['MulticlassAccuracy'][-1], flush=True)

#     if verbose:
#         print('Evaluation Score: ')
#         print(tabulate(score, headers='keys', tablefmt='github'), flush=True)
#     model.train()
#     torch.enable_grad()
#     return score


@ torch.no_grad()
def test(
    model: nn.Module,
    test_dataloader: DataLoader,
    device='cuda:0',
    fast_dev_run=False,
    verbose=True,
) -> Dict[str, torch.Tensor]:

    num_batch = len(test_dataloader)
    model.eval()
    global metrics

    metrics = metrics.to(device)
    progress_bar = tqdm(enumerate(test_dataloader),
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
        if fast_dev_run:
            break

    outputs = metrics.compute()
    metrics.reset()
    model.train(True)
    outputs = {k: [v.item()] for k, v in outputs.items()}

    if verbose:
        print(tabulate(outputs, headers='keys', tablefmt='github'))
    return outputs


def l1_prune(model, amount=0.00, name='weight', verbose=False, glob=False):
    """
        Prunes the model param by param by given amount
    """
    params_to_prune = get_prune_params(model, name)
    if glob:
        prune.global_unstructured(
            params_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount)
    else:
        for params, name in params_to_prune:
            prune.l1_unstructured(params, name, amount)
    if verbose:
        info = get_prune_summary(model, name)
        global_pruning = info['global']
        info.pop('global')
        print(tabulate(info, headers='keys', tablefmt='github'))
        print("Total Pruning: {}%".format(global_pruning))


"""
Hadamard Mult of Mask and Attributes,
then return zeros
"""


@ torch.no_grad()
def summarize_prune(model: nn.Module, name: str = 'weight') -> tuple:
    """
        returns (pruned_params,total_params)
    """
    num_pruned = 0
    params, num_global_weights, _ = get_prune_params(model)
    for param, _ in params:
        if hasattr(param, name+'_mask'):
            data = getattr(param, name+'_mask')
            num_pruned += int(torch.sum(data == 0.0).item())
    return (num_pruned, num_global_weights)


def get_prune_params(model, name='weight') -> List[Tuple[nn.Parameter, str]]:
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


def custom_save(model, path):
    """
    https://pytorch.org/docs/stable/generated/torch.save.html#torch.save
    Custom save utility function
    Compresses the model using gzip
    Helpfull if model is highly pruned
    """
    bufferIn = io.BytesIO()
    torch.save(model.state_dict(), bufferIn)
    bufferOut = gzip.compress(bufferIn.getvalue())
    with gzip.open(path, 'wb') as f:
        f.write(bufferOut)


def custom_load(path) -> Dict:
    """
    returns saved_dictionary
    """
    with gzip.open(path, 'rb') as f:
        bufferIn = f.read()
        bufferOut = gzip.decompress(bufferIn)
        state_dict = torch.load(io.BytesIO(bufferOut))
    return state_dict


def log_obj(path, obj):
    # pass
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    #
    with open(path, 'wb') as file:
        if isinstance(obj, nn.Module):
            torch.save(obj, file)
        else:
            pickle.dump(obj, file)


class CustomPruneMethod(prune.BasePruningMethod):

    PRUNING_TYPE = 'unstructured'

    def __init__(self, amount, orig_weights):
        super().__init__()
        self.amount = amount
        self.original_signs = self.get_signs_from_tensor(orig_weights)

    def get_signs_from_tensor(self, t: torch.Tensor):
        return torch.sign(t).view(-1)

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        large_weight_mask = t.view(-1).mul(self.original_signs)
        large_weight_mask_ranked = F.relu(large_weight_mask)
        nparams_toprune = int(torch.numel(t) * self.amount)  # get this val
        if nparams_toprune > 0:
            bottom_k = torch.topk(
                large_weight_mask_ranked.view(-1), k=nparams_toprune, largest=False)
            mask.view(-1)[bottom_k.indices] = 0.00
            return mask
        else:
            return mask


def customPrune(module, orig_module, amount=0.1, name='weight'):
    """
        Taken from https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
        Takes: current module (module), name of the parameter to prune (name)

    """
    CustomPruneMethod.apply(module, name, amount, orig_module)
    return module


def super_prune(
    model: nn.Module,
    init_model: nn.Module,
    amount: float = 0.0,
    name: str = 'weight'
) -> None:
    """

    """
    params_to_prune = get_prune_params(model)
    init_params = get_prune_params(init_model)

    for idx, (param, name) in enumerate(params_to_prune):
        orig_params = getattr(init_params[idx][0], name)

        # original params are sliced by the pruned model's mask
        # this is because pytorch's pruning container slices the mask by
        # non-zero params
        if hasattr(param, 'weight_mask'):
            mask = getattr(param, 'weight_mask')
            sliced_params = orig_params[mask.to(torch.bool)]
            customPrune(param, sliced_params, amount, name)
        else:
            customPrune(param, orig_params, amount, name)

def calc_mask_from_model_with_mask_object(model):
    layer_to_mask = {}
    for layer, module in model.named_children():
        for name, mask in module.named_buffers():
            if 'mask' in name:
                layer_to_mask[layer] = mask
    return layer_to_mask


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

def check_mask_object_from_model(model):
    for layer, module in model.named_children():
        for name, mask in module.named_buffers():
            if 'mask' in name:
                return True
    return False


def produce_mask_from_model(model):
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

def get_num_total_model_params(model):
    total_num_model_params = 0
    # not including bias
    for layer_name, params in model.named_parameters():
        if 'weight' in layer_name:
            total_num_model_params += params.numel()
    return total_num_model_params

def get_pruned_amount_by_weights(model):
    if check_mask_object_from_model(model):
        sys.exit("\033[91m" + "Warning - get_pruned_amount_by_weights() is called when the model has mask." + "\033[0m")
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

def get_pruned_amount(model):
    if check_mask_object_from_model(model):
        return get_pruned_amount_by_mask(model)
    return get_pruned_amount_by_weights(model)

def get_pruned_amount_by_mask(model):
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
