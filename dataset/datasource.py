import numpy as np
import random
import torch
import torchvision as tv
from torchvision import transforms
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
# from dataset.cifar10_noniid import get_dataset_cifar10_extr_noniid, cifar_extr_noniid
# from dataset.mnist_noniid import get_dataset_mnist_extr_noniid, mnist_extr_noniid


def DataLoaders(n_devices, dataset_name, total_samples, log_dirpath, seed, mode="non-iid", batch_size=32, alpha=1.0, dataloader_workers=1, need_test_loaders=False):
    if mode == "non-iid":
        if dataset_name == "mnist":
            return get_data_noniid_mnist(n_devices,
                                         total_samples,
                                         log_dirpath,
                                         seed,
                                         batch_size,
                                         alpha,
                                         dataloader_workers,
                                         mode,
                                         need_test_loaders)
        elif dataset_name == "cifar10":
            return get_data_noniid_cifar10(n_devices,
                                           total_samples,
                                           log_dirpath,
                                           seed,
                                           batch_size,
                                           alpha,
                                           dataloader_workers,
                                           need_test_loaders)
    elif mode == 'iid':
        if dataset_name == 'cifar10':
            data_dir = './data'
            apply_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            train_dataset = tv.datasets.CIFAR10(data_dir, train=True, download=True,
                                                transform=apply_transform)

            test_dataset = tv.datasets.CIFAR10(data_dir, train=False, download=True,
                                               transform=apply_transform)
            return iid_split(n_devices, train_dataset, batch_size, test_dataset, dataloader_workers, log_dirpath, total_samples, need_test_loaders)
        elif dataset_name == 'mnist':
            data_dir = './data'
            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])
            train_dataset = tv.datasets.MNIST(data_dir, train=True, download=True,
                                              transform=apply_transform)

            test_dataset = tv.datasets.MNIST(data_dir, train=False, download=True,
                                             transform=apply_transform)
            return iid_split(n_devices, train_dataset, batch_size, test_dataset, dataloader_workers, log_dirpath, total_samples, need_test_loaders)


def iid_split(n_clients,
              train_data,
              batch_size, test_data, dataloader_workers, log_dirpath, total_samples, need_test_loaders):

    labels = np.array(train_data.targets)
    samples_per_label = total_samples // 10  # assume 10 labels
    
    # ===== TRAINING DATA ASSIGNMENT (UNCHANGED) =====
    idx_by_label = {l: np.where(labels == l)[0] for l in range(10)}
    for l in idx_by_label:
        np.random.shuffle(idx_by_label[l])
    
    sample_train_idx = [[] for _ in range(n_clients)]
    for i in range(n_clients):
        for l in range(10):
            take = idx_by_label[l][:samples_per_label]
            sample_train_idx[i].extend(take)
            idx_by_label[l] = idx_by_label[l][samples_per_label:]

    # ===== TEST DATA ASSIGNMENT (NEW - MATCHING CELL BEHAVIOR) =====
    # Create test index pools for proportional allocation
    test_idx_by_class = {k: np.where(np.array(test_data.targets) == k)[0].tolist() for k in range(10)}
    for k in test_idx_by_class:
        np.random.shuffle(test_idx_by_class[k])
    
    sample_test_idx = [[] for _ in range(n_clients)]
    
    for i in range(n_clients):
        # Get the actual training distribution for this device
        train_indices = sample_train_idx[i]
        train_labels_arr = np.array(train_data.targets)[train_indices]
        train_class_counts = np.bincount(train_labels_arr, minlength=10)
        train_total = len(train_indices)
        
        if train_total == 0:
            continue
            
        # Calculate test sample size as 25% of training samples
        test_total_needed = int(train_total * 0.25)
        
        # Assign test data with same proportions as training data
        assigned_test_samples = 0
        for label in range(10):
            if train_class_counts[label] == 0:
                continue
                
            # Calculate how many test samples needed for this label
            label_proportion = train_class_counts[label] / train_total
            label_test_needed = int(label_proportion * test_total_needed)
            
            # Take samples from test data for this label
            available_test_samples = len(test_idx_by_class[label])
            take = min(label_test_needed, available_test_samples)
            
            if take > 0:
                selected_test = test_idx_by_class[label][:take]
                test_idx_by_class[label] = test_idx_by_class[label][take:]
                sample_test_idx[i].extend(selected_test)
                assigned_test_samples += take
        
        # Fill any remaining slots if we're short due to rounding
        remaining_needed = test_total_needed - assigned_test_samples
        if remaining_needed > 0:
            # Fill from labels that this device actually has in training
            device_labels = [label for label in range(10) if train_class_counts[label] > 0]
            for label in device_labels:
                if remaining_needed <= 0:
                    break
                if len(test_idx_by_class[label]) > 0:
                    sample_test_idx[i].append(test_idx_by_class[label].pop())
                    remaining_needed -= 1

    # ===== CREATE DATA LOADERS =====
    user_train_loaders = []
    user_test_loaders = []

    for idx in sample_train_idx:
        user_train_loaders.append(torch.utils.data.DataLoader(train_data,
                                                              sampler=torch.utils.data.SubsetRandomSampler(
                                                                  idx),
                                                              batch_size=batch_size, num_workers=dataloader_workers))
    
    for idx in sample_test_idx:
        if len(idx) > 0:
            user_test_loaders.append(torch.utils.data.DataLoader(test_data,
                                                                 sampler=torch.utils.data.SubsetRandomSampler(
                                                                     idx),
                                                                 batch_size=batch_size, num_workers=dataloader_workers))
        else:
            # If no test data assigned, create empty loader
            user_test_loaders.append(torch.utils.data.DataLoader(test_data,
                                                                 sampler=torch.utils.data.SubsetRandomSampler([]),
                                                                 batch_size=batch_size, num_workers=dataloader_workers))
    
    # ===== LOGGING =====
    user_label_to_qty = {}
    for i, (train_loader, test_idx) in enumerate(zip(user_train_loaders, sample_test_idx)):
        labels = []
        for batch in train_loader:
            _, targets = batch
            labels.extend(targets.numpy().tolist())
        unique_labels = sorted(list(set(labels)))
        class_counts = np.bincount(np.array(labels), minlength=10)
        
        # Also log test distribution for verification
        if len(test_idx) > 0:
            test_labels_arr = np.array(test_data.targets)[test_idx]
            test_class_counts = np.bincount(test_labels_arr, minlength=10)
        else:
            test_class_counts = np.zeros(10)
        
        msg = f"Device {i + 1} train label distribution: {dict(enumerate(class_counts))}"
        if need_test_loaders:
            msg += f", test label distribution: {dict(enumerate(test_class_counts))}"
        with open(f"{log_dirpath}/dataset_assigned.txt", "a") as f:
            f.write(f"{msg}\n")
        print(msg)
        user_label_to_qty[i] = dict(enumerate(class_counts))
    
    global_test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=dataloader_workers)
    return user_train_loaders, user_test_loaders, user_label_to_qty, global_test_loader


def get_data_noniid_cifar10(n_devices, total_samples, log_dirpath, seed, batch_size=32, alpha=1.0, dataloader_workers=1, need_test_loaders=False):
    data_dir = './data'
    apply_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_data = tv.datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)
    test_data = tv.datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)

    np.random.seed(seed)
    K = 10  # total labels

    idx_by_class = {k: np.where(np.array(train_data.targets) == k)[0].tolist() for k in range(K)}
    for k in idx_by_class:
        np.random.shuffle(idx_by_class[k])

    idx_batch = [[] for _ in range(n_devices)]
    user_label_to_qty = {}

    for device_id in range(n_devices):
        proportions = np.random.dirichlet(np.repeat(alpha, K))
        proportions = proportions / proportions.sum()
        assigned_counts = {k: int(p * total_samples) for k, p in enumerate(proportions)}

        selected_labels = [label for label in assigned_counts if assigned_counts[label] > 0]

        for label in selected_labels:
            take = min(assigned_counts[label], len(idx_by_class[label]))
            selected = idx_by_class[label][:take]
            idx_by_class[label] = idx_by_class[label][take:]
            idx_batch[device_id].extend(selected)

        while len(idx_batch[device_id]) < total_samples:
            for label in selected_labels:
                if len(idx_by_class[label]) == 0:
                    continue
                idx_batch[device_id].append(idx_by_class[label].pop())
                if len(idx_batch[device_id]) == total_samples:
                    break

    train_loaders, test_loaders = [], []
    for i, idx in enumerate(idx_batch):
        sampler_train = torch.utils.data.SubsetRandomSampler(idx)
        loader_train = torch.utils.data.DataLoader(train_data, sampler=sampler_train, batch_size=batch_size, num_workers=dataloader_workers)
        train_loaders.append(loader_train)

        labels_arr = np.array(train_data.targets)[idx]
        class_counts = np.bincount(labels_arr, minlength=10)
        msg = f"Device {i + 1} label distribution: {dict(enumerate(class_counts))}"
        with open(f"{log_dirpath}/dataset_assigned.txt", "a") as f:
            f.write(f"{msg}\n")
        print(msg)

        user_label_to_qty[i] = dict(enumerate(class_counts))

        idx_test = [j for j, label in enumerate(test_data.targets) if class_counts[label] > 0]
        sampler_test = torch.utils.data.SubsetRandomSampler(idx_test)
        loader_test = torch.utils.data.DataLoader(test_data, sampler=sampler_test, batch_size=batch_size, num_workers=dataloader_workers)
        test_loaders.append(loader_test)

    global_test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=dataloader_workers)

    return train_loaders, test_loaders, user_label_to_qty, global_test_loader

# NBFL - Modified to match CELL test_loader behavior while preserving exact same train_loaders

def get_data_noniid_mnist(n_devices, total_samples, log_dirpath, seed, batch_size=32, alpha=1.0, dataloader_workers=1, dataset_mode="non-iid", need_test_loaders=False):
    import os
    data_dir = './data'
    apply_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = tv.datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)
    test_data = tv.datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)

    np.random.seed(seed)
    K = 10  # total labels
    N = len(train_data)

    # ===== TRAINING DATA ASSIGNMENT (UNCHANGED) =====
    # Index pool by class - EXACTLY as before
    idx_by_class = {k: np.where(np.array(train_data.targets) == k)[0].tolist() for k in range(K)}
    for k in idx_by_class:
        np.random.shuffle(idx_by_class[k])
    
    # Pre-compute IID partitions if needed - EXACTLY as before
    if dataset_mode == "iid":
        all_indices = np.arange(len(train_data))
        np.random.shuffle(all_indices)
        iid_chunks = np.array_split(all_indices, n_devices)

    idx_batch = [[] for _ in range(n_devices)]
    user_labels = [[] for _ in range(n_devices)]

    # Training data assignment - EXACTLY as before
    for device_id in range(n_devices):
        if dataset_mode == "iid":
            selected = iid_chunks[device_id]
            idx_batch[device_id].extend(selected)
            user_labels[device_id] = list(np.unique(np.array(train_data.targets)[selected]))
            continue
        elif alpha == 0:
            # Assign data from a single randomly chosen label to each device
            label = np.random.choice(K)
            user_labels[device_id] = [label]
            take = min(total_samples, len(idx_by_class[label]))
            selected = idx_by_class[label][:take]
            idx_by_class[label] = idx_by_class[label][take:]
            idx_batch[device_id].extend(selected)
 
            while len(idx_batch[device_id]) < total_samples:
                if len(idx_by_class[label]) == 0:
                    break
                idx_batch[device_id].append(idx_by_class[label].pop())
            continue
        else:
            proportions = np.random.dirichlet(np.repeat(alpha, K))
            proportions = proportions / proportions.sum()
            assigned_counts = {k: int(p * total_samples) for k, p in enumerate(proportions)}
 
            selected_labels = [label for label in assigned_counts if assigned_counts[label] > 0]
            user_labels[device_id] = selected_labels
 
            for label in selected_labels:
                take = min(assigned_counts[label], len(idx_by_class[label]))
                selected = idx_by_class[label][:take]
                idx_by_class[label] = idx_by_class[label][take:]
                idx_batch[device_id].extend(selected)
 
            while len(idx_batch[device_id]) < total_samples:
                for label in selected_labels:
                    if len(idx_by_class[label]) == 0:
                        continue
                    idx_batch[device_id].append(idx_by_class[label].pop())
                    if len(idx_batch[device_id]) == total_samples:
                        break

    # ===== TEST DATA ASSIGNMENT (NEW - MATCHING CELL BEHAVIOR) =====
    # Create test index pools that exclude training indices to ensure non-overlap
    test_idx_by_class = {k: np.where(np.array(test_data.targets) == k)[0].tolist() for k in range(K)}
    for k in test_idx_by_class:
        np.random.shuffle(test_idx_by_class[k])
    
    test_idx_batch = [[] for _ in range(n_devices)]
    
    for device_id in range(n_devices):
        # Get the actual training distribution for this device
        train_indices = idx_batch[device_id]
        train_labels_arr = np.array(train_data.targets)[train_indices]
        train_class_counts = np.bincount(train_labels_arr, minlength=10)
        train_total = len(train_indices)
        
        if train_total == 0:
            continue
            
        # Calculate test sample size as 25% of training samples
        test_total_needed = int(train_total * 0.25)
        
        # Assign test data with same proportions as training data
        assigned_test_samples = 0
        for label in range(K):
            if train_class_counts[label] == 0:
                continue
                
            # Calculate how many test samples needed for this label
            label_proportion = train_class_counts[label] / train_total
            label_test_needed = int(label_proportion * test_total_needed)
            
            # Take samples from test data for this label
            available_test_samples = len(test_idx_by_class[label])
            take = min(label_test_needed, available_test_samples)
            
            if take > 0:
                selected_test = test_idx_by_class[label][:take]
                test_idx_by_class[label] = test_idx_by_class[label][take:]
                test_idx_batch[device_id].extend(selected_test)
                assigned_test_samples += take
        
        # Fill any remaining slots if we're short due to rounding
        remaining_needed = test_total_needed - assigned_test_samples
        if remaining_needed > 0:
            # Fill from labels that this device actually has in training
            device_labels = [label for label in range(K) if train_class_counts[label] > 0]
            for label in device_labels:
                if remaining_needed <= 0:
                    break
                if len(test_idx_by_class[label]) > 0:
                    test_idx_batch[device_id].append(test_idx_by_class[label].pop())
                    remaining_needed -= 1
    
    # ===== CREATE DATA LOADERS =====
    user_label_to_qty = {}
    train_loaders, test_loaders = [], []
    
    for i, (train_idx, test_idx) in enumerate(zip(idx_batch, test_idx_batch)):
        # Training loader - EXACTLY as before
        sampler_train = torch.utils.data.SubsetRandomSampler(train_idx)
        loader_train = torch.utils.data.DataLoader(train_data, sampler=sampler_train, batch_size=batch_size, num_workers=dataloader_workers)
        train_loaders.append(loader_train)

        # Log training distribution - EXACTLY as before
        labels_arr = np.array(train_data.targets)[train_idx]
        class_counts = np.bincount(labels_arr, minlength=10)
        
        # Also log test distribution for verification
        if len(test_idx) > 0:
            test_labels_arr = np.array(test_data.targets)[test_idx]
            test_class_counts = np.bincount(test_labels_arr, minlength=10)
        else:
            test_class_counts = np.zeros(10)
        
        msg = f"Device {i + 1} train label distribution: {dict(enumerate(class_counts))}"
        if need_test_loaders:
            msg += f", test label distribution: {dict(enumerate(test_class_counts))}"
        with open(f"{log_dirpath}/dataset_assigned.txt", "a") as f:
            f.write(f"{msg}\n")
        print(msg)

        # Test loader - NEW: using proportional indices like CELL
        if len(test_idx) > 0:
            sampler_test = torch.utils.data.SubsetRandomSampler(test_idx)
            loader_test = torch.utils.data.DataLoader(test_data, sampler=sampler_test, batch_size=batch_size, num_workers=dataloader_workers)
        else:
            # If no test data assigned, create empty loader
            loader_test = torch.utils.data.DataLoader(test_data, sampler=torch.utils.data.SubsetRandomSampler([]), batch_size=batch_size, num_workers=dataloader_workers)
        test_loaders.append(loader_test)

        user_label_to_qty[i] = dict(enumerate(class_counts))
    
    # Global test loader - using original approach
    global_test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=dataloader_workers)

    return train_loaders, test_loaders, user_label_to_qty, global_test_loader