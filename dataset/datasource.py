import numpy as np
import random
import torch
import torchvision as tv
from torchvision import transforms
from sklearn.utils import shuffle
from matplotlib import pyplot as plt


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


def iid_split(n_clients, train_data, batch_size, test_data, dataloader_workers, log_dirpath, total_samples, need_test_loaders):
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

    # ===== TEST DATA ASSIGNMENT (MODIFIED TO MATCH LOTTERYFL) =====
    # Sort test data by labels like LotteryFL does
    test_labels = np.array(test_data.targets)
    test_idxs = np.arange(len(test_data))
    
    # Sort test indices by labels
    test_idxs_labels = np.vstack((test_idxs, test_labels))
    test_idxs_labels = test_idxs_labels[:, test_idxs_labels[1, :].argsort()]
    sorted_test_idxs = test_idxs_labels[0, :]
    sorted_test_labels = test_idxs_labels[1, :]
    
    # Create test index pools by class (like LotteryFL)
    test_idxs_splits = [[] for _ in range(10)]
    for i in range(len(sorted_test_labels)):
        test_idxs_splits[sorted_test_labels[i]].append(sorted_test_idxs[i])
    
    sample_test_idx = [[] for _ in range(n_clients)]
    
    for i in range(n_clients):
        # Get training labels for this client
        train_indices = sample_train_idx[i]
        train_labels_arr = np.array(train_data.targets)[train_indices]
        user_labels_set = set(train_labels_arr)
        
        # Assign ALL test samples from the same classes as training (LotteryFL style)
        for label in user_labels_set:
            sample_test_idx[i].extend(test_idxs_splits[label])

    # ===== CREATE DATA LOADERS =====
    user_train_loaders = []
    user_test_loaders = []

    for idx in sample_train_idx:
        user_train_loaders.append(torch.utils.data.DataLoader(train_data,
                                                              sampler=torch.utils.data.SubsetRandomSampler(idx),
                                                              batch_size=batch_size, num_workers=dataloader_workers))
    
    for idx in sample_test_idx:
        if len(idx) > 0:
            user_test_loaders.append(torch.utils.data.DataLoader(test_data,
                                                                 sampler=torch.utils.data.SubsetRandomSampler(idx),
                                                                 batch_size=batch_size, num_workers=dataloader_workers))
        else:
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
        
        # Log test distribution for verification
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

    # ===== TRAINING DATA ASSIGNMENT (UNCHANGED) =====
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

    # ===== TEST DATA ASSIGNMENT (MODIFIED TO MATCH LOTTERYFL) =====
    # Sort test data by labels like LotteryFL does
    test_labels = np.array(test_data.targets)
    test_idxs = np.arange(len(test_data))
    
    # Sort test indices by labels
    test_idxs_labels = np.vstack((test_idxs, test_labels))
    test_idxs_labels = test_idxs_labels[:, test_idxs_labels[1, :].argsort()]
    sorted_test_idxs = test_idxs_labels[0, :]
    sorted_test_labels = test_idxs_labels[1, :]
    
    # Create test index pools by class (like LotteryFL)
    test_idxs_splits = [[] for _ in range(K)]
    for i in range(len(sorted_test_labels)):
        test_idxs_splits[sorted_test_labels[i]].append(sorted_test_idxs[i])
    
    # ===== CREATE DATA LOADERS =====
    train_loaders, test_loaders = [], []
    for i, idx in enumerate(idx_batch):
        # Training loader (unchanged)
        sampler_train = torch.utils.data.SubsetRandomSampler(idx)
        loader_train = torch.utils.data.DataLoader(train_data, sampler=sampler_train, batch_size=batch_size, num_workers=dataloader_workers)
        train_loaders.append(loader_train)

        # Get training labels for this device
        labels_arr = np.array(train_data.targets)[idx]
        class_counts = np.bincount(labels_arr, minlength=10)
        user_labels_set = set(labels_arr)
        
        # Test data assignment: ALL test samples from training classes (LotteryFL style)
        idx_test = []
        for label in user_labels_set:
            idx_test.extend(test_idxs_splits[label])
        
        # Log distributions
        if len(idx_test) > 0:
            test_labels_arr = np.array(test_data.targets)[idx_test]
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

        # Test loader
        sampler_test = torch.utils.data.SubsetRandomSampler(idx_test)
        loader_test = torch.utils.data.DataLoader(test_data, sampler=sampler_test, batch_size=batch_size, num_workers=dataloader_workers)
        test_loaders.append(loader_test)

    global_test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=dataloader_workers)

    return train_loaders, test_loaders, user_label_to_qty, global_test_loader


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
    idx_by_class = {k: np.where(np.array(train_data.targets) == k)[0].tolist() for k in range(K)}
    for k in idx_by_class:
        np.random.shuffle(idx_by_class[k])
    
    # Pre-compute IID partitions if needed
    if dataset_mode == "iid":
        all_indices = np.arange(len(train_data))
        np.random.shuffle(all_indices)
        iid_chunks = np.array_split(all_indices, n_devices)

    idx_batch = [[] for _ in range(n_devices)]
    user_labels = [[] for _ in range(n_devices)]

    # Training data assignment (unchanged)
    for device_id in range(n_devices):
        if dataset_mode == "iid":
            selected = iid_chunks[device_id]
            idx_batch[device_id].extend(selected)
            user_labels[device_id] = list(np.unique(np.array(train_data.targets)[selected]))
            continue
        elif alpha == 0:
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

    # ===== TEST DATA ASSIGNMENT (MODIFIED TO MATCH LOTTERYFL) =====
    # Sort test data by labels like LotteryFL does
    test_labels = np.array(test_data.targets)
    test_idxs = np.arange(len(test_data))
    
    # Sort test indices by labels
    test_idxs_labels = np.vstack((test_idxs, test_labels))
    test_idxs_labels = test_idxs_labels[:, test_idxs_labels[1, :].argsort()]
    sorted_test_idxs = test_idxs_labels[0, :]
    sorted_test_labels = test_idxs_labels[1, :]
    
    # Create test index pools by class (like LotteryFL)
    test_idxs_splits = [[] for _ in range(K)]
    for i in range(len(sorted_test_labels)):
        test_idxs_splits[sorted_test_labels[i]].append(sorted_test_idxs[i])
    
    # ===== CREATE DATA LOADERS =====
    user_label_to_qty = {}
    train_loaders, test_loaders = [], []
    
    for i, train_idx in enumerate(idx_batch):
        # Training loader (unchanged)
        sampler_train = torch.utils.data.SubsetRandomSampler(train_idx)
        loader_train = torch.utils.data.DataLoader(train_data, sampler=sampler_train, batch_size=batch_size, num_workers=dataloader_workers)
        train_loaders.append(loader_train)

        # Get training labels for this device
        labels_arr = np.array(train_data.targets)[train_idx]
        class_counts = np.bincount(labels_arr, minlength=10)
        user_labels_set = set(labels_arr)
        
        # Test data assignment: ALL test samples from training classes (LotteryFL style)
        test_idx = []
        for label in user_labels_set:
            test_idx.extend(test_idxs_splits[label])
        
        # Log distributions
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

        # Test loader
        if len(test_idx) > 0:
            sampler_test = torch.utils.data.SubsetRandomSampler(test_idx)
            loader_test = torch.utils.data.DataLoader(test_data, sampler=sampler_test, batch_size=batch_size, num_workers=dataloader_workers)
        else:
            loader_test = torch.utils.data.DataLoader(test_data, sampler=torch.utils.data.SubsetRandomSampler([]), batch_size=batch_size, num_workers=dataloader_workers)
        test_loaders.append(loader_test)

        user_label_to_qty[i] = dict(enumerate(class_counts))
    
    # Global test loader
    global_test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=dataloader_workers)

    return train_loaders, test_loaders, user_label_to_qty, global_test_loader