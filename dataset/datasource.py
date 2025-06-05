import numpy as np
import random
import torch
import torchvision as tv
from torchvision import transforms
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
# from dataset.cifar10_noniid import get_dataset_cifar10_extr_noniid, cifar_extr_noniid
# from dataset.mnist_noniid import get_dataset_mnist_extr_noniid, mnist_extr_noniid


def DataLoaders(n_devices, dataset_name, total_samples, log_dirpath, seed, mode="non-iid", batch_size=32, alpha=1.0, dataloader_workers=1):
    if mode == "non-iid":
        if dataset_name == "mnist":
            return get_data_noniid_mnist(n_devices,
                                         total_samples,
                                         log_dirpath,
                                         seed,
                                         batch_size,
                                         alpha,
                                         dataloader_workers,
                                         mode)
        elif dataset_name == "cifar10":
            return get_data_noniid_cifar10(n_devices,
                                           total_samples,
                                           log_dirpath,
                                           seed,
                                           batch_size,
                                           alpha,
                                           dataloader_workers)
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
            return iid_split(n_devices, train_dataset, batch_size, test_dataset, dataloader_workers, log_dirpath, total_samples)
        elif dataset_name == 'mnist':
            data_dir = './data'
            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])
            train_dataset = tv.datasets.MNIST(data_dir, train=True, download=True,
                                              transform=apply_transform)

            test_dataset = tv.datasets.MNIST(data_dir, train=False, download=True,
                                             transform=apply_transform)
            return iid_split(n_devices, train_dataset, batch_size, test_dataset, dataloader_workers, log_dirpath, total_samples)


def iid_split(n_clients,
              train_data,
              batch_size, test_data, dataloader_workers, log_dirpath, total_samples):

    labels = np.array(train_data.targets)
    samples_per_label = total_samples // 10  # assume 10 labels
    
    idx_by_label = {l: np.where(labels == l)[0] for l in range(10)}
    for l in idx_by_label:
        np.random.shuffle(idx_by_label[l])
    
    sample_train_idx = [[] for _ in range(n_clients)]
    for i in range(n_clients):
        for l in range(10):
            take = idx_by_label[l][:samples_per_label]
            sample_train_idx[i].extend(take)
            idx_by_label[l] = idx_by_label[l][samples_per_label:]

    all_test_idx = np.arange(test_data.data.shape[0])

    sample_test_idx = np.array_split(all_test_idx, n_clients)

    user_train_loaders = []
    user_test_loaders = []

    for idx in sample_train_idx:
        user_train_loaders.append(torch.utils.data.DataLoader(train_data,
                                                              sampler=torch.utils.data.SubsetRandomSampler(
                                                                  idx),
                                                              batch_size=batch_size, num_workers=dataloader_workers))
    for idx in sample_test_idx:
        user_test_loaders.append(torch.utils.data.DataLoader(test_data,
                                                             sampler=torch.utils.data.SubsetRandomSampler(
                                                                 idx),
                                                             batch_size=batch_size, num_workers=dataloader_workers))
    user_label_to_qty = {}
    for i, loader in enumerate(user_train_loaders):
        labels = []
        for batch in loader:
            _, targets = batch
            labels.extend(targets.numpy().tolist())
        unique_labels = sorted(list(set(labels)))
        # user_labels.append(unique_labels)
        class_counts = np.bincount(np.array(labels), minlength=10)
        msg = f"Device {i + 1} label distribution: {dict(enumerate(class_counts))}"
        with open(f"{log_dirpath}/dataset_assigned.txt", "a") as f:
            f.write(f"{msg}\n")
        print(msg)
        user_label_to_qty[i] = dict(enumerate(class_counts))
    global_test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=dataloader_workers)
    return user_train_loaders, user_test_loaders, user_label_to_qty, global_test_loader


def get_data_noniid_cifar10(n_devices, n_labels, total_samples, log_dirpath, seed, batch_size=32, alpha=1.0, dataloader_workers=1):
    data_dir = './data'
    apply_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_data = tv.datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)
    test_data = tv.datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)

    np.random.seed(seed)
    K = 10  # number of total labels
    N = len(train_data)

    # Get index list per class
    idx_by_class = {k: np.where(np.array(train_data.targets) == k)[0].tolist() for k in range(K)}
    for k in idx_by_class:
        np.random.shuffle(idx_by_class[k])

    idx_batch = [[] for _ in range(n_devices)]
    user_labels = [[] for _ in range(n_devices)]

    for device_id in range(n_devices):
        chosen_labels = np.random.choice(K, size=n_labels, replace=False)
        user_labels[device_id] = list(chosen_labels)

        for label in chosen_labels:
            take = min(total_samples // n_labels, len(idx_by_class[label]))
            selected = idx_by_class[label][:take]
            idx_by_class[label] = idx_by_class[label][take:]
            idx_batch[device_id].extend(selected)
 
    train_loaders, test_loaders = [], []
    for i, idx in enumerate(idx_batch):
        sampler_train = torch.utils.data.SubsetRandomSampler(idx)
        loader_train = torch.utils.data.DataLoader(train_data, sampler=sampler_train, batch_size=batch_size, num_workers=dataloader_workers)
        train_loaders.append(loader_train)
 
        labels_arr = np.array(train_data.targets)[idx]
        # Added code block to log label distribution
        class_counts = np.bincount(labels_arr, minlength=10)
        msg = f"Device {i + 1} label distribution: {dict(enumerate(class_counts))}"
        with open(f"{log_dirpath}/dataset_assigned.txt", "a") as f:
            f.write(f"{msg}\n")
        print(msg)

        curr_labels = user_labels[i]
        idx_test = [j for j, label in enumerate(test_data.targets) if label in curr_labels]
        sampler_test = torch.utils.data.SubsetRandomSampler(idx_test)
        loader_test = torch.utils.data.DataLoader(test_data, sampler=sampler_test, batch_size=batch_size, num_workers=dataloader_workers)
        test_loaders.append(loader_test)

    global_test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=dataloader_workers)

    return train_loaders, test_loaders, user_labels, global_test_loader

def get_data_noniid_mnist(n_devices, total_samples, log_dirpath, seed, batch_size=32, alpha=1.0, dataloader_workers=1, dataset_mode="non-iid"):
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

    # Index pool by class
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
    
    user_label_to_qty = {}
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

        curr_labels = user_labels[i]
        idx_test = [j for j, label in enumerate(test_data.targets) if label in curr_labels]
        sampler_test = torch.utils.data.SubsetRandomSampler(idx_test)
        loader_test = torch.utils.data.DataLoader(test_data, sampler=sampler_test, batch_size=batch_size, num_workers=dataloader_workers)
        test_loaders.append(loader_test)

        user_label_to_qty[i] = dict(enumerate(class_counts))
    global_test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=dataloader_workers)

    return train_loaders, test_loaders, user_label_to_qty, global_test_loader