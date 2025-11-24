import numpy as np
import random
import torch
import torchvision as tv
from torchvision import transforms
from sklearn.utils import shuffle
from matplotlib import pyplot as plt


def select_fixed_dataset(train_data, total_samples_per_device, n_devices, seed):
    """
    Select a fixed subset of data points that will be used for both IID and Non-IID.
    This ensures the same data points are used, just distributed differently.
    
    Args:
        train_data: The full training dataset
        total_samples_per_device: Number of samples each device should get
        n_devices: Number of devices
        seed: Random seed for reproducibility
    
    Returns:
        selected_indices: Dictionary mapping class labels to selected indices
        total_per_class: Dictionary showing how many samples per class were selected
    """
    np.random.seed(seed)
    
    total_samples_needed = total_samples_per_device * n_devices
    K = 10  # Number of classes
    
    # Get all indices organized by class
    all_labels = np.array(train_data.targets)
    idx_by_class = {k: np.where(all_labels == k)[0].tolist() for k in range(K)}
    
    # Shuffle indices within each class
    for k in idx_by_class:
        np.random.shuffle(idx_by_class[k])
    
    # For IID, we want roughly equal representation of each class
    # So we'll select equal amounts from each class
    samples_per_class = total_samples_needed // K
    remaining_samples = total_samples_needed % K
    
    selected_indices = {}
    total_per_class = {}
    
    for k in range(K):
        # Take base amount plus one extra for some classes to handle remainder
        take_count = samples_per_class + (1 if k < remaining_samples else 0)
        take_count = min(take_count, len(idx_by_class[k]))  # Don't take more than available
        
        selected_indices[k] = idx_by_class[k][:take_count]
        total_per_class[k] = take_count
    
    # If we couldn't get enough samples (dataset too small), fill from available classes
    total_selected = sum(total_per_class.values())
    if total_selected < total_samples_needed:
        for k in range(K):
            remaining_in_class = len(idx_by_class[k]) - total_per_class[k]
            if remaining_in_class > 0:
                extra_needed = min(remaining_in_class, total_samples_needed - total_selected)
                selected_indices[k].extend(idx_by_class[k][total_per_class[k]:total_per_class[k] + extra_needed])
                total_per_class[k] += extra_needed
                total_selected += extra_needed
                if total_selected >= total_samples_needed:
                    break
    
    return selected_indices, total_per_class


def distribute_iid(selected_indices, n_devices, total_samples_per_device):
    """
    Distribute the selected data points evenly across devices (IID).
    
    Args:
        selected_indices: Dictionary mapping class labels to selected indices
        n_devices: Number of devices
        total_samples_per_device: Number of samples each device should get
    
    Returns:
        device_indices: List of indices for each device
    """
    K = 10
    device_indices = [[] for _ in range(n_devices)]
    
    # For IID, distribute each class's samples evenly across all devices
    for k in range(K):
        class_indices = selected_indices[k].copy()
        samples_per_device_from_class = len(class_indices) // n_devices
        remainder = len(class_indices) % n_devices
        
        idx_position = 0
        for device_id in range(n_devices):
            # Give base amount plus one extra to handle remainder
            take = samples_per_device_from_class + (1 if device_id < remainder else 0)
            device_indices[device_id].extend(class_indices[idx_position:idx_position + take])
            idx_position += take
    
    # Shuffle indices within each device to mix classes
    for device_id in range(n_devices):
        np.random.shuffle(device_indices[device_id])
    
    return device_indices


def distribute_noniid(selected_indices, n_devices, total_samples_per_device, alpha, seed):
    """
    Distribute the selected data points using Dirichlet distribution (Non-IID).
    
    Args:
        selected_indices: Dictionary mapping class labels to selected indices
        n_devices: Number of devices
        total_samples_per_device: Number of samples each device should get
        alpha: Dirichlet distribution parameter
        seed: Random seed for Dirichlet sampling
    
    Returns:
        device_indices: List of indices for each device
    """
    np.random.seed(seed)
    K = 10
    device_indices = [[] for _ in range(n_devices)]
    
    # Create pools of available indices for each class
    available_by_class = {k: selected_indices[k].copy() for k in range(K)}
    
    for device_id in range(n_devices):
        if alpha == 0:
            # Special case: each device gets only one class
            # Try to assign different classes to different devices if possible
            assigned_classes = [len(device_indices[i]) > 0 for i in range(device_id)]
            available_classes = [k for k in range(K) if len(available_by_class[k]) > 0]
            
            if available_classes:
                label = np.random.choice(available_classes)
                take = min(total_samples_per_device, len(available_by_class[label]))
                device_indices[device_id].extend(available_by_class[label][:take])
                available_by_class[label] = available_by_class[label][take:]
        else:
            # Sample from Dirichlet distribution
            proportions = np.random.dirichlet(np.repeat(alpha, K))
            proportions = proportions / proportions.sum()
            
            # Calculate how many samples from each class for this device
            assigned_counts = {k: int(p * total_samples_per_device) for k, p in enumerate(proportions)}
            
            # Adjust to ensure we assign exactly total_samples_per_device
            total_assigned = sum(assigned_counts.values())
            if total_assigned < total_samples_per_device:
                # Add remaining samples to classes with highest proportions
                sorted_classes = sorted(range(K), key=lambda k: proportions[k], reverse=True)
                for k in sorted_classes:
                    if len(available_by_class[k]) > assigned_counts[k]:
                        diff = min(total_samples_per_device - total_assigned, 
                                 len(available_by_class[k]) - assigned_counts[k])
                        assigned_counts[k] += diff
                        total_assigned += diff
                        if total_assigned >= total_samples_per_device:
                            break
            
            # Assign samples from each class based on calculated counts
            for k in range(K):
                if assigned_counts[k] > 0 and len(available_by_class[k]) > 0:
                    take = min(assigned_counts[k], len(available_by_class[k]))
                    device_indices[device_id].extend(available_by_class[k][:take])
                    available_by_class[k] = available_by_class[k][take:]
            
            # If we couldn't assign enough samples, take from any available class
            while len(device_indices[device_id]) < total_samples_per_device:
                available_classes = [k for k in range(K) if len(available_by_class[k]) > 0]
                if not available_classes:
                    break
                k = np.random.choice(available_classes)
                device_indices[device_id].append(available_by_class[k].pop(0))
    
    return device_indices


def DataLoaders(n_devices, dataset_name, total_samples, log_dirpath, seed, mode="non-iid", batch_size=32, alpha=1.0, dataloader_workers=1, need_test_loaders=False):
    """
    Modified DataLoaders function that uses the same data points for both IID and Non-IID distributions.
    """
    
    # Load the dataset
    if dataset_name == "mnist":
        data_dir = './data'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = tv.datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)
        test_dataset = tv.datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)
    elif dataset_name == "cifar10":
        data_dir = './data'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = tv.datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)
        test_dataset = tv.datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Step 1: Select a fixed subset of data points based on seed
    # This ensures the same data points are used for both IID and Non-IID
    selected_indices, total_per_class = select_fixed_dataset(train_dataset, total_samples, n_devices, seed)
    
    # Log the selected data distribution
    with open(f"{log_dirpath}/selected_data_distribution.txt", "w") as f:
        f.write(f"Total samples selected per class (same for both IID and Non-IID):\n")
        for k in range(10):
            f.write(f"Class {k}: {total_per_class.get(k, 0)} samples\n")
        f.write(f"Total: {sum(total_per_class.values())} samples\n")
    
    # Step 2: Distribute the selected data points based on mode
    if mode == "iid":
        device_indices = distribute_iid(selected_indices, n_devices, total_samples)
    elif mode == "non-iid":
        device_indices = distribute_noniid(selected_indices, n_devices, total_samples, alpha, seed)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Step 3: Create data loaders
    user_train_loaders = []
    user_test_loaders = []
    user_label_to_qty = {}
    
    # Prepare test data splits (same as original implementation)
    K = 10
    test_labels = np.array(test_dataset.targets)
    test_idxs = np.arange(len(test_dataset))
    test_idxs_labels = np.vstack((test_idxs, test_labels))
    test_idxs_labels = test_idxs_labels[:, test_idxs_labels[1, :].argsort()]
    sorted_test_idxs = test_idxs_labels[0, :]
    sorted_test_labels = test_idxs_labels[1, :]
    test_idxs_splits = [[] for _ in range(K)]
    for i in range(len(sorted_test_labels)):
        test_idxs_splits[sorted_test_labels[i]].append(sorted_test_idxs[i])
    
    # Create loaders for each device
    for i, train_idx in enumerate(device_indices):
        # Training loader
        sampler_train = torch.utils.data.SubsetRandomSampler(train_idx)
        loader_train = torch.utils.data.DataLoader(train_dataset, sampler=sampler_train, 
                                                   batch_size=batch_size, num_workers=dataloader_workers)
        user_train_loaders.append(loader_train)
        
        # Get training labels for this device
        if len(train_idx) > 0:
            labels_arr = np.array(train_dataset.targets)[train_idx]
            class_counts = np.bincount(labels_arr, minlength=10)
            user_labels_set = set(labels_arr)
        else:
            class_counts = np.zeros(10, dtype=int)
            user_labels_set = set()
        
        # Test data assignment: ALL test samples from training classes
        test_idx = []
        for label in user_labels_set:
            test_idx.extend(test_idxs_splits[label])
        
        # Test loader
        if len(test_idx) > 0:
            sampler_test = torch.utils.data.SubsetRandomSampler(test_idx)
            loader_test = torch.utils.data.DataLoader(test_dataset, sampler=sampler_test, 
                                                      batch_size=batch_size, num_workers=dataloader_workers)
        else:
            loader_test = torch.utils.data.DataLoader(test_dataset, 
                                                      sampler=torch.utils.data.SubsetRandomSampler([]), 
                                                      batch_size=batch_size, num_workers=dataloader_workers)
        user_test_loaders.append(loader_test)
        
        # Log distributions
        if len(test_idx) > 0:
            test_labels_arr = np.array(test_dataset.targets)[test_idx]
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
    
    # Global test loader
    global_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                                     num_workers=dataloader_workers)
    
    return user_train_loaders, user_test_loaders, user_label_to_qty, global_test_loader


# Legacy functions for backward compatibility (using new implementation internally)
def iid_split(n_clients, train_data, batch_size, test_data, dataloader_workers, log_dirpath, total_samples, need_test_loaders):
    """Legacy wrapper - uses new unified implementation"""
    # Create a default seed if not provided
    seed = 42
    dataset_name = "cifar10" if hasattr(train_data, 'classes') and len(train_data.classes) == 10 else "mnist"
    return DataLoaders(n_clients, dataset_name, total_samples, log_dirpath, seed, 
                      mode="iid", batch_size=batch_size, dataloader_workers=dataloader_workers, 
                      need_test_loaders=need_test_loaders)


def get_data_noniid_cifar10(n_devices, total_samples, log_dirpath, seed, batch_size=32, alpha=1.0, dataloader_workers=1, need_test_loaders=False):
    """Legacy wrapper - uses new unified implementation"""
    return DataLoaders(n_devices, "cifar10", total_samples, log_dirpath, seed,
                      mode="non-iid", batch_size=batch_size, alpha=alpha,
                      dataloader_workers=dataloader_workers, need_test_loaders=need_test_loaders)


def get_data_noniid_mnist(n_devices, total_samples, log_dirpath, seed, batch_size=32, alpha=1.0, dataloader_workers=1, dataset_mode="non-iid", need_test_loaders=False):
    """Legacy wrapper - uses new unified implementation"""
    mode = dataset_mode if dataset_mode in ["iid", "non-iid"] else "non-iid"
    return DataLoaders(n_devices, "mnist", total_samples, log_dirpath, seed,
                      mode=mode, batch_size=batch_size, alpha=alpha,
                      dataloader_workers=dataloader_workers, need_test_loaders=need_test_loaders)
