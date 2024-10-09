import numpy as np
from torchvision import datasets, transforms


def get_dataset_mnist_extr_noniid(n_devices, n_classes, nsamples, rate_unbalance, log_dirpath):
    data_dir = './data'
    apply_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                   transform=apply_transform)

    test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                  transform=apply_transform)

    # Chose euqal splits for every user
    user_groups_train, user_groups_test, user_groups_labels = mnist_extr_noniid(
        train_dataset, test_dataset, n_devices, n_classes, nsamples, rate_unbalance, log_dirpath)
    return train_dataset, test_dataset, user_groups_train, user_groups_test, user_groups_labels


def mnist_extr_noniid(train_dataset, test_dataset, n_devices, n_classes, num_samples, rate_unbalance, log_dirpath): # BUG: test data labels not corresponding to the train! trainig sharding starting from line 73 may need to change
    num_shards_train, num_imgs_train = int(60000/num_samples), num_samples
    num_classes = 10
    num_imgs_perc_test, num_imgs_test_total = 1000, 10000

    assert(n_classes * n_devices <= num_shards_train)
    assert(n_classes <= num_classes)

    idx_class = [i for i in range(num_classes)]
    idx_shard = np.array([i for i in range(num_shards_train)])

    dict_users_train = {i: np.array([]) for i in range(n_devices)}
    dict_users_test = {i: np.array([]) for i in range(n_devices)}
    dict_users_labels = {i: np.array([]) for i in range(n_devices)}
    
    idxs = np.arange(num_shards_train*num_imgs_train) # make indices from 0 to 59999
    labels = np.array(train_dataset.targets) # labels of the corresponding 60000 training samples
    
    idxs_test = np.arange(num_imgs_test_total) # make indices from 0 to 9999
    labels_test = np.array(test_dataset.targets) # labels of the corresponding 10000 test samples

    idxs_labels = np.vstack((idxs, labels)) # stack the indices and labels vertically, becomes a 2D array, first row is indices, second row is corresponding labels
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()] # sort by labels from 0 to 9, and ensures that all samples with the same label are grouped together.
    idxs = idxs_labels[0, :] # extract indices
    labels = idxs_labels[1, :] # extract labels

    idxs_labels_test = np.vstack((idxs_test, labels_test)) # do the same for test set
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_test = idxs_labels_test[0, :]
    labels_test = idxs_labels_test[1, :]

    idxs_test_splits = [[] for i in range(num_classes)]
    for i in range(len(labels_test)):
        idxs_test_splits[labels_test[i]].append(idxs_test[i]) # append the index of the test sample to the corresponding list of the label of the test sample -> key: label, value: list of indices of samples with that label

    idx_shards = np.split(idx_shard, 10) # split the training sample shards into 10 groups
    for i in range(n_devices):
        user_labels = np.array([])
        temp_set = list(set(np.random.choice(10, n_classes, replace=False))) # randomly get n_classes number of labels without duplicates
        dict_users_labels[i] = temp_set # record the labels assigned to the device
        rand_set = []
        for j in temp_set:
            choice = np.random.choice(idx_shards[j], 1)[0] # randomly pick a shard
            rand_set.append(int(choice))
            idx_shards[j] = np.delete(
                idx_shards[j], np.where(idx_shards[j] == choice))
        unbalance_flag = 0
        label_to_qty = {}
        for rand_iter in range(len(rand_set)):	
            rand = rand_set[rand_iter]
            if unbalance_flag == 0:
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], idxs[rand*num_imgs_train:(rand+1)*num_imgs_train]), axis=0)
                label_to_qty[temp_set[rand_iter]] = len(idxs[rand*num_imgs_train:(rand+1)*num_imgs_train])
                user_labels = np.concatenate(
                    (user_labels, labels[rand*num_imgs_train:(rand+1)*num_imgs_train]), axis=0)
            else:
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], idxs[rand*num_imgs_train:int((rand+rate_unbalance)*num_imgs_train)]), axis=0)
                label_to_qty[temp_set[rand_iter]] = len(idxs[rand*num_imgs_train:int((rand+rate_unbalance)*num_imgs_train)])
                user_labels = np.concatenate(
                    (user_labels, labels[rand*num_imgs_train:int((rand+rate_unbalance)*num_imgs_train)]), axis=0)
            unbalance_flag = 1

        display_text = f"Device {i + 1}  - labels {list(label_to_qty.keys())}, corresponding qty {list(label_to_qty.values())}"
        with open(f'{log_dirpath}/dataset_assigned.txt', 'a') as f:
            f.write(f'{display_text}\n')
        print(display_text)

        user_labels_set = set(user_labels)

        for label in user_labels_set:
            dict_users_test[i] = np.concatenate(
                (dict_users_test[i], idxs_test_splits[int(label)]), axis=0)
    return dict_users_train, dict_users_test, dict_users_labels
