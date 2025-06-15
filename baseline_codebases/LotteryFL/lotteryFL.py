import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
from pytorch_lightning import seed_everything
import random

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from baseline_codebases.LotteryFL.lotteryfl_utils import *

from dataset.datasource import DataLoaders
from model.mnist.cnn import CNN as CNNMnist

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference


from datetime import datetime
import pickle

if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    exp_details(args)

    device = 'cuda' if args.gpu else 'cpu'

    if not args.n_malicious or not args.attack_type:
        args.n_malicious, args.attack_type = 0, 0
    
    if args.dataset_mode == 'iid':
        args.alpha = '∞'

    exe_date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
    log_root_name = f"LotteryFL_{args.dataset}_seed_{args.seed}_{args.dataset_mode}_alpha_{args.alpha}_{exe_date_time}_ndevices_{args.n_clients}_nsamples_{args.total_samples}_rounds_{args.epochs}_mal_{args.n_malicious}_attack_{args.attack_type}"

    args.log_dir = f"{args.log_dir}/{log_root_name}"
    os.makedirs(args.log_dir)

    print('building model...\n')
    global_model = CNNMnist()
    

    print('model built\n')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # read the init_global_model generated from NBFL
    state_dict = torch.load(f"logs/init_global_model_seed_{args.seed}.pth")
    global_model.load_state_dict(state_dict)

    # copy weights
    global_weights = global_model.state_dict()
    init_weights = copy.deepcopy(global_model.state_dict())

    #make masks
    masks = []
    init_mask = make_mask(global_model)
    for i in range(args.n_clients):
        masks.append(copy.deepcopy(init_mask))
   

    #list to document the pruning rate of each local model
    pruning_rate = []
    for i in range(args.n_clients):
        pruning_rate.append(1)

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0
    best_acc = [0 for i in range(args.n_clients)]

    # Hang adds necessary
    def set_seed(seed):
        seed_everything(seed, workers=True)
        os.environ['PYTHONHASHSEED']=str(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def poison_model(model, noise_variance):
        layer_to_mask = calc_mask_from_model_without_mask_object(model) # introduce noise to unpruned weights
        for layer, module in model.named_children():
            for name, weight_params in module.named_parameters():
                if "weight" in name:
                    noise = noise_variance * torch.randn(weight_params.size()).to(device) * torch.from_numpy(layer_to_mask[layer]).to(device)
                    weight_params.data.add_(noise.to(device))
        print(f"User {idx} poisoned the whole neural network with variance {noise_variance}.") # or should say, unpruned weights?
        
    set_seed(args.seed)
    print(f"Seed set: {args.seed}")

    exe_date_time = datetime.now().strftime("%m%d%Y_%H%M%S")

    train_loaders, test_loaders, user_labels, global_test_loader = DataLoaders(n_devices=args.n_clients,
                                              dataset_name=args.dataset,
                                              total_samples=args.total_samples,
                                              log_dirpath=args.log_dir,
                                              seed=args.seed,
                                              mode=args.dataset_mode,
                                              batch_size=args.batch_size,
                                              alpha=args.alpha,
                                              dataloader_workers=args.num_workers)


    logger = {}
    logger['global_test_acc'] = {r: {} for r in range(1, args.epochs + 1)}
    logger['global_model_sparsity'] = {r: {} for r in range(1, args.epochs + 1)}
    logger['local_max_acc'] = {r: {} for r in range(1, args.epochs + 1)}
    logger['local_test_acc'] = {r: {} for r in range(1, args.epochs + 1)}

    # save args
    with open(f'{args.log_dir}/args.pickle', 'wb') as f:
        pickle.dump(args, f)

    if args.n_malicious == 3:
        noise_variances = [0.05]
    elif args.n_malicious == 6:
        noise_variances = [0.05, 0.5, 1.0]
    elif args.n_malicious == 9:
        noise_variances = [0.05, 0.25, 0.5, 1.0]
    elif args.n_malicious == 10:
        noise_variances = [0.05, 0.25, 0.5, 0.75, 1.0]

    for epoch in tqdm(range(args.epochs)):
        users_in_epoch = []
        local_weights, local_losses , local_masks, local_prune = [], [], [], []
        local_acc = []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.n_clients), 1)
        #sample users for training
        idxs_users = np.random.choice(range(args.n_clients), m, replace=False)
        #train local models
        noise_index = 0
        for idx in idxs_users:
            print(f"User {idx + 1} is training")

            local_model = LocalUpdate(args=args, trainloader=train_loaders[idx],
                                        validloader=test_loaders[idx], testloader=global_test_loader)
            
            # Hang - for malicious user with label flipping attack
            # if idx >= args.n_clients - args.n_malicious and args.attack_type == 2 and epoch == 0:
            #     local_model.trainloader.dataset.targets = 9 - local_model.trainloader.dataset.targets  
                
            #test global model before train
            train_model = copy.deepcopy(global_model)

            # Hang's hack
            logger['global_test_acc'][epoch + 1][idx] = test_by_data_set(train_model, global_test_loader, device)['MulticlassAccuracy'][0]

            #mask the model
            mask_model(train_model, masks[idx], train_model.state_dict())
            acc_beforeTrain, _ = local_model.inference(model = train_model)
            
            logger['global_model_sparsity'][epoch + 1][idx] = 1 - get_pruned_amount(train_model)

            #if test acc is not bad, prune it
            if(acc_beforeTrain > args.prune_start_acc and pruning_rate[idx] > args.prune_end_rate):
                #prune it
                prune_by_percentile(train_model, masks[idx], args.prune_percent)
                #update pruning rate
                pruning_rate[idx] = pruning_rate[idx] * (1 - args.prune_percent/100)
                #reset to initial value to make lottery tickets
                mask_model(train_model, masks[idx], init_weights)

            # Hang
            if idx >= args.n_clients - args.n_malicious:
                if (idx + 1) % 2 == 1:
                    # for malicious user with model poisoning attack, skip training and poison
                    print(f"Malicious user {idx} is poisoning the model")
                    poison_model(train_model, noise_variances[noise_index])
                    noise_index += 1
                    w, loss = local_model.update_weights(
                        model=train_model, epochs=0, device = device) # no train
                else:
                    # lazy attack
                    w, loss = local_model.update_weights(
                    model=train_model, epochs=int(args.local_ep * 0.1), device = device)
            else:
                w, loss = local_model.update_weights(
                    model=train_model, epochs=args.local_ep, device = device)
            #model used for test
            temp_model = copy.deepcopy(global_model)
            temp_model.load_state_dict(w)
            mask_model(temp_model, masks[idx], temp_model.state_dict())
            acc, _ = local_model.inference(model = temp_model)
            #print("user {} is trained, acc_beforeTrain = {}, acc = {}, loss = {}, parameter pruned = {}%".format(idx, acc_beforeTrain, acc, loss, (1 - pruning_rate[idx]) * 100))
            if(args.prune_percent != 0):
                users_in_epoch.append(idx)
                if(acc > best_acc[idx]):
                    best_acc[idx] = acc
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            local_masks.append(copy.deepcopy(masks[idx]))
            local_prune.append(pruning_rate[idx])
            #if acc < 0.95:
            local_acc.append(acc)      

            logger['local_max_acc'][epoch + 1][idx] = test_by_data_set(temp_model, local_model.trainloader, device)['MulticlassAccuracy'][0]
            logger['local_test_acc'][epoch + 1][idx] = test_by_data_set(temp_model, local_model.validloader, device)['MulticlassAccuracy'][0]
       
        print("local accuracy: {}\n".format(sum(local_acc)/len(local_acc)))
        
        # update global weights
        #global_weights = average_weights(local_weights)
        global_weights_epoch = average_weights_with_masks(local_weights, local_masks, device)
        global_weights = mix_global_weights(global_weights, global_weights_epoch, local_masks, device)

        # updatc global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        #compute communication cost rate in this epoch
        communication_cost_epoch = sum(local_prune) / len(local_prune)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        
        with open(f'{args.log_dir}/logger.pickle', 'wb') as f:
            pickle.dump(logger, f)

