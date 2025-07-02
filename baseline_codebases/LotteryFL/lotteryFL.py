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
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision as tv
from torchvision import transforms
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference

from datetime import datetime
import pickle


class LotteryFLLocalUpdate(object):
    """
    Simplified LocalUpdate class using train/test loaders from DataLoaders function
    """
    def __init__(self, args, trainloader, testloader, global_testloader, device_id):
        self.args = args
        self.trainloader = trainloader
        self.testloader = testloader  # Local test set from DataLoaders
        self.global_testloader = global_testloader  # Global test set
        self.device_id = device_id
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = nn.NLLLoss().to(self.device)

    def update_weights(self, model, epochs, device):
        """
        Original LotteryFL update_weights logic
        """
        EPS = 1e-6
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates (same as original LotteryFL)
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        
        iter = 0
        max_model = model
        max_acc = test_by_data_set(model, self.trainloader, device, verbose=False)['MulticlassAccuracy'][0]

        # Create progress bar for local epochs
        pbar = tqdm(range(epochs), desc=f"Device {self.device_id + 1}", 
                   leave=False, disable=(epochs == 0))

        while iter < epochs:
            batch_loss = []
            
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                
                # Freezing Pruned weights by making their gradients Zero (original LotteryFL logic)
                for name, p in model.named_parameters():
                    if 'weight' in name:
                        tensor = p.data.cpu().numpy()
                        grad_tensor = p.grad.data.cpu().numpy()
                        grad_tensor = np.where(abs(tensor) < EPS, 0, grad_tensor)
                        p.grad.data = torch.from_numpy(grad_tensor).to(device)
                
                optimizer.step()
                batch_loss.append(loss.item())
            
            # Evaluate on training set to find best model (original LotteryFL logic)
            acc = test_by_data_set(model, self.trainloader, device, verbose=False)['MulticlassAccuracy'][0]
            if acc > max_acc:
                max_model = copy.deepcopy(model)
                max_acc = acc
            
            # Update progress bar
            pbar.set_postfix({
                'epoch': f'{iter + 1}/{epochs}',
                'loss': f'{sum(batch_loss)/len(batch_loss):.4f}',
                'acc': f'{acc:.4f}'
            })
            pbar.update(1)
            
            iter += 1
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            if max_acc == 1.0: 
                break

        pbar.close()
        
        if epochs == 0:
            return model.state_dict(), 0
        return max_model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """
        Original LotteryFL inference method - uses local test set and limits to 2 batches
        """
        if self.testloader is None:
            return 0.0, 0.0
            
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            # Original LotteryFL: each client test 100 images at most for running time
            if batch_idx > 1:
                break
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            
        accuracy = correct/total if total > 0 else 0.0
        return accuracy, loss

    # Additional evaluation methods
    def evaluate_on_train(self, model):
        """Evaluate model on training set"""
        return test_by_data_set(model, self.trainloader, self.device, verbose=False)['MulticlassAccuracy'][0]
    
    def evaluate_on_local_test(self, model):
        """Evaluate model on local test set (full evaluation, not limited to 2 batches)"""
        if self.testloader is None:
            return 0.0
        return test_by_data_set(model, self.testloader, self.device, verbose=False)['MulticlassAccuracy'][0]
    
    def evaluate_on_global_test(self, model):
        """Evaluate model on global test set"""
        return test_by_data_set(model, self.global_testloader, self.device, verbose=False)['MulticlassAccuracy'][0]


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
    log_root_name = f"LotteryFL_{args.dataset}_seed_{args.seed}_{args.dataset_mode}_alpha_{args.alpha}_{exe_date_time}_ndevices_{args.n_clients}_nsamples_{args.total_samples}_rounds_{args.epochs}_mal_{args.n_malicious}_attack_{args.attack_type}_rewind_{args.rewind}"

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
        layer_to_mask = calc_mask_from_model_without_mask_object(model)
        for layer, module in model.named_children():
            for name, weight_params in module.named_parameters():
                if "weight" in name:
                    noise = noise_variance * torch.randn(weight_params.size()).to(device) * torch.from_numpy(layer_to_mask[layer]).to(device)
                    weight_params.data.add_(noise.to(device))
        print(f"User {idx} poisoned the whole neural network with variance {noise_variance}.")
        
    set_seed(args.seed)
    print(f"Seed set: {args.seed}")

    # Get train and test loaders from DataLoaders function
    print("\n=== Getting Data from DataLoaders Function ===")
    train_loaders, test_loaders, user_labels, global_test_loader = DataLoaders(
        n_devices=args.n_clients,
        dataset_name=args.dataset,
        total_samples=args.total_samples,
        log_dirpath=args.log_dir,
        seed=args.seed,
        mode=args.dataset_mode,
        batch_size=args.batch_size,
        alpha=args.alpha,
        dataloader_workers=args.num_workers,
        need_test_loaders=True
    )

    # Logger structure following original LotteryFL
    logger = {}
    logger['global_test_acc'] = {r: {} for r in range(1, args.epochs + 1)}
    logger['global_model_sparsity'] = {r: {} for r in range(1, args.epochs + 1)}
    logger['local_train_acc'] = {r: {} for r in range(1, args.epochs + 1)}
    logger['local_test_acc'] = {r: {} for r in range(1, args.epochs + 1)}
    logger['local_max_acc'] = {r: {} for r in range(1, args.epochs + 1)}

    # save args
    with open(f'{args.log_dir}/args.pickle', 'wb') as f:
        pickle.dump(args, f)

    # Malicious user noise variances (same as original)
    if args.n_malicious == 3:
        noise_variances = [0.05]
    elif args.n_malicious == 6:
        noise_variances = [0.05, 0.5, 1.0]
    elif args.n_malicious == 9:
        noise_variances = [0.05, 0.25, 0.5, 1.0]
    elif args.n_malicious == 10:
        noise_variances = [0.05, 0.25, 0.5, 0.75, 1.0]

    # Main training loop (following original LotteryFL structure)
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
        
        # Create progress bar for users in this round
        user_pbar = tqdm(idxs_users, desc=f"Round {epoch+1} Users", leave=False)
        
        for idx in user_pbar:
            user_pbar.set_description(f"Round {epoch+1} - User {idx + 1}")

            # Use simplified LocalUpdate with train/test loaders from DataLoaders
            local_model = LotteryFLLocalUpdate(
                args=args, 
                trainloader=train_loaders[idx],
                testloader=test_loaders[idx],
                global_testloader=global_test_loader,
                device_id=idx
            )
            
            #test global model before train
            train_model = copy.deepcopy(global_model)

            # Log global test accuracy before training (same as original)
            logger['global_test_acc'][epoch + 1][idx] = local_model.evaluate_on_global_test(train_model)

            #mask the model
            mask_model(train_model, masks[idx], train_model.state_dict())
            acc_beforeTrain, _ = local_model.inference(model=train_model)
            
            logger['global_model_sparsity'][epoch + 1][idx] = 1 - get_pruned_amount(train_model)

            #if test acc is not bad, prune it (original LotteryFL logic)
            if(acc_beforeTrain > args.prune_start_acc and pruning_rate[idx] > args.prune_end_rate):
                #prune it
                prune_by_percentile(train_model, masks[idx], args.prune_percent)
                #update pruning rate
                pruning_rate[idx] = pruning_rate[idx] * (1 - args.prune_percent/100)
                #reset to initial value to make lottery tickets
                if args.rewind:
                    mask_model(train_model, masks[idx], init_weights)

            # Handle malicious users (same as original)
            if idx >= args.n_clients - args.n_malicious:
                if (idx + 1) % 2 == 1:
                    # for malicious user with model poisoning attack, skip training and poison
                    user_pbar.set_postfix_str("Poisoning model")
                    poison_model(train_model, noise_variances[noise_index])
                    noise_index += 1
                    w, loss = local_model.update_weights(
                        model=train_model, epochs=0, device=device) # no train
                else:
                    # lazy attack
                    user_pbar.set_postfix_str("Lazy attack")
                    w, loss = local_model.update_weights(
                    model=train_model, epochs=int(args.local_ep * 0.1), device=device)
            else:
                user_pbar.set_postfix_str("Training")
                w, loss = local_model.update_weights(
                    model=train_model, epochs=args.local_ep, device=device)
                    
            #model used for test
            temp_model = copy.deepcopy(global_model)
            temp_model.load_state_dict(w)
            mask_model(temp_model, masks[idx], temp_model.state_dict())
            acc, _ = local_model.inference(model=temp_model)  # This uses local test set with 2-batch limit
            
            # Original LotteryFL tracking
            if(args.prune_percent != 0):
                users_in_epoch.append(idx)
                if(acc > best_acc[idx]):
                    best_acc[idx] = acc
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            local_masks.append(copy.deepcopy(masks[idx]))
            local_prune.append(pruning_rate[idx])
            local_acc.append(acc)      

            # Enhanced logging
            logger['local_train_acc'][epoch + 1][idx] = local_model.evaluate_on_train(temp_model)
            logger['local_test_acc'][epoch + 1][idx] = local_model.evaluate_on_local_test(temp_model)
            logger['local_max_acc'][epoch + 1][idx] = test_by_data_set(temp_model, local_model.trainloader, device, verbose=False)['MulticlassAccuracy'][0]

        user_pbar.close()
        print("local accuracy: {}\n".format(sum(local_acc)/len(local_acc)))
        
        # update global weights (original LotteryFL logic)
        global_weights_epoch = average_weights_with_masks(local_weights, local_masks, device)
        global_weights = mix_global_weights(global_weights, global_weights_epoch, local_masks, device)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        #compute communication cost rate in this epoch
        communication_cost_epoch = sum(local_prune) / len(local_prune)

        # Save logger after each epoch
        with open(f'{args.log_dir}/logger.pickle', 'wb') as f:
            pickle.dump(logger, f)

        # Print summary statistics
        if (epoch + 1) % print_every == 0:
            avg_train_acc = np.mean([logger['local_train_acc'][epoch + 1][idx] 
                                   for idx in logger['local_train_acc'][epoch + 1]])
            avg_local_test_acc = np.mean([logger['local_test_acc'][epoch + 1][idx] 
                                        for idx in logger['local_test_acc'][epoch + 1] 
                                        if logger['local_test_acc'][epoch + 1][idx] > 0])
            avg_global_test_acc = np.mean([logger['global_test_acc'][epoch + 1][idx] 
                                         for idx in logger['global_test_acc'][epoch + 1]])
            
            print(f"\n=== Epoch {epoch + 1} Summary ===")
            print(f"Average Train Accuracy: {avg_train_acc:.4f}")
            print(f"Average Local Test Accuracy: {avg_local_test_acc:.4f}")
            print(f"Average Global Test Accuracy: {avg_global_test_acc:.4f}")
            print(f"Communication Cost Saved: {100*(1-communication_cost_epoch):.2f}%")

    print(f'\nTotal Run Time: {time.time()-start_time:.4f} seconds')