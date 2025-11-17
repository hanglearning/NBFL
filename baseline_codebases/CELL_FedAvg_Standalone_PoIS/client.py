import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module
import numpy as np
import os
from typing import Dict
import copy
import math
from torch.nn.utils import prune
from baseline_utils import *
from baseline_utils import train as util_train
from baseline_utils import test as util_test

import random
from util import *

class Client():
    def __init__(
        self,
        idx,
        args,
        is_malicious,
        attack_type,
        noise_variance,
        init_global_model,
        train_loader=None,
        test_loader=None,
        user_labels = None,
        global_test_loader=None,
        **kwargs
    ):
        self.idx = idx
        self.args = args
        self._is_malicious = is_malicious
        self.attack_type = attack_type
        self._test_loader = test_loader
        self._train_loader = train_loader
        self._user_labels = user_labels
        self.global_test_loader = global_test_loader

        self.eita_hat = self.args.eita
        self.eita = self.eita_hat
        self.alpha = self.args.alpha
        self.num_data = len(train_loader)

        self.elapsed_comm_rounds = 0

        self.accuracies = []
        self.losses = []
        self.prune_rates = []
        self.cur_prune_rate = 0.00

        self.model = copy_model(init_global_model, args.dev_device)
        self.global_model = copy_model(init_global_model, args.dev_device)
        self.init_global_model = copy_model(init_global_model, args.dev_device)

        self.noise_variance = noise_variance
    
    def model_learning_max(self, comm_round, logger):

        produce_mask_from_model_in_place(self.model)

        logger['global_test_acc'][comm_round][self.idx] = self.eval_model_by_global_test(self.model)
        logger['global_model_sparsity'][comm_round][self.idx] = 1 - get_pruned_ratio(self.model)

        print()
        L_or_M = "M" if self._is_malicious else "L"
        attack_type = 'Legitimate'
        if L_or_M == 'M':
            if self.attack_type == 1:
                attack_type = 'Poison Attack'
            if self.attack_type == 3:
                attack_type = 'Lazy'
        
        print(f"\n---------- {L_or_M} {attack_type} Worker:{self.idx} Train to Max Acc Update ---------------------")

        if comm_round > 1 and self.args.reset:
        # reinitialize model with init_params
            source_params = dict(self.init_global_model.named_parameters())
            for name, param in self.model.named_parameters():
                param.data.copy_(source_params[name].data)
        
        max_epoch = self.args.epochs

        # lazy worker
        if self._is_malicious and self.attack_type == 3:
            max_epoch = int(max_epoch * 0.1)

        # init max_acc as the initial global model acc on local training set
        max_acc = self.eval_model_by_train(self.model)

        if self._is_malicious and self.attack_type == 1:
            # skip training and poison local model on trainable weights before submission
            self.poison_model(self.model)
            poinsoned_acc = self.eval_model_by_train(self.model)
            print(f'Poisoned accuracy: {poinsoned_acc}, decreased {max_acc - poinsoned_acc}.')
            self.max_model_acc = poinsoned_acc
        else:
            max_model = copy_model(self.model, self.args.dev_device)
            max_model_epoch = epoch = 0
            # train to max accuracy
            while epoch < max_epoch and max_acc != 1.0:
                if self.args.train_verbose:
                    print(f"Worker={self.idx}, epoch={epoch + 1}")

                util_train(model = self.model,
                            train_dataloader = self._train_loader,
                            optimizer_type = self.args.optimizer,
                            lr = self.args.lr,
                            device = self.args.dev_device,
                            verbose = self.args.train_verbose)
                
                acc = self.eval_model_by_train(self.model)
                if acc > max_acc:
                    max_model = copy_model(self.model, self.args.dev_device)
                    max_acc = acc
                    max_model_epoch = epoch + 1

                epoch += 1


            print(f"Worker {self.idx} with max training acc {max_acc} arrived at epoch {max_model_epoch}.")
            logger['local_max_epoch'][comm_round][self.idx] = max_model_epoch

            self.model = max_model
            self.max_model_acc = max_acc

        logger['local_max_acc'][comm_round][self.idx] = self.max_model_acc
        logger['local_test_acc'][comm_round][self.idx] = self.eval_model_by_local_test(self.model)

    def worker_prune(self, comm_round, logger):

        # model prune percentage
        init_pruned_ratio = get_pruned_ratio(self.model) # pruned_ratio = 0s/total_params = 1 - sparsity
        if self.attack_type != 1 and 1 - init_pruned_ratio <= self.args.target_sparsity:
            print(f"Worker {self.idx}'s model at sparsity {1 - init_pruned_ratio}, which is already <= the target sparsity {self.args.target_sparsity}. Skip pruning.")
            return
        
        max_intended_prune_amount = self.max_model_acc
        if max_intended_prune_amount < get_pruned_ratio(self.model):
            max_intended_prune_amount = get_pruned_ratio(self.model)
        
        print()
        L_or_M = "M" if self._is_malicious else "L"
        print(f"\n---------- {L_or_M} Worker:{self.idx} starts pruning ---------------------")

        init_model_acc = self.eval_model_by_train(self.model)
        accs = [init_model_acc]

        to_prune_amount = init_pruned_ratio
        last_pruned_model = copy_model(self.model, self.args.dev_device)

        while True:
            if self._is_malicious and self.attack_type == 1:
                to_prune_amount = 1 - self.args.target_sparsity # noise attacker tries to maximize the overlapping mask reward
            else:
                to_prune_amount += random.uniform(0, self.args.max_prune_step)
                to_prune_amount = min(to_prune_amount, max_intended_prune_amount, 1 - self.args.target_sparsity) # ensure the pruned amount is not larger than the target sparsity
            pruned_model = copy_model(self.model, self.args.dev_device)
            make_prune_permanent(pruned_model)
            l1_prune(model=pruned_model,
                        amount=to_prune_amount,
                        name='weight',
                        verbose=self.args.prune_verbose)
            
            model_acc = self.eval_model_by_train(pruned_model)
            
            # prune until the accuracy drop exceeds the threshold or below the target sparsity
            if not (self._is_malicious and self.attack_type == 1) and init_model_acc - model_acc > self.args.acc_drop_threshold: # or 1 - to_prune_amount <= self.args.target_sparsity:
                # revert to the last pruned model
                self.model = copy_model(last_pruned_model, self.args.dev_device) # copy mask as well
                self.max_model_acc = accs[-1]
                break

            if to_prune_amount == max_intended_prune_amount or 1 - to_prune_amount <= self.args.target_sparsity:
                # reached intended prune amount, stop
                self.model = copy_model(pruned_model, self.args.dev_device)
                self.max_model_acc = model_acc
                break

            accs.append(model_acc)
            last_pruned_model = copy_model(pruned_model, self.args.dev_device)

        after_pruned_ratio = get_pruned_ratio(self.model) # pruned_ratio = 0s/total_params = 1 - sparsity
        after_pruning_acc = self.eval_model_by_train(self.model)

        self._worker_pruned_ratio = after_pruned_ratio

        print(f"Model sparsity: {1 - after_pruned_ratio:.2f}")
        print(f"Pruned model before and after accuracy: {init_model_acc:.2f}, {after_pruning_acc:.2f}")
        print(f"Pruned amount: {after_pruned_ratio - init_pruned_ratio:.2f}")

        logger['worker_pruned_amount'][comm_round][self.idx] = after_pruned_ratio - init_pruned_ratio
        logger['after_prune_sparsity'][comm_round][self.idx] = 1 - after_pruned_ratio
        logger['after_prune_acc'][comm_round][self.idx] = after_pruning_acc
        logger['after_prune_local_test_acc'][comm_round][self.idx] = self.eval_model_by_local_test(self.model)
        logger['after_prune_global_test_acc'][comm_round][self.idx] = self.eval_model_by_global_test(self.model)
    
    def update_standalone_LTH(self, logger) -> None:
        
        print(f"\nClient {self.idx} doing standalone LTH")
        
        for comm_round in range(1, self.args.rounds + 1):
            
            print("\nRound", comm_round)
            
            self.model_learning_max(comm_round, logger)
            self.worker_prune(comm_round, logger)


    def update_fedavg_no_prune_max_acc(self, comm_round, logger):

        L_or_M = "M" if self._is_malicious else "L"
        print(f"\n----------Client[{L_or_M}]:{self.idx} FedAvg without Pruning Update---------------------")

        self.model_learning_max(comm_round, logger)

    def update_CELL(self, comm_round, logger) -> None:
        """
            Interface to Server
        """
        print(f"\n----------Client:{self.idx} Update---------------------")

        print(f"Evaluating Global model ")
        metrics = self.eval(self.global_model)
        accuracy = metrics['MulticlassAccuracy'][0]
        print(f'Global model accuracy: {accuracy}')

        prune_rate = get_prune_summary(model=self.global_model,
                                       name='weight')['global']
        print('Global model prune percentage: {}'.format(prune_rate))
           
        if self.cur_prune_rate < self.args.prune_threshold:
            if accuracy > self.eita:
                self.cur_prune_rate = min(self.cur_prune_rate + self.args.prune_step,
                                          self.args.prune_threshold)
                if self.cur_prune_rate > prune_rate:
                    l1_prune(model=self.global_model,
                             amount=self.cur_prune_rate - prune_rate,
                             name='weight',
                             verbose=self.args.prune_verbose)
                    self.prune_rates.append(self.cur_prune_rate)
                else:
                    self.prune_rates.append(prune_rate)
                # reinitialize model with init_params
                source_params = dict(self.global_init_model.named_parameters())
                for name, param in self.global_model.named_parameters():
                    param.data.copy_(source_params[name].data)

                self.model = self.global_model
                self.eita = self.eita_hat

            else:
                self.eita *= self.alpha
                self.model = self.global_model
                self.prune_rates.append(prune_rate)
        else:
            if self.cur_prune_rate > prune_rate:
                l1_prune(model=self.global_model,
                         amount=self.cur_prune_rate-prune_rate,
                         name='weight',
                         verbose=self.args.prune_verbose)
                self.prune_rates.append(self.cur_prune_rate)
            else:
                self.prune_rates.append(prune_rate)
            self.model = self.global_model


        print(f"\nTraining local model")
        # self.train(self.elapsed_comm_rounds)
        self.model_learning_max(comm_round, logger)

        print(f"\nEvaluating Trained Model")
        metrics = self.eval(self.model)
        print(f'Trained model accuracy: {metrics['MulticlassAccuracy'][0]}')

        logger['local_test_acc'][comm_round][self.idx] = self.eval_model_by_local_test(self.model)

        if (self.elapsed_comm_rounds+1) % self.args.save_freq == 0:
            self.save(self.model)

        self.elapsed_comm_rounds += 1

    def update_PoIS(self, comm_round, logger):

        self.model_learning_max(comm_round, logger)


    def train(self, round_index):
        """
            Train NN
        """
        losses = []

        for epoch in range(self.args.epochs):
            if self.args.train_verbose:
                print(
                    f"Client={self.idx}, epoch={epoch}, round:{round_index}")

            util_train(model = self.model,
                            train_dataloader = self._train_loader,
                            optimizer_type = self.args.optimizer,
                            lr = self.args.lr,
                            device = self.args.dev_device,
                            verbose = self.args.train_verbose)
            losses.append(metrics['Loss'][0])

            if self.args.fast_dev_run:
                break
        self.losses.extend(losses)

    @torch.no_grad()
    def download(self, global_model, global_init_model, *args, **kwargs):
        """
            Download global model from server
        """
        self.global_model = global_model
        self.global_init_model = global_init_model

        params_to_prune = get_prune_params(self.global_model)
        for param, name in params_to_prune:
            weights = getattr(param, name)
            masked = torch.eq(weights.data, 0.00).sum().item()
            # masked = 0.00
            prune.l1_unstructured(param, name, amount=int(masked))

        params_to_prune = get_prune_params(self.global_init_model)
        for param, name in params_to_prune:
            weights = getattr(param, name)
            masked = torch.eq(weights.data, 0.00).sum().item()
            # masked = 0.00
            prune.l1_unstructured(param, name, amount=int(masked))

    def eval(self, model):
        """
            Eval self.model
        """
        eval_score = util_test(model,
                               self._test_loader,
                               self.args.dev_device,
                               self.args.fast_dev_run,
                               self.args.test_verbose)
        self.accuracies.append(eval_score['MulticlassAccuracy'][0])
        return eval_score

    def save(self, *args, **kwargs):
        pass

    def upload(self, *args, **kwargs) -> Dict[nn.Module, float]:
        """
            Upload self.model
        """
        upload_model = copy_model(model=self.model, device=self.args.dev_device)
        params_pruned = get_prune_params(upload_model, name='weight')
        for param, name in params_pruned:
            prune.remove(param, name)
        return {
            'model': upload_model,
            # 'acc': self.accuracies[-1]
        }

    def poison_model(self, model):
        layer_to_mask = calc_mask_from_model_with_mask_object(model) # introduce noise to unpruned weights
        for layer, module in model.named_children():
            for name, weight_params in module.named_parameters():
                if "weight" in name:
                    # noise = self.args.noise_variance * torch.randn(weight_params.size()).to(self.args.dev_device) * torch.from_numpy(layer_to_mask[layer]).to(self.args.dev_device)
                    noise = self.noise_variance * torch.randn(weight_params.size()).to(self.args.dev_device) * layer_to_mask[layer].to(self.args.dev_device)
                    weight_params.data.add_(noise.to(self.args.dev_device))  # Modify weights in place
        print(f"Client {self.idx} poisoned the whole network with variance {self.noise_variance}.") # or should say, unpruned weights?

    def eval_model_by_local_test(self, model):
        """
            Eval self.model by local test dataset - containing all samples corresponding to the device's training labels
        """
        return test_by_data_set(model,
                               self._test_loader,
                               self.args.dev_device,
                               self.args.test_verbose)['MulticlassAccuracy'][0]
    
    def eval_model_by_global_test(self, model):
        """
            Eval self.model by global test dataset - containing all samples of the original test dataset
        """
        return test_by_data_set(model,
                self.global_test_loader,
                self.args.dev_device,
                self.args.test_verbose)['MulticlassAccuracy'][0]


    def eval_model_by_train(self, model):
        """
            Eval self.model by local training dataset
        """
        return test_by_data_set(model,
                               self._train_loader,
                               self.args.dev_device,
                               self.args.test_verbose)['MulticlassAccuracy'][0]
