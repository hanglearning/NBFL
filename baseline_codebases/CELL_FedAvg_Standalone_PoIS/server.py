from typing import List, Dict, Tuple
import torch.nn.utils.prune as prune
import numpy as np
import random
import os
import copy
from tabulate import tabulate
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module
from baseline_utils import get_prune_params, super_prune, fed_avg, l1_prune, create_model, copy_model, get_prune_summary
import pickle
import shap
from scipy import stats
from util import make_prune_permanent

class Server():
    """
        Central Server
    """

    def __init__(
        self,
        args,
        model,
        clients
    ):
        super().__init__()
        self.clients = clients
        self.n_clients = len(self.clients)
        self.args = args
        self.model = model
        self.init_model = copy_model(model, self.args.dev_device)

        self.elapsed_comm_rounds = 0
        self.curr_prune_step = 0.00

    def aggr(
        self,
        models,
        clients,
        *args,
        **kwargs
    ):
        weights_per_client = np.array(
            [client.num_data for client in clients], dtype=np.float32)
        weights_per_client /= np.sum(weights_per_client)

        aggr_model = fed_avg(
            models=models,
            weights=weights_per_client,
            device=self.args.dev_device
        )
        pruned_percent = get_prune_summary(aggr_model, name='weight')['global']
        # pruned by the earlier zeros in the model
        l1_prune(aggr_model, amount=pruned_percent, name='weight')

        return aggr_model
    
    def aggr_pois(
        self,
        global_model, models_to_aggregate):
        global_model = make_prune_permanent(global_model)
        if models_to_aggregate: 
            global_dict = global_model.state_dict()   
            for k in global_dict.keys():
                global_dict[k] = torch.stack([models_to_aggregate[i].state_dict()[k].float() for i in range(len(models_to_aggregate))], 0).mean(0)
            global_model.load_state_dict(global_dict)
        l1_prune(global_model, amount=0.00, name='weight', verbose=False)
        return global_model

    def update(
        self,
        comm_round,
        logger,
        background,
        test_images,
        *args,
        **kwargs
    ):
        """
            Interface to server and clients
        """
        self.elapsed_comm_rounds += 1
        print('-----------------------------', flush=True)
        print(
            f'| Communication Round: {self.elapsed_comm_rounds}  | ', flush=True)
        print('-----------------------------', flush=True)

        # global_model pruned at fixed freq
        # with a fixed pruning step
        if (self.args.server_prune == True and
                (self.elapsed_comm_rounds % self.args.server_prune_freq) == 0):
            # prune the model using super_mask
            self.curr_prune_step += self.args.prune_step
            super_prune(
                model=self.model,
                init_model=self.init_model,
                amount=self.curr_prune_step,
                name='weight'
            )
            # reinitialize model with std.dev of init_model
            source_params = dict(self.init_model.named_parameters())
            for name, param in self.model.named_parameters():
                std_dev = torch.std(source_params[name].data)
                param.data.copy_(std_dev*torch.sign(source_params[name].data))

        client_idxs = np.random.choice(
            self.n_clients, int(
                self.args.frac_clients_per_round*self.n_clients),
            replace=False,
        )
        clients = [self.clients[i] for i in client_idxs]

        # upload model to selected clients
        self.upload(clients)

         # call training loop on all clients
        for client in clients:
            if self.args.standalone_LTH:
                client.update_standalone_LTH(logger)
            if self.args.fedavg_no_prune_max_acc:
                client.update_fedavg_no_prune_max_acc(comm_round, logger)
            if self.args.PoIS:
                client.update_PoIS(comm_round, logger)
            if self.args.CELL:
                client.update_CELL(comm_round, logger)
        
        if self.args.standalone_LTH:
            with open(f'{self.args.log_dir}/logger.pickle', 'wb') as f:
                pickle.dump(logger, f)
            import sys
            sys.exit() 


        # download models from selected clients
        # models, accs = self.download(clients)
        models = self.download(clients)

        if self.args.PoIS:
            torch.set_grad_enabled(True)
            # https://github.com/harshkasyap/DecFL/blob/master/Non%20IID/dirichlet%20distribution/vary%20attacker/fm_noniid_ba_9.py
            models_to_aggregate = []
            threshold = 1.8
            id = 0
            shap_data_temp = []
            for model in models:
                e = shap.DeepExplainer(model, background)
                shap_values = e.shap_values(test_images, check_additivity=False)
                #print(shap_values)
                print('client id : {}'.format(id))
                id += 1
                shap_data_temp_model = []
                # for i in range(10):
                shap_data_temp_model.append(torch.sum(torch.tensor(shap_values[0])))
                shap_data_temp.append(shap_data_temp_model)
                temp = []
                for shap_label in shap_data_temp_model:
                    temp.append(shap_label.detach().item())
                z_score = np.abs(stats.zscore(temp))
                print(z_score)
                flag = True
                for j in range(len(z_score)):
                    if z_score[j] > threshold:
                        print('client {} not appended'.format(id))
                        flag = False
                        break
                
                if flag == True:
                    print('client {} is aggregated'.format(id))
                    models_to_aggregate.append(copy.copy(model))

        # compute average-model
        if self.args.PoIS:
            aggr_model = self.aggr_pois(self.model, models_to_aggregate)
        else:
            aggr_model = self.aggr(models, clients)

        # copy aggregated-model's params to self.model (keep buffer same)
        source_params = dict(aggr_model.named_parameters())
        for name, param in self.model.named_parameters():
            param.data.copy_(source_params[name])

    def download(
        self,
        clients,
        *args,
        **kwargs
    ):
        # downloaded models are non pruned (taken care of in fed-avg)
        uploads = [client.upload() for client in clients]
        models = [upload["model"] for upload in uploads]
        # accs = [upload["acc"] for upload in uploads]
        return models #, accs

    def save(
        self,
        *args,
        **kwargs
    ):
        # """
        #     Save model,meta-info,states
        # """
        # eval_log_path1 = f"./log/full_save/server/round{self.elapsed_comm_rounds}_model.pickle"
        # eval_log_path2 = f"./log/full_save/server/round{self.elapsed_comm_rounds}_dict.pickle"
        # if self.args.report_verbosity:
        #     log_obj(eval_log_path1, self.model)
        pass

    def upload(
        self,
        clients,
        *args,
        **kwargs
    ) -> None:
        """
            Upload global model to clients
        """
        for client in clients:
            # make pruning permanent and then upload the model to clients
            model_copy = copy_model(self.model, self.args.dev_device)
            init_model_copy = copy_model(self.init_model, self.args.dev_device)

            params = get_prune_params(model_copy, name='weight')
            for param, name in params:
                prune.remove(param, name)

            init_params = get_prune_params(init_model_copy)
            for param, name in init_params:
                prune.remove(param, name)
            # call client method
            client.download(model_copy, init_model_copy)
