# TODO - record accuracy and pruned amount after training and pruning
# NOTE - pick_winning_block(), resync_chain(), validate_chain() and check_resync_eligibility_when_picking() are related to each other

import torch
import numpy as np
from util import *
from util import train as util_train
from util import test_by_data_set
from pathlib import Path
from sklearn.cluster import KMeans

from copy import copy, deepcopy
from Crypto.PublicKey import RSA
from hashlib import sha256
from Block import Block
from Blockchain import Blockchain
import random
import math

from collections import defaultdict
from torch.utils.data import DataLoader


class Device():
    def __init__(
        self,
        idx,
        is_malicious,
        args,
        train_loader,
        test_loader,
        user_labels,
        global_test_loader,
        init_global_model,
    ):
        self.args = args
        
        # blockchain variables
        self.idx = idx
        self.role = None
        self._is_malicious = is_malicious
        self.online = True
        self.has_appended_block = False
        self.peers = set()
        self.blockchain = Blockchain()
        self._received_blocks = {}
        self._resync_to = None # record the last round's picked winning validator to resync chain
        self.verified_winning_block = None
        # for workers
        self._worker_tx = None
        self.layer_to_model_sig_row = {}
        self.layer_to_model_sig_col = {}
        self._train_loader = train_loader
        self._test_loader = test_loader
        self._user_labels = user_labels
        self.global_test_loader = global_test_loader
        self.init_global_model = copy_model(init_global_model, args.dev_device)
        self.model = copy_model(init_global_model, args.dev_device)
        self.max_model_acc = 0
        self._worker_pruned_ratio = 0
        # for validators
        self._validator_tx = None
        self._verified_worker_txs = {} # signature verified
        self._verified_validator_txs = {}
        self._final_global_model = None
        self.produced_block = None
        self._pos_book = {}
        self.worker_to_model_sig = {}
        self.worker_to_acc = {}
        self._device_to_ungranted_reward = defaultdict(float)
        self.worker_to_acc_weight = {}  # used to show the validation performance against the malicious devices in its block
        # init key pair
        self._modulus = None
        self._private_key = None
        self.public_key = None
        self.generate_rsa_key()
    
    ''' Workers' Method '''        

    def model_learning_max(self, comm_round, logger):

        produce_mask_from_model_in_place(self.model)

        logger['global_test_acc'][comm_round][self.idx] = self.eval_model_by_global_test(self.model)
        logger['global_model_sparsity'][comm_round][self.idx] = 1 - get_pruned_ratio(self.model)

        print()
        L_or_M = "M" if self._is_malicious else "L"
        print(f"\n---------- {L_or_M} Worker:{self.idx} {self._user_labels} Train to Max Acc Update ---------------------")

        if comm_round > 1 and self.args.rewind:
        # reinitialize model with initial params
            source_params = dict(self.init_global_model.named_parameters())
            for name, param in self.model.named_parameters():
                param.data.copy_(source_params[name].data)

        max_epoch = self.args.epochs

        # lazy worker
        if self._is_malicious and self.args.attack_type == 3:
            max_epoch = int(max_epoch * 0.1)


        # init max_acc as the initial global model acc on local training set
        max_acc = self.eval_model_by_train(self.model)

        if self._is_malicious and self.args.attack_type == 1:
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

                util_train(self.model,
                            self._train_loader,
                            self.args.optimizer,
                            self.args.lr,
                            self.args.dev_device,
                            self.args.train_verbose)
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

        if not self._is_malicious and self.max_model_acc < self.args.worker_prune_acc_trigger:
            print(f"Worker {self.idx}'s local model max accuracy is < the prune acc trigger {self.args.worker_prune_acc_trigger}. Skip pruning.")
            return

        # model prune percentage
        init_pruned_ratio = get_pruned_ratio(self.model) # pruned_ratio = 0s/total_params = 1 - sparsity
        if not self._is_malicious and 1 - init_pruned_ratio <= self.args.target_sparsity:
            print(f"Worker {self.idx}'s model at sparsity {1 - init_pruned_ratio}, which is already <= the target sparsity {self.args.target_sparsity}. Skip pruning.")
            return
        
        print()
        L_or_M = "M" if self._is_malicious else "L"
        print(f"\n---------- {L_or_M} Worker:{self.idx} starts pruning ---------------------")

        init_model_acc = self.eval_model_by_train(self.model)
        accs = [init_model_acc]
        # models = deque([copy_model(self.model, self.args.dev_device)]) - used if want the model with the best accuracy arrived at an intermediate pruned amount

        to_prune_amount = init_pruned_ratio
        last_pruned_model = copy_model(self.model, self.args.dev_device)

        while True:
            to_prune_amount += random.uniform(0, self.args.max_prune_step)
            pruned_model = copy_model(self.model, self.args.dev_device)
            make_prune_permanent(pruned_model)
            l1_prune(model=pruned_model,
                        amount=to_prune_amount,
                        name='weight',
                        verbose=self.args.prune_verbose)
            
            model_acc = self.eval_model_by_train(pruned_model)

            # prune until the accuracy drop exceeds the threshold or below the target sparsity
            if init_model_acc - model_acc > self.args.prune_acc_drop_threshold or 1 - to_prune_amount <= self.args.target_sparsity:
                # revert to the last pruned model
                self.model = copy_model(last_pruned_model, self.args.dev_device)
                self.max_model_acc = accs[-1]
                break
            
            accs.append(model_acc)
            last_pruned_model = copy_model(pruned_model, self.args.dev_device)

        after_pruned_ratio = get_pruned_ratio(self.model) # pruned_ratio = 0s/total_params = 1 - sparsity
        after_pruning_acc = self.eval_model_by_train(self.model)

        self._worker_pruned_ratio = after_pruned_ratio

        print(f"Model sparsity: {1 - after_pruned_ratio:.2f}")
        print(f"Pruned model before and after accuracy: {init_model_acc:.2f}, {after_pruning_acc:.2f}")
        print(f"Pruned amount: {after_pruned_ratio - init_pruned_ratio:.2f}")

        logger['after_prune_sparsity'][comm_round][self.idx] = 1 - after_pruned_ratio
        logger['after_prune_acc'][comm_round][self.idx] = after_pruning_acc
        logger['after_prune_local_test_acc'][comm_round][self.idx] = self.eval_model_by_local_test(self.model)
        logger['after_prune_global_test_acc'][comm_round][self.idx] = self.eval_model_by_global_test(self.model)

    def poison_model(self, model):
        layer_to_mask = calc_mask_from_model_with_mask_object(model) # introduce noise to unpruned weights
        for layer, module in model.named_children():
            for name, weight_params in module.named_parameters():
                if "weight" in name:
                    # noise = self.args.noise_variance * torch.randn(weight_params.size()).to(self.args.dev_device) * torch.from_numpy(layer_to_mask[layer]).to(self.args.dev_device)
                    noise = self.args.noise_variance * torch.randn(weight_params.size()).to(self.args.dev_device) * layer_to_mask[layer].to(self.args.dev_device)
                    weight_params = weight_params + noise.to(self.args.dev_device)
        print(f"Device {self.idx} poisoned the whole neural network with variance {self.args.noise_variance}.") # or should say, unpruned weights?

    def generate_model_sig(self):
        
        make_prune_permanent(self.model)
        # zero knowledge proof of model ownership
        self.layer_to_model_sig_row, self.layer_to_model_sig_col = sum_over_model_params(self.model)
        
    def make_worker_tx(self):
                
        worker_tx = {
            'worker_idx' : self.idx,
            'rsa_pub_key': self.return_rsa_pub_key(),
            'model' : self.model,
            # 'model_path' : self.last_local_model_path, # in reality could be IPFS
            'model_sig_row': self.layer_to_model_sig_row,
            'model_sig_col': self.layer_to_model_sig_col
        }
        worker_tx['model_sig_row_sig'] = self.sign_msg(str(worker_tx['model_sig_row']))
        worker_tx['model_sig_col_sig'] = self.sign_msg(str(worker_tx['model_sig_col']))
        worker_tx['tx_sig'] = self.sign_msg(str(worker_tx))
        self._worker_tx = worker_tx
    
    def broadcast_worker_tx(self):
        # worker broadcast tx, imagine the tx is broadcasted to all peers volunterring to become validators
        # see receive_and_verify_worker_tx_sig()
        return
   
    ''' Validators' Methods '''

    def receive_and_verify_worker_tx_sig(self, online_workers):
        for worker in online_workers:
            if worker.idx not in self.peers:
                continue
            self.update_peers(worker.peers)
            if self.verify_tx_sig(worker._worker_tx):
                if self.args.validation_verbose:
                    print(f"Validator {self.idx} has received and verified the signature of the tx from worker {worker.idx}.")
                self._verified_worker_txs[worker.idx] = worker._worker_tx
            else:
                print(f"Signature of tx from worker {worker['idx']} is invalid.")

    def receive_and_verify_validator_tx_sig(self, online_validators):
        for validator in online_validators:
            # if validator == self:
            #     continue
            if validator.idx not in self.peers:
                continue
            self.update_peers(validator.peers)
            if self.verify_tx_sig(validator._validator_tx):
                if self.args.validation_verbose:
                    print(f"Validator {self.idx} has received and verified the signature of the tx from validator {validator.idx}.")
                self._verified_validator_txs[validator.idx] = validator._validator_tx
            else:
                print(f"Signature of tx from worker {validator['idx']} is invalid.")

    def validate_models(self, idx_to_device):

        # validate model siganture
        for widx, wtx in self._verified_worker_txs.items():
            worker_model = wtx['model']
            worker_model_sig_row_sig = wtx['model_sig_row_sig']
            worker_model_sig_col_sig = wtx['model_sig_col_sig']
            worker_rsa = wtx['rsa_pub_key']

            worker_layer_to_model_sig_row, worker_layer_to_model_sig_col = sum_over_model_params(worker_model)
            if self.compare_dicts_of_tensors(wtx['model_sig_row'], worker_layer_to_model_sig_row)\
                  and self.compare_dicts_of_tensors(wtx['model_sig_col'], worker_layer_to_model_sig_col)\
                  and self.verify_msg(wtx['model_sig_row'], worker_model_sig_row_sig, worker_rsa['pub_key'], worker_rsa['modulus'])\
                  and self.verify_msg(wtx['model_sig_col'], worker_model_sig_col_sig, worker_rsa['pub_key'], worker_rsa['modulus']):
                
                if self.args.validation_verbose:
                    print(f"Worker {widx} has valid model signature.")
            
                self.worker_to_model_sig[widx] = {'model_sig_row': wtx['model_sig_row'], 'model_sig_row_sig': wtx['model_sig_row_sig'], 'model_sig_col': wtx['model_sig_col'], 'model_sig_col_sig': wtx['model_sig_col_sig'], 'worker_rsa': worker_rsa}
            
        for widx, wtx in self._verified_worker_txs.items():
            worker_model = wtx['model']
            # calculate accuracy by validator's local dataset
            self.worker_to_acc[widx] = self.eval_model_by_train(worker_model)

        # add itself's accuracy
        self.worker_to_acc[self.idx] = self.max_model_acc
        if self._is_malicious:
            # malicious validator always assigns itself the maximum accuracy (1.0)
            self.worker_to_acc[self.idx] = 1.0
        
        if self.args.show_all_validation_performance:
            print(f"\nShowing validator {self.idx}'s validation performance against malicious workers out of total {len(self.worker_to_acc)} workers:")
            i = 1
            for widx, acc_weight in sorted(self.worker_to_acc.items(), key=lambda x: x[1]):
                if acc_weight == 0:
                    continue
                if idx_to_device[widx]._is_malicious:
                    if i < len(self.worker_to_acc) // 2:
                        print(f"Malicious worker {widx} has validated accuracy {acc_weight:.3f}, ranked {i}, in the lower half (lower rank means smaller weight).")
                    else:
                        print("\033[91m" + f"Malicious worker {widx} has validated accuracy {acc_weight:.2f}, ranked {i}, in the higher half (higher rank means heavier weight)." + "\033[0m")
                i += 1

        # sometimes may inverse the accuracy weights to account for minority workers
        if self.args.inverse_acc_weights and random.random() <= 0.5:
            self.worker_to_acc = {worker_idx: 1 - acc for worker_idx, acc in self.worker_to_acc.items()}
            print(f"Validator {self.idx} has inversed its accuracy weights.")

    def make_validator_tx(self):
         
        validator_tx = {
            'validator_idx' : self.idx,
            'rsa_pub_key': self.return_rsa_pub_key(),
            'worker_to_acc' : self.worker_to_acc,
        }
        validator_tx['tx_sig'] = self.sign_msg(str(validator_tx))
        self._validator_tx = validator_tx

    def broadcast_validator_tx(self, online_validators):
        return

    def calc_ungranted_reward(self, worker_acc, worker_pruned_ratio):
        # using harmonic mean with linear shift emphasize from accuracy to pruned_ratio
        latest_block_global_model_pruned_ratio = get_pruned_ratio(self.blockchain.get_last_block().global_model) if self.blockchain.get_chain_length() > 0 else 0
        curr_chain_length = self.blockchain.get_chain_length()

        denominator = (1 - self.args.target_sparsity) / (latest_block_global_model_pruned_ratio / (curr_chain_length + np.nextafter(0, 1))) # this part is used to estimate the number of remaining communication rounds
        acc_weight = 1 - ( (curr_chain_length + 1) / denominator ) # (curr_chain_length + 1) is technically the current communication round, but sometimes no device appended a block in the previous round, so this is more precise
        acc_weight = max(0, acc_weight)
        '''As the pruning reaches the target sparsity, such as 0.1, and say the current chain length is 20
            then, 1 - self.args.target_sparsity = 0.9, and 0.9 / 20 = 0.045, 0.9 / 0.045 = 20, and the acc_weight = 1 - (21 / 20) = -0.05, max(0, -0.05) = 0, so we gradually emphasize on pruning'''

        pruned_ratio_weight = 1 - acc_weight

        # worker_pruned_ratio needs to be controlled by the difference of the pruned ratio between the worker's model and the global model, so that the worker has no rewards to gain after reaching the target sparsity
        controlled_worker_pruned_ratio = max(np.nextafter(0, 1), worker_pruned_ratio - latest_block_global_model_pruned_ratio)

        reward = 1 / (acc_weight / (worker_acc + np.nextafter(0, 1)) + pruned_ratio_weight / controlled_worker_pruned_ratio)
        return reward
        

    def produce_global_model_and_reward(self, idx_to_device, comm_round):

        # NOTE - if change the aggregation rule, also need to change in verify_winning_block()

        # aggregate votes and accuracies - normalize by validator_power, defined by the historical stake of the validator + 1 (to avoid float point number and 0 division)
        # for the reward of the validator itself, it is directly adding its own max model accuracy, as an incentive to become a validator
        worker_to_acc_weight = defaultdict(float)
        for validator_idx, validator_tx in self._verified_validator_txs.items():
            validator_power = self._pos_book[validator_idx] + 1
            for worker_idx, worker_acc in validator_tx['worker_to_acc'].items():
                worker_to_acc_weight[worker_idx] += worker_acc * validator_power
                worker_pruned_ratio = get_pruned_ratio(self._verified_worker_txs[worker_idx]['model'])
                self._device_to_ungranted_reward[worker_idx] += self.calc_ungranted_reward(worker_acc, worker_pruned_ratio)
        
        # get models for aggregation
        worker_to_model = {worker_idx: self._verified_worker_txs[worker_idx]['model'] for worker_idx in worker_to_acc_weight}

        # normalize weights to between 0 and 1
        worker_to_acc_weight = {worker_idx: acc/sum(worker_to_acc_weight.values()) for worker_idx, acc in worker_to_acc_weight.items()}
        self.worker_to_acc_weight = worker_to_acc_weight
        
        # produce final global model
        self._final_global_model = weighted_fedavg(worker_to_acc_weight, worker_to_model, device=self.args.dev_device)

    def validator_post_prune(self): # prune by the weighted average of the pruned amount of the selected models

        init_pruned_ratio = get_pruned_ratio(self._final_global_model) # pruned_ratio = 0s/total_params = 1 - sparsity
        
        if 1 - init_pruned_ratio <= self.args.target_sparsity:
            print(f"\nValidator {self.idx}'s model at sparsity {1 - init_pruned_ratio}, which is already <= the target sparsity. Skip post-pruning.")
            return
        
        print()
        L_or_M = "M" if self._is_malicious else "L"
        print(f"\n---------- {L_or_M} Validator:{self.idx} post pruning ---------------------")

        worker_to_pruned_ratio = {}
        worker_to_power = {}
            
        for worker_idx in self._device_to_ungranted_reward:
            if worker_idx == self.idx:
                worker_to_pruned_ratio[self.idx] = self._worker_pruned_ratio
                worker_to_power[self.idx] = self._pos_book[self.idx] + 1
            else:
                worker_model = self._verified_worker_txs[worker_idx]['model']
                worker_to_pruned_ratio[worker_idx] = get_pruned_ratio(worker_model)
                worker_to_power[worker_idx] = self._pos_book[worker_idx] + 1
        
        worker_to_prune_weight = {worker_idx: power/sum(worker_to_power.values()) for worker_idx, power in worker_to_power.items()}

        need_pruned_ratio = sum([worker_to_pruned_ratio[worker_idx] * weight for worker_idx, weight in worker_to_prune_weight.items()])
        if self._is_malicious:
            need_pruned_ratio *= 2

        if need_pruned_ratio <= init_pruned_ratio:
            print(f"The need_pruned_ratio value ({need_pruned_ratio}) <= init_pruned_ratio ({init_pruned_ratio}). Validator {self.idx} skips post-pruning.")
            return

        need_pruned_ratio = min(need_pruned_ratio, 1 - self.args.target_sparsity)
        to_prune_amount = need_pruned_ratio
        if check_mask_object_from_model(self._final_global_model):
            to_prune_amount = (need_pruned_ratio - init_pruned_ratio) / (1 - init_pruned_ratio)

        # post_prune the model
        l1_prune(model=self._final_global_model,
                        amount=to_prune_amount,
                        name='weight',
                        verbose=self.args.prune_verbose)

        print(f"{L_or_M} Validator {self.idx} has pruned {need_pruned_ratio - init_pruned_ratio:.2f} of the model. Final sparsity: {1 - need_pruned_ratio:.2f}.")


    def produce_block(self):
        
        def sign_block(block_to_sign):
            block_to_sign.block_signature = self.sign_msg(str(block_to_sign.__dict__))

        # self-assign validator's reward to itself
        last_block = self.blockchain.get_last_block() # before appending the winning block

        # assign reward to itself by the difference of pruned ratio between last block and this block
        last_block_global_model_pruned_ratio = get_pruned_ratio(last_block.global_model) if last_block else 0
        new_global_model_pruned_ratio = get_pruned_ratio(self._final_global_model)
        self._device_to_ungranted_reward[self.idx] = self._device_to_ungranted_reward.get(self.idx, 0) + max(0, new_global_model_pruned_ratio - last_block_global_model_pruned_ratio)

        last_block_hash = self.blockchain.get_last_block_hash()

        block = Block(last_block_hash, self._final_global_model, self._device_to_ungranted_reward, self.idx, self.worker_to_model_sig, self._verified_validator_txs, self.return_rsa_pub_key())
        sign_block(block)

        self.produced_block = block
        
    def broadcast_block(self):
        # see receive_blocks()
        return
        
    ''' Blockchain Operations '''
        
    def generate_rsa_key(self):
        keyPair = RSA.generate(bits=1024)
        self._modulus = keyPair.n
        self._private_key = keyPair.d
        self.public_key = keyPair.e
        
    def assign_peers(self, idx_to_device):
        peer_size = math.ceil(len(idx_to_device) * self.args.peer_percent)
        self.peers = set(random.sample(list(idx_to_device.keys()), peer_size))
        self.peers.add(self.idx) # include itself
        self._pos_book = {key: 0 for key in self.peers}
        
    def update_peers(self, peers_of_other_device):
        new_peers = peers_of_other_device.difference(self.peers)
        self._pos_book.update({key: 0 for key in new_peers})
        self.peers.update(new_peers)

    def set_online(self):
        if self.args.malicious_always_online and self._is_malicious:
            self.online = True
            return True
        
        self.online = random.random() <= self.args.network_stability
        if not self.online:
            print(f"Device {self.idx} is offline in this communication round.")
        return self.online
    
    def is_online(self):
        return self.online
    
    def recalc_stake(self):
        self._pos_book = {idx: 0 for idx in self._pos_book}
        for block in self.blockchain.chain:
            for idx in block.device_to_reward:
                self._pos_book[idx] += block.device_to_reward[idx]

    def check_resync_eligibility_when_picking(self, idx_to_device, winning_block):
        to_resync_chain = idx_to_device[self._resync_to].blockchain
        if to_resync_chain.get_chain_length() == 0:
            print(f"resync_to_device {self._resync_to}'s chain length is 0. Chain not resynced. Resync next round.") # may resync to the same device, but the device may have appended other blocks to make its chain valid at the beginning of the next round
            return False
        if len(to_resync_chain.chain) >= 2 and to_resync_chain.chain[-2].produced_by == to_resync_chain.chain[-1].produced_by == winning_block.produced_by:
            print(f"resync_to_device {self._resync_to}'s chain's last two blocks' producer is identical to the winning_block's ({winning_block.produced_by}). Chain not resynced. Resync next round.")
            return False
        return True
    
    def resync_chain(self, comm_round, idx_to_device, skip_check_peers=False):
        # NOTE - if change logic of resync_chain(), also need to change logic in pick_winning_block()
        """ 
            Return:
            - True if the chain needs to be resynced, False otherwise
        """
        if comm_round == 1:
            # validator not applicable to resync chain
            return False

        if skip_check_peers:
            resync_to_device = idx_to_device[self._resync_to]
            # came from verify_winning_block() when hash is inconsistant rather than in the beginning, direct resync
            if self.validate_chain(resync_to_device.blockchain):
                # update chain
                self.blockchain.replace_chain(resync_to_device.blockchain.chain)
                print(f"\n{self.role} {self.idx}'s chain is resynced from the picked winning validator {self._resync_to}.")                    
                return True
        
        longer_chain_peers = set()
        # check peer's longest chain length
        online_peers = [peer for peer in self.peers if idx_to_device[peer].is_online()]
        for peer in online_peers:
            if idx_to_device[peer].blockchain.get_chain_length() > self.blockchain.get_chain_length():
                # if any online peer's chain is longer, may need to resync. "may" because the longer chain may not be legitimate since this is not PoW, but pos similar to PoS
                longer_chain_peers.add(peer)        
        if not longer_chain_peers:
            return False
        
        # may need to resync - devices may be left behind due to offline
        if self._resync_to:
            if self._resync_to in longer_chain_peers:
                # _resync_to specified to the last time's picked winning validator
                resync_to_device = idx_to_device[self._resync_to]
                # if self.blockchain.get_last_block_hash() == resync_to_device.blockchain.get_last_block_hash():   # redundant, if longer, chain must be different
                #     # if two chains are the same, no need to resync. assume the last block's hash is valid
                #     return False
                if self.validate_chain(resync_to_device.blockchain):
                    # update chain
                    self.blockchain.replace_chain(resync_to_device.blockchain.chain)
                    print(f"\n{self.role} {self.idx}'s chain is resynced from last time's picked winning validator {self._resync_to}.")                    
                    return True
                else:
                    print(f"\nDevice {self.idx}'s _resync_to device's ({self._resync_to}) chain is invalid, resync to another online peer with the top stake obtained.")
            else: # resync_to device may also be left behind
                print(f"\nDevice {self.idx}'s _resync_to device ({self._resync_to})'s chain is not longer than its own chain. May need to resync to another online peer with the top stake obtained.") # in the case both devices were offline
        else:
            print(f"\nDevice {self.idx}'s does not have a _resync_to device, resync to another online peer with the top stake obtained.")


        # resync chain from online peers using the same logic in pick_winning_block()
        online_peers = [peer for peer in self.peers if idx_to_device[peer].is_online() and idx_to_device[peer].blockchain.get_last_block()]
        online_peer_to_stake = {peer: self._pos_book[peer] for peer in online_peers}
        top_stake = max(online_peer_to_stake.values())
        candidates = [peer for peer, stake in online_peer_to_stake.items() if stake == top_stake]
        self._resync_to = random.choice(candidates)
        resync_to_device = idx_to_device[self._resync_to]

        # compare chain difference, assume the last block's hash is valid
        if self.blockchain.get_last_block_hash() == resync_to_device.blockchain.get_last_block_hash():
            return False
        else:
            # validate chain
            if not self.validate_chain(resync_to_device.blockchain):
                print(f"resync_to device {resync_to_device.idx} chain validation failed. Chain not resynced.")
                return False
            # update chain
            self.blockchain.replace_chain(resync_to_device.blockchain.chain)
            print(f"\n{self.role} {self.idx}'s chain is resynced from {resync_to_device.idx}, who picked {resync_to_device.blockchain.get_last_block().produced_by}'s block.")
            return True

            
    def post_resync(self, idx_to_device):
        # update global model from the new block
        self.model = deepcopy(self.blockchain.get_last_block().global_model)
        # update peers
        self.update_peers(idx_to_device[self._resync_to].peers)
        # recalculate stake
        self.recalc_stake()
    
    def validate_chain(self, chain_to_check):
        # TODO - should also verify the block signatures and the model signatures
        blockchain_to_check = chain_to_check.get_chain()
        for i in range(1, len(blockchain_to_check)):
            if blockchain_to_check[i].previous_block_hash != blockchain_to_check[i-1].compute_hash():
                return False
            if i >= 2 and blockchain_to_check[i].produced_by == blockchain_to_check[i-1].produced_by == blockchain_to_check[i-2].produced_by:
                return False
        return True
        
    def verify_tx_sig(self, tx):
        tx_before_signed = copy(tx)
        del tx_before_signed["tx_sig"]
        modulus = tx['rsa_pub_key']["modulus"]
        pub_key = tx['rsa_pub_key']["pub_key"]
        signature = tx["tx_sig"]
        # verify
        hash = int.from_bytes(sha256(str(tx_before_signed).encode('utf-8')).digest(), byteorder='big')
        hashFromSignature = pow(signature, pub_key, modulus)
        return hash == hashFromSignature
    
    def return_rsa_pub_key(self):
        return {"modulus": self._modulus, "pub_key": self.public_key}
    
    def sign_msg(self, msg):
        # TODO - sorted migjt be a bug when signing. need to change in VBFL as well
        hash = int.from_bytes(sha256(str(msg).encode('utf-8')).digest(), byteorder='big')
        # pow() is python built-in modular exponentiation function
        signature = pow(hash, self._private_key, self._modulus)
        return signature

    def verify_msg(self, msg, signature, public_key, modulus):
        hash = int.from_bytes(sha256(str(msg).encode('utf-8')).digest(), byteorder='big')
        hashFromSignature = pow(signature, public_key, modulus)
        return hash == hashFromSignature
    
    def verify_block_sig(self, block):
        # assume block signature is not disturbed
        # return True
        block_to_verify = copy(block)
        block_to_verify.block_signature = None
        modulus = block.validator_rsa_pub_key["modulus"]
        pub_key = block.validator_rsa_pub_key["pub_key"]
        signature = block.block_signature
        # verify
        hash = int.from_bytes(sha256(str(block_to_verify.__dict__).encode('utf-8')).digest(), byteorder='big')
        hashFromSignature = pow(signature, pub_key, modulus)
        return hash == hashFromSignature
    
    def receive_blocks(self, online_validators):
        for validator in online_validators:
            if validator.idx in self.peers:
                self.update_peers(validator.peers)
                self._received_blocks[validator.idx] = validator.produced_block

    def pick_winning_block(self, idx_to_device):

        # NOTE - if change logic of pick_winning_block(), also need to change logic in resync_chain(), validate_chain() and check_resync_eligibility_when_picking()

        picked_block = None

        if not self._received_blocks:
            print(f"\n{self.idx} has not received any block. Resync chain next round.")
            return picked_block
        
        received_validators_to_blocks = {block.produced_by: block for block in self._received_blocks.values()}
        # the same validator cannot win two times consecutively - mitigate monopoly
        if self.blockchain.get_chain_length() >= 1:
            received_validators_to_blocks.pop(self.blockchain.chain[-1].produced_by, None)
        received_validators_pos_book = {block.produced_by: self._pos_book[block.produced_by] for block in received_validators_to_blocks.values()}
        if not received_validators_pos_book:
            print(f"\n{self.idx} has no valid received block. Resync chain next round.")
            return picked_block
        top_stake = max(received_validators_pos_book.values())
        candidates = [validator for validator, stake in received_validators_pos_book.items() if stake == top_stake]
        # get the winning validator
        winning_validator = random.choice(candidates) # may cause forking in the 1st round and some middle rounds
        if self.idx in candidates:
            # oppourtunistic validator
            winning_validator = self.idx
        # winning_validator = max(validator_to_stake_criterion, key=validator_to_stake_criterion.get)        

        print(f"\n{self.role} {self.idx} ({self._user_labels}) picks {winning_validator}'s ({idx_to_device[winning_validator]._user_labels}) block.")

        return received_validators_to_blocks[winning_validator]

    def check_block_when_resyncing(self, block, last_block):
        # 1. check block signature
        if not self.verify_block_sig(block):
            return False
        # 2. check last block hash match
        if block.previous_block_hash != last_block.compute_hash():
            return False
        # block checked
        return True
    
    def check_last_block_hash_match(self, block):
        if not self.blockchain.get_last_block_hash():
            return True
        else:
            last_block_hash = self.blockchain.get_last_block_hash()
            if block.previous_block_hash == last_block_hash:
                return True
        return False

    def verify_winning_block(self, winning_block, comm_round, idx_to_device):

        # NOTE - change the aggregation rule if there's change in produce_global_model_and_reward()
        
        # verify block signature
        if not self.verify_block_sig(winning_block):
            print(f"{self.role} {self.idx}'s picked winning block has invalid signature. Block discarded.")
            return False
        
        # verify validator transactions signature
        for val_tx in winning_block.validator_txs.values():
            if not self.verify_tx_sig(val_tx):
                # TODO - may compare validator's signature with itself's received validator_txs, but a validator may send different transactions to sabortage this process
                print(f"{self.role} {self.idx}'s picked winning block has invalid validator transaction signature. Block discarded.")
                return False

        # check last block hash match
        if not self.check_last_block_hash_match(winning_block):
            print(f"{self.role} {self.idx}'s last block hash conflicts with {winning_block.produced_by}'s block. Checking chain resyncing eligibility...")
            self._resync_to = winning_block.produced_by
            if not self.check_resync_eligibility_when_picking(idx_to_device, winning_block):
                return False
            self.resync_chain(comm_round, idx_to_device, skip_check_peers = True)
            self.post_resync(idx_to_device)
        

        ''' validate model signature to make sure the validator is performing model aggregation honestly '''
        layer_to_model_sig_row, layer_to_model_sig_col = sum_over_model_params(winning_block.global_model)
        
        # perform model signature aggregation by the same rule in produce_global_model_and_reward()
        worker_to_acc_weight = defaultdict(float)
        device_to_should_reward = defaultdict(float)
        for validator_idx, validator_tx in winning_block.validator_txs.items():
            validator_power = self._pos_book[validator_idx] + 1
            for worker_idx, worker_acc in validator_tx['worker_to_acc'].items():
                worker_to_acc_weight[worker_idx] += worker_acc * validator_power
                worker_pruned_ratio = get_pruned_ratio(self._verified_worker_txs[worker_idx]['model'])
                device_to_should_reward[worker_idx] += self.calc_ungranted_reward(worker_acc, worker_pruned_ratio)
                # device_to_should_reward[worker_idx] += worker_acc
        
        
        # (1) verify if validator honestly assigned reward to devices
        # validator_self_assigned_stake = winning_block.device_to_reward[winning_block.produced_by] - self._pos_book[winning_block.produced_by]
        last_block = self.blockchain.get_last_block() # before appending the winning block

        last_block_global_model_pruned_ratio = get_pruned_ratio(last_block.global_model) if last_block else 0
        new_global_model_pruned_ratio = get_pruned_ratio(winning_block.global_model)
        validator_should_self_assigned_reward = max(0, new_global_model_pruned_ratio - last_block_global_model_pruned_ratio)
        device_to_should_reward[winning_block.produced_by] += validator_should_self_assigned_reward

        if device_to_should_reward != winning_block.device_to_reward:
            print(f"{self.role} {self.idx}'s picked winning block has invalid reward assignment. Block discarded.")
            return False

        # (2) verify if validator honestly aggregated the models
        # normalize weights to between 0 and 1
        worker_to_acc_weight = {worker_idx: acc/sum(worker_to_acc_weight.values()) for worker_idx, acc in worker_to_acc_weight.items()}
        
        # apply weights to worker's model signatures
        workers_layer_to_model_sig_row = {}
        workers_layer_to_model_sig_col = {}
        for worker_idx, acc_weight in worker_to_acc_weight.items():
            model_sig_row = winning_block.worker_to_model_sig[worker_idx]['model_sig_row']
            model_sig_col = winning_block.worker_to_model_sig[worker_idx]['model_sig_col']
            for layer in model_sig_row:
                if layer not in workers_layer_to_model_sig_row:
                    workers_layer_to_model_sig_row[layer] = model_sig_row[layer] * acc_weight
                    workers_layer_to_model_sig_col[layer] = model_sig_col[layer] * acc_weight
                else:
                    workers_layer_to_model_sig_row[layer] += model_sig_row[layer] * acc_weight
                    workers_layer_to_model_sig_col[layer] += model_sig_col[layer] * acc_weight
        
        if not self.compare_dicts_of_tensors(layer_to_model_sig_row, workers_layer_to_model_sig_row) or not self.compare_dicts_of_tensors(layer_to_model_sig_col, workers_layer_to_model_sig_col):
            print(f"{self.role} {self.idx}'s picked winning block has invalid workers' model signatures or PoS book is inconsistent with the block producer's.") # this could happen if the owner of this winning block A had resynced to another device B's chain, so when this device C actually resynced to B's chain, got B's pos book, but still gets A's block, which is not valid anymore. Resync in next round.
            return False

        self.verified_winning_block = winning_block
        return True
        
    def process_and_append_block(self, comm_round):

        if not self.verified_winning_block:
            print(f"\nNo verified winning block to append. Device {self.idx} may resync to last time's picked winning validator({self._resync_to})'s chain.")
            return

        self._resync_to = self.verified_winning_block.produced_by # in case of offline, resync to this validator's chain
        
        self.old_pos_book = deepcopy(self._pos_book) # helper used in check_validation_performance()

        # grant reward to devices
        for device_idx, reward in self.verified_winning_block.device_to_reward.items():
            self._pos_book[device_idx] += reward

        # used to record if a block is produced by a malicious device
        self.has_appended_block = True
        # update global model
        self.model = deepcopy(self.verified_winning_block.global_model)

        self.blockchain.chain.append(deepcopy(self.verified_winning_block))

        print(f"\n{self.role} {self.idx} has appended the winning block produced by {self.verified_winning_block.produced_by}.")

    ''' Helper Functions '''

    def compare_dicts_of_tensors(self, dict1, dict2, atol=1e-3, rtol=1e-3):
            """
            Compares two dictionaries with torch.Tensor values.

            Parameters:
            - dict1, dict2: The dictionaries to compare.
            - atol: Absolute tolerance.
            - rtol: Relative tolerance.

            Returns:
            - True if the dictionaries are equivalent, False otherwise.
            """
            if dict1.keys() != dict2.keys():
                return False
            
            for key in dict1:
                if not torch.allclose(dict1[key], dict2[key], atol=atol, rtol=rtol):
                    return False
            
            return True
    
    def check_validation_performance(self, idx_to_device, comm_round):

        if not self.verified_winning_block:
            print(f"\nNo verified winning block to check validation performance. Device {self.idx} may resync to last time's picked winning validator({self._resync_to})'s chain.")
            return

        worker_to_acc_weight = idx_to_device[self.verified_winning_block.produced_by].worker_to_acc_weight

        ''' BELOW MUST BE CONSISTENT WITH THE LOGIC IN produce_global_model_and_reward() '''
        i = 1
        for widx, acc_weight in sorted(worker_to_acc_weight.items(), key=lambda x: x[1]):
            if acc_weight == 0: continue
            if idx_to_device[widx]._is_malicious:
                if i < len(worker_to_acc_weight) // 2:
                    print(f"Malicious worker {widx} has accuracy-model-weight {acc_weight:.3f}, ranked {i}, in the lower half (lower rank means smaller weight).")
                else:
                    print("\033[91m" + f"Malicious worker {widx} has accuracy-model-weight {acc_weight:.2f}, ranked {i}, in the higher half (higher rank means heavier weight)." + "\033[0m")
            i += 1


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
