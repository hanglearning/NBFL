# TODO - prevent free or lazy rider - 

''' logger
1. log latest model accuracy in test_indi_accuracy()
2. log validation mechanism performance in check_validation_performance()
3. log forking event at the end of main.py
4. log pos book at the end of main.py
5. log when a malicious block has been added by any device in the network 
'''
import os
import torch
import argparse
import pickle
import random
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime
from pytorch_lightning import seed_everything
from collections import defaultdict
from copy import deepcopy

from device import Device
from util import *
from dataset.datasource import DataLoaders

from model.cifar10.cnn import CNN as CIFAR_CNN
from model.cifar10.mlp import MLP as CIFAR_MLP
from model.mnist.cnn import CNN as MNIST_CNN
from model.mnist.mlp import MLP as MNIST_MLP


''' abbreviations:
    tx: transaction
    pos: proof of stake
'''

models = {
    'cifar10': {
        'cnn': CIFAR_CNN,
        'mlp': CIFAR_MLP
    },
    'mnist': {
        'cnn': MNIST_CNN,
        'mlp': MNIST_MLP
    }
}

parser = argparse.ArgumentParser(description='LBFL')

####################### system setting #######################
parser.add_argument('--train_verbose', type=bool, default=False)
parser.add_argument('--test_verbose', type=bool, default=False)
parser.add_argument('--prune_verbose', type=bool, default=False)
parser.add_argument('--resync_verbose', type=bool, default=True)
parser.add_argument('--validation_verbose', type=int, default=0, help='show validation process detail')
parser.add_argument('--seed', type=int, default=40)
parser.add_argument('--log_dir', type=str, default="./logs")
parser.add_argument('--peer_percent', type=float, default=1, help='this indicates the percentage of peers to assign. See assign_peers() in device.py. As the communication goes on, a device should be able to know all other devices in the network.')

####################### federated learning setting #######################
parser.add_argument('--arch', type=str, default='cnn', help='cnn|mlp')
parser.add_argument('--dataset', help="mnist|cifar10",type=str, default="mnist")
parser.add_argument('--dataset_mode', type=str,default='non-iid', help='non-iid|iid')
parser.add_argument('--alpha', type=float, default=1.0, help='unbalance between labels')
parser.add_argument('--dataloader_workers', type=int, default=0, help='num of pytorch dataloader workers')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--rounds', type=int, default=25)
parser.add_argument('--epochs', type=int, default=50, help="local max training epochs to get the max accuracy")
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--optimizer', type=str, default="Adam", help="SGD|Adam")
parser.add_argument('--total_samples', type=int, default=40)
parser.add_argument('--n_malicious', type=int, default=8, help="number of malicious nodes in the network")

####################### validation and rewards setting #######################
parser.add_argument('--pass_all_models', type=int, default=0, help='turn off validation and pass all models, used for debug or create baselines')
parser.add_argument('--validate_center_threshold', type=float, default=0.1, help='only recognize malicious devices if the difference of two centers of KMeans exceed this threshold')
parser.add_argument('--inverse_acc_weights', type=int, default=0, help='sometimes may inverse the accuracy weights to give more weights to minority workers. ideally, malicious workers should have been filtered out and not be considered here')

####################### attack setting #######################
parser.add_argument('--attack_type', type=int, default=0, help='0 - no attack, 1 - model poisoning attack, 2 - label flipping attack, 3 - lazy attack, 4 - model poisoning and lazy attack')

####################### pruning setting #######################
parser.add_argument('--rewind', type=int, default=1, help="reinit ticket model parameters before training")
parser.add_argument('--target_sparsity', type=float, default=0.1, help='target sparsity for pruning, stop pruning if below this threshold')
parser.add_argument('--max_prune_step', type=float, default=0.05, help='max increment of pruning step')
parser.add_argument('--acc_drop_threshold', type=float, default=0.05, help='if the accuracy drop is larger than this threshold, stop prunning')
parser.add_argument('--prune_acc_trigger', type=float, default=0.8, help='must achieve this accuracy to trigger worker to post prune its local model')
# parser.add_argument('--validator_prune_acc_trigger', type=float, default=0.8, help='must achieve this accuracy to trigger validator to post prune the global model')


####################### blockchain setting #######################
parser.add_argument('--n_devices', type=int, default=10)
parser.add_argument('--n_validators', type=str, default='*', 
                    help='if input * to this argument, the number of validators is random from round to round')
parser.add_argument('--check_signature', type=int, default=0, 
                    help='if set to 0, all signatures are assumed to be verified to save execution time')
parser.add_argument('--network_stability', type=float, default=1.0, 
                    help='the odds a device can be reached')
parser.add_argument('--malicious_always_online', type=int, default=1, 
                    help='1 - malicious devices are always online; 0 - malicious devices can be online or offline depending on network_stability')
parser.add_argument('--top_percent_winning', type=int, default=0.3, 
                    help='when picking the winning block, considering the validators having the stake within this top percent. see pick_winning_block()')

####################### debug setting #######################
parser.add_argument('--show_all_validation_performance', type=int, default=0, help='0 - do not show, 1 - show the validation performance of EVERY validator against the malicious devices in produce_global_model_and_reward()')
parser.add_argument('--show_validation_performance_in_block', type=int, default=1, help='0 - do not show, 1 - show the validation performance against the malicious devices in its block')

####################### other settings #######################
parser.add_argument('--save_init_global_model', type=int, default=0, help='0 - do not save, 1 - save')


args = parser.parse_args()

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

def main(): 

    set_seed(args.seed)

    if not args.attack_type:
        args.n_malicious = 0
    
    args.dev_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device {args.dev_device}")

    exe_date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
    log_root_name = f"LBFL_seed_{args.seed}_{args.dataset_mode}_alpha_{args.alpha}_{exe_date_time}_ndevices_{args.n_devices}_nsamples_{args.total_samples}_rounds_{args.rounds}_mal_{args.n_malicious}_attack_{args.attack_type}"

    ######## initiate global model ########
    init_global_model = create_init_model(cls=models[args.dataset]
                         [args.arch], device=args.dev_device)
    if args.save_init_global_model:
        # save init_global_model to be used by baseline methods and LotteryFL
        torch.save(init_global_model.state_dict(), f"{args.log_dir}/init_global_model_seed_{args.seed}.pth")
    # pruned by 0.00 %. This is necessary to create buffer masks and to be consistent with create_model() in util.py
    l1_prune(init_global_model, amount=0.00, name='weight', verbose=False)

    try:
        # on Google Colab with Google Drive mounted
        import google.colab
        args.log_dir = f"/content/drive/MyDrive/LBFL/{log_root_name}"
    except:
        # local
        args.log_dir = f"{args.log_dir}/{log_root_name}"
    os.makedirs(args.log_dir)
    
    ######## initiate devices ########
    
    train_loaders, test_loaders, user_labels, global_test_loader = DataLoaders(n_devices=args.n_devices,
    dataset_name=args.dataset,
    total_samples=args.total_samples,
    log_dirpath=args.log_dir,
    seed=args.seed,
    mode=args.dataset_mode,
    batch_size=args.batch_size,
    alpha=args.alpha,
    dataloader_workers=args.dataloader_workers)
    
    idx_to_device = {}
    n_malicious = args.n_malicious

    if args.n_malicious == 3:
        noise_variances = [0.05]
    elif args.n_malicious == 6:
        noise_variances = [0.05, 0.5, 1.0]
    elif args.n_malicious == 10:
        noise_variances = [0.05, 0.05, 0.5, 0.5, 1.0]

    noise_attacker_idx = 0
    for i in range(1, args.n_devices + 1):
        is_malicious = True if args.n_devices - i < n_malicious else False
        attack_type = 0
        noise_variance = 0
        if is_malicious and args.attack_type == 4:
            if i % 2 == 1:
                attack_type = 1 # odd number assign model poisoning attack - why not random? Because we need to keep it consistent across runs so comparison can reflect the effect of the attack
                noise_variance = noise_variances[noise_attacker_idx]
                noise_attacker_idx += 1
            elif i % 2 == 0:
                attack_type = 3
        elif is_malicious:
            attack_type = args.attack_type
        device = Device(i, is_malicious, attack_type, args, train_loaders[i - 1], test_loaders[i - 1], user_labels[i - 1], global_test_loader, init_global_model, noise_variance)
        if is_malicious:
            if attack_type == 1:
                print(f"Assigned device {i} malicious as noise attacker with noise variance: {noise_variance}.")
            elif attack_type == 3:
                print(f"Assigned device {i} malicious as lazy attacker.")
            # label flipping attack
            if args.attack_type == 2:
                device._train_loader.dataset.targets = 9 - device._train_loader.dataset.targets
        idx_to_device[i] = device
    
    devices_list = list(idx_to_device.values())
    for device in devices_list:
        device.assign_peers(idx_to_device)

    logger = {} # used to log accuracy, stake, forking events, etc.

    logger['global_test_acc'] = {r: {} for r in range(1, args.rounds + 1)}
    logger['local_max_epoch'] = {r: {} for r in range(1, args.rounds + 1)}
    logger['global_model_sparsity'] = {r: {} for r in range(1, args.rounds + 1)}
    logger['local_max_acc'] = {r: {} for r in range(1, args.rounds + 1)}
    logger['local_test_acc'] = {r: {} for r in range(1, args.rounds + 1)}

    logger['after_prune_sparsity'] = {r: {} for r in range(1, args.rounds + 1)}
    logger['after_prune_acc'] = {r: {} for r in range(1, args.rounds + 1)}
    logger['after_prune_local_test_acc'] = {r: {} for r in range(1, args.rounds + 1)}
    logger['after_prune_global_test_acc'] = {r: {} for r in range(1, args.rounds + 1)}

    logger['n_online_devices'] = {r: 0 for r in range(1, args.rounds + 1)}
    logger['n_validators'] = {r: 0 for r in range(1, args.rounds + 1)}
    logger['forking_event'] = {r: 0 for r in range(1, args.rounds + 1)}
    logger['malicious_winning_count'] = {r: 0 for r in range(1, args.rounds + 1)}

    logger["pos_book"] = {r: {} for r in range(1, args.rounds + 1)}

    logger["validator_to_worker_acc_diff"] = {r: {} for r in range(1, args.rounds + 1)}
    logger["pruned_amount"] = {r: {} for r in range(1, args.rounds + 1)}

    # DEBUG for worker_15 lazy+noisy
    logger["worker_15_val_rank"] = {r: {} for r in range(1, args.rounds + 1)}
    
    # save args
    with open(f'{args.log_dir}/args.pickle', 'wb') as f:
        pickle.dump(args, f)

    ######## LBFL ########

    for comm_round in range(1, args.rounds + 1):
        
        print_text = f"Comm Round: {comm_round}"
        print()
        print("=" * len(print_text))
        print(print_text)
        print("=" * len(print_text))
        
        random.shuffle(devices_list)
        
        ''' find online devices '''
        init_online_devices = [device for device in devices_list if device.set_online()]
        if len(init_online_devices) < 2:
            print(f"Total {len(init_online_devices)} device online, skip this round.")
            continue
        
        logger['n_online_devices'][comm_round] = len(init_online_devices)

        ''' reset params '''
        for device in init_online_devices:
            device._received_blocks = {}
            device.has_appended_block = False
            device.verified_winning_block = None
            # workers
            device.layer_to_model_sig_row = {}
            device.layer_to_model_sig_col = {}
            device.max_model_acc = 0
            device._worker_pruned_ratio = 0
            # validators
            device._validator_tx = None
            device._verified_worker_txs = {}
            device._verified_validator_txs = {}
            device._final_global_model = None
            device.produced_block = None 
            device.worker_to_model_sig = {}           
            device.worker_to_acc_diff = {}
            device.worker_to_euc_dist = {}
            device._device_to_ungranted_reward = defaultdict(float)
            device.worker_to_model_weight = {}
            device.worker_to_direction_percent = {}
            device.worker_to_mask_overlap_percent = {}
            
        ''' Device Starts LBFL '''

        ''' Phase 1 - Worker Learning and Pruning '''
        # all online devices become workers in this phase
        online_workers = []
        for device in init_online_devices:
            device.role = 'worker'
            online_workers.append(device)

        ### worker starts learning and pruning ###

        for worker_iter in range(len(online_workers)):
            worker = online_workers[worker_iter]
            # resync chain - especially offline devices from last round
            if worker.resync_chain(comm_round, idx_to_device):
                worker.post_resync(idx_to_device)
            # perform training
            worker.model_learning_max(comm_round, logger)
            # perform pruning
            worker.worker_prune(comm_round, logger)
            # generate model signature
            worker.generate_model_sig()
            # make tx
            worker.make_worker_tx()
            # broadcast tx to the network
            worker.broadcast_worker_tx()

        print(f"\nWorkers {[worker.idx for worker in online_workers]} have broadcasted worker transactions to validators.")

        ''' Phase 2 - Validators Model Validation and Exchange Votes '''
        # workers volunteer to become validators
        if args.n_validators == '*':
            n_validators = random.randint(1, len(online_workers))
        else:
            n_validators = int(args.n_validators)
        
        print(f"\nRound {comm_round}, {n_validators} validators selected.")
        logger['n_validators'][comm_round] = n_validators

        online_validators = []
        random.shuffle(online_workers)
        
        for worker in online_workers:
            if worker.is_online():
                # receive worker tx and verify signature
                worker.receive_and_verify_worker_tx_sig(online_workers) # worker also receives other workers' tx due to verifying pruning reward
                if n_validators > 0:
                    worker.role = 'validator'
                    online_validators.append(worker)
                    n_validators -= 1
            
        for validator_iter in range(len(online_validators)):
            validator = online_validators[validator_iter]
            # validate model based on accuracy
            validator.validate_models(idx_to_device, comm_round)
            # make validator transaction
            validator.make_validator_tx()
            # broadcast tx to all the validators
            validator.broadcast_validator_tx(online_validators)
        
        print(f"\nValidators {[validator.idx for validator in online_validators]} have broadcasted validator transactions to other validators.")


        ''' Phase 3 - Validators Perform FedAvg and Produce Blocks '''
        for validator_iter in range(len(online_validators)):
            validator = online_validators[validator_iter]
            # verify validator tx signature
            validator.receive_and_verify_validator_tx_sig(online_validators)
            # validator produces global model
            validator.produce_global_model_and_reward(idx_to_device, comm_round, logger)
            # validator post prune the global model
            validator.validator_post_prune()
            # validator produce block
            validator.produce_block()
            # validator broadcasts block
            validator.broadcast_block()
        print(f"\nValidators {[validator.idx for validator in online_validators]} have broadcasted their blocks to the network.")

        ''' Phase 4 - All Online Devices Pick and Process Winning Block '''
        for device in online_workers:
            # receive blocks from validators
            device.receive_blocks(online_validators)
            # pick winning block based on pos
            winning_block = device.pick_winning_block(idx_to_device)
            if not winning_block:
                # no winning_block found, perform chain_resync next round
                continue
            # check block
            if not device.verify_winning_block(winning_block, comm_round, idx_to_device):
                # block check failed, perform chain_resync next round
                continue
        
        ''' Phase 5 - All Online Devices Process and Append Winning Block '''
        for device in online_workers:
            # append and process block
            if not device.process_and_append_block(comm_round):
                continue
            # DEBUG - check performance of the validation mechanism
            if args.show_validation_performance_in_block:
                device.check_validation_performance(idx_to_device, comm_round)

        ''' End of LBFL '''

        ''' Evaluation '''
        ### record forking events ###
        forking = 0
        blocks_produced_by = set()
        for device in init_online_devices:
            if device.has_appended_block:
                blocks_produced_by.add(device.blockchain.get_last_block().produced_by)
                if len(blocks_produced_by) > 1:
                    forking = 1
                    break
        logger['forking_event'][comm_round] = forking

        ### record validation accuracies ###
        for device in init_online_devices:
            logger["validator_to_worker_acc_diff"][comm_round][device.idx] = deepcopy(device.worker_to_acc_diff)

        ### record pos book ###
        for device in devices_list:
            logger["pos_book"][comm_round][device.idx] = deepcopy(device._pos_book)

        ### record when a winning block from a malicious validator is accepted in network ###
        logger['malicious_winning_count'][comm_round] = 0
        for device in init_online_devices:
            if device.has_appended_block:
                block_produced_by = device.blockchain.get_last_block().produced_by
                if idx_to_device[block_produced_by]._is_malicious:
                   logger['malicious_winning_count'][comm_round] += 1

        # save logger
        with open(f'{args.log_dir}/logger.pickle', 'wb') as f:
            pickle.dump(logger, f)

if __name__ == "__main__":
    main()