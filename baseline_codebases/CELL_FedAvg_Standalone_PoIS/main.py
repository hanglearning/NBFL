import os
import torch
import argparse
import pickle
import random
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pytorch_lightning import seed_everything
from model.cifar10.cnn import CNN as CIFAR_CNN
from model.cifar10.mlp import MLP as CIFAR_MLP
from model.mnist.cnn import CNN as MNIST_CNN
from model.mnist.mlp import MLP as MNIST_MLP
from server import Server
from client import Client
from baseline_utils import *
from dataset.datasource import DataLoaders

from datetime import datetime
import pickle

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="mnist|cifar10",
                        type=str, default="mnist")
    parser.add_argument('--arch', type=str, default='cnn', help='cnn|mlp')
    parser.add_argument('--dataset_mode', type=str,
                        default='non-iid', help='non-iid|iid')
    parser.add_argument('--alpha_dirichlet', type=float, default=1.0, help='dirichlet distribution parameter for non-iid data')
    parser.add_argument('--n_clients', type=int, default=20)
    parser.add_argument('--rounds', type=int, default=25)
    parser.add_argument('--prune_threshold', type=float, default=0.9) # 1 - sparsity
    parser.add_argument('--server_prune', type=bool, default=False)
    parser.add_argument('--server_prune_step', type=float, default=0.2)
    parser.add_argument('--server_prune_freq', type=int, default=10)
    parser.add_argument('--frac_clients_per_round', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--total_samples', type=int, default=200)
    parser.add_argument('--eita', type=float, default=0.5,
                        help="accuracy threshold")
    parser.add_argument('--alpha', type=float, default=0.5,
                        help="accuracy reduction factor")
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--log_dir', type=str, default="logs")
    parser.add_argument('--train_verbose', type=bool, default=False)
    parser.add_argument('--test_verbose', type=bool, default=False)
    parser.add_argument('--prune_verbose', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=40)
    parser.add_argument('--fast_dev_run', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=0) # for pytorch data loader
    parser.add_argument('--rewind', type=int, default=0, help="reinit ticket model parameters before training")
    parser.add_argument('--optimizer', type=str, default="Adam", help="SGD|Adam")

    parser.add_argument('--n_malicious', type=int, default=0)

    parser.add_argument('--acc_drop_threshold', type=float, default=0.05, help='if the accuracy drop is larger than this threshold, stop prunning')
    parser.add_argument('--target_sparsity', type=float, default=0.1, help='target sparsity for pruning, stop pruning if below this threshold')

    # Run Type
    parser.add_argument('--standalone_LTH', type=int, default=0)
    parser.add_argument('--fedavg_no_prune_max_acc', type=int, default=0)
    parser.add_argument('--PoIS', type=int, default=0)
    parser.add_argument('--attack_type', type=int, default=0, help='0 - no attack, 1 - model poisoning attack, 2 - label flipping attack, 3 - lazy attack')

    
    # for CELL
    parser.add_argument('--CELL', type=int, default=0)
    parser.add_argument('--prune_step', type=float, default=0.15)

    # NBFL needs
    parser.add_argument('--prune_acc_trigger', type=float, default=0.8, help='must achieve this accuracy to trigger worker to post prune its local model')
    parser.add_argument('--max_prune_step', type=float, default=0.05)

    args = parser.parse_args()

    seed_everything(args.seed, workers=True)
    os.environ['PYTHONHASHSEED']=str(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.dev_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if not args.n_malicious or not args.attack_type:
        args.n_malicious, args.attack_type = 0, 0
    
    if args.dataset_mode == 'iid':
        args.alpha_dirichlet = '∞'

    need_test_loaders = False
    # from POLL_march_23_toy
    if args.standalone_LTH:
        run_name = "STANDALONE_LTH" # Pure Centralized
    if args.fedavg_no_prune_max_acc:
        run_name = "FEDAVG_NO_PRUNE" # Pure FedAvg without Pruning
    if args.PoIS:
        run_name = "PoIS"
    if args.CELL:
        run_name = "CELL"
        need_test_loaders = True

    exe_date_time = datetime.now().strftime("%m%d%Y_%H%M%S")

    args.log_dir = f"{args.log_dir}/{run_name}_{args.dataset}_seed_{args.seed}_{args.dataset_mode}_alpha_{args.alpha_dirichlet}_{exe_date_time}_ndevices_{args.n_clients}_nsamples_{args.total_samples}_rounds_{args.rounds}_mal_{args.n_malicious}_attack_{args.attack_type}_rewind_{int(args.rewind)}"

    os.makedirs(args.log_dir)

    init_global_model = create_init_model(cls=models[args.dataset]
                         [args.arch], device=args.dev_device)

    train_loaders, test_loaders, user_labels, global_test_loader = DataLoaders(n_devices=args.n_clients,
                                              dataset_name=args.dataset,
                                              total_samples=args.total_samples,
                                              log_dirpath=args.log_dir,
                                              seed=args.seed,
                                              mode=args.dataset_mode,
                                              batch_size=args.batch_size,
                                              alpha=args.alpha_dirichlet,
                                              dataloader_workers=args.num_workers,
                                              need_test_loaders=need_test_loaders)
    background = None
    test_images = None
    ''' PoIS code '''        
    if args.PoIS:
        # https://github.com/harshkasyap/DecFL/blob/master/Non%20IID/dirichlet%20distribution/vary%20attacker/fm_noniid_ba_9.py
        from torchvision import datasets, transforms
        transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])
        test_data = datasets.MNIST(root='./data',train=False,transform=transform,download=True)
        _, test_data_2 = torch.utils.data.random_split(test_data, [8000, 2000])
        test_data_bd, shap_background = torch.utils.data.random_split(test_data_2, [1500, 500])

        shap_tr_loader = torch.utils.data.DataLoader(shap_background, batch_size = 128, shuffle=True) 
        batch_shap = next(iter(shap_tr_loader))
        images_shap, _ = batch_shap
        background = images_shap[:100]
        test_images = torch.zeros(1,1,28,28)

    ''' PoIS code '''
    
            
    # read init_global_model from file, which was generated from the NBFL run to ensure consistency
    # Construct the file path
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f'../../logs/init_global_model_seed_{args.seed}.pth'))

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file or directory: '{file_path}'")

    # Load the state dictionary
    state_dict = torch.load(file_path)
    init_global_model.load_state_dict(state_dict)

    # pruned by 0.00 %. This is necessary to create buffer masks and to be consistent with create_model() in util.py of the NBFL codebase
    l1_prune(init_global_model, amount=0.00, name='weight', verbose=False)


    clients = []
    n_malicious = args.n_malicious

    if args.n_malicious == 3:
        noise_variances = [0.05]
    elif args.n_malicious == 6:
        noise_variances = [0.05, 0.5, 1.0]
    elif args.n_malicious == 9:
        noise_variances = [0.05, 0.25, 0.5, 1.0]
    elif args.n_malicious == 10:
        noise_variances = [0.05, 0.25, 0.5, 0.75, 1.0]

    noise_attacker_idx = 0
    for i in range(1, args.n_clients + 1):
        is_malicious = True if args.n_clients - i < n_malicious else False
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
        client = Client(i, args, is_malicious, attack_type, noise_variance, init_global_model, train_loaders[i - 1], test_loaders[i - 1], user_labels[i - 1], global_test_loader)
        if is_malicious:
            if attack_type == 1:
                print(f"Assigned client {i} malicious as noise attacker with noise variance: {noise_variance}.")
            elif attack_type == 3:
                print(f"Assigned client {i} malicious as lazy attacker.")
            # label flipping attack
            if args.attack_type == 2:
                client._train_loader.dataset.targets = 9 - client._train_loader.dataset.targets
        clients.append(client)
    
    server = Server(args, init_global_model, clients)

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
    logger["worker_pruned_amount"] = {r: {} for r in range(1, args.rounds + 1)}

    # save args
    with open(f'{args.log_dir}/args.pickle', 'wb') as f:
        pickle.dump(args, f)

    for comm_round in range(1, args.rounds+1):
        server.update(comm_round, logger, background, test_images)
        # save logger
        with open(f'{args.log_dir}/logger.pickle', 'wb') as f:
            pickle.dump(logger, f)
