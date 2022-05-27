

import sys
sys.path.append(r'D:\Shaoru\GithubDesk\auto_LiRPA')

import socket
import random
import pickle
import os
import time
import gc
import csv
import torch
import numpy as np
from collections import defaultdict

import arguments
from beta_CROWN_solver import LiRPAConvNet
from batch_branch_and_bound import relu_bab_parallel
from utils import load_model_onnx, convert_test_model
from attack_pgd import pgd_attack
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import reduction_min, stop_criterion_min

import torch
import torch.nn as nn
from bab_verification_general import bab
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import arguments
import matplotlib.pyplot as plt
torch.set_grad_enabled(True)
from tqdm import tqdm


def config_args():
    # Add arguments specific for this front-end.
    h = ["general"]
    arguments.Config.add_argument("--mode", type=str, default="verified-acc", choices=["verified-acc", "runnerup", "clean-acc", "crown-only-verified-acc", "alpha-crown-only-verified-acc", "ibp-only-verified-acc", "attack-only", "specify-target"],
            help='Verify against all labels ("verified-acc" mode), or just the runnerup labels ("runnerup" mode), or using a specified label in dataset ("speicify-target" mode, only used for oval20). Mode can also be set as "crown-only-verified-acc" or "alpha-crown-only-verified-acc", which quickly computes the verified accuracy over the entire dataset via CROWN or alpha-CROWN.', hierarchy=h + ["mode"])
    arguments.Config.add_argument('--complete_verifier', choices=["bab", "mip", "bab-refine", "skip"], default="bab",
            help='Complete verification verifier. "bab": branch and bound with beta-CROWN; "mip": mixed integer programming (MIP) formulation; "bab-refine": branch and bound with intermediate layer bounds computed by MIP.', hierarchy=h + ["complete_verifier"])
    arguments.Config.add_argument('--no_incomplete', action='store_false', dest='incomplete',
            help='Enable/Disable initial alpha-CROWN incomplete verification (this can save GPU memory when disabled).', hierarchy=h + ["enable_incomplete_verification"])
    arguments.Config.add_argument("--crown", action='store_true', help='Compute CROWN verified accuracy before verification (not used).', hierarchy=h + ["get_crown_verified_acc"])

    h = ["model"]
    arguments.Config.add_argument("--model", type=str, default="please_specify_model_name", help='Name of model. Model must be defined in the load_verification_dataset() function in utils.py.', hierarchy=h + ["name"])

    h = ["data"]
    arguments.Config.add_argument("--dataset", type=str, default="CIFAR", choices=["MNIST", "CIFAR", "CIFAR_SDP_FULL", "CIFAR_RESNET", "CIFAR_SAMPLE", "MNIST_SAMPLE", "CIFAR_ERAN", "MNIST_ERAN",
                                 "MNIST_ERAN_UN", "MNIST_SDP", "MNIST_MADRY_UN", "CIFAR_SDP", "CIFAR_UN"], help='Dataset name. Dataset must be defined in utils.py.', hierarchy=h + ["dataset"])
    arguments.Config.add_argument("--filter_path", type=str, default=None, help='A filter in pkl format contains examples that will be skipped (not used).', hierarchy=h + ["data_filter_path"])
    arguments.Config.add_argument("--data_idx_file", type=str, default=None, help='A text file with a list of example IDs to run.', hierarchy=h + ["data_idx_file"])
    # arguments.Config.add_argument("--num_outputs", type = int, default = 4)

    h = ["attack"]
    arguments.Config.add_argument("--mip_attack", action='store_true', help='Use MIP (Gurobi) based attack if PGD cannot find a successful adversarial example.', hierarchy=h + ["enable_mip_attack"])
    arguments.Config.add_argument('--pgd_steps', type=int, default=100, help="Steps of PGD attack.", hierarchy=h + ["pgd_steps"])
    arguments.Config.add_argument('--pgd_restarts', type=int, default=30, help="Number of random PGD restarts.", hierarchy= h + ["pgd_restarts"])
    arguments.Config.add_argument('--no_pgd_early_stop', action='store_false', dest='pgd_early_stop', help="Early stop PGD when an adversarial example is found.", hierarchy=h + ["pgd_early_stop"])
    arguments.Config.add_argument('--pgd_lr_decay', type=float, default=0.99, help='Learning rate decay factor used in PGD attack.', hierarchy= h + ["pgd_lr_decay"])
    arguments.Config.add_argument('--pgd_alpha', type=str, default="auto", help='Step size of PGD attack. Default (auto) is epsilon/4.', hierarchy=h + ["pgd_alpha"])

    h = ["debug"]
    arguments.Config.add_argument("--lp_test", type=str, default=None, choices=["MIP", "LP", "LP_intermediate_refine", "MIP_intermediate_refine", None], help='Debugging option, do not use.', hierarchy=h + ['lp_test'])

    arguments.Config.parse_config()

def main():
    nx = 4

    nn_width = 100

    nn_file_name = 'nn_model_2_layer_' + str(nn_width) + '_neuron.pt'

    torch.set_grad_enabled(False)
    nn_system = torch.load(nn_file_name)

    # original test bed
    # x0_lb = torch.tensor([[2.0, 1.0, -10 * np.pi / 180, -1.0]]).to(device)
    # x0_ub = torch.tensor([[2.2, 1.2, -6 * np.pi / 180, -0.8]]).to(device)


    horizon = 20

    nn_system = torch.load(nn_file_name)
    nn_layers_list = list(nn_system) * horizon
    k_nn_system = nn.Sequential(*nn_layers_list)
    model_ori = k_nn_system
    model_ori.train()

    # example 3
    pre_fix = 'exp_3_'
    x0_lb = torch.tensor([[-0.1, -0.5, -10*np.pi/180, 0.05]]).to(device)
    x0_ub = torch.tensor([[0.1, -0.3, -8*np.pi/180, 0.25]]).to(device)

    x0 = (x0_lb + x0_ub) / 2
    perturb_eps = (x0_ub - x0_lb) / 2

    # find upper bounds on all 4 outputs
    for target in tqdm(range(nx), desc='bab'):
        l, u, nodes, glb_record = bab(model_ori, x0, target, y=None, eps=perturb_eps)

        print(f'lower bounds: {l}, upper bounds: {u}, time: {glb_record[-1][0]}')

        result = {'l': l, 'u': u, 'nodes': nodes, 'glb_record': glb_record, 'target': target}
        torch.save(result, pre_fix + 'bab_result_horizon_' + str(horizon) + '_target_' + str(target) + '_lb.pt')


    # example 4
    pre_fix = 'exp_4_'
    x0_lb = torch.tensor([[1.4, 0.2, 6*np.pi/180, 0.1]]).to(device)
    x0_ub = torch.tensor([[1.6, 0.4, 8*np.pi/180, 0.3]]).to(device)

    x0 = (x0_lb + x0_ub) / 2
    perturb_eps = (x0_ub - x0_lb) / 2

    # find upper bounds on all 4 outputs
    for target in tqdm(range(nx), desc='bab'):
        l, u, nodes, glb_record = bab(model_ori, x0, target, y=None, eps=perturb_eps)

        print(f'lower bounds: {l}, upper bounds: {u}, time: {glb_record[-1][0]}')

        result = {'l': l, 'u': u, 'nodes': nodes, 'glb_record': glb_record, 'target': target}
        torch.save(result, pre_fix + 'bab_result_horizon_' + str(horizon) + '_target_' + str(target) + '_lb.pt')

    # example 5
    pre_fix = 'exp_5_'
    x0_lb = torch.tensor([[-0.1, -0.2, -3*np.pi/180, -0.1]]).to(device)
    x0_ub = torch.tensor([[0.1, 0.2, 3*np.pi/180, 0.1]]).to(device)

    x0 = (x0_lb + x0_ub) / 2
    perturb_eps = (x0_ub - x0_lb) / 2

    # find upper bounds on all 4 outputs
    for target in tqdm(range(nx), desc='bab'):
        l, u, nodes, glb_record = bab(model_ori, x0, target, y=None, eps=perturb_eps)

        print(f'lower bounds: {l}, upper bounds: {u}, time: {glb_record[-1][0]}')

        result = {'l': l, 'u': u, 'nodes': nodes, 'glb_record': glb_record, 'target': target}
        torch.save(result, pre_fix + 'bab_result_horizon_' + str(horizon) + '_target_' + str(target) + '_lb.pt')

if __name__ == '__main__':
    config_args()
    main()

