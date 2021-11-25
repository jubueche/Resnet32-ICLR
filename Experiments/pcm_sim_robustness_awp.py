"""
Experiment for testing robustness when virtually deployed on chip
"""
from architectures import Resnet
from datajuicer import run, split, run, query
from datajuicer.utils import configure
from Experiments.eta_train_vs_eta_inf import eta_train_vs_eta_inf
from Experiments.fp_baseline import fp_baseline
import os
from Utils.utils import pcm_acc_over_time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np

seeds = [0,1]
eta_train = 0.075
eps_pgas = [0.0,0.001,0.01]
attack_size_mismatches = [0.01,0.025,0.05,0.1]
base_path = os.path.dirname(os.path.abspath(__file__))

class pcm_sim_robustness_awp:

    @staticmethod
    def train_grid():
        resnet = [Resnet.make()]
        resnet = configure(resnet, {"n_attack_steps":3, "batch_size":256, "clipping_alpha":2.0,
                                    "start_epoch":60, "pretrained":True, "epochs":200, "gamma":0.1})
        resnet = split(resnet, "eps_pga", eps_pgas)
        resnet = split(resnet, "attack_size_mismatch", attack_size_mismatches)
        # - Baseline and models w/o adversary
        baseline = [Resnet.make()]
        baseline = configure(baseline, {"eta_mode":"range", "batch_size":256,
                                    "clipping_alpha":2.0, "start_epoch":60, "pretrained":True,
                                    "eta_train":eta_train,"epochs":200})
        baseline = baseline + fp_baseline.train_grid()
        resnet = split(resnet + baseline, "seed", seeds)
        return resnet

    @staticmethod
    def visualize():
        grid = pcm_sim_robustness_awp.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")
        grid = configure(grid, {"mode":"direct"})
        times = [25.0, 3600.0, 86400.0, 2592000.0, 31536000.0]
        N_rep = 25
        grid = run(
            grid,
            pcm_acc_over_time,
            n_threads=1,
            run_mode="normal",
            store_key="pcm_acc_over_time")(
                                "{*}",
                                "{data_dir}",
                                N_rep,
                                times
                            )
