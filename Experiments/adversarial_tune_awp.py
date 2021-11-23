import numpy as np
from architectures import Resnet
from datajuicer import run, split, run, query
from datajuicer.utils import configure
from Experiments.fp_baseline import fp_baseline
from Experiments.eta_train_vs_eta_inf import eta_train_vs_eta_inf
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import matplotlib.cm as cm
import os

eps_pgas = [0.001,0.01,0.1,0.3]
eps_pgas_str = ["0.001","0.01","0.1","0.3"]
base_path = os.path.dirname(os.path.abspath(__file__))

class adversarial_tune_awp:

    @staticmethod
    def train_grid():
        seeds = [0]
        resnet = [Resnet.make()]
        resnet = configure(resnet, {"n_attack_steps":3, "batch_size":256, "clipping_alpha":2.0,
                                    "start_epoch":60, "pretrained":True, "epochs":200, "gamma":0.1})
        resnet = split(resnet, "eps_pga", eps_pgas)
        resnet = split(resnet, "seed", seeds)
        return eta_train_vs_eta_inf.train_grid()

    @staticmethod
    def visualize():
        grid = adversarial_tune_awp.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")
