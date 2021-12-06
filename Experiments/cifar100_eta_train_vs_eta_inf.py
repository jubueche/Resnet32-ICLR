from architectures import Resnet
from datajuicer import run, split, run, query
from datajuicer.utils import configure
import matplotlib.pyplot as plt
from Experiments.fp_baseline import fp_baseline
import numpy as np
import os

eta_trains = [0.026,0.037,0.056,0.075,0.11]
eta_trains_str = ["2.6","3.7","5.6","7.5","11.0"]
eta_modes = ["range"]
base_path = os.path.dirname(os.path.abspath(__file__))

class cifar100_eta_train_vs_eta_inf:

    @staticmethod
    def train_grid():
        seeds = [0]
        resnet = [Resnet.make()]
        # ...
        return resnet

    @staticmethod
    def visualize():
        grid = cifar100_eta_train_vs_eta_inf.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")
        