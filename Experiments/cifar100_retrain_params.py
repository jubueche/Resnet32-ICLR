from architectures import Resnet
from datajuicer import run, split, run, query
from datajuicer.utils import configure
import matplotlib.pyplot as plt
import numpy as np
import os

lrs = [0.01,0.1]
clipping_alphas = [2.0,2.5]
base_path = os.path.dirname(os.path.abspath(__file__))

class cifar100_retrain_params:

    @staticmethod
    def train_grid():
        seeds = [0]
        resnet = [Resnet.make()]
        resnet = configure(resnet,
            {
                "eta_mode":"range", "eta_train":0.0, "batch_size":256,
                "start_epoch":60, "pretrained":True, "epochs":200, "dataset":"cifar100",
                "architecture":"resnet34"
            }
        )
        resnet = split(resnet, "clipping_alpha", clipping_alphas)
        resnet = split(resnet, "lr", lrs)
        resnet = split(resnet, "seed", seeds)
        return resnet

    @staticmethod
    def visualize():
        grid = cifar100_retrain_params.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")
        
