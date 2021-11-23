from architectures import Resnet
from datajuicer import run, split, run, query
from datajuicer.utils import configure
import os

inner_lrs = [0.05,0.1,0.2]
base_path = os.path.dirname(os.path.abspath(__file__))
epsilons = [0.0025,0.005,0.01,0.02]

class amp_experiment:

    @staticmethod
    def train_grid():
        seeds = [0]
        resnet = [Resnet.make()]
        resnet = configure(resnet, {"n_attack_steps":3, "batch_size":256, "clipping_alpha":2.0,
                                    "start_epoch":60, "pretrained":True, "epochs":200, "gamma":0.0})
        resnet = split(resnet, "inner_lr", inner_lrs)
        resnet = split(resnet, "attack_size_mismatch", epsilons)
        resnet = split(resnet, "seed", seeds)
        return resnet

    @staticmethod
    def visualize():
        grid = amp_experiment.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")