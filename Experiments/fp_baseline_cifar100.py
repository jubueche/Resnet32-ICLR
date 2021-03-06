from architectures import Resnet
from datajuicer import run, split, run, configure

class fp_baseline_cifar100:
    
    @staticmethod
    def train_grid():
        seeds = [0]
        resnet = [Resnet.make()]
        resnet = configure(resnet, {"architecture":"resnet34", "batch_size":256, "dataset":"cifar100", "pretrained":False})
        resnet = split(resnet, "seed", seeds)
        return resnet

    @staticmethod
    def visualize():
        grid = fp_baseline_cifar100.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")
