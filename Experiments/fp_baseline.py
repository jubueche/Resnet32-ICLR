from architectures import Resnet
from datajuicer import run, split, run, configure

class fp_baseline:
    
    @staticmethod
    def train_grid():
        seeds = [0]
        resnet = [Resnet.make()]
        resnet = configure(resnet, {"batch_size":256})
        resnet = split(resnet, "seed", seeds)
        return resnet

    @staticmethod
    def visualize():
        grid = fp_baseline.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")