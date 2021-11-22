from architectures import Resnet
from datajuicer import run, split, run, query
from datajuicer.utils import configure
import matplotlib.pyplot as plt
from Experiments.fp_baseline import fp_baseline
import numpy as np
import os

eta_trains = [0.026,0.037,0.056,0.075,0.11]
eta_trains_str = ["2.6","3.7","5.6","7.5","11.0"]
eta_modes = ["range","ind"]
base_path = os.path.dirname(os.path.abspath(__file__))

class eta_train_vs_eta_inf:

    @staticmethod
    def train_grid():
        seeds = [0,1]
        resnet = [Resnet.make()]
        resnet = configure(resnet, {"batch_size":256, "clipping_alpha":2.0, "start_epoch":60, "pretrained":True, "epochs":200})
        resnet = split(resnet, "eta_mode", values=eta_modes)
        resnet = split(resnet, "eta_train", values=eta_trains)
        resnet = split(resnet, "seed", seeds)
        return resnet + fp_baseline.train_grid()

    @staticmethod
    def visualize():
        grid = eta_train_vs_eta_inf.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")

        fig = plt.figure(figsize=(13,4), constrained_layout=True)
        
        subfigs = fig.subfigures(nrows=2, ncols=1)
        
        baseline_val_acc = np.max(np.mean(np.array(query(grid, "val_acc", where={"eta_train":0.0})),axis=0))
        baseline_test_acc = np.max(np.mean(np.array(query(grid, "test_acc", where={"eta_train":0.0})),axis=0))
        for i,eta_mode in enumerate(eta_modes):
            subfig = subfigs[i]
            title = r"$\sigma_{w^{l}_{i,j}} = \eta_{train} \cdot w^{l}_{max}$" if eta_mode == "range" else r"$\sigma_{w^{l}_{i,j}} = \eta_{train} \cdot |w^{l}_{i,j}|$"
            subfig.suptitle(title)
            axs = subfig.subplots(nrows=1, ncols=2)
            for ax in axs:
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.tick_params(axis='both', which='major', labelsize=8)
                
            ax = axs[0]
            ax.set_ylabel(r"Val. Acc. ($\zeta=\eta_{train}$)")
            ax.set_xlabel("Epochs")
            ax.set_ylim([85,95])
            ax.set_yticks(np.arange(85,95,3))
            ax.plot([query(grid, "start_epoch", where={"eta_train":eta_trains[0]})[0],
                query(grid, "n_epochs", where={"eta_train":eta_trains[0]})[0]],
                2*[baseline_val_acc], linestyle="--", color="k", label="Baseline")
            # - Draw baseline line
            for j,eta in enumerate(eta_trains):
                start_epoch = query(grid, "start_epoch", where={"eta_train":eta})[0]
                stop_epoch = query(grid, "n_epochs", where={"eta_train":eta})[0]
                x = np.arange(start=start_epoch, stop=stop_epoch, step=1, dtype=int)
                eta_val_acc = np.mean(np.array(query(grid, "eta_val_acc", where={"eta_mode":eta_mode, "eta_train":eta})), axis=0)
                eta_val_acc_std = np.std(np.array(query(grid, "eta_val_acc_std", where={"eta_mode":eta_mode, "eta_train":eta})), axis=0)
                color = ("C%d" % j)
                ax.plot(x, eta_val_acc, color=color, linestyle="--", label=r"$\eta_{train}=$" + eta_trains_str[j] + "%")
                ax.fill_between(x, eta_val_acc-eta_val_acc_std, eta_val_acc+eta_val_acc_std, color=color, alpha=0.2)
            ax.legend(fontsize=4)

            ax = axs[1]
            ax.set_ylabel("Test. Acc.")
            ax.set_xlabel(r"$\eta_{train}=\zeta$")
            ax.plot([eta_trains[0],eta_trains[-1]],2*[baseline_test_acc], linestyle="--", color="k", label="Baseline")
            accs = []; acc_stds = []
            for j,eta in enumerate(eta_trains):
                eta_test_acc = np.mean(np.array(query(grid, "noisy_test_acc_mean", where={"eta_mode":eta_mode, "eta_train":eta})))
                eta_test_acc_std = np.mean(np.array(query(grid, "noisy_test_acc_std", where={"eta_mode":eta_mode, "eta_train":eta})))
                accs.append(eta_test_acc)
                acc_stds.append(eta_test_acc_std)
            
            ax.errorbar(x=eta_trains, y=accs, yerr=acc_stds)
            ax.set_xticks(eta_trains)
        
        figures_path = os.path.join(base_path,"../Resources/Figures")
        if not os.path.exists(figures_path):
            os.mkdir(figures_path)
        plt.savefig(os.path.join(figures_path, "eta_train_vs_eta_inf.pdf"))
        plt.show()
        