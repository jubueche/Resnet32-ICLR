import numpy as np
from Utils.utils import noisy_test_acc
from architectures import Resnet
from datajuicer import run, split, run, query
from datajuicer.utils import configure
from Experiments.eta_train_vs_eta_inf import eta_train_vs_eta_inf
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import matplotlib.cm as cm
import os

eps_pgas = [0.0,0.001,0.01]
attack_size_mismatches = [0.01,0.025,0.05,0.1]
eps_pgas_str = ["0.001","0.01","0.1","0.3"]
base_path = os.path.dirname(os.path.abspath(__file__))

class adversarial_tune_awp:

    @staticmethod
    def train_grid():
        seeds = [0,1]
        resnet = [Resnet.make()]
        resnet = configure(resnet, {"n_attack_steps":3, "batch_size":256, "clipping_alpha":2.0,
                                    "start_epoch":60, "pretrained":True, "epochs":200, "gamma":0.1})
        resnet = split(resnet, "eps_pga", eps_pgas)
        resnet = split(resnet, "attack_size_mismatch", attack_size_mismatches)
        resnet = split(resnet, "seed", seeds)
        return resnet + eta_train_vs_eta_inf.train_grid()

    @staticmethod
    def visualize():
        grid = adversarial_tune_awp.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")

        eta_infs = [0.026,0.037,0.056,0.075,0.11]

        fig, axes = plt.subplots(ncols=len(attack_size_mismatches), nrows=1, constrained_layout=True, figsize=(13,3))
        axes = np.reshape(axes, newshape=(1,len(attack_size_mismatches)))
        for ax in axes.reshape(-1):
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        baseline_test_acc = np.max(np.mean(np.array(query(grid, "test_acc", where={"eta_train":0.0, "gamma":0.0})),axis=0))
        green_vals = np.linspace(242. / 255., 0.3 ,num=len(eps_pgas))

        def plot_errorbar(ax, eta_trains, color, grid, eta_mode, eps_pga, gamma, label=None):
            accs = []; acc_stds = []
            for eta in eta_trains:
                eta_test_acc = np.mean(np.array(query(grid, "noisy_test_acc_mean",
                    where={"eta_mode":eta_mode, "eta_train":eta, "eps_pga":eps_pga, "gamma":gamma})))
                eta_test_acc_std = np.mean(np.array(query(grid, "noisy_test_acc_std",
                    where={"eta_mode":eta_mode, "eta_train":eta, "eps_pga":eps_pga, "gamma":gamma})))
                accs.append(100*eta_test_acc)
                acc_stds.append(100*eta_test_acc_std)
            ax.errorbar(x=eta_trains, y=accs, yerr=acc_stds, color=color, label=label)

        for i,attack_size_mismatch in enumerate(attack_size_mismatches):
            ax = axes[0,i]
            # - Plot the baseline
            ax.set_xticks(eta_infs)
            ax.plot([eta_infs[0],eta_infs[-1]],2*[baseline_test_acc], linestyle="--", color="k", label="FP Baseline")
            ax.set_ylabel("Test acc. (%)")
            ax.set_title(r"$\zeta_{attack}=$" + ("%.2f"%attack_size_mismatch))
            ax.set_xlabel(r"$\eta_{train}=\zeta$")
            ax.tick_params(axis='both', which='major', labelsize=6)

            # - Get the list of AWP models with this mismatch level
            grid_awp = [g for g in grid if g["gamma"]>0.0 and g["attack_size_mismatch"]==attack_size_mismatch]

            # - Split the grid and write to eta_train
            grid_awp = split(grid_awp, "eta_train", eta_infs)
            grid_awp = configure(grid_awp, {"mode":"direct"})

            # - Recalculate the noisy test acc
            grid_awp = run(
                grid_awp,
                noisy_test_acc,
                n_threads=1,
                run_mode="normal",
                store_key="noisy_test_acc")(
                                    "{*}",
                                    "{data_dir}",
                                    25,
                                    "{eta_train}",
                                    "{eta_mode}"
                                )
            for g in grid_awp:
                test_acc_mean = g["noisy_test_acc"][0]
                test_acc_std = g["noisy_test_acc"][1]
                g["noisy_test_acc_mean"] = test_acc_mean
                g["noisy_test_acc_std"] = test_acc_std

            for k,eps_pga in enumerate(eps_pgas):
                color = (0.,green_vals[k],1.)
                plot_errorbar(
                    ax=ax,
                    eta_trains=eta_infs,
                    color=color,
                    grid=grid_awp,
                    eta_mode="ind",
                    eps_pga=eps_pga,
                    gamma=grid_awp[0]["gamma"]
                )
        

        norm = mpl.colors.Normalize(vmin=eps_pgas[0],vmax=eps_pgas[-1])
        cmap = ListedColormap([(0.3,g,1.0) for g in green_vals])
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, aspect=30)
        cbar.ax.tick_params(labelsize=6)
        cbar.outline.set_visible(False)
        cbar.set_label(r"$\epsilon_{pga}$")
        cbar.set_ticks(np.linspace(eps_pgas[0],eps_pgas[-1],len(eps_pgas)))
        cbar.set_ticklabels([str(br) for br in eps_pgas])
        figures_path = os.path.join(base_path,"../Resources/Figures")
        if not os.path.exists(figures_path):
            os.mkdir(figures_path)
        plt.savefig(os.path.join(figures_path, "awp_sweep_eps_pga.pdf"), dpi=1200)
