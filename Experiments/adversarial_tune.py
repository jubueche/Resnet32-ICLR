import numpy as np
from architectures import Resnet
from Utils.utils import noisy_test_acc
from datajuicer import run, split, run, query
from datajuicer.utils import configure
from Experiments.fp_baseline import fp_baseline
from Experiments.eta_train_vs_eta_inf import eta_train_vs_eta_inf
from Experiments.adversarial_tune_awp import adversarial_tune_awp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import matplotlib.cm as cm
import os

eta_trains = [0.026,0.037,0.056,0.075,0.11]
beta_robustnesses = [0.01,0.025,0.05,0.1]
attack_size_mismatches = [0.01,0.025,0.05,0.1]
eta_trains_str = ["2.6","3.7","5.6","7.5","11.0"]
eta_modes = ["ind"]
base_path = os.path.dirname(os.path.abspath(__file__))

class adversarial_tune:

    @staticmethod
    def train_grid():
        seeds = [0,1]
        resnet = [Resnet.make()]
        resnet = configure(resnet, {"n_attack_steps":3, "batch_size":256, "clipping_alpha":2.0,
                                    "start_epoch":60, "pretrained":True, "epochs":200})
        resnet = split(resnet, "eta_mode", values=eta_modes)
        resnet = split(resnet, "eta_train", values=eta_trains)
        resnet = split(resnet, "beta_robustness", beta_robustnesses)
        resnet = split(resnet, "attack_size_mismatch", attack_size_mismatches)
        resnet = split(resnet, "seed", seeds)
        return resnet + eta_train_vs_eta_inf.train_grid() + adversarial_tune_awp.train_grid()

    @staticmethod
    def visualize():
        grid = adversarial_tune.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")

        fig, axes = plt.subplots(ncols=len(attack_size_mismatches), nrows=len(eta_modes), constrained_layout=True, figsize=(13,3))
        axes = np.reshape(axes, newshape=(len(eta_modes),len(attack_size_mismatches)))
        for ax in axes.reshape(-1):
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        def plot_errorbar(
            ax,
            eta_trains,
            color,
            grid,
            eta_mode,
            beta_robustness,
            attack_size_mismatch,
            gamma,
            eps_pga,
            label=None
        ):
            accs = []; acc_stds = []
            for eta in eta_trains:
                eta_test_acc = np.mean(np.array(query(grid, "noisy_test_acc_mean",
                    where={"eta_mode":eta_mode, "eta_train":eta, "beta_robustness":beta_robustness,
                    "attack_size_mismatch":attack_size_mismatch, "gamma":gamma, "eps_pga":eps_pga})))
                eta_test_acc_std = np.mean(np.array(query(grid, "noisy_test_acc_std",
                    where={"eta_mode":eta_mode, "eta_train":eta, "beta_robustness":beta_robustness,
                    "attack_size_mismatch":attack_size_mismatch, "gamma":gamma, "eps_pga":eps_pga})))
                accs.append(eta_test_acc)
                acc_stds.append(eta_test_acc_std)
            ax.errorbar(x=eta_trains, y=accs, yerr=acc_stds, color=color, label=label)

        baseline_test_acc = np.max(np.mean(np.array(query(grid, "test_acc", where={"eta_train":0.0, "gamma":0.0})),axis=0))
        green_vals = np.linspace(242. / 255., 0.3 ,num=len(beta_robustnesses))
        for i,eta_mode in enumerate(eta_modes):
            for j,attack_size_mismatch in enumerate(attack_size_mismatches):

                # - Get the awp grid
                # - Get the list of AWP models with this mismatch level
                grid_awp = [g for g in grid if g["gamma"]>0.0 and g["attack_size_mismatch"]==attack_size_mismatch]

                # - Split the grid and write to eta_train
                grid_awp = split(grid_awp, "eta_train", eta_trains)
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
                    test_acc_mean = 100*g["noisy_test_acc"][0]
                    test_acc_std = 100*g["noisy_test_acc"][1]
                    g["noisy_test_acc_mean"] = test_acc_mean
                    g["noisy_test_acc_std"] = test_acc_std


                ax = axes[i,j]
                ax.set_xticks(eta_trains)
                ax.plot([eta_trains[0],eta_trains[-1]],2*[baseline_test_acc], linestyle="--", color="k", label="FP Baseline")
                ax.set_ylabel("Test acc. (%)")
                ax.set_title(r"$\zeta_{attack}=$" + ("%.2f"%attack_size_mismatch))
                ax.set_xlabel(r"$\eta_{train}=\zeta$")
                ax.tick_params(axis='both', which='major', labelsize=6)
                for k,beta_robustness in enumerate(beta_robustnesses):
                    color = (0.,green_vals[k],1.)
                    plot_errorbar(
                        ax=ax,
                        eta_trains=eta_trains,
                        color=color,
                        grid=grid,
                        eta_mode=eta_mode,
                        beta_robustness=beta_robustness,
                        gamma=0.0,
                        eps_pga=0.0,
                        attack_size_mismatch=attack_size_mismatch
                    )
                plot_errorbar(
                        ax=ax,
                        eta_trains=eta_trains,
                        color="g",
                        grid=grid_awp,
                        eta_mode=eta_mode,
                        beta_robustness=0.0,
                        attack_size_mismatch=attack_size_mismatch,
                        gamma=0.1,
                        eps_pga=0.001,
                        label=r"AWP ($\epsilon_{pga}=0.001$)"
                    )
                plot_errorbar(
                        ax=ax,
                        eta_trains=eta_trains,
                        color="r",
                        grid=grid,
                        eta_mode=eta_mode,
                        beta_robustness=0.0,
                        attack_size_mismatch=0.2,
                        gamma=0.0,
                        eps_pga=0.0,
                        label=r"$\beta_{rob}=0.0$"
                    )
                if j == 0:
                    ax.legend()
        
        norm = mpl.colors.Normalize(vmin=beta_robustnesses[0],vmax=beta_robustnesses[-1])
        cmap = ListedColormap([(0.3,g,1.0) for g in green_vals])
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, aspect=30)
        cbar.ax.tick_params(labelsize=6)
        cbar.outline.set_visible(False)
        cbar.set_label(r"$\beta_{rob}$")
        cbar.set_ticks(np.linspace(beta_robustnesses[0],beta_robustnesses[-1],len(beta_robustnesses)))
        cbar.set_ticklabels([str(br) for br in beta_robustnesses])
        figures_path = os.path.join(base_path,"../Resources/Figures")
        if not os.path.exists(figures_path):
            os.mkdir(figures_path)
        plt.savefig(os.path.join(figures_path, "adversarial_tune.pdf"))
        plt.show()
