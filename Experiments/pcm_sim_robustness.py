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
eta_trains = [0.11]
beta_robustnesses = [0.01,0.025,0.05,0.1]
attack_size_mismatches = [0.01,0.025,0.05,0.1]
mode = "range"
base_path = os.path.dirname(os.path.abspath(__file__))

class pcm_sim_robustness:

    @staticmethod
    def train_grid():
        resnet = [Resnet.make()]
        resnet = configure(resnet, {"eta_mode":mode, "n_attack_steps":3,
                                    "batch_size":256, "clipping_alpha":2.0, "start_epoch":60,
                                    "pretrained":True, "epochs":200})
        resnet = split(resnet, "beta_robustness", beta_robustnesses)
        resnet = split(resnet, "attack_size_mismatch", attack_size_mismatches)
        resnet = split(resnet, "eta_train", eta_trains)
        # - Baseline and models w/o adversary
        baseline = [Resnet.make()]
        baseline = configure(baseline, {"eta_mode":mode, "batch_size":256,
                                    "clipping_alpha":2.0, "start_epoch":60, "pretrained":True,
                                    "epochs":200})
        baseline = split(baseline, "eta_train", eta_trains)
        baseline = baseline + fp_baseline.train_grid()
        resnet = split(resnet + baseline, "seed", seeds)
        return resnet

    @staticmethod
    def visualize():
        grid = pcm_sim_robustness.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")
        grid = configure(grid, {"mode":"bsub"})
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

        fig, axes = plt.subplots(ncols=len(attack_size_mismatches), nrows=len(eta_trains), constrained_layout=True, figsize=(13,3))
        axes = np.reshape(axes, newshape=(len(eta_trains),len(attack_size_mismatches)))
        for ax in axes.reshape(-1):
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        def plot_errorbar(ax, times, eta_train, color, grid, eta_mode, beta_robustness, attack_size_mismatch, label=None):
            mean_acc_over_time, std_acc_over_time = query(grid, "pcm_acc_over_time",
                where={"eta_mode":eta_mode, "eta_train":eta_train,
                       "beta_robustness":beta_robustness,
                       "attack_size_mismatch":attack_size_mismatch})[0]

            ax.fill_between(
                x=times,
                y1=100*(mean_acc_over_time-std_acc_over_time),
                y2=100*(mean_acc_over_time+std_acc_over_time),
                color=color,
                label=label,
                alpha=0.3
            )
            ax.plot(times,100*mean_acc_over_time, color=color)

        baseline_test_acc = np.max(np.mean(np.array(query(grid, "test_acc", where={"eta_train":0.0})),axis=0))
        green_vals = np.linspace(242. / 255., 0.3 ,num=len(beta_robustnesses))
        for i,eta_train in enumerate(eta_trains):
            for j,attack_size_mismatch in enumerate(attack_size_mismatches):
                ax = axes[i,j]
                ax.set_xticks(times)
                ax.set_xscale("log")
                ax.plot([times[0],times[-1]],2*[baseline_test_acc], linestyle="--", color="k", label="FP Baseline")
                ax.set_ylabel("Test acc. (%)")
                ax.set_title(r"$\zeta_{attack}=$" + ("%.2f"%attack_size_mismatch))
                ax.set_xlabel(r"$T_{inf}$ (s)")
                ax.tick_params(axis='both', which='major', labelsize=6)
                for k,beta_robustness in enumerate(beta_robustnesses):
                    color = (0.,green_vals[k],1.)
                    plot_errorbar(
                        ax=ax,
                        times=times,
                        eta_train=eta_train,
                        color=color,
                        grid=grid,
                        eta_mode=mode,
                        beta_robustness=beta_robustness,
                        attack_size_mismatch=attack_size_mismatch
                    )
                plot_errorbar(
                        ax=ax,
                        times=times,
                        eta_train=eta_train,
                        color="r",
                        grid=grid,
                        eta_mode=mode,
                        beta_robustness=0.0,
                        attack_size_mismatch=0.2,
                        label=r"$\beta_{rob}=0.0,\eta_{train}=$"+("%.3f"%eta_train)
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
        plt.savefig(os.path.join(figures_path, "pcm_simulation_robustness_%s_11.pdf" % mode))
        plt.show()
