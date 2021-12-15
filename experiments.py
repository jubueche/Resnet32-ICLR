import datajuicer as dj
if __name__ == "__main__":
    dj.setup(max_workers=1)

from defaults import defaults
from trainer_resnet import train


@dj.Task.make(version=2)
def adversarial_tune_awp(seeds = [0, 1], eta_modes = ["range","ind"], eta_trains = [0.026,0.037,0.056,0.075,0.11], eps_pgas = [0.0,0.001,0.01], attack_size_mismatches = [0.01,0.025,0.05,0.1]):
    fp_baseline = dj.Frame()
    fp_baseline["hyperparams"] = defaults
    fp_baseline["hyperparams"]["batch_size"] = 256

    grid1 = dj.Frame()
    grid1["hyperparams"] = defaults
    grid1["hyperparams"].configure({"batch_size":256, "clipping_alpha":2.0, "start_epoch":60, "pretrained":True, "epochs":200, "eta_mode":dj.Vary(eta_modes), "eta_train":dj.Vary(eta_trains), "seed":dj.Vary(seeds)})

    
    grid2 = dj.Frame()
    grid2["hyperparams"] = defaults
    grid2["hyperparams"].configure({"n_attack_steps":3, "batch_size":256, "clipping_alpha":2.0,
                                "start_epoch":60, "pretrained":True, "epochs":200, "gamma":0.1})
    grid2["hyperparams"]["eps_pga"] = dj.Vary(eps_pgas)
    grid2["hyperparams"]["attack_size_mismatch"] = dj.Vary(attack_size_mismatches)
    grid2["hyperparams"]["seed"] = dj.Vary(seeds)

    grid = dj.Frame(list(fp_baseline)+ list(grid1)+ list(grid2))

    grid["hyperparams"]["batch_size"] = 1
    grid["train_run"] = grid.map(train).join()



if __name__ == "__main__":
    fp_baseline = dj.Frame()
    fp_baseline["hyperparams"] = defaults
    fp_baseline["hyperparams"]["batch_size"] = 64
    fp_baseline["hyperparams"]["workers"] = 1
    fp_baseline.map(train).join()
    #adversarial_tune_awp().join()