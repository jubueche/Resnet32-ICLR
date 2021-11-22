import numpy as np
import Architectures.cifar_resnet as cifar_resnet
import torch
torch.manual_seed(0)
from ais.utils import Config
import os
import Utils.cifar_dataloader as DataLoader
from datajuicer import cachable

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_dataloaders(network_dict, data_dir, batch_size=None):
    train_laoder, val_loader = DataLoader.get_train_valid_loader(
        data_dir=os.path.join(data_dir,network_dict["dataset"]),
        batch_size=network_dict["batch_size"] if batch_size is None else batch_size,
        augment=True,
        random_seed=network_dict["seed"],
        valid_size=0.1,
        shuffle=True,
        num_workers=network_dict["workers"],
        pin_memory=("cuda" in device)
    )
    test_loader = DataLoader.get_test_loader(
        data_dir=os.path.join(data_dir,network_dict["dataset"]),
        batch_size=128,
        shuffle=False,
        num_workers=network_dict['workers'],
        pin_memory=("cuda" in device)
    )
    return train_laoder,val_loader,test_loader

def _single_pcm_acc_over_time(
    model,
    times,
    test_loader
):
    accs = np.empty(shape=(len(times),))
    for t_idx,time in enumerate(times):
        model.set_time(time)
        accs[t_idx] = get_acc(model, test_loader)
    return accs

def get_acc(model, data_loader):
    counter = 0; correct = 0
    for X,y in data_loader:
        X, y = X.to(device),y.to(device)
        correct += (y == torch.argmax(model(X), dim=1)).int().sum()
        counter += len(y)
        # print("Batch acc is %.3f" % (correct/counter))
    print("Acc. is %.4f" % (correct/counter))
    return correct / counter

@cachable(dependencies = ["network_dict:cnn_session_id", "N_rep", "times"])
def pcm_acc_over_time(
    network_dict,
    data_dir,
    N_rep,
    times
):
    config = {
        "USE_ABS_MAX_FOR_GT": True,
        "N_WSTD": 2.5,
        "T0": 25.0,
        "BITS_DAC":8,
        "BITS_ADC":8,
        "USE_ABS_MAX_FOR_INPUT_RANGE": False,
        "PERCENTILE_INPUT_RANGE": 0.99995,
        "SIZE_CROSSBAR": 256,
        "N_STD_DAC": 3.89, # - Number of std's corresponding to 99.995 under N(0,1) assumption
        "N_STD_ADC": 4.0, # - Tunable hyperparameter
        "ETA_TRAIN": network_dict["eta_train"],
        "ETA_MODE": network_dict["eta_mode"]
    }
    config = Config(**config)
    model = cifar_resnet.__dict__[network_dict["architecture"]]()
    state_dict_model = network_dict["checkpoint"]["state_dict"]
    new_state_dict = {}
    for k,v in state_dict_model.items():
        new_state_dict[k[7:]] = v # - Remove module.
    model = model.to(device)
    model.set_config(config)
    model.load_state_dict(new_state_dict)
    _,val_loader,test_loader = get_dataloaders(network_dict, data_dir, batch_size=512)
    
    # model.eval()
    # print(get_acc(model, test_loader))
    
    model.turn_off_ADCDAC()
    model.pcm()
    model.set_time(25)
    X_cal,_ = next(iter(val_loader))
    model.calibrate(input=X_cal.to(device))
    accuracies = np.empty(shape=(N_rep,len(times)))
    for rep in range(N_rep):
        accs = _single_pcm_acc_over_time(model, times, test_loader)
        accuracies[rep,:] = accs
    return np.mean(accuracies, axis=0), np.std(accuracies, axis=0)
