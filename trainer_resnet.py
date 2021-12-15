import os
import time
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import Architectures.cifar_resnet as cifar10_resnet
import Architectures.cifar100_resnet as cifar100_resnet
from Losses.torch_loss import AdversarialLoss, AWP_Loss, AMP_Loss
import Utils.cifar_dataloader as DataLoader
from copy import deepcopy
from Utils.utils import device
import datajuicer as dj
import json
import numpy as np

def log(key, value):
    with dj.open("training_results.json", "w+") as f:
        contents = f.read()
    if contents != "":
        results = json.loads(contents)
    if not key in results:
        results[key] = []
    results[key].append(value)
    with dj.open("training_results.json", "w+") as f:
        f.seek(0)
        json.dump(results, f)

@dj.Task.make(mode="process", hyperparams=dj.Depend("dataset", "architecture", "momentum", "seed", "weight_decay", "start_epoch"))
def init(hyperparams, pretrained_path):
    if pretrained_path is not None:
        print(f"Loading pretrained Model from path {pretrained_path}")
        if os.path.isfile(pretrained_path):
            checkpoint = torch.load(pretrained_path)
    
    if hyperparams["dataset"] == "cifar10":
        model = cifar10_resnet.__dict__[hyperparams["architecture"]]()
    elif hyperparams["dataset"] == "cifar100":
        model = cifar100_resnet.__dict__[hyperparams["architecture"]]()
    else:
        raise Exception("Unknown dataset")
    model = torch.nn.DataParallel(model).to(device)
    
    model.load_state_dict(checkpoint["state_dict"])

    optimizer = torch.optim.SGD(
        model.parameters(),
        hyperparams["lr"],
        momentum=hyperparams["momentum"],
        nesterov=True,
        weight_decay=hyperparams["weight_decay"]
    )
    epoch = hyperparams["start_epoch"]
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[60, 120, 160],
        gamma=0.2,
        last_epoch=-1
    )
    
    for _ in range(epoch):
        lr_scheduler.step()
    
    torch.manual_seed(hyperparams["seed"])
    np.random.seed(hyperparams["seed"])

    with dj.open("checkpoint.th", "wb+") as f:
        torch.save(
            {
                "rng_state":torch.get_rng_state(),
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f)
    print(f"Successfully initialized Model with run id {dj.run_id()}")

@dj.Task.make(mode = "process", hyperparams = dj.Depend(num_workers=dj.Ignore, data_dir=dj.Ignore, save_every=dj.Ignore))
def train(hyperparams):
    t_start = time.time()
    base_path = os.path.dirname(os.path.abspath(__file__))
    resources_path = os.path.join(base_path, "Resources")
    run = init(hyperparams, os.path.join(resources_path,"%s_pretrained_models/%s.th" % (hyperparams["dataset"],hyperparams["architecture"]))).join()

    if hyperparams["dataset"] == "cifar10":
        model = cifar10_resnet.__dict__[hyperparams["architecture"]]()
    elif hyperparams["dataset"] == "cifar100":
        model = cifar100_resnet.__dict__[hyperparams["architecture"]]()
    else:
        raise Exception("Unknown dataset")
    model = torch.nn.DataParallel(model).to(device)
    
    _, val_loader = DataLoader.get_train_valid_loader(
        data_dir=os.path.join(hyperparams["data_dir"],hyperparams["dataset"]),
        batch_size=hyperparams["batch_size"],
        augment=True,
        random_seed=hyperparams["seed"],
        valid_size=0.1,
        shuffle=True,
        num_workers=hyperparams["workers"],
        pin_memory=("cuda" in device),
        dataset=hyperparams["dataset"]
    )

    val_acc = []
    eta_val_acc = []
    eta_val_acc_std = []


    best_prec1 = 0.0
    for i in range(0, hyperparams["n_epochs"]- hyperparams["start_epoch"]):
        next_run = train_epoch(run, hyperparams).join()
        if i % hyperparams["save_every"] != 0:
            run.delete()
        with next_run.open("model.th", "rb") as f:
            checkpoint = torch.load(f)
        model.load_state_dict(checkpoint["state_dict"])

        prec1 = validate(val_loader, model)
        mean_prec1, std_prec1 = validate_noisy(val_loader, model, eta_inf=hyperparams["eta_train"], eta_mode=hyperparams["eta_mode"], n_inf=3)

        val_acc.append(prec1)
        eta_val_acc.append(mean_prec1)
        eta_val_acc_std.append(std_prec1)

        if prec1 > best_prec1:
            print("\t * New best: %.5f" % best_prec1)
            checkpoint["val_acc"] = val_acc
            checkpoint["eta_val_acc"] = eta_val_acc
            checkpoint["eta_val_acc_std"] = eta_val_acc_std
            with dj.open("best_checkpoint.th", "wb+") as f:
                torch.save(checkpoint, f)

        run = next_run
    
    test_loader = DataLoader.get_test_loader(
        data_dir=os.path.join(hyperparams["data_dir"],hyperparams["dataset"]),
        batch_size=128,
        shuffle=False,
        num_workers=hyperparams["workers"],
        pin_memory=("cuda" in device),
        dataset = hyperparams["dataset"]
    )
    # - Load the best model
    with dj.open("best_checkpoint.th", "rb+") as f:
        best_model = torch.load(f)
    model.load_state_dict(best_model["state_dict"])
    # - Get normal test acc.
    prec1_test = validate(test_loader, model)
    # - Get noisy test acc. using eta_inf = eta_train
    prec1_test_noisy_mean, prec1_test_noisy_std = validate_noisy(test_loader, model,  eta_inf=hyperparams["eta_train"], eta_mode=hyperparams["eta_mode"], n_inf=25) 
    time_passed = time.time() - t_start
    print("Training finished in %.4f hours. Test accuracy is %.5f. Run Id = %s" %\
        (time_passed / 3600., float(prec1_test),dj.run_id()))
    log("test_acc", float(prec1_test))
    log("noisy_test_acc_mean", float(prec1_test_noisy_mean))
    log("noisy_test_acc_std", float(prec1_test_noisy_std))
    log("best_noisy_val_acc", float(best_prec1))
    log("time_passed", float(time_passed / 3600.))
    log("completed", True)
    
@dj.Task.make(mode = "process", hyperparams = dj.Depend(num_workers=dj.Ignore, data_dir=dj.Ignore, save_every=dj.Ignore, n_epochs = dj.Ignore))
def train_epoch(previous_epoch, hyperparams):
    @torch.no_grad()
    def clip_weights(model):
        if hyperparams["clipping_alpha"] == 0.0 : return 
        for n,v in model.named_parameters():
            if "bn" in n or "bias" in n : continue
            clamp_val = float(hyperparams["clipping_alpha"] * torch.std(v.view(-1), dim=0))
            mean_v = float(torch.mean(v.view(-1), dim=0))
            v.clamp_(min=mean_v-clamp_val, max=mean_v+clamp_val)

    train_loader, _ = DataLoader.get_train_valid_loader(
        data_dir=os.path.join(hyperparams["data_dir"],hyperparams["dataset"]),
        batch_size=hyperparams["batch_size"],
        augment=True,
        random_seed=hyperparams["seed"],
        valid_size=0.1,
        shuffle=True,
        num_workers=hyperparams["workers"],
        pin_memory=("cuda" in device),
        dataset=hyperparams["dataset"]
    )
    
    if hyperparams["dataset"] == "cifar10":
        model = cifar10_resnet.__dict__[hyperparams["architecture"]]()
    elif hyperparams["dataset"] == "cifar100":
        model = cifar100_resnet.__dict__[hyperparams["architecture"]]()
    else:
        raise Exception("Unknown dataset")
    model = torch.nn.DataParallel(model).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        hyperparams["lr"],
        momentum=hyperparams["momentum"],
        nesterov=True,
        weight_decay=hyperparams["weight_decay"]
    )
    with previous_epoch.open("checkpoint.th", "rb") as f:
        checkpoint = torch.load(f)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    torch.set_rng_state(checkpoint["rng_state"])
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[60, 120, 160],
        gamma=0.2,
        last_epoch=-1
    )
    
    for _ in range(epoch):
        lr_scheduler.step()
    
    nat_loss = nn.CrossEntropyLoss(reduction="mean")

    if hyperparams["gamma"] > 0.0:
        criterion = AWP_Loss(
            model=model,
            loss_func=nat_loss,
            device=device,
            n_attack_steps=hyperparams["n_attack_steps"],
            mismatch_level=hyperparams["attack_size_mismatch"],
            initial_std=hyperparams["initial_std"],
            gamma=hyperparams["gamma"],
            eps_pga=hyperparams["eps_pga"],
            burn_in=hyperparams["burn_in"]
        )
    elif hyperparams["beta_robustness"] >= 999:
        criterion = AMP_Loss(
            model = model,
            loss_func=nat_loss,
            device = device,
            n_attack_steps= hyperparams["n_attack_steps"],
            epsilon = hyperparams["attack_size_mismatch"], #misnomer
            inner_lr = hyperparams["inner_lr"]
        )
    else:
        criterion = AdversarialLoss(
            model=model,
            natural_loss=nat_loss,
            robustness_loss=torch.nn.KLDivLoss(reduction="batchmean"),
            device=device,
            n_attack_steps=hyperparams["n_attack_steps"],
            mismatch_level=hyperparams["attack_size_mismatch"],
            initial_std=hyperparams["initial_std"],
            beta_robustness=hyperparams["beta_robustness"],
            burn_in=hyperparams["burn_in"]
        )

    """
    Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    print_freq = len(train_loader) // 10
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.to(device)
        input_var = input.to(device)
        target_var = target

        loss = criterion.compute_gradient_and_backward(
                model=model,
                X=input_var,
                y=target_var,
                epoch=epoch
            )

        optimizer.step()
        optimizer.zero_grad()

        clip_weights(model)

        with torch.no_grad():
            output = model(input_var) #! This is inefficient, right?

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
        
    with dj.open("checkpoint.th", "wb+") as f:
        torch.save({
            "rng_state":torch.get_rng_state(),
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, f)


def validate_noisy(val_loader, model, eta_inf, eta_mode, n_inf):
    accs = torch.empty(size=(n_inf,))
    for i in range(n_inf):
        model_noisy = deepcopy(model)
        with torch.no_grad():
            for name,v in model_noisy.named_parameters():
                if "bn" in name or "bias" in name : continue
                noise = eta_inf * 0.5*(v.max()-v.min()) * torch.randn_like(v) if eta_mode == "range"\
                    else eta_inf * v.abs() * torch.randn_like(v)
                v.add_(noise.detach())
        accs[i] = validate(val_loader, model_noisy)
    print(" * Noisy Prec@1 %.2f" % float(accs.mean()))
    return (accs.mean(), accs.std())


def validate(val_loader, model):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.to(device)
            input_var = input.to(device)
            target_var = target.to(device)

            # compute output
            output = model(input_var)

            output = output.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target_var)[0]
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res