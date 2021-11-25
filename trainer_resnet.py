import os
import time
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import Architectures.cifar_resnet as cifar_resnet
from Losses.torch_loss import AdversarialLoss, AWP_Loss, AMP_Loss
import Utils.cifar_dataloader as DataLoader
from architectures import Resnet as arch
from architectures import log
from copy import deepcopy
from ais.utils import Config
from Utils.utils import device

def main():
    t_start = time.time()
    best_prec1 = 0.0
    args = arch.get_flags()

    # - Create AIS configuration
    config = {
        "USE_ABS_MAX_FOR_GT": True,
        "N_WSTD": 2.5,
        "T0": 25.0,
        "BITS_DAC":7,
        "BITS_ADC":6,
        "USE_ABS_MAX_FOR_INPUT_RANGE": False,
        "PERCENTILE_INPUT_RANGE": 0.99995,
        "SIZE_CROSSBAR": 256,
        "N_STD_DAC": 4.0,
        "N_STD_ADC": 4.0,
        "ETA_TRAIN": args.eta_train,
        "ETA_MODE": args.eta_mode
    }
    config = Config(**config)

    torch.manual_seed(args.seed)
    if "cuda" in device:
        torch.cuda.manual_seed(args.seed)

    base_path = os.path.dirname(os.path.abspath(__file__))
    resources_path = os.path.join(base_path, "Resources")
    if not os.path.exists(resources_path):
        os.mkdir(resources_path)
    models_path = os.path.join(base_path, "Resources/Models")
    if not os.path.exists(models_path):
        os.mkdir(models_path)
    model_save_path = os.path.join(models_path, f"{args.session_id}_model.th")
    checkpoint_save_path = os.path.join(models_path, f"{args.session_id}_checkpoint.th")

    model = cifar_resnet.__dict__[args.architecture]()
    model.set_config(config)
    model = torch.nn.DataParallel(model).to(device)
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        nesterov=True,
        weight_decay=args.weight_decay
    )

    if args.pretrained:
        pretrained_path = os.path.join(resources_path,"%s_pretrained_models/%s.th" % (args.dataset,args.architecture))
        if os.path.isfile(pretrained_path):
            checkpoint = torch.load(pretrained_path)
            model.load_state_dict(checkpoint["state_dict"])
            print("Loaded pretrained model from %s"%pretrained_path)
        else:
            print("No pretrained model found at %s"%pretrained_path)
            sys.exit(0)

    train_loader, val_loader = DataLoader.get_train_valid_loader(
        data_dir=os.path.join(args.data_dir,args.dataset),
        batch_size=args.batch_size,
        augment=True,
        random_seed=args.seed,
        valid_size=0.1,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=("cuda" in device),
        dataset=args.dataset
    )
    args.print_freq = len(train_loader) // 10.

    test_loader = DataLoader.get_test_loader(
        data_dir=os.path.join(args.data_dir,args.dataset),
        batch_size=128,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=("cuda" in device),
        dataset = args.dataset
    )

    nat_loss = nn.CrossEntropyLoss(reduction="mean")

    if args.gamma > 0.0:
        criterion = AWP_Loss(
            model=model,
            loss_func=nat_loss,
            device=device,
            n_attack_steps=args.n_attack_steps,
            mismatch_level=args.attack_size_mismatch,
            initial_std=args.initial_std,
            gamma=args.gamma,
            eps_pga=args.eps_pga,
            burn_in=args.burn_in
        )
    elif args.beta_robustness >= 999:
        criterion = AMP_Loss(
            model = model,
            loss_func=nat_loss,
            device = device,
            n_attack_steps= args.n_attack_steps,
            epsilon = args.attack_size_mismatch, #misnomer
            inner_lr = args.inner_lr
        )
    else:
        criterion = AdversarialLoss(
            model=model,
            natural_loss=nat_loss,
            robustness_loss=torch.nn.KLDivLoss(reduction="batchmean"),
            device=device,
            n_attack_steps=args.n_attack_steps,
            mismatch_level=args.attack_size_mismatch,
            initial_std=args.initial_std,
            beta_robustness=args.beta_robustness,
            burn_in=args.burn_in
        )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[60, 120, 160],
        gamma=0.2,
        last_epoch=-1
    )
    
    for _ in range(args.start_epoch):
        lr_scheduler.step()

    if args.architecture in ['resnet1202', 'resnet110']:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1

    @torch.no_grad()
    def clip_weights(model):
        if args.clipping_alpha == 0.0 : return 
        for n,v in model.named_parameters():
            if "bn" in n or "bias" in n : continue
            clamp_val = float(args.clipping_alpha * torch.std(v.view(-1), dim=0))
            v.clamp_(min=-clamp_val, max=clamp_val)

    # - Clip the weights initially. No-op if no clipping
    p1_nc = validate(val_loader, model, args=args)
    clip_weights(model)
    p1_wc = validate(val_loader, model, args=args)
    mean_prec1, std_prec1 = validate_noisy(val_loader, model, args, eta_inf=args.eta_train, eta_mode=args.eta_mode, n_inf=3)
    print("After loading, with clipping@%.2f %.2f w/o %.2f noisy: %.2fpm%.2f" %\
        (args.clipping_alpha,p1_wc,p1_nc,mean_prec1,std_prec1))

    for epoch in range(args.start_epoch, args.n_epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch, args=args, clip_fn=clip_weights)
        lr_scheduler.step()

        prec1 = validate(val_loader, model, args=args)
        mean_prec1, std_prec1 = validate_noisy(val_loader, model, args, eta_inf=args.eta_train, eta_mode=args.eta_mode, n_inf=3)

        log(args.session_id, "val_acc", float(prec1))
        log(args.session_id, "eta_val_acc", float(mean_prec1))
        log(args.session_id, "eta_val_acc_std", float(std_prec1))

        # remember best prec@1 and save checkpoint
        is_best = mean_prec1 > best_prec1
        best_prec1 = max(mean_prec1, best_prec1)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_prec1': best_prec1,
            }, filename=checkpoint_save_path)

        if is_best:
            print("\t * New best: %.5f" % best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_prec1': best_prec1,
            }, filename=model_save_path)

    # - Load the best model
    best_model = torch.load(model_save_path)
    model.load_state_dict(best_model["state_dict"])
    # - Get normal test acc.
    prec1_test = validate(test_loader, model, args=args)
    # - Get noisy test acc. using eta_inf = eta_train
    prec1_test_noisy_mean, prec1_test_noisy_std = validate_noisy(test_loader, model, args, args.eta_train, args.eta_mode, n_inf=25) 
    time_passed = time.time() - t_start
    print("Training finished in %.4f hours. Test accuracy is %.5f. Saved under %s" %\
        (time_passed / 3600., float(prec1_test),model_save_path))
    log(args.session_id, "test_acc", float(prec1_test))
    log(args.session_id, "noisy_test_acc_mean", float(prec1_test_noisy_mean))
    log(args.session_id, "noisy_test_acc_std", float(prec1_test_noisy_std))
    log(args.session_id, "best_noisy_val_acc", float(best_prec1))
    log(args.session_id, "time_passed", float(time_passed / 3600.))
    log(args.session_id, "completed", True)


def train(train_loader, model, criterion, optimizer, epoch, args, clip_fn):
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
                epoch=epoch - args.start_epoch
            )

        optimizer.step()
        optimizer.zero_grad()

        clip_fn(model)

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

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate_noisy(val_loader, model, args, eta_inf, eta_mode, n_inf):
    accs = torch.empty(size=(n_inf,))
    for i in range(n_inf):
        model_noisy = deepcopy(model)
        with torch.no_grad():
            for name,v in model_noisy.named_parameters():
                if "bn" in name or "bias" in name : continue
                noise = eta_inf * v.abs().max() * torch.randn_like(v) if eta_mode == "range"\
                    else eta_inf * v.abs() * torch.randn_like(v)
                v.add_(noise.detach())
        accs[i] = validate(val_loader, model_noisy, args)
    print(" * Noisy Prec@1 %.2f" % float(accs.mean()))
    return (accs.mean(), accs.std())


def validate(val_loader, model, args):
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

def save_checkpoint(state, filename):
    torch.save(state, filename)

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


if __name__ == '__main__':
    main()
