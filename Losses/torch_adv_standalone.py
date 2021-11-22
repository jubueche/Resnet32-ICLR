# - Deterministic linear layer
from copy import deepcopy
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

# - Import the adversarial loss
from torch_loss import AdversarialLoss

def eval_test_set(
    test_dataloader,
    net,
):
    net.eval()
    N_correct = 0
    N = 0
    for (X,y) in test_dataloader:
        X, y = X.to(device), y.to(device)
        y_hat = torch.argmax(net(X), axis=1)
        N += len(y)
        N_correct += (y_hat == y).int().sum()
    net.train()
    return N_correct / N

def eval_test_set_mismatch(
    test_dataloader,
    net,
    mismatch,
    n_reps,
    device
):
    net_theta_star = deepcopy(net)
    test_acc_no_noise = eval_test_set(test_dataloader, net)
    test_accs = []
    for idx in range(n_reps):
        print("Test eval. mismatch rob. %d/%d" % (idx,n_reps))
        theta_star = {}
        for name,v in net.named_parameters():
            theta_star[name] = v + v.abs() * mismatch * torch.randn(size=v.shape, device=device)
        net_theta_star.load_state_dict(theta_star)
        test_accs.append(eval_test_set(test_dataloader, net_theta_star))
    return float(test_acc_no_noise), float(sum(test_accs)/len(test_accs))

def init_weights(lyr):
    if isinstance(lyr, (torch.nn.Linear,torch.nn.Conv2d)):
        torch.nn.init.xavier_uniform(lyr.weight)
        lyr.bias.data.fill_(0.01)

class TorchCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, out_channels=64, kernel_size=(4,4), stride=(1,1), padding="same")
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4,4), stride=(1,1), padding="valid")
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.linear1 = torch.nn.Linear(in_features=1600, out_features=256)
        self.linear2 = torch.nn.Linear(in_features=256, out_features=64)
        self.linear3 = torch.nn.Linear(in_features=64, out_features=10)

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 1600)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

if __name__ == '__main__':
    torch.manual_seed(0)
    # - Avoid reprod. issues caused by GPU
    torch.use_deterministic_algorithms(True)
    # - Select device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # - Select which device
    if torch.cuda.device_count() == 2:
        device = "cuda:1"

    # - Fixed parameters
    BATCH_SIZE_TRAIN = 100
    BATCH_SIZE_TEST = 500
    N_EPOCHS = 5
    LR = 1e-4

    base_dir = os.path.dirname(os.path.abspath(__file__))

    download_path = os.path.join(base_dir, "fmnist")
    train_set = torchvision.datasets.FashionMNIST(
        download_path,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5],std=[0.25])])
    )
    test_set = torchvision.datasets.FashionMNIST(
        download_path,
        download=True,
        train=False,
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5],std=[0.25])])
    )
    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=4
    )
    test_dataloader = DataLoader(
        dataset=test_set,
        batch_size=BATCH_SIZE_TEST,
        shuffle=False,
        num_workers=4
    )

    # - Create Torch network
    cnn = TorchCNN().to(device)
    cnn.apply(init_weights)

    # - Create adam instance for torch
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)

    # - Adversarial loss
    adv_loss = AdversarialLoss(
        model=cnn,
        natural_loss=torch.nn.CrossEntropyLoss(reduction="mean"),
        robustness_loss=torch.nn.KLDivLoss(reduction="batchmean"),
        device=device,
        n_attack_steps=10,
        mismatch_level=0.025,
        initial_std=1e-3,
        beta_robustness=0.25
    )

    for epoch_id in range(N_EPOCHS):
        for idx,(X,y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            robustness_loss = adv_loss.compute_gradient_and_backward(
                model=cnn,
                X=X,
                y=y
            )

            # - Backward does not need to be called
            # - Update the weights
            optimizer.step()

            # - Zero out the grads of the optimizer
            optimizer.zero_grad()

            if idx % 100 == 0:
                test_acc_no_noise, mean_noisy_test_acc = eval_test_set_mismatch(
                    test_dataloader,
                    cnn,
                    mismatch=0.2,
                    n_reps=5,
                    device=device
                )
                print("\n\nTest acc %.5f Mean noisy test acc %.5f" % (test_acc_no_noise,mean_noisy_test_acc))

            print("Epoch %d Batch %d/%d Loss %.5f" % (epoch_id,idx,len(train_dataloader),robustness_loss))