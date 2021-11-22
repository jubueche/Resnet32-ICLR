"""
Script for testing the correctness of the implementation of the loss function in torch compared to the one in Jax.
"""
# - For debugging
from copy import deepcopy
from jax import config
from numpy.lib.arraysetops import isin
config.FLAGS.jax_log_compiles=False
config.update('jax_disable_jit', True)
# - Specify hat platform to use
config.update('jax_platform_name', 'cpu')

# - Are we using GPU?
from jax.lib import xla_bridge
print("Using GPU", xla_bridge.get_backend().platform == "gpu")

from jax import lax
from jax.lax import stop_gradient
from jax.nn import softmax
from jax import device_put, value_and_grad, grad
from jax import random as jax_random
from jax import jit, partial
import numpy as onp
import jax.numpy as jnp
from jax.nn import relu
from jax.experimental import optimizers
# - Deterministic linear layer
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


class CNN:
    def call(self,input,K1,CB1,K2,CB2,W1,W2,W3,B1,B2,B3):
        if(input.size == 0):
            return jnp.array([]), jnp.array([[0]])
        cnn_out = _evolve_CNN(K1, CB1, K2, CB2, W1, W2, W3, B1, B2, B3, input)
        return cnn_out

@jit
def _evolve_CNN(K1,CB1,K2,CB2,W1,W2,W3,B1,B2,B3,P_input,):
    def MaxPool(mat,ksize=(2,2),method='max',pad=False):
        m, n = mat.shape[2:]
        ky,kx=ksize
        _ceil=lambda x,y: int(jnp.ceil(x/float(y)))
        if pad:
            ny=_ceil(m,ky)
            nx=_ceil(n,kx)
            size=(ny*ky, nx*kx)+mat.shape[2:]
            mat_pad=jnp.full(size,onp.nan)
            mat_pad[:m,:n,...]=mat
        else:
            ny=m//ky
            nx=n//kx
            mat_pad=mat[:, :, :ny*ky, :nx*kx]
        new_shape= mat.shape[:2] + (ny,ky,nx,kx)
        if method=='max':
            result=jnp.nanmax(mat_pad.reshape(new_shape),axis=(3,5))
        else:
            result=jnp.nanmean(mat_pad.reshape(new_shape),axis=(3,5))
        return result

    batch_size = P_input.shape[0]
    x = P_input
    strides = (1,1)
    x = lax.conv_general_dilated(
        x,
        K1,
        strides,
        padding ="SAME"
    )
    x = x + CB1
    x = relu(x)
    x = MaxPool(x)
    x = lax.conv_general_dilated(
        x,
        K2,
        strides,
        padding="VALID"
    )
    x = x + CB2
    x = relu(x)
    x = MaxPool(x)
    x = x.reshape(batch_size,-1)
    x = x @ W1 + B1
    x = relu(x)
    x = x @ W2 + B2
    x = relu(x)
    x = x @ W3 + B3
    return x

@jit
def categorical_cross_entropy(y, logits):
    """ Calculates cross entropy. y assumed to be vector of labels  (len=BS), i.e. [1,2,2,5,3]. logits are shape [BS,output_dim]"""
    logits_s = jnp.log(softmax(logits))
    nll = jnp.take_along_axis(logits_s, jnp.expand_dims(y, axis=1), axis=1)
    cce = -jnp.mean(nll)
    return cce

@jit
def loss_kl(logits, logits_theta_star):
    # - Apply softmax
    logits_s = softmax(logits)
    logits_theta_star_s = softmax(logits_theta_star)
    # - Assumes [BatchSize,Output] shape
    kl = jnp.mean(jnp.sum(logits_s * jnp.log(logits_s / jnp.where(logits_theta_star_s >= 1e-6, logits_theta_star_s, 1e-6) ), axis=1))
    return kl

def _split_and_sample_normal(key, shape):
    key, subkey = jax_random.split(key)
    val = jax_random.normal(subkey, shape=shape)
    return key, val

def _eval_target_loss(
    parameters,
    inputs,
    target,
    net,
    loss):
    # - Evolve the network
    output = net.call(inputs, **parameters)
    # - Calc. the loss
    return loss(target, output) # - outputs_theta, outputs_theta_star

@partial(
    jit,
    static_argnames=(
        "net",
        "mismatch_loss",
        "attack_steps",
        "mismatch_level",
        "initial_std",
    ),
)
def pga_attack(
    params,
    net,
    rng_key,
    inputs,
    net_out_original,
    mismatch_loss,
    attack_steps,
    mismatch_level = 0.025,
    initial_std = 1e-3,
):
    # - Create verbose dict
    verbose = {"grads": [], "losses": []}

    # - Initialize Theta* by adding Gaussian noise to each parameter
    theta_star = {}
    step_size = {}
    random_vars = {}
    for key in params:
        rng_key, random_normal_var = _split_and_sample_normal(rng_key, params[key].shape)
        # print("!! WARNING Using jnp.ones_like as random initial perturbation in pga attack")
        # random_normal_var = jnp.ones_like(params[key])
        random_vars[key] = random_normal_var
        theta_star[key] = params[key] + jnp.abs(params[key]) * initial_std * random_normal_var
        step_size[key] = (mismatch_level * jnp.abs(params[key])) / attack_steps

    # - Needed for analytical calc. of the gradient
    verbose["random_init"] = random_vars

    # - Perform gradient ascent on the parameters Theta*, with respect to the provided mismatch loss
    for _ in range(attack_steps):
        # - Compute loss and gradients
        loss, grads_theta_star = value_and_grad(_eval_target_loss)(
            theta_star, inputs, net_out_original, net, mismatch_loss
        )

        # - Store the loss and gradients for this iteration
        verbose["losses"].append(loss)
        verbose["grads"].append(grads_theta_star)

        # - Step each parameter in the direction of the gradient, scaled to the parameter scale
        for key in theta_star:
            theta_star[key] = theta_star[key] + step_size[key] * jnp.sign(
                grads_theta_star[key]
            )

    # - Return the attacked parameters
    return theta_star, verbose

def adversarial_loss(
    params,
    net,
    inputs,
    target,
    task_loss,
    mismatch_loss,
    rng_key,
    noisy_forward_std = 0.0,
    initial_std = 1e-3,
    mismatch_level = 0.025,
    beta_robustness = 0.25,
    attack_steps = 10,
):

    params_gaussian = {}
    for key in params:
        rng_key, random_normal_var = _split_and_sample_normal(rng_key, params[key].shape)
        params_gaussian[key] = params[key] + stop_gradient(jnp.abs(params[key]) * noisy_forward_std * random_normal_var)
        

    # - Evaluate the task loss using the perturbed parameters
    loss_n = _eval_target_loss(params_gaussian, inputs, target, net, task_loss)

    # - Get the network output using the original parameters
    output_theta = net.call(inputs, **params)

    # - Perform the adversarial attack to obtain the attacked parameters `theta_star`
    theta_star, _ = pga_attack(
        params=params,
        net=net,
        rng_key=rng_key,
        attack_steps=attack_steps,
        mismatch_level=mismatch_level,
        initial_std=initial_std,
        inputs=inputs,
        net_out_original=output_theta,
        mismatch_loss=mismatch_loss,
    )

    # - Compute robustness loss using the attacked parameters `theta_star`
    loss_r = _eval_target_loss(
        theta_star, inputs, output_theta, net, mismatch_loss
    )

    # - Add the robustness loss as a regularizer
    return loss_n + beta_robustness * loss_r

def value_and_compute_gradient_and_update(
    batch_id,
    X,
    y,
    opt_state,
    opt_update,
    get_params,
    net,
    rng_key,
    use_numerical
):
    params = get_params(opt_state)
    if use_numerical:
        v_and_g = numerical_value_and_grad
    else:
        v_and_g = value_and_grad(adversarial_loss)
    value, grads = v_and_g(
        params,
        net,
        X,
        y,
        categorical_cross_entropy,
        loss_kl,
        rng_key
    )
    return value, opt_update(batch_id, grads, opt_state)

def dict_op(op,A,B=None):
    C = {}
    for key in A:
        if B == None:
            C[key] = op(A[key])
        else:
            if isinstance(B, dict):
                C[key] = op(A[key],B[key])
            else:
                C[key] = op(A[key],B)
    return C

def numerical_value_and_grad(
    params,
    net,
    X,
    y,
    task_loss,
    mismatch_loss,
    rng_key,
    noisy_forward_std = 0.0,
    initial_std = 1e-3,
    mismatch_level = 0.025,
    beta_robustness = 0.25,
    attack_steps = 10
):
    params_gaussian = {}
    for key in params:
        rng_key, random_normal_var = _split_and_sample_normal(rng_key, params[key].shape)
        params_gaussian[key] = params[key] + stop_gradient(jnp.abs(params[key]) * noisy_forward_std * random_normal_var)
        
    # - Evaluate the task loss using the perturbed parameters
    value_task_loss, grads_task_loss = value_and_grad(_eval_target_loss)(
        params_gaussian,
        X,
        y,
        net,
        task_loss
    )

    # - Perform the pga attack and save the gradients during the forward pass of the attack
    # - Get the network output using the original parameters
    output_theta = net.call(X, **params)

    # - Perform the adversarial attack to obtain the attacked parameters `theta_star`
    theta_star, verbose = pga_attack(
        params=params,
        net=net,
        rng_key=rng_key,
        attack_steps=attack_steps,
        mismatch_level=mismatch_level,
        initial_std=initial_std,
        inputs=X,
        net_out_original=output_theta,
        mismatch_loss=mismatch_loss,
    )
    # - verbose = {"grads": ..., "losses": ...}

    value_robustness_loss, grads_theta_star = value_and_grad(_eval_target_loss)(
        theta_star, X, output_theta, net, mismatch_loss
    )

    # - Compute d L / d Theta
    def f(theta):
        output_theta = net.call(X, **theta)
        return _eval_target_loss(theta_star, X, output_theta, net, mismatch_loss)

    grads_theta_rob_loss = grad(f)(params)

    # - Calculate the gradients analytically
    sum_signed_grads = {k: jnp.zeros_like(params[k]) for k in params}
    for t in range(len(verbose["grads"])):
        sum_signed_grads = dict_op(jnp.add, sum_signed_grads, dict_op(jnp.sign,verbose["grads"][t]))

    J = dict_op(
        jnp.add,
        dict_op(
        jnp.multiply,
        dict_op(jnp.sign,params),
        dict_op(
            jnp.add,
            dict_op(
                jnp.multiply,
                sum_signed_grads,
                mismatch_level / attack_steps
            ),
            dict_op(
                jnp.multiply,
                verbose["random_init"],
                initial_std
            )
        )),
        1.0
    )
    
    grads_robustness = dict_op(jnp.multiply, dict_op(jnp.add, dict_op(jnp.multiply, J, grads_theta_star), grads_theta_rob_loss), beta_robustness)
    final_value = value_task_loss + beta_robustness * value_robustness_loss
    final_grads = dict_op(jnp.add, grads_task_loss, grads_robustness)
    return final_value, final_grads

def jax_eval_test_set(
    test_dataloader,
    net,
    params
):
    N_correct = 0
    N = 0
    for (X,y) in test_dataloader:
        X_jax = device_put(X.numpy())
        y_jax = device_put(y.numpy())
        y_hat = jnp.argmax(net.call(X_jax, **params), axis=1)
        N += len(y_jax)
        N_correct += jnp.sum(jnp.array(y_hat == y_jax, dtype=int))
    return N_correct / N

def jax_eval_test_set_mismatch(
    test_dataloader,
    net,
    params,
    mismatch,
    n_reps,
    rng_key
):
    test_acc_no_noise = jax_eval_test_set(test_dataloader, net, params)
    test_accs = []
    for idx in range(n_reps):
        print("Test eval. mismatch rob. %d/%d" % (idx,n_reps))
        theta_star = {}
        for key in params:
            rng_key, random_normal_var = _split_and_sample_normal(rng_key, params[key].shape)
            theta_star[key] = params[key] + jnp.abs(params[key]) * mismatch * random_normal_var
        test_accs.append(jax_eval_test_set(test_dataloader, net, theta_star))
    return test_acc_no_noise, onp.mean(test_accs), rng_key

if __name__ == '__main__':
    torch.manual_seed(0)
    # - Set the Jax seed
    rng_key = jax_random.PRNGKey(0)
    # - Set numpy seed
    onp.random.seed(0)
    # - Avoid reprod. issues caused by GPU
    torch.use_deterministic_algorithms(True)
    # - Select device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # - Select which device
    if torch.cuda.device_count() == 2:
        device = "cuda:1"

    # - Fixed parameters
    BATCH_SIZE_TRAIN = 500
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

    # - Create CNN
    cnn_jax = CNN()

    # - Parameter initialization
    _, *sks = jax_random.split(rng_key, 10)
    K1 = onp.array(jax_random.truncated_normal(sks[1],-2,2,(64,1,4,4))* (onp.sqrt(6/(64*4+64*4))))
    K2 = onp.array(jax_random.truncated_normal(sks[2],-2,2,(64,64,4,4))* (onp.sqrt(6/(64*64*4 + 64*64*4))))
    CB1 = onp.array(jax_random.truncated_normal(sks[6],-2,2,(1,64,1,1))* (onp.sqrt(6/(64*1*4 + 64*1*4))))
    CB2 = onp.array(jax_random.truncated_normal(sks[7],-2,2,(1,64,1,1))* (onp.sqrt(6/(64*64*4 + 64*64*4))))
    W1 = onp.array(jax_random.truncated_normal(sks[3],-2,2,(1600, 256))* (onp.sqrt(6/(1600 + 256))))
    W2 = onp.array(jax_random.truncated_normal(sks[4],-2,2,(256, 64))* (onp.sqrt(6/(256 + 64))))
    W3 = onp.array(jax_random.truncated_normal(sks[4],-2,2,(64, 10))* (onp.sqrt(6/(64 + 10))))
    B1 = onp.zeros((256,))
    B2 = onp.zeros((64,))
    B3 = onp.zeros((10,))

    # - Init params for the CNN
    init_params = {"K1": K1, "CB1": CB1, "K2": K2, "CB2": CB2, "W1": W1, "W2": W2, "W3": W3, "B1": B1, "B2": B2, "B3": B3}

    opt_init, opt_update, get_params = optimizers.adam(LR, 0.9, 0.999, 1e-08)
    opt_state = opt_init(init_params)

    for epoch_id in range(N_EPOCHS):

        for idx,(X,y) in enumerate(train_dataloader):
            params = get_params(opt_state)
            X_jax = device_put(X.numpy())
            y_jax = device_put(y.numpy())

            loss, opt_state = value_and_compute_gradient_and_update(
                batch_id=idx,
                X=X_jax,
                y=y_jax,
                opt_state=opt_state,
                opt_update=opt_update,
                get_params=get_params,
                net=cnn_jax,
                rng_key=rng_key,
                use_numerical=False
            )

            # - Split the random key
            rng_key = jax_random.split(rng_key)

            # if idx % 100 == 0:
            #     test_acc_no_noise, mean_noisy_test_acc, rng_key = jax_eval_test_set_mismatch(
            #         test_dataloader,
            #         cnn_jax,
            #         params,
            #         mismatch=0.2,
            #         n_reps=5,
            #         rng_key=rng_key
            #     )
            #     print("\n\nTest acc %.5f Mean noisy test acc %.5f" % (test_acc_no_noise,mean_noisy_test_acc))

            print("Epoch %d Batch %d/%d Loss %.5f" % (epoch_id,idx,len(train_dataloader),loss))