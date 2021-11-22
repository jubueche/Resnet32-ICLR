# Repository for MSc thesis

## Setup
We suggest to setup a conda environment:
```
conda create -n msc python=3.7
pip install torch==1.9 torchvision ujson
```
### Optional:
The code is written mainly for torch. The original code for training robust networks was written in Jax however. The folder `loss_implementation_test` contains code for making sure the implementation in torch is correct. For this Jax is required:
```
pip install --upgrade pip
pip install --upgrade "jax[cuda111]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```
Please not that this donwloads Jax for the GPU (supported for linux only). If you need the CPU version, just do `pip install jax`.