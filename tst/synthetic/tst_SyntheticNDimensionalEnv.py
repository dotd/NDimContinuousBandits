import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor

from definitions_NDimContinuousBandits import ROOT_DIR
from src.synthetic.SyntheticNDimensionalEnv import SyntheticGaussianEnv


def show_2D_gaussians(x, y, z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(x, y, z, color='white', edgecolors='grey', alpha=0.5)
    ax.scatter(x, y, z, c='red')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def show_gaussians(env):
    grid = env.snd.create_grid(31)
    print(f"grid shape {grid.shape}\n{grid}")
    value = env.snd.compute_function_on_batch(grid)
    print(f"value shape {value.shape}")
    x = grid[:, 0].view(-1).numpy()
    y = grid[:, 1].view(-1).numpy()
    z = value.view(-1).numpy()
    show_2D_gaussians(x, y, z)


def tst_env():
    tb_writer = SummaryWriter(
        log_dir=ROOT_DIR + "/tensorboard/runs/gauss_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    gaussian_params = dict(dim=2,
                           num_gauss=6,
                           std_gauss_min=0.01,
                           std_gauss_max=0.01,
                           dependance_factor=0,
                           random=np.random.RandomState(2),
                           std_factor=0.01)
    env = SyntheticGaussianEnv(random=np.random.RandomState(2),
                               synthetic_gaussian_params=gaussian_params)
    # show_gaussians(env)
    env.reset()
    action = env.action_space.sample()
    print(action)
    grid = env.snd.create_grid(31)
    value = env.snd.compute_function_on_batch(grid)
    for i in range(grid.shape[0]):
        action = grid[i]
        _, reward, _, _ = env.step(action)
        print(f"reward = {reward} value={value[i]}")


if __name__ == "__main__":
    tst_env()
