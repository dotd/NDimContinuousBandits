import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting

from torch.utils.tensorboard import SummaryWriter

from definitions_NDimContinuousBandits import ROOT_DIR
from src.synthetic.SyntheticNDimensional import SyntheticGaussianNumpy, SyntheticGaussianTorch


def tst_gauss_torch():
    tb_writer = SummaryWriter(
        log_dir=ROOT_DIR + "/tensorboard/runs/cusum_test_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    snd = SyntheticGaussianTorch(dim=2,
                                 num_gauss=6,
                                 std_gauss_min=0.01,
                                 std_gauss_max=0.01,
                                 dependance_factor=0,
                                 random=np.random.RandomState(2),
                                 std_factor=0.01)
    print(snd.centers)
    print(snd.vars)
    grid = snd.create_grid(31)
    print(f"grid shape {grid.shape}\n{grid}")
    value = snd.compute_function_on_batch(grid)
    print(f"value shape {value.shape}")

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = grid[:, 0].view(-1).numpy()
    y = grid[:, 1].view(-1).numpy()
    z = value.view(-1).numpy()

    ax.plot_trisurf(x, y, z, color='white', edgecolors='grey', alpha=0.5)
    ax.scatter(x, y, z, c='red')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def tst_gauss_numpy():
    tb_writer = SummaryWriter(
        log_dir=ROOT_DIR + "/tensorboard/runs/cusum_test_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    snd = SyntheticGaussianNumpy(dim=2,
                                 num_gauss=3,
                                 std_gauss_min=0.01,
                                 std_gauss_max=0.01,
                                 dependance_factor=0,
                                 random=np.random.RandomState(2),
                                 std_factor=0.01)
    print(snd.centers)
    print(snd.vars)
    grid = snd.create_grid(31)
    print(f"grid shape {grid.shape}\n{grid}")
    value = snd.compute_function_on_batch(grid)
    print(f"value shape {value.shape}")

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = grid[:, 0].reshape(-1)
    y = grid[:, 1].reshape(-1)
    z = value.reshape(-1)

    ax.plot_trisurf(x, y, z, color='white', edgecolors='grey', alpha=0.5)
    ax.scatter(x, y, z, c='red')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


if __name__ == "__main__":
    tst_gauss_numpy()
