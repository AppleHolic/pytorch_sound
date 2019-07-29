import matplotlib.pyplot as plt
import numpy as np


plt.switch_backend('Agg')


def plot_to_buf(x: np.ndarray, align: bool = True) -> np.ndarray:
    fig, ax = plt.subplots()
    ax.plot(x)
    if align:
        ax.set_ylim([-1, 1])
    fig.canvas.draw()
    im = np.array(fig.canvas.renderer._renderer)
    plt.clf()
    plt.close('all')
    return np.rollaxis(im[..., :3], 2)


def imshow_to_buf(x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 3:
        x = x[0]
    fig, ax = plt.subplots()
    ax.imshow(x, cmap='magma', aspect='auto')
    fig.canvas.draw()
    im = np.array(fig.canvas.renderer._renderer)
    plt.clf()
    plt.close('all')
    return np.rollaxis(im[..., :3], 2)
