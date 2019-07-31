import matplotlib.pyplot as plt
import numpy as np


plt.switch_backend('Agg')


#
# Make image array with using matplotlib
#
def plot_to_buf(x: np.ndarray, align: bool = True) -> np.ndarray:
    """
    make plotted image given array
    :param x: an array to be plotted
    :param align: make limit from -1 to +1 on array
    :return: plotted image
    """
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
    """
    make image given array
    :param x: an array to be painted
    :return: painted image
    """
    if len(x.shape) == 3:
        x = x[0]
    fig, ax = plt.subplots()
    ax.imshow(x, cmap='magma', aspect='auto')
    fig.canvas.draw()
    im = np.array(fig.canvas.renderer._renderer)
    plt.clf()
    plt.close('all')
    return np.rollaxis(im[..., :3], 2)
