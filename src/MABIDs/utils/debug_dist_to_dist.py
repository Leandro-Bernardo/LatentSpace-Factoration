import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

from typing import List, Tuple
from matplotlib.gridspec import GridSpec

def format_axes(fig, titles:str):
    for i, ax in enumerate(fig.axes):
        ax.text(-0.9, -0.9, titles[i], va="center", ha="center")
        ax.tick_params(labelbottom=True, labelleft=True)

def plt_pmfs(data:List[Tuple[str,np.array]], fname:str, roi:List):
    fig = plt.figure(layout="constrained")
    data_len = len(data)
    gs = GridSpec(2, data_len, figure=fig)

    axs = list()
    for i in range(data_len):
        axs.append(fig.add_subplot(gs[0,i]))

    for i in range(data_len):
        axs.append(fig.add_subplot(gs[1,i]))

    fig.suptitle("GridSpec")

    titles = list(map(lambda x: f"{x[0]}", data))
    titles += [f"mask_{s}" for s in titles]
    format_axes(fig, titles=titles)

    masked = list(map(lambda d: (d[0], np.where(d[1]>0, 255, 0)), data))
    data_ = data+masked

    # roi = normalized[..., self.input_roi[0, 0]:self.input_roi[0, 1] + 1, self.input_roi[1, 0]:self.input_roi[1, 1] + 1]
    # get_padding é recortado dessa maneira
    ys, xs = tuple(map(lambda x: (x[0],x[1]), roi))
    left = xs[0]
    top = ys[0]
    height = ys[1]+1-ys[0]
    width = xs[1]+1-xs[0]

    for i, ax in enumerate(axs):
        ax.add_patch(patches.Rectangle((left, top), width=width, height=height, linewidth=1, edgecolor='r', facecolor='none'))
        ax.imshow(data_[i][1])

    path = os.path.join(os.path.dirname(__file__), "debug_pmfs")
    if not os.path.isdir(path):
        os.makedirs(path)

    # plt.savefig(os.path.join(path, fname))
    plt.show()


# roi = [[108, 265], [248, 340]]

plt_pmfs(data=[
    ("lower", np.random.random((511,511))),
    ("target", np.random.random((511,511))),
    ("upper", np.random.random((511,511)))
], fname="testo", roi=[[108, 265], [248, 340]])