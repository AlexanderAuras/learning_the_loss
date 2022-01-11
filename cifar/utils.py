import collections
import itertools
import random
from typing import Any, Dict, Tuple

import numpy as np

import torch

import matplotlib.colors
import matplotlib.pyplot as plt

def make_deterministic(seed: int) -> None:
    if seed == -1:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def render_matrix(matrix: torch.tensor, title: str) -> torch.tensor:
    figure = plt.figure()
    plt.imshow(matrix.transpose(), interpolation="nearest", cmap=plt.cm.inferno)
    plt.title(title)
    plt.colorbar()
    normalized = (matrix-matrix.min())/(matrix.max()-matrix.min())
    matrix = np.around(matrix, decimals=2)
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(i, j, matrix[i,j], horizontalalignment="center", color=("white" if normalized[i,j] < 0.5 else "black"))
    plt.tight_layout()
    plt.xticks(np.arange(matrix.shape[0]), np.arange(matrix.shape[0]))
    plt.yticks(np.arange(matrix.shape[1]), np.arange(matrix.shape[1]))
    return figure