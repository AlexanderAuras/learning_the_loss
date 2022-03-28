import functools
import itertools
import random
import typing

import numpy as np

import torch

import higher

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



def translate_to_zero(matrix: np.ndarray) -> torch.tensor:
    return matrix-np.tile(matrix.min(axis=1),(matrix.shape[0],1)).T



def render_matrix(matrix: np.ndarray, title: str, x_label: str = "", y_label:str = "") -> torch.tensor:
    figure = plt.figure()
    plt.imshow(matrix.transpose(), interpolation="nearest", cmap=plt.cm.inferno)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.colorbar()
    normalized = (matrix-matrix.min())/(matrix.max()-matrix.min())
    matrix = np.around(matrix, decimals=2)
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(i, j, matrix[i,j], horizontalalignment="center", color=("white" if normalized[i,j] < 0.5 else "black"))
    plt.tight_layout()
    plt.xticks(np.arange(matrix.shape[0]), np.arange(matrix.shape[0]))
    plt.yticks(np.arange(matrix.shape[1]), np.arange(matrix.shape[1]))
    return figure



def copy_higher_to_torch(higher_fmodel_state: higher.patch._MonkeyPatchBase, higher_diffopt_state: higher.optim.DifferentiableOptimizer, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    with torch.no_grad():
        model.load_state_dict(higher_fmodel_state)
    with torch.no_grad():
        for group_idx, entries in higher_diffopt_state.items():
            for entry_key, entry_value in entries.items():
                if torch.is_tensor(entry_value):
                    optimizer.state[optimizer.param_groups[0]["params"][group_idx]][entry_key].copy_(entry_value)
                else:
                    optimizer.state[optimizer.param_groups[0]["params"][group_idx]][entry_key] = entry_value



#https://arxiv.org/pdf/2007.08199v6.pdf
def label_gaussian_noise_transform(noise_transition_matrix: torch.tensor) -> typing.Callable[[torch.tensor,int],int]:
    return functools.partial(_lgnt, noise_transition_matrix)

def _lgnt(noise_transition_matrix: torch.tensor, y: int) -> int:
    p = torch.rand((1,))
    sum = 0.0
    for i in range(noise_transition_matrix.shape[1]):
        sum += noise_transition_matrix[y,i]
        if p <= sum:
            return i
    return noise_transition_matrix.shape[1]-1