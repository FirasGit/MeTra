import torch
import numpy as np


def _get_balanced_weights(labels):
    eps = 1e-7  # Added for numeric stability
    pos = np.sum(labels)
    neg = len(labels) - pos
    weights = torch.FloatTensor([neg / (pos + eps)])
    return weights
