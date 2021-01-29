import numpy as np


def transit(t, t0, duration, depth=1, c=20, period=1):
    _t = period * np.sin(np.pi * (t - t0) / period) / (np.pi * duration)
    return ((1 - depth) + (depth / 2) * (
            2 - np.tanh(c * (_t + 1 / 2)) + np.tanh(c * (_t - 1 / 2))
    ) - 1)[:, None]


def constant(x):
    return np.ones_like(x)[:, None]

def polynomial(x, order):
    return x[:, None]**np.arange(1, order+1)


def design_matrix(model_list):
    dm = np.hstack(model_list)
    dm /= np.mean(dm, 0)
    return dm
