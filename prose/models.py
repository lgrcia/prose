import numpy as np
from functools import reduce

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


try:
    import theano
    import theano.tensor as tt
    theano.config.compute_test_value = "warn"
except ModuleNotFoundError:
    pass


class LinearModel:

    def __init__(self, design_matrix):
        X = np.array(design_matrix).transpose()

        try:
            self.tX = tt.as_tensor(X)
        except NameError:
            print("exoplanet must be installed")

        svd = tt.nlinalg.SVD(full_matrices=False)

        self.U, S, self.V = svd(self.tX)
        S = tt.diag(S)
        self.S_ = tt.set_subtensor(S[tt.eq(S, 0.)], 1e10)

    def __call__(self, y):
        ty = y
        coeffs = reduce(tt.dot, [self.U.T, ty.T, 1.0 / self.S_, self.V])
        return tt.dot(coeffs, self.tX.T)

