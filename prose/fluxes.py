import numpy as np
from . import utils
from dataclasses import dataclass, asdict
from copy import deepcopy
import matplotlib.pyplot as plt
import warnings
import pickle


def binned_white_function(x, bins: int=12):
    # set binning idxs for white noise evaluation
    bins = np.min([x.shape[-1], bins])
    n = x.shape[-1] // bins
    idxs = np.arange(n * bins)

    def compute(f):
            return np.nanmean(np.nanstd(np.array(np.split(f.take(idxs, axis=-1), n, axis=-1)), axis=-1), axis=0)
    
    return compute

def weights(fluxes: np.ndarray, tolerance: float=1e-3, max_iteration: int=200, bins: int=5):
    """Returns the weights computed using Broeg 2005

    Parameters
    ----------
    fluxes : np.ndarray
        fluxes matrix with dimensions (star, flux) or (aperture, star, flux)
    tolerance : float, optional
        the minimum standard deviation of weights difference to attain (meaning weights are stable), by default 1e-3
    max_iteration : int, optional
        maximum number of iterations to compute weights, by default 200
    bins : int, optional
        binning size (in number of points) to compute the white noise, by default 5

    Returns
    -------
    np.ndarray
        Broeg weights
    """
    
    # normalize
    dfluxes = fluxes/np.expand_dims(np.nanmean(fluxes, -1), -1)
    binned_white = binned_white_function(fluxes, bins=bins)
    
    i = 0
    evolution = 1e25
    lcs = None
    weights = None
    last_weights = np.zeros(dfluxes.shape[0:len(dfluxes.shape) - 1])

    # Broeg 2004 algorithm to find weights of comp stars
    # --------------------------------------------------
    while evolution > tolerance and i < max_iteration:
        if i == 0:
            weights = 1 / binned_white(dfluxes)
        else:
            # This metric is preferred from std to optimize over white noise and not red noise
            std = binned_white(lcs)
            weights = 1 / std

        weights[~np.isfinite(weights)] = 0

        # Keep track of weights
        evolution = np.nanstd(np.abs(np.nanmean(weights, axis=-1) - np.nanmean(last_weights, axis=-1)))

        last_weights = weights
        lcs = diff(dfluxes, weights=weights)
        i += 1
        
    return weights

def diff(fluxes: np.ndarray, weights: np.ndarray=None):
    """Returns differential fluxes. 
    
    If weights are specified, they are used to produce an artificial light curve by which all flux are differentiated (see Broeg 2005)

    Parameters
    ----------
    fluxes : np.ndarray
        fluxes matrix with dimensions (star, flux) or (aperture, star, flux)
    weights :np.ndarray, optional
        weights matrix with dimensions (star) or (aperture, star), by default None which simply returns normalized fluxes

    Returns
    -------
    np.ndarray
        Differential fluxes if weights is provided, else normalized fluxes
    """
    diff_fluxes = fluxes/np.expand_dims(np.nanmean(fluxes, -1), -1)
    if weights is not None:
        # not to divide flux by itself
        sub = np.expand_dims((~np.eye(fluxes.shape[-2]).astype(bool)).astype(int), 0)
        weighted_fluxes = diff_fluxes * np.expand_dims(weights, -1)
        # see broeg 2005
        artificial_light_curve = (sub @ weighted_fluxes) / np.expand_dims(weights @ sub[0], -1)
        diff_fluxes = diff_fluxes / artificial_light_curve
    return diff_fluxes

def auto_diff_1d(fluxes, i=None):
    dfluxes = fluxes / np.expand_dims(np.nanmean(fluxes, -1), -1)
    w = weights(dfluxes)
    if i is not None:
        idxs = np.argsort(w)[::-1]
        white_noise = binned_white_function(dfluxes)
        last_white_noise = 1e10
        
        def best_weights(j):
            _w = w.copy()
            _w[idxs[j::]] = 0.
            _w[i] = 0.
            return _w
        
        for j in range(w.shape[-1]):
            _w = best_weights(j)
            _df = diff(dfluxes, _w)
            _white_noise = np.take(white_noise(_df), i, axis=-1)[0]
            if not np.isfinite(_white_noise):
                continue
            if _white_noise < last_white_noise:
                last_white_noise = _white_noise
            else:
                break
        
        w = best_weights(j-1)

    df = diff(dfluxes, w)
        
    return df.reshape(fluxes.shape), w

def auto_diff(fluxes: np.array, i:int=None):
    if fluxes.ndim == 3:
        auto_diffs = [auto_diff_1d(f, i) for f in fluxes]
        w = [a[1] for a in auto_diffs]
        fluxes = np.array([a[0] for a in auto_diffs])
        return fluxes, np.array(w)
    else:
        return auto_diff_1d(fluxes, i)


def optimal_flux(diff_fluxes, method="stddiff"):
    if method == "binned":
        white_noise = binned_white_function(diff_fluxes)
        criterion = white_noise(diff_fluxes)
    elif method == "stddiff":
        criterion = utils.std_diff_metric(diff_fluxes)
    elif method == "stability":
        criterion = utils.stability_aperture(diff_fluxes)
    else:
        raise ValueError("{} is not a valid method".format(method))

    i = np.argmin(criterion)
    return i


@dataclass
class Fluxes:
    fluxes: np.ndarray
    time: np.ndarray = None
    errors: np.ndarray = None
    data: dict = None
    apertures: np.ndarray = None
    weights: np.ndarray = None
    target: int = None
    aperture: int = None
    
    def __post_init__(self):
        assert self.fluxes.ndim in [2, 3], "fluxes must be 2 or 3 dimensional"
        if self.data is None:
            self.data = {}
        
    def _target_attr(self, name, full=False):
        assert self.__dict__[name] is not None, f"{name} not provided"
        assert self.target is not None, "target must be set"
        # if self.ndim == 1:
        #    return self.__dict__[name]
        if self.ndim == 2:
            if full:
                return self.__dict__[name]
            else:
                return self.__dict__[name][self.target]
        else:
            if full:
                return self.__dict__[name][:, self.target]
            else:
                assert self.aperture is not None, "aperture must be set"
                return self.__dict__[name][self.aperture, self.target]
            
    @property
    def flux(self):
        return self._target_attr("fluxes")

    @property
    def error(self):
        return self._target_attr("errors")
    
    @property
    def shape(self):
        return self.fluxes.shape
    
    @property
    def ndim(self):
        return self.fluxes.ndim
    
    def vander(consant=True, **kwargs):
        pass

    def diff(self, comps: np.ndarray=None):
        if comps is not None:
            if self.ndim == 2:
                weights = np.zeros((self.fluxes[0]))
                weights[comps] = 1
            elif self.ndim == 3:
                weights = np.zeros(self.fluxes[0:2])
                weights[:, comps] = 1
        else:
            weights = None
        
        diff_fluxes = diff(self.fluxes, weights)
        _new = deepcopy(self)
        _new.fluxes = diff_fluxes
        _new.weights = weights
        return _new
    
    def autodiff(self, target=None):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            diff_fluxes, weights = auto_diff(self.fluxes, target)
            
        _new = deepcopy(self)
        _new.fluxes = diff_fluxes
        _new.weights = weights
        _new.target = target
        _new.aperture = _new.best_aperture_index()
        
        return _new
    
    def best_aperture_index(self, method="stddiff"):
        i = optimal_flux(self._target_attr("fluxes", full=True), method)
        return i
    
    def estimate_best_aperture(self, target=None, method="stddiff"):
        self.target = target
        self.aperture = self.best_aperture_index(method=method)

    def estimate_error(self):
        pass
    
    def plot(self, marker=".", color="0.8", ls="", **kwargs):
        kwargs.update(dict(marker=marker, color=color, ls=ls))
        if self.time is None:
            plt.plot(self.flux, **kwargs)
        else:
            plt.plot(self.time, self.flux, **kwargs)
    
    def errorbar(self, color="k", fmt=".", **kwargs):
        kwargs.update(dict(color=color, fmt=fmt))
        plt.errorbar(self.time, self.flux, self.error, **kwargs)
    
    def bin(self, size, estimate_error=False):
        if isinstance(size, float):
            assert self.time is not None, "using a float bin size requires time to be set"
            
        time = self.time if self.time is not None else np.arange(self.fluxes.shape[-1])
        idxs = utils.index_binning(time, size)
        _new = deepcopy(self)
        
        _new.fluxes = np.array([np.mean(self.fluxes.T[i], 0) for i in idxs]).T
        
        if self.time is not None:
            _new.time = np.array([np.mean(self.time[i], 0) for i in idxs])
        if self.errors is not None:
            _new.errors = np.array([np.mean(self.errors.T[i], 0)/np.sqrt(len(i)) for i in idxs]).T
        
        if estimate_error:
            _new.errors = np.array([np.std(self.fluxes.T[i], 0)/np.sqrt(len(i)) for i in idxs]).T
        elif self.errors is not None:
            _new.errors = np.array([np.sqrt(np.sum(np.power(self.errors[i], 2)))/len(i) for i in idxs]).T
            
        return _new

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(asdict(self), f)
        
    def load(path):
        with open(path, "rb") as f:
            return Fluxes(**pickle.load(f))