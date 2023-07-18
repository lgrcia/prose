import pickle
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prose import utils


def weights(
    fluxes: np.ndarray, tolerance: float = 1e-3, max_iteration: int = 200, bins: int = 5
):
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
    dfluxes = fluxes / np.expand_dims(np.nanmean(fluxes, -1), -1)

    def weight_function(fluxes):
        return 1 / np.std(fluxes, axis=-1)

    i = 0
    evolution = 1e25
    lcs = None
    weights = None
    last_weights = np.zeros(dfluxes.shape[0 : len(dfluxes.shape) - 1])

    # Broeg 2004 algorithm to find weights of comp stars
    # --------------------------------------------------
    while evolution > tolerance and i < max_iteration:
        if i == 0:
            weights = weight_function(dfluxes)
            mask = np.where(~np.isfinite(weights))
        else:
            # This metric is preferred from std to optimize over white noise and not red noise
            weights = weight_function(lcs)

        weights[~np.isfinite(weights)] = 0

        evolution = np.abs(
            np.nanmean(weights, axis=-1) - np.nanmean(last_weights, axis=-1)
        )

        last_weights = weights
        lcs = diff(dfluxes, weights=weights)

        i += 1

    weights[0, mask] = 0

    return weights[0]


def diff(fluxes: np.ndarray, weights: np.ndarray = None):
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
    diff_fluxes = fluxes / np.expand_dims(np.nanmean(fluxes, -1), -1)
    if weights is not None:
        # not to divide flux by itself
        sub = np.expand_dims((~np.eye(fluxes.shape[-2]).astype(bool)).astype(int), 0)
        weighted_fluxes = diff_fluxes * np.expand_dims(weights, -1)
        # see broeg 2005
        artificial_light_curve = (sub @ weighted_fluxes) / np.expand_dims(
            weights @ sub[0], -1
        )
        diff_fluxes = diff_fluxes / artificial_light_curve
    return diff_fluxes


def auto_diff_1d(fluxes, i=None):
    dfluxes = fluxes / np.expand_dims(np.nanmean(fluxes, -1), -1)
    w = weights(dfluxes)
    if i is not None:
        idxs = np.argsort(w)[::-1]
        white_noise = utils.binned_nanstd(dfluxes)
        last_white_noise = 1e10

        def best_weights(j):
            _w = w.copy()
            _w[idxs[j::]] = 0.0
            _w[i] = 0.0
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

        w = best_weights(j - 1)

    df = diff(dfluxes, w)

    return df.reshape(fluxes.shape), w


def auto_diff(fluxes: np.array, i: int = None):
    if fluxes.ndim == 3:
        auto_diffs = [auto_diff_1d(f, i) for f in fluxes]
        w = [a[1] for a in auto_diffs]
        fluxes = np.array([a[0] for a in auto_diffs])
        return fluxes, np.array(w)
    else:
        return auto_diff_1d(fluxes, i)


def optimal_flux(diff_fluxes, method="stddiff", sigma=4):
    fluxes = diff_fluxes.copy()
    fluxes = fluxes[
        ...,
        np.all(
            (fluxes - np.median(fluxes, 1)[..., None])
            < sigma * np.std(fluxes, 1)[..., None],
            0,
        ),
    ]
    if method == "binned":
        white_noise = utils.binned_nanstd(fluxes)
        criterion = white_noise(fluxes)
    elif method == "stddiff":
        criterion = utils.std_diff_metric(fluxes)
    elif method == "stability":
        criterion = utils.stability_aperture(fluxes)
    else:
        raise ValueError("{} is not a valid method".format(method))

    i = np.argmin(criterion)
    return i


@dataclass
class Fluxes:
    """Photometric fluxes, from single to multiple stars and apertures.

    Can hold other measurements time-series, errors and apertures properties.
    """

    fluxes: np.ndarray
    """Fluxes either as 1, 2, or 3 dimensional arrays, with following dimensions
        - 1: (time)
        - 2: (star, time)
        - 3: (aperture, star, time)
    """
    time: np.ndarray = None
    """Array of observed time"""
    errors: np.ndarray = None
    """Errors with same shape as :code:`fluxes`"""
    data: dict = None
    """A dict of data time-series, each with the same shape as :code:`time`"""
    apertures: np.ndarray = None
    """Apertures radii"""
    weights: np.ndarray = None
    """Fluxes weights (from differential photometry)"""
    target: int = None
    """Index of selected target"""
    aperture: int = None
    """Index of selected aperture"""
    metadata: dict = None
    """Metadata"""

    @property
    def _is_target_aperture_set(self):
        """Check if target and aperture are set depending on `fluxes` dimensions.

        Returns
        -------
        bool
            whether target and aperture are set
        """
        if self.ndim == 1:
            return True
        if self.ndim == 2:
            return self.target is not None
        else:
            return self.target is not None and self.aperture is not None

    def __post_init__(self):
        assert self.fluxes.ndim in [1, 2, 3], "fluxes must be 1, 2 or 3 dimensional"
        if self.data is None:
            self.data = {}
        if self.ndim == 1:
            self.target = 0
            self.aperture = 0
            self.fluxes = self.fluxes.copy()[None, None, :]
            if self.errors is not None:
                self.errors = self.errors.copy()[None, None, :]
        elif self.ndim == 2:
            self.aperture = 0
            self.fluxes = self.fluxes.copy()[None, :]
            if self.errors is not None:
                self.errors = self.errors.copy()[None, :]
        if self.metadata is None:
            self.metadata = {}

    def _target_attr(self, name, full=False):
        assert self.__dict__[name] is not None, f"{name} not provided"
        assert self.target is not None, "target must be set"
        if full:
            return self.__dict__[name][:, self.target]
        else:
            assert self.aperture is not None, "aperture must be set"
            return self.__dict__[name][self.aperture, self.target]

    @property
    def flux(self) -> np.array:
        """Main flux

        Returns
        -------
        np.array
            Main flux
        """
        return self._target_attr("fluxes")

    @property
    def error(self) -> np.array:
        """Error of the target flux"""
        return self._target_attr("errors")

    @property
    def shape(self):
        """shape of fluxes"""
        return self.fluxes.shape

    @property
    def ndim(self):
        """Number of dimensions of fluxes"""
        return self.fluxes.ndim

    @property
    def comparisons(self):
        """Comparison stars indices ordered from most to less weighted"""
        if self.weights is None:
            return None
        else:
            if self.aperture is None:
                raise ValueError("aperture must be set")

            idxs = np.argsort(self.weights[self.aperture])[::-1]
            return idxs[np.flatnonzero(self.weights[self.aperture][idxs] > 0.0)]

    def vander(consant=True, **kwargs):
        pass

    def diff(self, comps: np.ndarray = None):
        """Differential photometry

        Parameters
        ----------
        comps : np.ndarray, optional
            index of comparison stars, by default None

        Returns
        -------
        differential :code:`Fluxes`
        """
        if comps is not None:
            weights = np.zeros(self.fluxes[0:2])
            weights[:, comps] = 1
        else:
            weights = None

        diff_fluxes = diff(self.fluxes, weights)
        _new = deepcopy(self)
        _new.fluxes = diff_fluxes
        _new.weights = weights
        return _new

    def autodiff(self):
        """Automatic differential photometry with Broeg et al. 2005

        Returns
        -------
        differential :code:`Fluxes`
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            diff_fluxes, weights = auto_diff(self.fluxes, self.target)

        _new = deepcopy(self)
        _new.fluxes = diff_fluxes
        _new.weights = weights
        _new.aperture = _new.best_aperture_index()

        return _new

    def best_aperture_index(self, method="stddiff", sigma=4):
        """Find index of best aperture

        Parameters
        ----------
        method : str, optional
            Method to find best aperture, by default "stddiff"

        Returns
        -------
        int
            index of best aperture
        """
        i = optimal_flux(self._target_attr("fluxes", full=True), method, sigma=sigma)
        return i

    def estimate_best_aperture(
        self, target: int = None, method: str = "stddiff", sigma=4
    ):
        """Inplace setting of best aperture.

        Parameters
        ----------
        target : int, optional
            Index of target, by default None
        method : str, optional
            Method to find best aperture, by default "stddiff"
        """
        if target is None:
            target = self.target
        self.aperture = self.best_aperture_index(method=method, sigma=sigma)

    def _estimate_error(self):
        pass

    def plot(self, marker=".", color="0.8", ls="", ax=None, **kwargs):
        """Plot light curve

        Parameters
        ----------
        marker : str, optional
            Marker style, by default "."
        color : str, optional
            Marker color, by default "0.8"
        ls : str, optional
            Line style, by default ""
        ax : _type_, optional
            Matplotlib axis, by default None which takes :code:`plt.gca()`
        """
        if ax is None:
            ax = plt.gca()
        kwargs.update(dict(marker=marker, color=color, ls=ls))
        if self.time is None:
            ax.plot(self.flux, **kwargs)
        else:
            ax.plot(self.time, self.flux, **kwargs)

    def errorbar(self, color="k", fmt=".", **kwargs):
        """Error bar plot of the light curve

        Parameters
        ----------
        color : str, optional
            Marker color, by default "k"
        fmt : str, optional
            Error bar plot style, by default "."
        """
        kwargs.update(dict(color=color, fmt=fmt))
        plt.errorbar(self.time, self.flux, self.error, **kwargs)

    def bin(self, size: float, estimate_error: bool = False) -> "Fluxes":
        """Returns a :code:`Fluxes` instance with binned time series

        Parameters
        ----------
        size : float
            bin size in same unit as :code:`self.time`
        estimate_error : bool, optional
            whether to estimate error as the standard deviation of flux in each bin,
            by default False

        Returns
        -------
        _type_
            _description_
        """
        if isinstance(size, float):
            assert (
                self.time is not None
            ), "using a float bin size requires time to be set"

        time = self.time if self.time is not None else np.arange(self.fluxes.shape[-1])
        idxs = utils.index_binning(time, size)
        _new = deepcopy(self)

        _new.fluxes = np.array([np.mean(self.fluxes.T[i], 0) for i in idxs]).T

        if self.time is not None:
            _new.time = np.array([np.mean(self.time[i], 0) for i in idxs])
        if self.errors is not None:
            _new.errors = np.array(
                [np.mean(self.errors.T[i], 0) / np.sqrt(len(i)) for i in idxs]
            ).T

        if estimate_error:
            _new.errors = np.array(
                [np.std(self.fluxes.T[i], 0) / np.sqrt(len(i)) for i in idxs]
            ).T
        elif self.errors is not None:
            _new.errors = np.array(
                [np.sqrt(np.sum(np.power(self.errors[i], 2))) / len(i) for i in idxs]
            ).T

        return _new

    def save(self, path: Union[str, Path]):
        """Save fluxes to file

        Parameters
        ----------
        path : Union[str, Path]
            path of the file
        """
        with open(path, "wb") as f:
            pickle.dump(asdict(self), f)

    def load(path: Union[str, Path]):
        """Load fluxes from file"""
        with open(path, "rb") as f:
            return Fluxes(**pickle.load(f))

    def copy(self):
        """Deep copy of the object"""
        return deepcopy(self)

    @property
    def dataframe(self):
        """Pandas dataframe of the fluxes and associated measurements"""
        df_dict = self.data.copy()
        df_dict.update({"time": self.time})
        if self._is_target_aperture_set:
            df_dict.update({"flux": self.flux})

        return pd.DataFrame(df_dict)

    @property
    def df(self):
        """Pandas dataframe of the fluxes and associated measurements"""
        return self.dataframe

    def mask(self, array):
        """Mask time-dependant fluxes attributes (time, fluxes, errors, data)

        Parameters
        ----------
        m : np.array of bool
            mask

        Returns
        -------
        Fluxes
            masked Fluxes
        """
        _new = self.copy()
        _new.data = {key: value[array] for key, value in self.data.items()}
        if self.fluxes is not None:
            _new.fluxes = self.fluxes[..., array]
        if self.errors is not None:
            _new.errors = self.errors[..., array]
        if self.time is not None:
            _new.time = self.time[array]

        return _new

    def sigma_clipping_data(self, iterations: int = 5, **kwargs):
        """Return a Fluxes instance masked using iteratively sigma clipped data.

        Parameters
        ----------
        it : int, optional
            iterations, by default 5
        **kwargs: dict
            dict where keys are the names of the data to sigma clip and value are the
            sigma

        Returns
        -------
        Fluxes
            sigma clipped Fluxes
        """
        mask = np.ones_like(self.time).astype(bool)
        for _ in range(iterations):
            for name, sigma in kwargs.items():
                value = self.data[name].copy()
                value[~mask] = np.nan
                m = np.abs(value - np.nanmean(value)) < np.nanstd(value) * sigma
                mask = mask & m
        return self.mask(mask)

    def sigma_clip_flux(self, iterations: int = 5, sigma: float = 4.0):
        """Return a Fluxes instance masked using iteratively sigma clipping.

        Parameters
        ----------
        it : int, optional
            iterations, by default 5
        sigma: float
            sigma, by default 4.0

        Returns
        -------
        Fluxes
            sigma clipped Fluxes
        """
        flux = self.flux.copy()
        mask = np.ones_like(self.time).astype(bool)
        for _ in range(iterations):
            mask &= np.abs(flux - np.nanmean(flux)) < np.nanstd(flux[mask]) * sigma
        return self.mask(mask)

    def mask_stars(self, mask: np.array, keep_indexing: bool = True):
        """Mask stars fluxes.

        In order to keep indexing, the fluxes are set to -1.

        Parameters
        ----------
        mask : np.array
            A boolean array of the same length as the number of stars, indicating which fluxes should be masked
        remove : bool, optional
            whether to keep indexing (recommended) and only set pixels to 1, by default False

        Returns
        -------
        Fluxes
            A new Fluxes instance with masked stars
        """
        copy = self.copy()
        if not keep_indexing:
            copy.fluxes = self.fluxes[..., mask, :]
            if copy.errors is not None:
                copy.errors = self.errors[..., mask, :]
            if copy.weights is not None:
                copy.weights = self.weights[..., mask, :]
        else:
            copy.fluxes[:, ~mask] = -1
            if copy.errors is not None:
                copy.errors[:, ~mask] = -1
            if copy.weights is not None:
                copy.errors[:, ~mask] = 0
        return copy
