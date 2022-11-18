import numpy as np
from . import utils
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import xarray as xr
from itertools import product
from tqdm import tqdm
from . import models
from . import viz
from .console_utils import info

np.seterr(divide='ignore', invalid='ignore')

def nu(n, sw, sr):
    return (sw ** 2) / n + sr ** 2


def append_tex_order(current, n, name):
    _str = r""
    if n == 0:
        return current
    elif n == 1:
        _str += name
    elif n > 1:
        _str += fr"{name}^{n}"
    current.append(_str)


def pont2006(x, y, n=30, plot=True, return_error=False):
    dt = np.median(np.diff(x))
    ns = np.arange(1, n)
    binned = [utils.fast_binning(x, y, dt * _n)[1] for _n in ns]
    _nu = [np.var(b) for b in binned]
    (sw, sr), pcov = curve_fit(nu, ns, _nu)
    if plot:
        plt.plot(ns, np.std(y) / np.sqrt(ns), ":k", label=r"$\sigma / \sqrt{n}$")
        plt.plot(ns, np.sqrt(nu(ns, sw, sr)), "k", label="fit")
        plt.plot(ns, np.sqrt(_nu), "d", markerfacecolor="w", markeredgecolor="k")
        plt.xlabel("n points")
        plt.ylabel("$\\nu^{1/2}(n)$")
        plt.legend()
    if return_error:
        return sw, sr, np.sqrt(np.diag(pcov))
    else:
        return sw, sr


def binned_std(fluxes, bins=12):
    bins = np.min([fluxes.shape[-1], bins])
    n = fluxes.shape[-1] // bins
    idxs = np.arange(n * bins)
    return np.array(np.split(fluxes.take(idxs, axis=-1), n, axis=-1)).std(-1).mean(0)


def diff(fluxes, errors=None, weights=None, comps=None, alc=False):
    
    # not to divide flux by itself
    sub = np.expand_dims((~np.eye(fluxes.shape[-2]).astype(bool)).astype(int), 0)

    assert (weights is None) ^ (comps is None), "either weights or comps must be specified"
    if comps is not None:
        weights = np.zeros(fluxes.shape[0:-1])
        weights[np.arange(fluxes.shape[0]), comps.T] = 1.

    # Formulae
    # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Standard_error

    weighted_fluxes = fluxes * np.expand_dims(weights, -1)
    art_lc = (sub @ weighted_fluxes) / np.expand_dims(weights @ sub[0], -1)
    lcs = fluxes / art_lc

    if not alc and errors is None:
        return lcs

    if errors is not None:
        weighted_errors = errors**2 * np.expand_dims(weights, -1)**2
        squarred_art_error = (sub @ weighted_errors) / np.expand_dims(weights**2 @ sub[0], -1)
        lcs_errors = np.sqrt(errors ** 2 + squarred_art_error)
        returns = [lcs, lcs_errors]

        if alc:
            return [lcs, lcs_errors, art_lc]
        else:
            return [lcs, lcs_errors]
    else:
        return [lcs, art_lc]

def broeg(fluxes, tolerance=1e-2, max_iteration=200, bins=12):

    bins = np.min([fluxes.shape[-1], bins])
    n = fluxes.shape[-1] // bins
    idxs = np.arange(n * bins)

    def error_estimate(f):
        return np.nanmean(np.nanstd(np.array(np.split(f.take(idxs, axis=-1), n, axis=-1)), axis=-1), axis=0)

    i = 0
    evolution = 1e25
    lcs = None
    weights = None
    last_weights = np.zeros(fluxes.shape[0:len(fluxes.shape) - 1])

    # Broeg 2004 algorithm to find weights of comp stars
    # --------------------------------------------------
    while evolution > tolerance and i < max_iteration:
        if i == 0:
            weights = 1 / error_estimate(fluxes)
        else:
            # This metric is preferred from std to optimize over white noise and not red noise
            std = error_estimate(lcs)
            weights = 1 / std

        weights[~np.isfinite(weights)] = 0

        # Keep track of weights
        evolution = np.nanstd(np.abs(np.nanmean(weights, axis=-1) - np.nanmean(last_weights, axis=-1)))

        last_weights = weights
        lcs = diff(fluxes, weights=weights)
        i += 1

    return weights


def best_stars(fluxes, weights, target, return_idxs=True, bins=12):

    bins = np.min([fluxes.shape[-1], bins])
    b = fluxes.shape[-1] // bins
    idxs = np.arange(b * bins)

    def error_estimate(f):
        return np.array(np.split(f.take(idxs, axis=-1), b, axis=-1)).std(-1).mean(0)

    whites = []
    white = 1e12

    _weights = weights.copy()
    _weights[:, target] = 0
    sorted_weights = np.argsort(_weights)[:, ::-1]

    for n in np.arange(1, fluxes.shape[-2]):
        w = np.zeros(fluxes.shape[-2])
        w[sorted_weights[:, 0:n][::-1]] = 1
        _white = error_estimate(diff(fluxes.copy(), weights=w)[:, target]).min()
        if _white < white:
            white = _white
        else:
            break

    if return_idxs:
        return sorted_weights[:, 0:n]
    else:
        sub = np.zeros(fluxes.shape[0:len(fluxes.shape) - 1])
        for i, j in enumerate(sorted_weights[:, 0:n][::-1]):
            sub[i, j] = 1.
        _weights *= sub
        return _weights


def scargle(time, flux, error, periods, X=None, n=1):

    if X is None:
        X = np.atleast_2d(np.ones_like(time))

    variability = lambda p: models.harmonics(time, p, n)

    chi2 = []

    for p in tqdm(periods):
        x = models.design_matrix([X.T, *variability(p)])
        residuals = x @ np.linalg.lstsq(x, flux, rcond=None)[0] - flux
        chi2.append(-np.sum((residuals / error) ** 2))

    chi2 = np.array(chi2)

    return chi2

    # idxs = np.argwhere(np.abs(chi2) < 30*np.std(chi2)).flatten()
    # periods = periods[idxs]
    # chi2 = chi2[idxs]


class ApertureFluxes:

    def __init__(self, xarray):
        if isinstance(xarray, (str, Path)):
            self.xarray = xr.load_dataset(xarray, engine="netcdf4")
        else:
            self.xarray = xarray

        # backward compatibility
        self._fix_fluxes()

    def __getattr__(self, name):
        if name in self.xarray or name in self.xarray.dims:
            return self.xarray.get(name).values
        elif name in self.xarray.attrs:
            return self.xarray.attrs[name]
        else:
            raise AttributeError(f"{self.__class__.__name__} object has no attribute '{name}'")

    @property
    def target(self):
        return self.xarray.attrs.get('target', -1)
        #TODO: target not set error over all Observation function using target


    @target.setter
    def target(self, value):
        self.xarray.attrs['target'] = value
        if "diff_fluxes" in self:
            self.pick_best_aperture()

    @property
    def aperture(self):
        return self.xarray.attrs.get('aperture', -1)
        #TODO: aperture not set error over all Observation 

    @aperture.setter
    def aperture(self, value):
        self.xarray.attrs['aperture'] = value

    def _repr_html_(self):
        return self.xarray._repr_html_()

    def __str__(self):
        return self.xarray.__str__()

    def __iter__(self):
        return self.xarray.__iter__()

    def mask(self, mask, dim="time"):
        new_self = self.copy()
        new_self.xarray = self.x.sel(**{dim: self.x[dim][mask]})
        return new_self

    @staticmethod
    def _binn(var, *bins):
        if var.dtype.name != 'object':
            if "time" in var.dims:
                if "errors" in var.name:
                    return xr.concat(
                        [(np.sqrt(np.power(var.isel(time=b), 2).sum(dim="time")) / len(b)).expand_dims('time', -1) for b
                         in bins], dim="time")
                else:
                    return xr.concat([var.isel(time=b).mean(dim="time").expand_dims('time', -1) for b in bins],
                                     dim="time")
            else:
                return var
        else:
            return xr.DataArray(np.ones(len(bins)) * -1, dims="time")

    def binn(self, dt, std=False, keep_coords=True):
        x = self.xarray.copy()
        bins = utils.index_binning(self.time, dt)
        new_time = np.array([self.time[b].mean() for b in bins])

        x = x.map(self._binn, args=(bins), keep_attrs=True)

        if std:
            if "diff_fluxes" in self.xarray:
                flu = self.xarray.diff_fluxes.copy()
                x['diff_errors'] = xr.concat(
                    [(flu.isel(time=b).std(dim="time") / np.sqrt(len(b))).expand_dims('time', -1) for b in bins],
                    dim="time")
            if "raw_fluxes" in self.xarray:
                raw_flu = self.xarray.raw_fluxes.copy()
                x['raw_errors'] = xr.concat(
                    [(raw_flu.isel(time=b).std(dim="time") / np.sqrt(len(b))).expand_dims('time', -1) for b in bins],
                    dim="time")

        x.coords['time'] = new_time

        if keep_coords:
            for name, value in self.xarray.coords.items():
                if "time" not in value.dims:
                    x.coords[name] = value

        new_self = self.copy()
        new_self.xarray = x

        return new_self

    @property
    def x(self):
        return self.xarray

    @property
    def has_diff(self):
        return hasattr(self, "diff_flux")

    @property
    def diff_flux(self):
        return self.xarray.diff_fluxes.isel(apertures=self.aperture, star=self.target).values

    @property
    def raw_flux(self):
        return self.xarray.raw_fluxes.isel(apertures=self.aperture, star=self.target).values

    @property
    def diff_error(self):
        return self.xarray.diff_errors.isel(apertures=self.aperture, star=self.target).values

    @property
    def raw_error(self):
        return self.xarray.raw_errors.isel(apertures=self.aperture, star=self.target).values

    @property
    def detrended_diff_flux(self):
        if "trends" in self:
            return self.diff_flux - self.trend + 1
        else:
            return None

    @property
    def trend(self):
        if "trends" in self:
            return self.xarray.trends[self.target].values
        else:
            return None

    @property
    def comparison_raw_fluxes(self):
        return self.raw_fluxes[self.aperture, self.comps[self.aperture]]

    @property
    def X(self):
        if "polynomial_trend_orders" in self.xarray.attrs:
            orders = list(zip(
                self.xarray.attrs["polynomial_trend_variables"],
                self.xarray.attrs["polynomial_trend_orders"])
            )
            return self.polynomial(**dict(orders))
        else:
            return None

    def __copy__(self):
        return self.__class__(self.xarray.copy())

    def copy(self):
        return self.__class__(self.xarray.copy())

    def pick_best_aperture(self, method="binned", return_criterion=False):

        diff_fluxes = self.xarray.sel(star=self.target).diff_fluxes.values

        if len(self.apertures) > 1:
            if method == "binned":
                criterion = binned_std(diff_fluxes)
            elif method == "stddiff":
                criterion = utils.std_diff_metric(diff_fluxes)
            elif method == "stability":
                criterion = utils.stability_aperture(diff_fluxes)
            elif method == "pont2006":
                criterion = []
                for a in range(len(self.apertures)):
                    self.aperture = a
                    criterion.append(self.pont2006(plot=False)[0])
            else:
                raise ValueError("{} is not a valid method".format(method))

            self.aperture = np.argmin(criterion)

            if return_criterion:
                return criterion
        else:
            self.aperture = 0

    def where(self, *args, **kwargs):
        new_self = self.copy()
        new_self.xarray = self.xarray.where(*args, **kwargs)
        return new_self

    def pont2006(self, plot=True):
        return pont2006(self.time, self.xarray.diff_fluxes.isel(apertures=self.aperture, star=self.target).values, plot=plot)

    def _fix_fluxes(self):
        if "raw_fluxes" not in self and "fluxes" in self:
            self.xarray = self.xarray.rename({
                "fluxes": "raw_fluxes",
                "errors": "raw_errors"
            })
        elif "fluxes" in self:
            self.xarray = self.xarray.rename({
                "fluxes": "diff_fluxes",
                "errors": "diff_errors"
            })

    # Differential photometry methods
    # ===============================
    def diff(self, comps, inplace=True):
        """Differential photometry based on a set of comparison stars

        The artificial light-curve is taken as the mean of comparison stars

        Parameters
        ----------
        comps : list
            indexes of the comparison stars (as shown in `show_stars`, same indexes as `stars`)
        inplace: bool, optional
            whether to perform the changes on current Observation or to return a new one, default True

        Returns
        -------
        [type]
            [description]
        """

        comps = np.array(comps)

        if inplace:
            new_self = self
        else:
            new_self = self.copy()

        # getting differential values
        raw_fluxes = self.raw_fluxes.copy()
        raw_errors = self.raw_errors.copy()

        mean_raw_fluxes = np.expand_dims(raw_fluxes.mean(-1), -1)
        raw_errors /= mean_raw_fluxes
        raw_fluxes /= mean_raw_fluxes

        comps = np.repeat(np.atleast_2d(comps), len(self.apertures), axis=0)

        diff_fluxes, diff_errors, alcs = diff(raw_fluxes, raw_errors, comps=comps, alc=True)

        dims = self.xarray.raw_fluxes.dims

        new_self.xarray["diff_fluxes"] = (dims, diff_fluxes)
        new_self.xarray["diff_errors"] = (dims, diff_errors)

        # Since we reset ncomps, older vars with ncomp in dims are removed
        new_self.xarray = new_self.xarray.drop_vars(
            [name for name, value in new_self.xarray.items() if 'ncomps' in value.dims])

        new_self.xarray['comps'] = (("apertures", "ncomps"), comps)
        new_self.xarray['weights'] = (("apertures", "ncomps"), np.ones_like(comps))
        new_self.xarray['alc'] = (('apertures', 'time'), alcs[:, self.target])
        new_self.xarray.attrs["method_diff"] = "manual"
        self.pick_best_aperture()

        if not inplace:
            return new_self

    def broeg2005(self, inplace=True, cut=True, nans=False):
        """
        The Broeg et al. 2005 differential photometry algorithm

        Compute an optimum weighted artificial light curve

        Parameters
        ----------
        inplace: bool, optional
            whether to perform the changes on current Observation or to return a new one, default True
        cut: bool or None, optional
            whether to pick the best comparison stars and apply unitary weights, default None which try both and
            cut if beneficial
        nans: bool
            whether to keep nans values in fluxes, otherwise set to 1 (arbitrarly low value) before computing the weights, default False, i.e. removing nans
        """
        # TODO: ignore apertures out of the image
        if inplace:
            new_self = self
        else:
            new_self = self.copy()

        # getting differential values
        raw_fluxes = self.raw_fluxes.copy()
        raw_errors = self.raw_errors.copy()

        if not nans:
            mask = np.isnan(raw_fluxes)
            raw_fluxes[mask] = 1
            raw_errors[mask] = 1

        mean_raw_fluxes = np.expand_dims(raw_fluxes.mean(-1), -1)
        raw_errors /= mean_raw_fluxes
        raw_fluxes /= mean_raw_fluxes

        # finding weights
        weights = broeg(raw_fluxes)
        weights[:, self.target] = 0

        comparisons = best_stars(raw_fluxes, weights, self.target)

        # We always try with and without a cut
        # -----————---------------------------
        # - with
        cut_comparisons = best_stars(raw_fluxes, weights, self.target)
        cut_weights = np.ones_like(comparisons)
        cut_diff_fluxes, cut_diff_errors, cut_alcs = diff(raw_fluxes, raw_errors, comps=comparisons, alc=True)

        # - without
        comparisons = np.repeat(np.expand_dims(np.arange(raw_fluxes.shape[-2]), -1).T, raw_fluxes.shape[0], axis=0)
        diff_fluxes, diff_errors, alcs = diff(raw_fluxes, raw_errors, weights=weights, alc=True)
        comparisons = np.delete(comparisons, self.target, axis=1)
        weights = np.delete(weights, self.target, axis=1)

        # cutting or not
        if cut is True:
            diff_fluxes, diff_errors, alcs = cut_diff_fluxes, cut_diff_errors, cut_alcs
            comparisons, weights = cut_comparisons, cut_weights
        elif cut is None:
            # we decide if cutting it beneficial
            bins = np.min([self.time.shape[-1], 12])
            b = self.time.shape[-1] // bins
            idxs = np.arange(b * bins)

            def error_estimate(f):
                return np.array(np.split(f.take(idxs, axis=-1), b, axis=-1)).std(-1).mean(0)

            cut_std = error_estimate(cut_diff_fluxes[:, self.target]).min()
            uncut_std = error_estimate(diff_fluxes[:, self.target]).min()

            if cut_std < uncut_std:
                diff_fluxes, diff_errors, alcs = cut_diff_fluxes, cut_diff_errors, cut_alcs
                comparisons, weights = cut_comparisons, cut_weights

        # setting xarray
        dims = self.xarray.raw_fluxes.dims

        new_self.xarray['diff_fluxes'] = (dims, diff_fluxes)
        new_self.xarray['diff_errors'] = (dims, diff_errors)

        # Since we reset ncomps, older vars with ncomp in dims are removed
        new_self.xarray = new_self.xarray.drop_vars(
            [name for name, value in new_self.xarray.items() if 'ncomps' in value.dims])

        new_self.xarray['comps'] = (("apertures", "ncomps"), comparisons)
        new_self.xarray['weights'] = (("apertures", "ncomps"), weights)
        new_self.xarray['alc'] = (('apertures', 'time'), alcs[:, self.target])
        new_self.xarray.attrs["method_diff"] = "broeg"
        self.pick_best_aperture()

        if not inplace:
            return new_self

    # io
    # ==

    @staticmethod
    def load(filepath):
        return ApertureFluxes(xr.load_dataset(filepath))

    def save(self, filepath):
        self.xarray.to_netcdf(filepath)

    # Plotting
    # ========

    def plot(self, star=None, bins=0.005, color="k", std=True, mask=True):
        if star is None:
            star = self.target
        viz.plot(self.time, self.x.diff_fluxes[self.aperture, star], std=std, bins=bins, bincolor=color)

    def plot_detrended(self, star=None, bins=0.005, color="k", std=True, fancy=False, ylim=None, label=True):
        if star is None:
            star = self.target
        diff_flux = self.diff_fluxes[self.aperture, star]
        trend = self.trends[star]

        if fancy:
            self.plot_systematics_signal(trend, signal=None, ylim=ylim)
            if label:
                viz.corner_text(f"{self.trend}", c="C0", loc=(0.05, 0.03))
        else:
            viz.plot(self.time, diff_flux - trend + 1, std=std, bins=bins, bincolor=color)
            if label:
                viz.corner_text(f"detrended ({self.trend})", c="C0")
            if ylim is not None:
                plt.ylim(ylim)
                
    def sigma_clip(self, sigma=3., star=None):
        """Sigma clipping

        Parameters
        ----------
        sigma : float, optional
            sigma clipping threshold, by default 3.

        star : int, optional
            star on which to apply the sigma clip, is target by default

        """
        new_self = self.copy()
        if star is None:
            new_self.xarray = new_self.xarray.sel(
                time=self.time[np.abs(self.diff_flux - np.median(self.diff_flux)) < sigma * np.std(self.diff_flux)])
        else:
            new_self.xarray = new_self.xarray.sel(
                time=self.time[np.abs(self.diff_fluxes[self.aperture, star] - np.median(
                    self.diff_fluxes[self.aperture, star])) < sigma * np.std(self.diff_fluxes[self.aperture, star])])

        return new_self

    # modeling

    def lstsq(self, X, star=None, split=None):
        """Given a design matrix return the fitted trend

        Parameters
        ----------
        X : np.ndarray
            design matrix of shape (time, n), n being the number of regressors.
        split : int or array, optional
            splitting indexes of the design matrix, passed to np.split and used to retrieve splitted models, by default None

        Returns
        -------
        [type]
            [description]
        """
        if star is None:
            star = self.target

        w, dw, _, _ = np.linalg.lstsq(X, self.diff_fluxes[self.aperture, star], rcond=None)
        if split is not None:
            if not isinstance(split, list):
                split = [split]
            split_w = np.split(w, [-1])
            split_dm_T = np.split(X.T, [-1])
            return [_w @ _dm for _w, _dm in zip(split_w, split_dm_T)]
        else:
            return X @ w

    def polynomial(self, **orders):
        """Return a design matrix representing a polynomial model

        Parameters
        ----------
        orders: dict
            dict which keys are the model variables and values are the polynomial orders use in their model against flux

        Returns
        -------
        [type]
            [description]
        """
        return models.design_matrix([
            models.constant(self.time),
            *[models.polynomial(self.xarray[name].values - self.xarray[name].values.min(), order) for name, order in orders.items() if order>0]
        ])

    def transit(self, t0, duration, depth=1):
        """A simple transit model

        Parameters
        ----------
        t0 : float
            transit midtime (in unit of time)
        duration : float
            transit duration (in unit of time)
        depth : int, optional
            transit depth, by default 1
        """
        return models.transit(self.time, t0, duration)

    def step(self, t0=None):
        """
        Two parameter step model model. f = a if time < t0 else b

        Parameters
        ----------
        t0: float, optional
            time when the step occur, default is None and corresponds to meridian flip
        """
        if t0 is None:
            assert self.meridian_flip, "please specify t0"
            t0 = self.meridian_flip

        return models.step(self.time, t0)

    def dm_ll(self, dm):
        n = len(self.time)
        chi2 = (self.diff_flux - self.lstsq(dm)) ** 2
        return np.sum(-(n / 2) * np.log(2 * np.pi * self.diff_error ** 2) - (1 / (2 * (self.diff_error ** 2))) * chi2)

    def dm_bic(self, dm):
        return np.log(len(self.time)) * dm.shape[1] - 2 * self.dm_ll(dm)

    def best_polynomial(self, add=None, verbose=False, **orders):
        """Return the best systematics polynomial model orders. 

        Parameters
        ----------
        orders: dict
            dict with keys being the systematic to consider and values are the maximum polynomial order to test
        add : np.ndarray, optional
            additional regressor to add to the design matrix, by default None
        verbose : bool, optional
            whether to show the progress bar, by default False
        """
        def progress(x):
            return tqdm(x) if verbose else x

        orders_ranges = [(key, np.arange(order+1)) for key, order in orders.items()]
        keys = [o[0] for o in orders_ranges]
        orders = [o[1] for o in orders_ranges]
        combs = list(product(*orders))
        dms_dicts = [dict(zip(keys, comb)) for comb in combs]
        if add is None:
            dms = [self.polynomial(**d) for d in dms_dicts]
        else:
            dms = [np.hstack([self.polynomial(**d), add]) for d in dms_dicts]
        bics = [self.dm_bic(dm) for dm in progress(dms)]
        return dms_dicts[np.argmin(bics)]

    def polynomial_trend(self, bic=True, verbose=True, **kwargs):
        if kwargs == dict() and bic:
            n = 3
            kwargs = dict(sky=n, dx=n, dy=n, airmass=n, fwhm=n)

        if bic:
            orders = self.best_polynomial(**kwargs)
        else:
            orders = kwargs

        # compute trend for all stars
        X = self.polynomial(**orders)
        trends = np.array([self.lstsq(X, star=s) for s in self.star])
        self.xarray["trends"] = (("stars", "time"), trends)

        # Display and save trend model
        polynomial_tex = [r"1"]
        original_orders = orders.copy()
        orders = list(orders.items())
        orders = sorted(orders, key=lambda x: x[1])

        for variable, order in orders:
            append_tex_order(polynomial_tex, order, variable)
        polynomial_tex = rf"${'+'.join(polynomial_tex)}$"

        if verbose:
            info(f"Polynomial trend:")
            viz.print_tex(polynomial_tex)

        self.xarray.attrs["trend"] = polynomial_tex
        self.xarray.attrs["trend_model"] = "polynomial"
        self.xarray.attrs["polynomial_trend_variables"] = list([o[0] for o in orders])
        self.xarray.attrs["polynomial_trend_orders"] = list([o[1] for o in orders])

    def noise_stats(self, bins=0.005, verbose=True):
        pont_w, pont_r = self.pont2006(plot=False)
        binned = self.binn(bins, std=True)
        binned_w = np.median(binned.diff_error)
        if verbose:
            print(f"white (pont2006)\t{pont_w:.3e}\nred   (pont2006)\t{pont_r:.3e}\nwhite (binned)\t\t{binned_w:.3e}\n")
        else:
            return {"binned_white": binned_w, "pont_white": pont_w, "pont_red": pont_r}

    @staticmethod
    def set_attribute(file, **kwargs):
        fluxes = ApertureFluxes(file)
        for name, value in kwargs.items():
            fluxes.xarray.attrs[name] = value
        fluxes.save(file)
