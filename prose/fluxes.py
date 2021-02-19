import numpy as np
from . import utils
import matplotlib.pyplot as plt
from . import visualisation as viz
from scipy.optimize import curve_fit
import xarray as xr
from astropy.stats import sigma_clip
from itertools import product
from tqdm import tqdm
from . import models

# TODO: check if working properly with a single aperture/lc
# TODO: a rms reject method and a sigma clip method in LightCurves
# TODO: test old broeg2005 method and see if returns same results (or simlar)


def differential_photometry(fluxes, errors, comps, return_alc=False):
    np.seterr(divide="ignore")  # Ignore divide by 0 warnings here

    comps = np.array(comps)

    # Normalization
    # ==============
    mean_fluxes = np.nanmean(fluxes, 2)[:, :, None]
    fluxes = fluxes.copy() / mean_fluxes
    errors = errors.copy() / mean_fluxes

    n_apertures, n_stars, n_points = np.shape(fluxes)

    # Compute the final light-curves
    # ==============================
    art_lc = np.array([fluxes[a, comps].mean(0) for a in range(n_apertures)])
    art_error = np.array([np.sqrt(errors[a, comps] ** 2).sum(0) / len(comps) for a in range(n_apertures)])
    lcs = fluxes / art_lc[:, None]
    lcs /= np.median(lcs, 2)[:, :, None]
    lcs_errors = np.sqrt(errors ** 2 + art_error[:, None, :] ** 2)

    np.seterr(divide="warn")  # Set warnings back

    return [lcs, lcs_errors, art_lc] if return_alc else [lcs, lcs_errors]


def nu(n, sw, sr):
    return (sw ** 2) / n + sr ** 2


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


def broeg2005(
        fluxes,
        errors,
        target,
        keep="float",
        max_iteration=50,
        tolerance=1e-8,
        n_comps=500,
        exclude=None,
        mask=None,
        sigclip=5,
):
    """[summary]

    Parameters
    ----------
    fluxes : np.ndarray
        fluxes with shape (apertures, stars, images)
    errors : np.ndarray
        fluxes errors with shape (apertures, stars, images)
    target : int
        target index
    keep : str, optional
        - if None: use a weighted artificial comparison star based on all stars (weighted mean)
        - if float: use a weighted artificial comparison star based on `keep` stars (weighted mean)
        - if int: use a simple artificial comparison star based on `keep` stars (mean)
        - if 'float': use a weighted artificial comparison star based on an optimal number of stars (weighted mean)
        - if 'int': use a simple artificial comparison star based on an optimal number of stars (mean)
        by default "float"
    max_iteration : int, optional
        maximum number of iteration to adjust weights, by default 50
    tolerance : float, optional
        mean difference between weights to reach, by default 1e-8
    n_comps : int, optional
        limit on the number of stars to keep, by default 500
    exclude : list, optional
        indexes of stars to exclude, by default None
    mask : boolean ndarray, optional
        a boolean mask of time length. Only points for which mask is True are considered when
        evaluating the comparison stars (then applied to all points), by default None
    sigclip: int, optional
        if a float is given, a sigma clipping of factor sigclip is temporarly applied to all
        fluxes before being used to choose the comparison stars, by default 5
    """

    np.seterr(divide="ignore")  # Ignore divide by 0 warnings here

    if exclude is not None:
        _exclude = np.array(exclude)
    else:
        _exclude = []

    original_fluxes = fluxes.copy()
    mean_fluxes = np.nanmean(fluxes, 2)[:, :, None]

    # Sigma clipping
    # --------------
    if sigclip is not None:
        f = fluxes.copy()

        for i in range(f.shape[0]):
            for j in range(f.shape[0]):
                _f = f[i, j]
                f[i, j, np.abs(_f - np.mean(_f)) > 5 * np.nanstd(_f)] = np.mean(_f)

        fluxes = f

    # Normalization
    # -------------
    fluxes = fluxes.copy() / mean_fluxes
    errors = errors.copy() / mean_fluxes

    # Cleaning
    # --------
    # To keep track of which stars are clean and get a light curve (if not containing infs, nans or zeros)
    bads = np.any(~np.isfinite(fluxes) | ~np.isfinite(errors), axis=(0, 2))
    fluxes = fluxes[:, ~bads, :]
    errors = errors[:, ~bads, :]

    clean_stars = np.arange(original_fluxes.shape[1])[~bads]
    target = np.argwhere(clean_stars == target).flatten()
    _exclude = [np.argwhere(clean_stars == ex).flatten() for ex in _exclude]

    n_apertures, n_stars, n_points = np.shape(fluxes)

    # Masking
    # -------
    if mask is None:
        mask = np.ones(fluxes.shape[-1]).astype(bool)

    unmasked_fluxes = fluxes.copy()
    unmasked_errors = errors.copy()
    fluxes = fluxes[:, :, mask]
    errors = errors[:, :, mask]

    i = 0
    evolution = 1e25
    lcs = None
    weights = None
    last_weights = np.zeros((n_apertures, n_stars))
    # record = []

    # Broeg 2004 algorithm to find weights of comp stars
    # --------------------------------------------------
    while evolution > tolerance and i < max_iteration:
        if i == 0:
            weights = 1 / np.mean(errors ** 2, axis=2)
        else:
            # This metric is preferred from std to optimize over white noise and not red noise
            std = utils.std_diff_metric(lcs)
            weights = 1 / std

        # Keep track of weights
        evolution = np.std(
            np.abs(np.mean(weights, axis=1) - np.mean(last_weights, axis=1))
        )
        # record.append(evolution)
        last_weights = weights

        weighted_fluxes = fluxes * weights[:, :, None]
        art_lc = np.sum(weighted_fluxes, 1) / weights.sum(1)[:, None]
        lcs = fluxes / art_lc[:, None, :]

        i += 1

    # Setting target weight to 0
    weights[:, target] = 0
    # Setting weights of stars to exclude to 0
    if exclude is not None:
        weights[:, _exclude] = 0

    ordered_stars = np.argsort(weights, axis=1)[:, ::-1]
    # Remove target and excluded stars (at the end of ordered since weight = 0
    ordered_stars = ordered_stars[:, 0:-(1 + len(_exclude))]

    # Find the best number of comp stars to keep if None
    # --------------------------------------------------
    if isinstance(keep, str):

        k_range = np.arange(1, np.min([np.shape(fluxes)[1], n_comps]))

        weighted_fluxes = fluxes * weights[:, :, None]
        ordered_weighted_fluxes = np.array([weighted_fluxes[a, ordered_stars[a], :] for a in range(n_apertures)])
        ordered_summed_weighted_fluxes = np.array([ordered_weighted_fluxes[:, 0:k, :].sum(1) for k in k_range])
        ordered_summed_weights = np.array([weights[:, 0:k].sum(1) for k in k_range])

        metric = 1e18
        last_metric = np.inf
        i = 0
        _keep = None

        while metric < last_metric:
            _keep = k_range[i]
            last_metric = metric
            _art_lc = ordered_summed_weighted_fluxes[i] / ordered_summed_weights[i][:, None]
            _lcs = fluxes / _art_lc[:, None, :]
            metric = np.std([utils.std_diff_metric(f) for f in _lcs[:, target, :]])

            i += 1

    elif keep is None:
        _keep = np.min([np.shape(fluxes)[1], n_comps])
        keep = "float"
    else:
        _keep = keep
        keep = "float" if isinstance(keep, float) else "int"

    # Compute the final lightcurves
    # -----------------------------
    if keep == "float":
        # Using the weighted sum
        ordered_weights = np.array([weights[a, ordered_stars[a, :]] for a in range(n_apertures)])
    elif keep == "int":
        # Using a simple mean
        ordered_weights = np.ones((n_apertures, _keep))
    else:
        raise AssertionError(f"Unknown issue: keep has value {keep}")

    keep = int(_keep)

    ordered_fluxes = np.array([unmasked_fluxes[a, ordered_stars[a], :] for a in range(n_apertures)])
    ordered_errors = np.array([unmasked_errors[a, ordered_stars[a], :] for a in range(n_apertures)])

    best_art_lc = (ordered_fluxes[:, 0:keep, :] * ordered_weights[:, 0:keep, None]).sum(1) / ordered_weights[:,
                                                                                             0:keep].sum(1)[:, None]
    best_art_error = (ordered_errors[:, 0:keep, :] ** 2 * ordered_weights[:, 0:keep, None] ** 2).sum(
        1) / ordered_weights[:, 0:keep].sum(1)[:, None]

    lcs = np.zeros(np.shape(original_fluxes))
    lcs_errors = np.zeros(np.shape(original_fluxes))

    for a in range(n_apertures):
        for s, cs in enumerate(clean_stars):
            lcs[a, cs, :] = unmasked_fluxes[a, s] / best_art_lc[a, :]
            lcs_errors[a, cs, :] = np.sqrt(
                unmasked_errors[a, s] ** 2 + best_art_error[a, :] ** 2
            )

    # Return
    # ------------------------------
    np.seterr(divide="warn")  # Set warnings back
    info = {
        "comps": clean_stars[ordered_stars[:, 0:keep]],
        "weights": np.array(ordered_weights[:, 0:keep]),
        "alc": best_art_lc
    }

    return lcs, lcs_errors, info


class ApertureFluxes:

    def __init__(self, xarray):
        if isinstance(xarray, str):
            self.xarray = xr.load_dataset(xarray)
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
        return self.xarray.target

    @target.setter
    def target(self, value):
        self.xarray.attrs['target'] = value
        if "diff_fluxes" in self:
            self._pick_best_aperture()

    @property
    def aperture(self):
        return self.xarray.aperture

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
        new_self.xarray = new_self.xarray.where(xr.DataArray(mask, dims=dim), drop=True)
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
    def comparison_raw_fluxes(self):
        return self.raw_fluxes[self.aperture, self.comps[self.aperture]]

    def __copy__(self):
        return self.__class__(self.xarray.copy())

    def copy(self):
        return self.__class__(self.xarray.copy())

    def _pick_best_aperture(self, method="stddiff", return_criterion=False):
        if len(self.apertures) > 1:
            if method == "stddiff":
                criterion = utils.std_diff_metric(self.xarray.diff_fluxes.sel(star=self.target))
            elif method == "stability":
                criterion = utils.stability_aperture(self.xarray.diff_fluxes.sel(star=self.target))
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
        if "raw_fluxes" not in self:
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
        if inplace:
            new_self = self
        else:
            new_self = self.copy()

        diff_fluxes, diff_errors, alc = differential_photometry(new_self.raw_fluxes, new_self.raw_errors, comps,
                                                                return_alc=True)
        dims = self.xarray.raw_fluxes.dims

        new_self.xarray["diff_fluxes"] = (dims, diff_fluxes)
        new_self.xarray["diff_errors"] = (dims, diff_errors)

        # Since we reset ncomps, older vars with ncomp in dims are removed
        new_self.xarray = new_self.xarray.drop_vars(
            [name for name, value in new_self.xarray.items() if 'ncomps' in value.dims])

        comps = np.repeat(np.atleast_2d(comps), len(self.apertures), axis=0)

        new_self.xarray['comps'] = (("apertures", "ncomps"), comps)
        new_self.xarray['weights'] = (("apertures", "ncomps"), np.ones_like(comps))
        new_self.xarray['alc'] = (("apertures", 'time'), alc)
        new_self._pick_best_aperture()

        if not inplace:
            return new_self

    def broeg2005(self, keep='float', exclude=None, inplace=True, mask=None, sigclip=5):
        """The Broeg et al. 2005 differential photometry algorithm

        Compute an optimum weighted artificial light curve

        Parameters
        ----------
        keep : str, optional
            - if `None`: use a weighted artificial comparison star based on all stars (weighted mean)
            - if `float`: use a weighted artificial comparison star based on `keep` stars (weighted mean)
            - if `int`: use a simple artificial comparison star based on `keep` stars (mean)
            - if `'float'`: use a weighted artificial comparison star based on an optimal number of stars (weighted mean)
            - if `'int'`: use a simple artificial comparison star based on an optimal number of stars (mean)
            by default "float"
        exclude : list, optional
            indexes of stars to exclude, by default None,
        inplace: bool, optional
            whether to perform the changes on current Observation or to return a new one, default True
        mask : boolean ndarray, optional
            a boolean mask of time length. Only points for which mask is True are considered when
            evaluating the comparison stars (then applied to all points), by default None
        sigclip: int, optional
            if a float is given, a sigma clipping of factor sigclip is temporarly applied to all
            fluxes before being used to choose the comparison stars, by default 5

        """
        if inplace:
            new_self = self
        else:
            new_self = self.copy()

        diff_fluxes, diff_errors, info = broeg2005(
            new_self.raw_fluxes, new_self.raw_errors, self.target,
            keep=keep, exclude=exclude,mask=mask, sigclip=sigclip)
        dims = self.xarray.raw_fluxes.dims

        new_self.xarray['diff_fluxes'] = (dims, diff_fluxes)
        new_self.xarray['diff_errors'] = (dims, diff_errors)

        # Since we reset ncomps, older vars with ncomp in dims are removed
        new_self.xarray = new_self.xarray.drop_vars(
            [name for name, value in new_self.xarray.items() if 'ncomps' in value.dims])

        new_self.xarray['comps'] = (("apertures", "ncomps"), info['comps'])
        new_self.xarray['weights'] = (("apertures", "ncomps"), info['weights'])
        new_self.xarray['alc'] = (('apertures', 'time'), info['alc'])
        new_self._pick_best_aperture()

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

    def plot(self, which="None", bins=0.005, color="k", std=True):
        binned = self.binn(bins, std=std)
        plt.plot(self.time, self.diff_flux, ".", c="gainsboro", zorder=0, alpha=0.6)
        plt.errorbar(binned.time, binned.diff_flux, yerr=binned.diff_error, fmt=".", zorder=1, color=color, alpha=0.8)

    def sigma_clip(self, sigma=3.):
        """Sigma clipping

        Parameters
        ----------
        sigma : float, optional
            sigma clipping threshold, by default 3.

        """
        new_self = self.copy()
        new_self.xarray = new_self.xarray.sel(
            time=self.time[self.diff_flux - np.median(self.diff_flux) < sigma * np.std(self.diff_flux)])
        return new_self

    # modeling

    def trend(self, dm, split=None):
        """Given a design matrix return the fitted trend

        Parameters
        ----------
        dm : np.ndarray
            design matrix of shape (time, n), n being the number of regressors.
        split : int or array, optional
            splitting indexes of the design matrix, passed to np.split and used to retrieve splitted models, by default None

        Returns
        -------
        [type]
            [description]
        """
        w, dw, _, _ = np.linalg.lstsq(dm, self.diff_flux, rcond=None)
        if split is not None:
            if not isinstance(split, list):
                split = [split]
            split_w = np.split(w, [-1])
            split_dm_T = np.split(dm.T, [-1])
            return [_w@_dm for _w, _dm in zip(split_w, split_dm_T)]
        else:
            return dm @ w

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
            *[models.polynomial(self.xarray[name].values, order) for name, order in orders.items() if order>0]
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
        chi2 = (self.diff_flux - self.trend(dm)) ** 2
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
            wether to show the progress bar, by default False
        """
        def progress(x):
            return tqdm(x) if verbose else x

        orders_ranges = [(key, np.arange(order)) for key, order in orders.items()]
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
