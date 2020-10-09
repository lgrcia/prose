import math
import numpy as np
from tqdm import tqdm
from os import path
from astropy.io import fits
from astropy.stats import sigma_clip
from prose import io
from prose import utils
import matplotlib.pyplot as plt
from prose import visualisation as viz
from scipy.optimize import curve_fit
from astropy.stats import sigma_clipped_stats, sigma_clip

def nu(n, sw, sr):
    return (sw**2)/n + sr**2


# TODO: check if working properly with a single aperture/lc
# TODO: a rms reject method and a sigma clip method in LightCurves

def Pont2006(x, y, n=30, plot=True, return_error=False):
    dt = np.median(np.diff(x))
    ns = np.arange(1, n)
    binned = [utils.fast_binning(x, y, dt*_n)[1] for _n in ns]
    _nu = [np.var(b) for b in binned]
    (sw, sr), pcov = curve_fit(nu, ns, _nu)
    if plot:
        plt.plot(ns, np.std(y)/np.sqrt(ns), ":k", label="$\sigma / \sqrt{n}$")
        plt.plot(ns, np.sqrt(nu(ns, sw, sr)), "k", label="fit")
        plt.plot(ns, np.sqrt(_nu), "d", markerfacecolor="w", markeredgecolor="k")
        plt.xlabel("n points")
        plt.ylabel("$\\nu^{1/2}(n)$")
        plt.legend()
    if return_error:
        return sw, sr, np.sqrt(np.diag(pcov))
    else:
        return sw, sr

def differential_photometry(fluxes, errors, comps, return_art_lc=False):

    np.seterr(divide="ignore")  # Ignore divide by 0 warnings here

    original_fluxes = fluxes.copy()

    _fluxes = fluxes.copy()
    _errors = errors.copy()

    # Normalization
    # -------------
    fluxes = np.array([[ff / np.nanmean(ff) for ff in f] for f in _fluxes])
    errors = np.array([[ee / np.nanmean(ff) for ee, ff in zip(e, f)]
                        for e, f in zip(_errors, _fluxes)])

    n_apertures, n_stars, n_points = np.shape(fluxes)

    # Compute the final light-curves
    # ------------------------------
    art_lc = np.zeros((n_apertures, n_points))
    art_error = np.zeros((n_apertures, n_points))

    for a in range(n_apertures):
        art_lc[a, :] = np.sum([fluxes[a, s] for s in comps], axis=0) / len(comps)
        art_error[a, :] = np.sqrt(np.sum([errors[a, s] ** 2 for s in comps], axis=0)) / len(comps)

    lcs = np.zeros(np.shape(original_fluxes))
    lcs_errors = np.zeros(np.shape(original_fluxes))

    for a in range(n_apertures):
        for s in range(n_stars):
            lcs[a, s, :] = fluxes[a, s] / art_lc[a, :]
            lcs_errors[a, s, :] = np.sqrt(errors[a, s] ** 2 + art_error[a, :] ** 2)

    # Final normalization and return
    # ------------------------------
    lcs = np.array([[ll / np.nanmedian(ll) for ll in l] for l in lcs])
    np.seterr(divide="warn")  # Set warnings back

    return_values = [lcs, lcs_errors]

    if return_art_lc:
        return_values.append(art_lc)

    return return_values


def Broeg2005(
        fluxes,
        errors,
        target,
        keep="float",
        max_iteration=50,
        tolerance=1e-8,
        n_comps=500,
        show_comps=False,
        return_art_lc=False,
        return_comps=False
):
    """
    Implementation of the Broeg (2004) algorithm to compute optimum weighted artificial comparison star
    Parameters
    ----------
    fluxes : np.ndarray
        Fluxes with shape (n_apertures, n_stars, n_images)
    errors: np.ndarray
        Errors on fluxes with shape (n_apertures, n_stars, n_images)
    target: int
        Target id
    keep: None, int, float, string or None (optional, default is 'float')
        - if None: use a weighted artificial comparison star based on all stars (weighted mean)
        - if float: use a weighted artificial comparison star based on `keep` stars (weighted mean)
        - if int: use a simple artificial comparison star based on `keep` stars (mean)
        - if 'float': use a weighted artificial comparison star based on an optimal number of stars (weighted mean)
        - if 'int': use a simple artificial comparison star based on an optimal number of stars (mean)
    max_iteration: int (optional, default is 50)
        maximum number of iteration to adjust weights
    tolerance: float (optional, default is 1e-8)
        mean difference between weights to reach
    n_comps: int (optional, default is 500)
        limit on the number of stars to keep (see keep kwargs)
    show_comps: bool (optional, default is False)
        show stars and weights used to build the artificial comparison star
    Returns
    -------
    lcs : np.ndarray
        Light curves with shape (n_apertures, n_stars, n_images)
    """

    np.seterr(divide="ignore")  # Ignore divide by 0 warnings here

    original_fluxes = fluxes.copy()
    initial_n_stars = np.shape(original_fluxes)[1]

    _fluxes = fluxes.copy()
    _errors = errors.copy()

    # Normalization
    # -------------
    fluxes = np.array([[ff / np.nanmean(ff) for ff in f] for f in _fluxes])
    errors = np.array([
        [ee / np.nanmean(ff) for ee, ff in zip(e, f)]
        for e, f in zip(_errors, _fluxes)
    ])

    # Cleaning
    # --------
    # To keep track of which stars are clean and get a light curve (if not containing infs, nans or zeros)
    clean_stars = np.arange(0, initial_n_stars)

    # Data cleaning, excluding stars with no finite flux (0, nan or inf)
    if not np.all(np.isfinite(fluxes)):
        idxs = np.unique(np.where(np.logical_not(np.isfinite(fluxes)))[1])
        fluxes = np.delete(fluxes, idxs, axis=1)
        errors = np.delete(errors, idxs, axis=1)
        clean_stars = np.delete(clean_stars, idxs)
    if not np.all(np.isfinite(errors)):
        idxs = np.unique(np.where(np.logical_not(np.isfinite(errors)))[1])
        fluxes = np.delete(fluxes, idxs, axis=1)
        errors = np.delete(errors, idxs, axis=1)
        clean_stars = np.delete(clean_stars, idxs)

    n_apertures, n_stars, n_points = np.shape(fluxes)

    i = 0
    evolution = 1e25
    # record = []

    # Broeg 2004 algorithm to find weights of comp stars
    # --------------------------------------------------
    while evolution > tolerance and i < max_iteration:
        if i == 0:
            last_weights = np.zeros((n_apertures, n_stars))
            weights = 1 / np.mean(errors ** 2, axis=2)
        else:
            # This metric is prefered from std to optimize over white noise and not red noise
            std = utils.std_diff_metric(lcs)
            weights = 1 / std

        # Keep track of weights
        evolution = np.std(
            np.abs(np.mean(weights, axis=1) - np.mean(last_weights, axis=1))
        )
        # record.append(evolution)
        last_weights = weights

        art_lc = np.zeros((n_apertures, n_points))

        for a in range(n_apertures):
            art_lc[a, :] = np.sum(
                [fluxes[a, s] * w for s, w in zip(range(n_stars), weights[a])], axis=0,
            ) / np.sum(weights[a])

        lcs = np.zeros((np.shape(fluxes)))

        for a in range(n_apertures):
            for s in range(np.shape(fluxes)[1]):
                lcs[a, s, :] = fluxes[a, s] / art_lc[a, :]

        i += 1

    ordered_stars = np.argsort(weights, axis=1)[:, ::-1]

    # Find the best number of comp stars to keep if None
    # --------------------------------------------------
    if isinstance(keep, str):

        metric = []
        k_range = np.arange(2, np.min([np.shape(fluxes)[1], n_comps]))

        for k in k_range:
            best_stars = ordered_stars[:, 0: int(k)]
            best_art_lc = np.zeros((n_apertures, n_points))

            for a in range(n_apertures):
                best_art_lc[a, :] = np.sum(
                    [
                        fluxes[a, s] * w
                        for s, w in zip(best_stars[a], weights[a, best_stars[a]])
                    ],
                    axis=0,
                ) / np.sum(weights[a, best_stars[a]])

            _lcs = np.zeros(np.shape(fluxes))

            for a in range(n_apertures):
                for s in range(np.shape(fluxes)[1]):
                    _lcs[a, s, :] = fluxes[a, s] / best_art_lc[a, :]

            metric.append(np.std([utils.std_diff_metric(f) for f in _lcs[:, target, :]]))

        if keep == "float":
            keep = float(k_range[np.argmin(metric)])
        elif keep == "int":
            keep = int(k_range[np.argmin(metric)])

    elif keep is None:
        keep = float(np.min([np.shape(fluxes)[1], n_comps]))

    # Compute the final lightcurves
    # -----------------------------
    # Keeping `keep` number of stars with the best weights
    ordered_stars = np.array(
        [np.delete(bs, np.where(bs == target)) for bs in ordered_stars]
    ).astype(int)
    best_stars = ordered_stars[:, 0: int(keep)]
    # Removing target from kept stars
    # best_stars = np.array([np.delete(bs, np.where(bs == target)) for bs in best_stars]).astype(int)
    best_art_lc = np.zeros((n_apertures, n_points))
    best_art_error = np.zeros((n_apertures, n_points))

    best_stars_n = np.shape(best_stars)[1]

    for a in range(n_apertures):
        if type(keep) is float:
            # Using the weighted sum
            _weights = weights[a, best_stars[a]]
        elif type(keep) is int:
            # Using a simple mean
            _weights = np.ones(best_stars_n)

        best_art_lc[a, :] = np.sum(
            [fluxes[a, s] * w for s, w in zip(best_stars[a], _weights)], axis=0,
        ) / np.sum(_weights)

        best_art_error[a, :] = np.sqrt(
            np.sum(
                [errors[a, s] ** 2 * w ** 2 for s, w in zip(best_stars[a], _weights)],
                axis=0,
            )
        ) / np.sum(_weights)

    lcs = np.zeros(np.shape(original_fluxes))
    lcs_errors = np.zeros(np.shape(original_fluxes))

    for a in range(n_apertures):
        for s, cs in enumerate(clean_stars):
            lcs[a, cs, :] = fluxes[a, s] / best_art_lc[a, :]
            lcs_errors[a, cs, :] = np.sqrt(
                errors[a, s] ** 2 + best_art_error[a, :] ** 2
            )

    # Final normalization and return
    # ------------------------------
    lcs = np.array([[ll / np.nanmedian(ll) for ll in l] for l in lcs])
    np.seterr(divide="warn")  # Set warnings back

    if show_comps:
        print(best_stars, weights)

    return_values = [lcs, lcs_errors]

    if return_art_lc:
        return_values.append(best_art_lc)
    if return_comps:
        return_values.append(best_stars)

    return return_values

def newBroeg2005(
    fluxes,
    errors,
    target,
    keep="float",
    max_iteration=50,
    tolerance=1e-8,
    n_comps=500,
    show_comps=False,
    return_art_lc=False,
    return_comps=False
):
    """

    Implementation of the Broeg (2004) algorithm to compute optimum weighted artificial comparison star

    Parameters
    ----------
    fluxes : np.ndarray
        Fluxes with shape (n_apertures, n_stars, n_images)
    errors: np.ndarray
        Errors on fluxes with shape (n_apertures, n_stars, n_images)
    target: int
        Target id
    keep: None, int, float, string or None (optional, default is 'float')
        - if None: use a weighted artificial comparison star based on all stars (weighted mean)
        - if float: use a weighted artificial comparison star based on `keep` stars (weighted mean)
        - if int: use a simple artificial comparison star based on `keep` stars (mean)
        - if 'float': use a weighted artificial comparison star based on an optimal number of stars (weighted mean)
        - if 'int': use a simple artificial comparison star based on an optimal number of stars (mean)
    max_iteration: int (optional, default is 50)
        maximum number of iteration to adjust weights
    tolerance: float (optional, default is 1e-8)
        mean difference between weights to reach
    n_comps: int (optional, default is 500)
        limit on the number of stars to keep (see keep kwargs)
    show_comps: bool (optional, default is False)
        show stars and weights used to build the artificial comparison star
    Returns
    -------
    lcs : np.ndarray
        Light curves with shape (n_apertures, n_stars, n_images)
    """

    np.seterr(divide="ignore")  # Ignore divide by 0 warnings here

    original_fluxes = fluxes.copy()
    initial_n_stars = np.shape(original_fluxes)[1]

    # Normalization
    # -------------

    mean_fluxes = np.nanmean(fluxes, 2)[:, :, None]
    fluxes = fluxes.copy() / mean_fluxes
    errors = errors.copy() / mean_fluxes

    # Cleaning
    # --------
    # To keep track of which stars are clean and get a light curve (if not containing infs, nans or zeros)
    clean_stars = np.arange(0, initial_n_stars)

    # Data cleaning, excluding stars with no finite flux (0, nan or inf)
    if not np.all(np.isfinite(fluxes)):
        idxs = np.unique(np.where(np.logical_not(np.isfinite(fluxes)))[1])
        fluxes = np.delete(fluxes, idxs, axis=1)
        errors = np.delete(errors, idxs, axis=1)
        clean_stars = np.delete(clean_stars, idxs)
    if not np.all(np.isfinite(errors)):
        idxs = np.unique(np.where(np.logical_not(np.isfinite(errors)))[1])
        fluxes = np.delete(fluxes, idxs, axis=1)
        errors = np.delete(errors, idxs, axis=1)
        clean_stars = np.delete(clean_stars, idxs)

    n_apertures, n_stars, n_points = np.shape(fluxes)

    i = 0
    evolution = 1e25
    # record = []

    # Broeg 2004 algorithm to find weights of comp stars
    # --------------------------------------------------
    while evolution > tolerance and i < max_iteration:
        if i == 0:
            last_weights = np.zeros((n_apertures, n_stars))
            weights = 1 / np.mean(errors ** 2, axis=2)
        else:
            # This metric is prefered from std to optimize over white noise and not red noise
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

    # Setting target weight to 0 and removing it from the set of comparable stars
    weights[:, target] = 0
    ordered_stars = np.argsort(weights, axis=1)[:, ::-1]
    ordered_stars = ordered_stars[:, 0:-1]

    # Find the best number of comp stars to keep if None
    # --------------------------------------------------
    if isinstance(keep, str):

        metric = []
        krange = np.arange(1, np.min([np.shape(fluxes)[1], n_comps]))

        weighted_fluxes = fluxes * weights[:, :, None]
        ordered_weighted_fluxes = np.array([weighted_fluxes[a, ordered_stars[a], :] for a in range(n_apertures)])
        ordered_summed_weighted_fluxes = np.array([ordered_weighted_fluxes[:, 0:k, :].sum(1) for k in krange])
        ordered_summed_weights = np.array([weights[:, 0:k].sum(1) for k in krange])

        metric = 1e18
        last_metric = np.inf
        i = 0

        while metric < last_metric:
            _keep = krange[i]
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

    keep = int(_keep)

    ordered_fluxes = np.array([fluxes[a, ordered_stars[a], :] for a in range(n_apertures)])
    ordered_errors = np.array([errors[a, ordered_stars[a], :] for a in range(n_apertures)])

    best_art_lc = (ordered_fluxes[:, 0:keep, :] * ordered_weights[:, 0:keep, None]).sum(1) / ordered_weights[:,
                                                                                             0:keep].sum(1)[:, None]
    best_art_error = (ordered_errors[:, 0:keep, :] ** 2 * ordered_weights[:, 0:keep, None] ** 2).sum(
        1) / ordered_weights[:, 0:keep].sum(1)[:, None]

    lcs = fluxes / best_art_lc[:, None]
    lcs /= np.median(lcs, 2)[:, :, None]
    lcs_errors = np.sqrt(errors ** 2 + best_art_error[:, None, :] ** 2)

    # Return
    # ------------------------------
    best_stars = ordered_stars[:, 0:keep]
    np.seterr(divide="warn")  # Set warnings back

    if show_comps:
        print(best_stars, weights)

    return_values = [lcs, lcs_errors]

    if return_art_lc:
        return_values.append(best_art_lc)
    if return_comps:
        return_values.append(best_stars)

    return return_values


class LightCurve:
    """
    Object holding time-series of flux and data from unique stellar observation

    If fluxes and errors have shape (apertures, times), best aperture is selected through
   ``self._pick_best_aperture`` for more details see TODO

    Parameters
    ----------
    time : array
        time of the time-serie
    fluxes : ndarray
        a 1D or 2D array of fluxes, if 2D shape should be (apertures, times)
    errors : ndarray, optional
        a 1D or 2D array of fluxes errors, if 2D shape should be (apertures, times)
    data : dict, optional
        a dict containing simultaneous time-series, e.g. {"fwhm": ndarray}. Each data should have shape (times)
    apertures : list or ndarray, optional
        apertures in pixels (if used to produce light curve), by default None
        
    """
    def __init__(self, time, fluxes, errors=None, data=None, apertures=None):
        
        n = np.shape(fluxes)
        self._n_aperture = n[0] if len(n) > 1 else 1
        self._n_time = n[1] if len(n) > 1 else n[0]
        
        self.time, self.data = time, data
        self.fluxes = fluxes
        self.errors = errors
        self.apertures = apertures
        
        self.best_aperture_id = None
        self._pick_best_aperture()
    
    @property
    def flux(self):
        """
        Flux from best aperture
        """
        return self.fluxes[self.best_aperture_id]
    
    @property
    def error(self):
        """
        Flux error from best aperture
        """
        return self.errors[self.best_aperture_id]

    @property
    def aslist(self):
        """
        Flux error from best aperture
        """
        return [self.time, self.flux, self.error, self.data]
    
    def _pick_best_aperture(self, method="stddiff"):
        """
        Picking of the best aperture and setting self.best_aperture_id

        Parameters
        ----------
        method: string
            Method to use to pick the best aperture (from "stddiff", "stability")

        """
        if self._n_aperture > 1:
            if method == "stddiff":
                criterion = utils.std_diff_metric(self.fluxes)
            elif method == "stability":
                criterion = utils.stability_aperture(self.fluxes)
            else:
                raise ValueError("{} is not a valid method".format(method))

            self.best_aperture_id = np.argmin(criterion)
        else:
            self.best_aperture_id = 0
    
    def to_tuple(self):
        return self.time, self.flux, self.error, self.data
    
    def plot(self, aperture=None, bins=0.005, std=False, offset=0):
        """
        Plot light curve along time

        Parameters
        ----------
        bins : float, optional
            bin size in unit of ``self.times`, by default 0.005
        std : bool, optional
            wether to show errors using std or ``self.error``, by default False
        offset: float, optional
            offset of y axis, by default 0

        Example
        -------
        .. image:: /guide/examples_images/plot_lc.png
           :align: center
        """
        if aperture is None:
            aperture = self.best_aperture_id
        flux = self.fluxes[aperture]
        error = self.errors[aperture]
        viz.plot_lc(self.time, flux+offset, bins=bins, error=error, std=std)
        plt.xlim(self.time.min(), self.time.max())
        
    def binned(self, bins):
        """
        Return a new ``LightCurve`` binned in time

        Parameters
        ----------
        bins : float
            bin size in unit of ``self.times``

        Returns
        -------
        LightCurve
        """
        binned_data = {
            key: utils.binning(np.array(self.time), np.array(value), bins)[1]
            for key, value in self.data.items()}
        
        binned_fluxes = []
        binned_errors = []
        
        for flux, error in zip(self.fluxes, self.errors):
            binned_time, binned_flux, binned_error = utils.binning(self.time, flux, bins, error=error, std=False)
            binned_fluxes.append(binned_flux)
            binned_errors.append(binned_error)

        binned_data = {}
        for key, value in self.data.items():
            binned_time, binned_value = utils.binning(self.time, value, bins)
            binned_data[key] = binned_value
            
        return LightCurve(binned_time, binned_fluxes, binned_errors, binned_data)
    
    def Pont2006(self, plot=True, n=35):
        return Pont2006(self.time, self.flux, n=n, plot=plot)

    
    def save(self, name):
        np.save("{}.lc".format(name), [{
            "fluxes": self.fluxes,
            "errors": self.errors,
            "data": self.data,
            "time": self.time,
            "best_aperture_id": self.best_aperture_id
        }], allow_pickle=True)
    
    @staticmethod
    def load(filename):
        data = np.load(filename, allow_pickle=True)[0]
        lc = LightCurve(data["time"], data["fluxes"], data["errors"], data["data"])
        lc.best_aperture_id = data["best_aperture_id"]
        return lc

    def mask(self, mask):
        lc = LightCurve(self.time[~mask], self.fluxes[:, ~mask], self.errors[:, ~mask], {
            key: np.array(value)[~mask] for key, value in self.data.items()
        })
        lc.best_aperture_id = self.best_aperture_id
        return lc

    def sigma_clip(self, **kwargs):
        # TODO: pre-smooth parameter to sigma-clip without trend (otherwise affecting the std)
        siglcip_mask = sigma_clip(self.flux, **kwargs).mask
        return self.mask(siglcip_mask)

    def plot_data(self, which, **kwargs):
        _, _, lc_std = sigma_clipped_stats(self.flux)
        _, median_data, std_data = sigma_clipped_stats(self.flux)
        rescaled_data = self.data[which]
        rescaled_data -= median_data
        rescaled_data /= std_data
        rescaled_data *= lc_std

        plt.plot(self.time, rescaled_data, **kwargs)
        

class LightCurves:
    """
    Object holding time-series of flux and data from multiple stellar observations. Act as a list of ``LightCurve``
    """
    def __init__(self, *lightcurves, **kwargs):
        if len(lightcurves) == 1:
            # Probably list(Lightucurves)
            self._lightcurves = lightcurves[0]
        elif len(lightcurves) == 3:
            # Probably list([time, fluxes, errors])
            time, fluxes, errors = lightcurves
            self._lightcurves = [
                LightCurve(time, flux, error) for flux, error in zip(fluxes, errors)
            ]
        self.apertures = None
        self.best_aperture_id = None
        
    def __getitem__(self, key):
        return self._lightcurves[key]

    def __len__(self):
        return len(self._lightcurves)
    
    def __iter__(self):
        return iter(self._lightcurves)

    @property
    def time(self):
        return np.hstack(self.times)
    
    @property
    def flux(self):
        return np.hstack(self.fluxes)

    @property
    def data(self):
        keys = [list(lc.data.keys()) for lc in self]
        n_keys = [len(k) for k in keys]
        if len(np.unique(n_keys)) != 1:
            raise AssertionError("LightCurves data should have same keys to be retrieved")

        n_keys = np.unique(n_keys)
        unique_keys = np.unique(keys)

        if len(unique_keys) != n_keys:
            raise AssertionError("LightCurves data should have same keys to be retrieved")

        return {key: np.hstack([lc.data[key] for lc in self]) for key in unique_keys}

    @property
    def error(self):
        return np.hstack(self.errors)
    
    @property
    def times(self):
        return [lc.time for lc in self._lightcurves]

    @property
    def fluxes(self):
        return [lc.flux for lc in self._lightcurves]

    @property
    def errors(self):
        return [lc.error for lc in self._lightcurves]

    def set_best_aperture_id(self, i):
        """
        Set a commonly share best aperture for all LightCurves

        Parameters
        ----------
        i : int
            index of the best aperture
        """
        if i is not None:
            for lc in self._lightcurves:
                lc.best_aperture_id = i
            self.best_aperture_id = i
    
    def from_ndarray(self, fluxes, errors):
        pass

    def as_array(self, aperture=None):
        """
        Return an ndarray with shape (n_apertures, n_lightcurves, n_times)

        Returns
        -------
        np.ndarray
        """
        if aperture is None:
            array = np.array([lc.fluxes for lc in self._lightcurves])
            error_array = np.array([lc.errors for lc in self._lightcurves])
            s, a, n = array.shape
            return np.moveaxis(array, 0, 1), np.moveaxis(error_array, 0, 1)
        else:
            array = np.array([lc.fluxes[aperture] for lc in self._lightcurves])
            error_array = np.array([lc.errors[aperture] for lc in self._lightcurves])
            return array, error_array

    def mask(self, mask):
        lcs = []
        for lc in self._lightcurves:
            _, _, idxs = np.intersect1d(lc.time, self.time, return_indices=True)
            sub_mask = mask[idxs]
            lcs.append(lc.mask(sub_mask))
        lcs = LightCurves(lcs)
        lcs.set_best_aperture_id(self.best_aperture_id)
        return lcs

    def sigma_clip(self, **kwargs):
        sigclip_mask = sigma_clip(self.flux, **kwargs).mask
        return self.mask(sigclip_mask)

    def binned(self, bins=0.005):
        return LightCurves([lc.binned(bins) for lc in self])

    def plot(self, **kwargs):
        viz.plot_lcs([[lc.time, lc.flux] for lc in self], **kwargs)

    def save(self, name):
        np.save("{}.lcs".format(name), [{
            "fluxes": lc.fluxes,
            "errors": lc.errors,
            "data": lc.data,
            "time": lc.time,
            "best_aperture_id": lc.best_aperture_id
        } for lc in self], allow_pickle=True)

    @staticmethod
    def load(filename):
        data = np.load(filename, allow_pickle=True)
        lcs = []
        for d in data:
            lc = LightCurve(d["time"], d["fluxes"], d["errors"], d["data"])
            lc.best_aperture_id = d["best_aperture_id"]
            lcs.append(lc)

        return LightCurves(lcs)

    def folded(self, t0, period):
        folded_time = utils.fold(self.time, t0, period)
        idxs = np.argsort(folded_time)
        folded_time = folded_time[idxs]
        flux = self.flux[idxs]
        error = self.error[idxs]

        lc = LightCurve(folded_time, [flux], [error])

        return lc