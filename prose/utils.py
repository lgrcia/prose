from astropy.time import Time
from datetime import datetime, timedelta
import numpy as np
from astropy.visualization import ZScaleInterval
from astropy.io import fits


def format_iso_date(date, night_date=True):
    """
    Return a datetime.date corresponding to the day 12 hours before given datetime.
    Used as a reference day, e.g. if a target is observed the 24/10 at 02:30, observation date
    is 23/10, day when night begin.

    Parameters
    ----------
    date : str or datetime
        if str: "fits" fromated date and time
        if datetime: datetime
    night_date : bool, optional
        return day 12 hours before given date and time, by default True

    Returns
    -------
    datetime.date
        formatted date
    """
    if isinstance(date, str):
        date = Time(date, format="fits").datetime
    elif isinstance(date, datetime):
        date = Time(date, format="datetime").datetime

    if night_date:
        return (date - timedelta(hours=12)).date()
    else:
        return date


def std_diff_metric(fluxes):
    k = len(list(np.shape(fluxes)))
    return np.std(np.diff(fluxes, axis=k - 1), axis=k - 1)


def stability_aperture(fluxes):
    lc_c = np.abs(np.diff(fluxes, axis=0))
    return np.mean(lc_c, axis=1)


def binning(jd, flux, bins, error=None, std=False, mean_method=np.mean,
            mean_error_method=lambda l: np.sqrt(np.sum(np.power(l, 2))) / len(l)):

    bins = np.arange(np.min(jd), np.max(jd), bins)
    d = np.digitize(jd, bins)

    final_bins = []
    binned_flux = []
    binned_error = []
    _std = []

    for i in range(1, np.max(d) + 1):
        s = np.where(d == i)
        if len(s[0]) > 0:
            binned_flux.append(mean_method(flux[s[0]]))
            final_bins.append(np.mean(jd[s[0]]))
            _std.append(np.std(flux[s[0]]) / np.sqrt(len(s[0])))
            if error is not None:
                binned_error.append(mean_error_method(error[s[0]]))

    if std:
        return np.array(final_bins), np.array(binned_flux), np.array(_std)
    elif error is not None and isinstance(error, (np.ndarray, list)):
        return np.array(final_bins), np.array(binned_flux), np.array(binned_error)
    else:
        return np.array(final_bins), np.array(binned_flux)


def z_scale(data, c=0.05):
    if type(data) == str:
        data = fits.getdata(data)
    interval = ZScaleInterval(contrast=c)
    return interval(data.copy())