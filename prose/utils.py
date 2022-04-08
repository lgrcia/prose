from datetime import timedelta
import numpy as np
from astropy.visualization import ZScaleInterval
from astropy.io import fits
import numba
import astropy.constants as c
import urllib
from astropy.time import Time
from astropy.table import Table
from datetime import datetime
import inspect
from scipy import ndimage

earth2sun = (c.R_earth / c.R_sun).value

def remove_sip(dict_like):

    for kw in [
        'A_ORDER',
        'A_0_2',
        'A_1_1',
        'A_2_0',
        'B_ORDER',
        'B_0_2',
        'B_1_1',
        'B_2_0',
        'AP_ORDER',
        'AP_0_0',
        'AP_0_1',
        'AP_0_2',
        'AP_1_0',
        'AP_1_1',
        'AP_2_0',
        'BP_ORDER',
        'BP_0_0',
        'BP_0_1',
        'BP_0_2',
        'BP_1_0',
        'BP_1_1',
        'BP_2_0'
    ]:
        if kw in dict_like:
            del dict_like[kw]


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
        return (date - timedelta(hours=15)).date()  # If obs goes up to 15pm it still belongs to day before
    else:
        return date


def std_diff_metric(fluxes):
    k = len(list(np.shape(fluxes)))
    return np.std(np.diff(fluxes, axis=k - 1), axis=k - 1)


def stability_aperture(fluxes):
    lc_c = np.abs(np.diff(fluxes, axis=0))
    return np.mean(lc_c, axis=1)


def binning(x, y, bins, error=None, std=False, mean_method=np.mean,
            mean_error_method=lambda x: np.sqrt(np.sum(np.power(x, 2))) / len(x)):

    bins = np.arange(np.min(x), np.max(x), bins)
    d = np.digitize(x, bins)

    final_bins = []
    binned_flux = []
    binned_error = []
    _std = []

    for i in range(1, np.max(d) + 1):
        s = np.where(d == i)
        if len(s[0]) > 0:
            binned_flux.append(mean_method(y[s[0]]))
            final_bins.append(np.mean(x[s[0]]))
            _std.append(np.std(y[s[0]]) / np.sqrt(len(s[0])))
            if error is not None:
                binned_error.append(mean_error_method(error[s[0]]))

    if std:
        return np.array(final_bins), np.array(binned_flux), np.array(_std)
    elif error is not None and isinstance(error, (np.ndarray, list)):
        return np.array(final_bins), np.array(binned_flux), np.array(binned_error)
    else:
        return np.array(final_bins), np.array(binned_flux)


# @numba.jit(fastmath=True, parallel=False, nopython=True)
# def fast_binning(x, y, bins, error=None, std=False):
#     bins = np.arange(np.min(x), np.max(x), bins)
#     d = np.digitize(x, bins)
#
#     binned_x = []
#     binned_y = []
#     binned_error = []
#
#     for i in range(1, np.max(d) + 1):
#         s = np.where(d == i)
#         if len(s[0]) > 0:
#             s = s[0]
#             binned_y.append(np.mean(y[s]))
#             binned_x.append(np.mean(x[s]))
#             binned_error.append(np.std(y[s]) / np.sqrt(len(s)))
#
#             if error is not None:
#                 err = error[s]
#                 binned_error.append(np.sqrt(np.sum(np.power(err, 2))) / len(err))
#             else:
#                 binned_error.append(np.std(y[s]) / np.sqrt(len(s)))
#
#     return np.array(binned_x), np.array(binned_y), np.array(binned_error)


@numba.jit(fastmath=True, parallel=False, nopython=True)
def fast_binning(x, y, bins, error=None, std=False):
    bins = np.arange(np.min(x), np.max(x), bins)
    d = np.digitize(x, bins)

    n = np.max(d) + 2

    binned_x = np.empty(n)
    binned_y = np.empty(n)
    binned_error = np.empty(n)

    binned_x[:] = -np.pi
    binned_y[:] = -np.pi
    binned_error[:] = -np.pi

    for i in range(0, n):
        s = np.where(d == i)
        if len(s[0]) > 0:
            s = s[0]
            binned_y[i] = np.mean(y[s])
            binned_x[i] = np.mean(x[s])
            binned_error[i] = np.std(y[s]) / np.sqrt(len(s))

            if error is not None:
                err = error[s]
                binned_error[i] = np.sqrt(np.sum(np.power(err, 2))) / len(err)
            else:
                binned_error[i] = np.std(y[s]) / np.sqrt(len(s))

    nans = binned_x == -np.pi

    return binned_x[~nans], binned_y[~nans], binned_error[~nans]


@numba.jit(fastmath=True, parallel=False, nopython=True)
def index_binning(x, bins):
    bins = np.arange(np.min(x), np.max(x), bins)
    d = np.digitize(x, bins)
    n = np.max(d) + 2
    indexes = []

    for i in range(0, n):
        s = np.where(d == i)
        if len(s[0]) > 0:
            s = s[0]
            indexes.append(s)

    return indexes


@numba.jit(fastmath=True, parallel=False, nopython=True)
def fast_points_binning(x, y, n):
    n = int(len(x) / n)
    bins = np.linspace(x.min(), x.max(), n)
    digitized = np.digitize(x, bins)
    binned_mean = np.zeros(n)
    binned_std = np.zeros(n)
    binned_time = np.zeros(n)
    for i in range(1, len(bins)):
        binned_mean[i] = y[digitized == i].mean()
        binned_std[i] = y[digitized == i].std()
        binned_time[i] = x[digitized == i].mean()

    return binned_time, binned_mean, binned_std


def z_scale(data, c=0.05):
    if type(data) == str:
        data = fits.getdata(data)
    interval = ZScaleInterval(contrast=c)
    return interval(data.copy())


def rescale(y):
    ry = y - np.mean(y)
    return ry/np.std(ry)


def check_class(_class, base, default):
    if _class is None:
        return default
    elif isinstance(_class, base):
        return _class
    else:
        raise TypeError("subclass of {} expected".format(base.__name__))


def divisors(n):
    _divisors = []
    i = 1
    while i <= n:
        if n % i == 0:
            _divisors.append(i)
        i = i + 1
    return np.array(_divisors)


def fold(t, t0, p):
    return (t - t0 + 0.5 * p) % p - 0.5 * p


def header_to_cdf4_dict(header):

    header_dict = {}

    for key, value in header.items():
        if isinstance(value, str):
            if len(key) > 0 and len(value) > 0:
                header_dict[key] = value
        elif isinstance(value, (float, np.ndarray, np.number)):
            header_dict[key] = float(value)
        elif isinstance(value, (int, bool)):
            header_dict[key] = int(value)
        else:
            pass

    return header_dict


def years_to_datetime(years):
    """
    https://stackoverflow.com/questions/19305991/convert-fractional-years-to-a-real-date-in-python
    Convert atime (a float) to DT.datetime
    This is the inverse of dt2t.
    assert dt2t(t2dt(atime)) == atime
    """
    year = int(years)
    remainder = years - year
    boy = datetime(year, 1, 1)
    eoy = datetime(year + 1, 1, 1)
    seconds = remainder * (eoy - boy).total_seconds()
    return boy + timedelta(seconds=seconds)


def datetime_to_years(adatetime):
    """
    https://stackoverflow.com/questions/19305991/convert-fractional-years-to-a-real-date-in-python
    Convert adatetime into a float. The integer part of the float should
    represent the year.
    Order should be preserved. If adate<bdate, then d2t(adate)<d2t(bdate)
    time distances should be preserved: If bdate-adate=ddate-cdate then
    dt2t(bdate)-dt2t(adate) = dt2t(ddate)-dt2t(cdate)
    """
    year = adatetime.year
    boy = datetime(year, 1, 1)
    eoy = datetime(year + 1, 1, 1)
    return year + ((adatetime - boy).total_seconds() / ((eoy - boy).total_seconds()))


def split(x, dt, fill=None):
    splits = np.argwhere(np.diff(x) > dt).flatten() + 1
    xs = np.split(x, splits)
    if fill is None:
        return xs
    else:
        ones = np.ones_like(x)
        filled_xs = [np.split(ones * fill, splits) for _ in xs]
        for i in range(len(xs)):
            filled_xs[i][i] = xs[i]
        for i in range(len(xs)):
            filled_xs[i] = np.hstack(filled_xs[i])
        return [np.hstack(fx) for fx in filled_xs]


def jd_to_bjd(jd, ra, dec):
    """
    Convert JD to BJD using http://astroutils.astronomy.ohio-state.edu (Eastman et al. 2010)
    """
    bjd = urllib.request.urlopen(f"http://astroutils.astronomy.ohio-state.edu/time/convert.php?JDS={','.join(jd.astype(str))}&RA={ra}&DEC={dec}&FUNCTION=utc2bjd").read()
    bjd = bjd.decode("utf-8")
    return np.array(bjd.split("\n"))[0:-1].astype(float)


def remove_arrays(d):
    copy = d.copy()
    for name, value in d.items():
        if isinstance(value, (list, np.ndarray)):
            del copy[name]
    return copy


def sigma_clip(y, sigma=5., return_mask=False, x=None):
    mask = np.abs(y - np.nanmedian(y)) < sigma * np.nanstd(y)

    if return_mask:
        return mask

    else:
        if x is None:
            return y[mask]
        else:
            return x[mask], y[mask]


def register_args(f):
    """
    When used within a class, saves args and kwargs passed to a function
    (mostly used to record __init__ inputs)
    """
    def inner(*args, **kwargs):
        self = args[0]
        self.args = args[1::]
        self.kwargs = kwargs
        return f(*args, **kwargs)
    return f
    

def nan_gaussian_filter(data, sigma=1., truncate=4.):
    """https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python

    Parameters
    ----------
    U : _type_
        _description_
    sigma : _type_, optional
        _description_, by default 1.
    truncate : _type_, optional
        _description_, by default 4.
    """

    V=data.copy()
    V[np.isnan(data)]=0
    VV=ndimage.gaussian_filter(V,sigma=sigma,truncate=truncate)

    W=0*data.copy()+1
    W[np.isnan(data)]=0
    WW=ndimage.gaussian_filter(W,sigma=sigma,truncate=truncate)

    return VV/WW

def clean_header(header_dict):
    return {key: value for key, value in header_dict.items() if not isinstance(value, (list, tuple)) and key.isupper()}


def easy_median(images):
    # To avoid memory errors, we split the median computation in 50
    images = np.array(images)
    shape_divisors = divisors(images.shape[1])
    n = shape_divisors[np.argmin(np.abs(50 - shape_divisors))]
    return np.concatenate([np.nanmedian(im, axis=0) for im in np.split(images, n, axis=1)])


def image_in_xarray(image, xarr, name="stack", stars=False):
    xarr.attrs.update(header_to_cdf4_dict(image.header))
    xarr.attrs.update(dict(
        telescope=image.telescope.name,
        filter=image.header.get(image.telescope.keyword_filter, ""),
        exptime=image.header.get(image.telescope.keyword_exposure_time, ""),
        name=image.header.get(image.telescope.keyword_object, ""),
    ))

    if image.telescope.keyword_observation_date in image.header:
        date = image.header[image.telescope.keyword_observation_date]
    else:
        date = Time(image.header[image.telescope.keyword_jd], format="jd").datetime

    xarr.attrs.update(dict(date=format_iso_date(date).isoformat()))
    xarr.coords[name] = (('w', 'h'), image.data)

    xarr = xarr.assign_coords(time=xarr.jd_utc)
    xarr = xarr.sortby("time")
    xarr.attrs["time_format"] = "jd_utc"
    
    if stars:
        xarr = xarr.assign_coords(stars=(("star", "n"), image.stars_coords))

    
    return xarr