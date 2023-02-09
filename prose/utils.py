from datetime import timedelta
import numpy as np
from astropy.visualization import ZScaleInterval
from astropy.io import fits
import astropy.constants as c
import urllib
from astropy.time import Time
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from datetime import datetime
import inspect
from scipy import ndimage
from functools import wraps
from collections import OrderedDict
import numpy as np
from astropy.stats import gaussian_sigma_to_fwhm

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


def index_binning(x, size):
    if isinstance(size, float):
        bins = np.arange(np.min(x), np.max(x), size)
    else:
        x = np.arange(0, len(x))
        bins = np.arange(0., len(x), size)
        
    d = np.digitize(x, bins)
    n = np.max(d) + 2
    indexes = []

    for i in range(0, n):
        s = np.where(d == i)
        if len(s[0]) > 0:
            s = s[0]
            indexes.append(s)

    return indexes


def z_scale(data, c=0.05):
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


def args_kwargs(f):
    s = inspect.signature(f)
    args = []
    kwargs = {}
    for p in s.parameters.values():
        if p.default != inspect._empty:
            kwargs[p.name] = p.default
        else:
            args.append(p.name)
    return args, kwargs

# todo: adapt to work with positional parameters like register
def register_args(f):
    """
    When used within a class, saves args and kwargs passed to a function
    (mostly used to record __init__ inputs)
    """
    @wraps(f)
    def inner(*_args, **_kwargs):
        self = _args[0]
        args, kwargs = args_kwargs(f)
        args = dict(zip(args[1::], _args[1::]))
        kwargs.update(_kwargs)
        self.args = args
        self.kwargs = kwargs
        return f(self, *args.values(), **kwargs)
    return inner
    

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

def check_skycoord(skycoord):
    """
    Check that skycoord is either:
    - a list of int (interpreted as deg)
    - a str (interpreted as houranlgle, deg)
    - a SkyCoord object

    and return a SkyCoord object

    Parameters
    ----------
    skycoord : list, tuple or SkyCoord
        coordinate of the image center

    Raise
    -----
    Raise an error if skycoord cannot be interpreted

    """
    if isinstance(skycoord, (tuple, list)):
        if isinstance(skycoord[0], (float, int)):
            skycoord = SkyCoord(*skycoord, unit=(u.deg, u.deg))
        elif isinstance(skycoord[0], str):
            skycoord = SkyCoord(*skycoord, unit=("hourangle", "deg"))
        else:
            if not isinstance(skycoord, SkyCoord):
                assert "'skycoord' must be a list of int (interpreted as deg), str (interpreted as houranlgle, deg) or SkyCoord object"

    return skycoord


def gaia_query(center, fov, *args, limit=10000, circular=True):
    """
    https://gea.esac.esa.int/archive/documentation/GEDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html
    """
    
    from astroquery.gaia import Gaia
    
    if isinstance(center, SkyCoord):
        ra = center.ra.to(u.deg).value
        dec = center.dec.to(u.deg).value
    
    if not isinstance(fov, u.Quantity):
        fov = fov * u.deg
    
    if fov.ndim == 1:
        ra_fov, dec_fov = fov.to(u.deg).value
    else:
        ra_fov = dec_fov = fov.to(u.deg).value

    radius = np.min([ra_fov, dec_fov])/2

    fields = ','.join(args) if isinstance(args, (tuple, list)) else args

    if circular:
        job = Gaia.launch_job(f"select top {limit} {fields} from gaiadr2.gaia_source where "
                            "1=CONTAINS("
                            f"POINT('ICRS', {ra}, {dec}), "
                            f"CIRCLE('ICRS',ra, dec, {radius}))"
                            "order by phot_g_mean_mag")
    else:
        job = Gaia.launch_job(f"select top {limit} {fields} from gaiadr2.gaia_source where "
                    f"ra BETWEEN {ra-ra_fov/2} AND {ra+ra_fov/2} AND "
                    f"dec BETWEEN {dec-dec_fov/2} AND {dec+dec_fov/2} "
                    "order by phot_g_mean_mag") 

    return job.get_results()

def sparsify(stars, radius):

    _stars = stars.copy()
    deleted_stars = np.zeros([], dtype=int)
    sparse_stars = []
    
    for i, s in enumerate(_stars):
        if not i in deleted_stars:
            distances = np.linalg.norm(_stars-s, axis=1)
            idxs = np.flatnonzero(distances<radius)
            sparse_stars.append(s)
            deleted_stars = np.hstack([deleted_stars, idxs])
    
    return np.array(sparse_stars)

def full_class_name(o):
    # https://stackoverflow.com/questions/2020014/get-fully-qualified-class-name-of-an-object-in-python
    klass = o.__class__
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__ # avoid outputs like 'builtins.str'
    return module + '.' + klass.__qualname__


def binn2D(arr, factor):
    new_shape = np.array(arr.shape) // factor
    shape = (new_shape[0], factor,
             new_shape[1], factor)
    return np.mean(arr.reshape(shape).mean(-1), 1)

import numpy as np
from scipy.spatial import KDTree
from twirl import utils as tutils
from skimage.transform import AffineTransform as skAT
from functools import partial

def distance(p1, p2):
    return np.sqrt(np.power(p1[0] - p2[0], 2) + np.power(p1[1] - p2[1], 2))


def distances(coords, coord):
    return [
        np.sqrt(((coord[0] - x)**2 + (coord[1] - y)**2))
        for x, y in zip(coords[0].flatten(), coords[1].flatten())
    ]

def clean_stars_positions(positions, tolerance=50, output_id=False):
    keep = []

    distance_to_others = np.array(
        [[distance(v, w) for w in positions] for v in positions]
    )
    for i, _distances in enumerate(distance_to_others):
        _distances[i] = np.inf
        close_stars = np.flatnonzero(_distances < tolerance)
        if len(close_stars) == 0:
            keep.append(i)

    if output_id:
        return positions[np.unique(keep)], np.unique(keep)
    else:
        return positions[np.unique(keep)]


def cross_match(S1, S2, tolerance=10, return_idxs=False, none=True):
    # cleaning
    s1 = S1.copy()
    s2 = S2.copy()
    
    s1[np.any(np.isnan(s1), 1)] = (1e15, 1e15)
    s2[np.any(np.isnan(s2), 1)] = (1e15, 1e15)
    
    # matching
    matches = []

    for i, s in enumerate(s1):
        distances = np.linalg.norm(s - s2, axis=1)
        closest = np.argmin(distances)
        if distances[closest] < tolerance:
            matches.append([i, closest])
        else:
            if none:
                matches.append([i, np.nan])

    matches = np.array(matches)

    if return_idxs:
        return matches
    else:
        if len(matches) > 0:
            return s1[matches[:, 0]], s2[matches[:, 1]]
        else:
            return np.array([]), np.array([])

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    height = data.max()
    background = data.min()
    data = data-np.min(data)
    total = data.sum()
    x, y = np.indices(data.shape)
    x = (x * data).sum() / total
    y = (y * data).sum() / total
    col = data[:, int(y)]
    width_x = np.sqrt(abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum())
    width_x /= gaussian_sigma_to_fwhm
    width_y /= gaussian_sigma_to_fwhm
    return {
        "amplitude": height, 
        "x": x, 
        "y": y, 
        "sigma_x": width_x, 
        "sigma_y": width_y, 
        "background": background,
        "theta": 0.
    }