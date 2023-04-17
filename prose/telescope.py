import inspect
from dataclasses import asdict, dataclass
from datetime import datetime

import astropy.units as u
import numpy as np
import yaml
from dateutil import parser as dparser

from prose import CONFIG
from prose.console_utils import info


def str_to_astropy_unit(unit_string):
    return u.__dict__[unit_string]


# TODO: add exposure time unit
@dataclass
class Telescope:
    """Save and store FITS header keywords definition for a given telescope

    This is a Python Data Class, so that all attributes described below can be used as
    keyword-arguments when instantiating a Telescope
    """

    name: str = "Unknown"
    """Name taken by the telescope if saved"""

    names: tuple = ()
    """Alternative names that the telescope may take in the fits header values of 
    `keyword_telescope`"""

    # Keywords
    # --------
    keyword_telescope: str = "TELESCOP"
    """FITS header keyword for telescope name, default is :code:`"TELESCOP"`"""

    keyword_object: str = "OBJECT"
    """FITS header keyword for observed object name, default is :code:`"OBJECT"`"""

    keyword_image_type: str = "IMAGETYP"
    """ FITS header keyword for image type (e.g. dark, bias, science),
        default is :code:`"IMAGETYP"`"""

    keyword_light_images: str = "light"
    """value of `keyword_image_type` associated to science (aka light) images,
        default is :code:`"light"`"""

    keyword_dark_images: str = "dark"
    """value of `keyword_image_type` associated to dark calibration images,
    Default is :code:`"dark"`"""

    keyword_flat_images: str = "flat"
    """value of `keyword_image_type` associated to flat calibration images,
        default is :code:`"flat"`"""

    keyword_bias_images: str = "bias"
    """value of `keyword_image_type` associated to flat calibration images,
        default is :code:`"bias"`"""

    keyword_observation_date: str = "DATE-OBS"
    """FITS header keyword for observation date, default is "DATE:code:`-OBS"`"""

    keyword_exposure_time: str = "EXPTIME"
    """ FITS header keyword for exposure time, default is :code:`"EXPTIME"`"""

    keyword_filter: str = "FILTER"
    """FITS header keyword for filter, default is :code:`"FILTER"`"""

    keyword_airmass: str = "AIRMASS"
    """FITS header keyword for airmass, default is :code:`"AIRMASS"`"""

    keyword_fwhm: str = "FWHM"
    """FITS header keyword for image full-width-half-maximum (fwhm),
        default is :code:`"FWHM"`"""

    keyword_seeing: str = "SEEING"
    """FITS header keyword for image seeing, default is :code:`"SEEING"`"""

    keyword_ra: str = "RA"
    """FITS header keyword for right ascension, default is :code:`"RA"`"""

    keyword_dec: str = "DEC"
    """FITS header keyword for declination, default is :code:`"DEC"`"""

    keyword_jd: str = "JD"
    """ FITS header keyword for julian day, default is :code:`"JD"`"""

    keyword_bjd: str = "BJD"
    """FITS header keyword for barycentric julian day, default is :code:`"BJD"`"""

    keyword_flip: str = "PIERSIDE"
    """FITS header keyword for meridian flip configuration,
        default is :code:`"PIERSIDE"`"""

    # Units, formats and scales
    # -------------------------
    ra_unit: str = "deg"
    """unit of the value of `keyword_ra`, default is :code:`"deg"`"""

    dec_unit: str = "deg"
    """unit of the value of `keyword_dec`, default is :code:`"deg"`"""

    jd_scale: str = "utc"
    """unit of the value of `JD`, default is :code:`"utc"`"""

    bjd_scale: str = "utc"
    """ unit of the value of `BJD`, default is :code:`"utc"`"""

    mjd: float = 0.0
    """value to subtract from the value of `keyword_jd`"""

    # Specs
    # -----
    trimming: tuple = (0, 0)
    """horizontal and vertical overscan of an image in pixels,
        default is :code:`(0, 0)`"""

    read_noise: float = 9
    """detector read noise in ADU, default is :code:`9`"""

    gain: float = 1
    """detector gain in electrons/ADU, default is :code:`1`"""

    altitude: float = 2000
    """altitude of the telescope in meters, default is :code:`2000`,"""

    diameter: float = 100
    """diameter of the telescope in centimeters, default is :code:`100`"""

    pixel_scale: float = None
    """pixel scale (or plate scale) of the detector in arcsec/pixel,
        default is :code:`None`"""

    latlong: tuple = (None, None)
    """latitude and longitude of the telescope, default is :code:`(None, None)`"""

    saturation: float = 55000
    """detector's pixels full depth (saturation) in ADU, default is :code:`55000`"""

    hdu: int = 0
    """index of the FITS HDU where to find image data, default is :code:`0`"""

    camera_name: str = None
    """name of the telescope camera, default is :code:`None`"""

    date_string_format: str = None
    """date string format, default is :code:`None`"""

    _default: bool = True
    save: bool = False

    def __post_init__(self):
        if self.save:
            telescope_dict = asdict(self)
            del telescope_dict["_default"]
            del telescope_dict["save"]
            CONFIG.save_telescope_file(telescope_dict)

    @classmethod
    def load(cls, filename):
        return cls.from_dict(**yaml.full_load(open(filename, "r")))

    @classmethod
    def from_dict(cls, env):
        """Load from a dict ensuring that only class attributes are used"""
        return cls(
            **{k: v for k, v in env.items() if k in inspect.signature(cls).parameters}
        )

    @property
    def earth_location(self):
        from astropy.coordinates import EarthLocation

        if self.latlong[0] is None or self.latlong[1] is None:
            return None
        else:
            return EarthLocation(self.latlong[1], self.latlong[0], self.altitude)

    # TODO keep?
    def error(self, signal, area, sky, exposure, airmass=None, scinfac=0.09):
        _signal = signal.copy()
        _squarred_error = _signal + area * (
            self.read_noise**2 + (self.gain / 2) ** 2 + sky
        )

        if airmass is not None:
            scintillation = (
                scinfac
                * np.power(self.diameter, -0.6666)
                * np.power(airmass, 1.75)
                * np.exp(-self.altitude / 8000.0)
            ) / np.sqrt(2 * exposure)

            _squarred_error += np.power(signal * scintillation, 2)

        return np.sqrt(_squarred_error)

    @classmethod
    def from_name(cls, name, verbose=True, strict=False):
        telescope_dict = CONFIG.match_telescope_name(name)
        if telescope_dict is not None:
            telescope = cls.from_dict(telescope_dict)
        else:
            if strict:
                return None

            telescope = cls()
            telescope.name = name
            if verbose:
                info(f"telescope {name} not found - using default")
        return telescope

    @staticmethod
    def from_names(instrument_name, telescope_name, verbose=True, strict=True):
        # we first check by instrument name
        telescope = Telescope.from_name(instrument_name, verbose=False, strict=True)
        # if not found we check telescope name
        if telescope is None:
            telescope = Telescope.from_name(telescope_name, verbose=verbose)

        if telescope is None:
            if not strict:
                telescope = Telescope()
                telescope.name = f"default_{telescope_name}"

        return telescope

    def date(self, header):
        header_date_str = header.get(self.keyword_observation_date, None)
        if header_date_str is not None:
            if self.date_string_format is not None:
                return datetime.strptime(header_date_str, self.date_string_format)
            else:
                return dparser.parse(header_date_str)
        else:
            return datetime(1800, 1, 2)

    def image_type(self, header):
        return header.get(self.keyword_image_type, "").lower()
