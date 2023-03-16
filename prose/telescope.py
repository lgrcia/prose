from os import path
import yaml
import numpy as np
from . import CONFIG
import astropy.units as u
from warnings import warn
from .console_utils import info
from .builtins import default
import astropy.units as u
from dateutil import parser as dparser
from dataclasses import dataclass, asdict
from datetime import datetime


def str_to_astropy_unit(unit_string):
    return u.__dict__[unit_string]


# TODO: add exposure time unit
@dataclass
class Telescope:
    """Save and store FITS header keywords definition for a given telescope

    Parameters
    ----------
    save: bool
        whether to save telescope in ~/.prose, by default False
    keyword_telescope: str
        FITS header keyword for telescope name, default is :code:`"TELESCOP"`
    keyword_object: str
        FITS header keyword for observed object name, default is :code:`"OBJECT"`
    keyword_image_type: str
        FITS header keyword for image type (e.g. dark, bias, science),
        default is :code:`"IMAGETYP"`
    keyword_light_images: str
        value of `keyword_image_type` associated to science (aka light) images,
        default is :code:`"light"`
    keyword_dark_images: str
        value of `keyword_image_type` associated to dark calibration images,
        default is :code:`"dark"`
    keyword_flat_images: str
        value of `keyword_image_type` associated to flat calibration images,
        default is :code:`"flat"`
    keyword_bias_images: str
        value of `keyword_image_type` associated to flat calibration images,
        default is :code:`"bias"`
    keyword_observation_date:
        FITS header keyword for observation date, default is "DATE:code:`-OBS"`
    keyword_exposure_time: str
        FITS header keyword for exposure time, default is :code:`"EXPTIME"`
    keyword_filter: str
        FITS header keyword for filter, default is :code:`"FILTER"`
    keyword_airmass: str
        FITS header keyword for airmass, default is :code:`"AIRMASS"`
    keyword_fwhm: str
        FITS header keyword for image full-width-half-maximum (fwhm),
        default is :code:`"FWHM"`
    keyword_seeing: str
        FITS header keyword for image seeing, default is :code:`"SEEING"`
    keyword_ra: str
        FITS header keyword for right ascension, default is :code:`"RA"`
    keyword_dec: str
        FITS header keyword for declination, default is :code:`"DEC"`
    keyword_jd: str
        FITS header keyword for julian day, default is :code:`"JD"`
    keyword_bjd: str
        FITS header keyword for barycentric julian day, default is :code:`"BJD"`
    keyword_flip: str
        FITS header keyword for meridian flip configuration,
        default is :code:`"PIERSIDE"`
    ra_unit: str
        unit of the value of `keyword_ra`, default is :code:`"deg"`
    dec_unit: str
        unit of the value of `keyword_dec`, default is :code:`"deg"`
    jd_scale: str
        unit of the value of `JD`, default is :code:`"utc"`
    bjd_scale: str
        unit of the value of `BJD`, default is :code:`"utc"`
    trimming: tuple
        horizontal and vertical overscan of an image in pixels,
        default is :code:`(0, 0)`
    read_noise: float
        detector read noise in ADU, default is :code:`9`
    gain: float
        detector gain in electrons/ADU, default is :code:`1`
    altitude: float
        altitude of the telescope in meters, default is :code:`2000`,
    diameter: float
        diameter of the telescope in centimeters, default is :code:`100`
    pixel_scale: float
        pixel scale (or plate scale) of the detector in arcsec/pixel,
        default is :code:`None`
    latlong: tuple
        latitude and longitude of the telescope, default is :code:`(None, None)`
    saturation: float
        detector's pixels full depth (saturation) in ADU, default is :code:`55000`
    hdu: int
        index of the FITS HDU where to find image data, default is :code:`0`
    camera_name: str
        name of the telescope camera, default is :code:`None`
    """

    name: str = "Unknown"
    names: tuple = ()

    # Keywords
    # --------
    keyword_telescope: str = "TELESCOP"
    keyword_object: str = "OBJECT"
    keyword_image_type: str = "IMAGETYP"
    keyword_light_images: str = "light"
    keyword_dark_images: str = "dark"
    keyword_flat_images: str = "flat"
    keyword_bias_images: str = "bias"
    keyword_observation_date: str = "DATE-OBS"
    keyword_exposure_time: str = "EXPTIME"
    keyword_filter: str = "FILTER"
    keyword_airmass: str = "AIRMASS"
    keyword_fwhm: str = "FWHM"
    keyword_seeing: str = "SEEING"
    keyword_ra: str = "RA"
    keyword_dec: str = "DEC"
    keyword_jd: str = "JD"
    keyword_bjd: str = "BJD"
    keyword_flip: str = "PIERSIDE"
    keyword_observation_time: str = None

    # Units, formats and scales
    # -------------------------
    ra_unit: str = "deg"
    dec_unit: str = "deg"
    jd_scale: str = "utc"
    bjd_scale: str = "utc"
    mjd: float = 0.0

    # Specs
    # -----
    trimming: tuple = (0, 0)  # pixels along y/x
    read_noise: float = 9  # ADU
    gain: float = 1  # e-/ADU
    altitude: float = 2000  # meters
    diameter: float = 100  # meters
    pixel_scale: float = None  # arcsec/pixel
    latlong: tuple = (None, None)
    saturation: float = 55000  # ADUs
    hdu: int = 0
    camera_name: str = None

    _default: bool = True
    save: bool = False
    """Object containing telescope information.

    Once a new telescope is instantiated its dictionary is permanantly saved by prose and automatically used whenever the telescope name is encountered in a fits header. Saved telescopesare located in ``~/.prose`` as ``.telescope`` files (yaml format).
    """

    def __post_init__(self):
        if self.save:
            telescope_dict = asdict(self)
            del telescope_dict["_default"]
            del telescope_dict["save"]
            CONFIG.save_telescope_file(telescope_dict)

    @classmethod
    def load(cls, filename):
        return cls(**yaml.full_load(open(filename, "r")))

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
            telescope = cls(**telescope_dict, _default=False)
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
        header_date = header.get(self.keyword_observation_date, None)
        if header_date is not None:
            return dparser.parse(header_date)
        else:
            return datetime(1800, 1, 2)

    def image_type(self, header):
        return header.get(self.keyword_image_type, "").lower()
