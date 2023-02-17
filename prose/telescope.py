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
from dataclasses import dataclass


def str_to_astropy_unit(unit_string):
    return u.__dict__[unit_string]


# TODO: add exposure time unit
@dataclass
class Telescope:
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
    mjd: float = 0.

    # Specs
    # -----
    trimming: tuple = (0, 0) # pixels along y/x
    read_noise: float = 9 # ADU
    gain: float = 1 # e-/ADU
    altitude: float = 2000, # meters
    diameter: float = 100, # meters
    pixel_scale: float = None, # arcsec/pixel
    latlong: tuple = (None, None), 
    saturation: float = 55000, # ADUs
    hdu: int = 0,
    camera_name: str = None

    default: bool = True
    """Object containing telescope information.

    Once a new telescope is instantiated its dictionary is permanantly saved by prose and automatically used whenever the telescope name is encountered in a fits header. Saved telescopesare located in ``~/.prose`` as ``.telescope`` files (yaml format).
    """

    def __post_init__(self):
        pass

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
        _squarred_error = _signal + area * (self.read_noise ** 2 + (self.gain / 2) ** 2 + sky)

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
            telescope = cls(**telescope_dict, default=False)
        else:
            if strict:
                return None

            telescope = cls()
            telescope.name = name
            if verbose:
                info(f"telescope {name} not found - using default")
        return telescope

    @staticmethod
    def from_names(instrument_name, telescope_name, verbose=True):
        # we first check by instrument name
        telescope = Telescope.from_name(instrument_name, verbose=False, strict=True)
        # if not found we check telescope name
        if telescope is None:
            telescope = Telescope.from_name(telescope_name, verbose=verbose)
        
        return telescope

    def date(self, header):
        return dparser.parse(header.get(self.keyword_observation_date, ""))

    def image_type(self, header):
        return header.get(self.keyword_image_type, "").lower()