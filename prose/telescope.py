from os import path
from astropy.coordinates import EarthLocation
import yaml
import numpy as np
from . import CONFIG
import astropy.units as u
from warnings import warn

def str_to_astropy_unit(unit_string):
    return u.__dict__[unit_string]


class Telescope:
    """Object containing telescope information.

    Parameters
    ----------
    telescope_file : dict or str, optional
        telescope dict or description file, by default None which load a "default" telescope

    """
    def __init__(self, telescope_file=None):

        # Keywords
        self.keyword_object = "OBJECT"
        self.keyword_image_type = "IMAGETYP"
        self.keyword_light_images = "light"
        self.keyword_dark_images = "dark"
        self.keyword_flat_images = "flat"
        self.keyword_bias_images = "bias"
        self.keyword_observation_date = "DATE-OBS"
        self.keyword_exposure_time = "EXPTIME"
        self.keyword_filter = "FILTER"
        self.keyword_observatory = "TELESCOP"
        self.keyword_airmass = "AIRMASS"
        self.keyword_fwhm = "FWHM"
        self.keyword_seeing = "SEEING"
        self.keyword_ra = "RA"
        self.keyword_dec = "DEC"
        self.ra_unit = "deg"
        self.dec_unit = "deg"
        self.jd_scale = "utc"
        self.bjd_scale = "utc"
        self.keyword_jd = "JD"
        self.mjd = 0
        self.keyword_bjd = "BJD"
        self.keyword_flip = "PIERSIDE"

        # Specs
        self.name = "Unknown"
        self.trimming = (0, 0)
        self.read_noise = 9
        self.gain = 1
        self.altitude = 2000
        self.diameter = 100
        self.pixel_scale = None
        self.latlong = [None, None]
        self.saturation = 55000

        if telescope_file is not None:
            success = self.load(telescope_file)
            if success:
                CONFIG.save_telescope_file(telescope_file)
                CONFIG.build_telescopes_dict()

    def __getattribute__(self, name):
        if name == "ra_unit":
            return str_to_astropy_unit(self.__dict__[name])
        elif name == "dec_unit":
            return str_to_astropy_unit(self.__dict__[name])
        return super(Telescope, self).__getattribute__(name)

    def load(self, file):
        if isinstance(file, str) and path.exists(file):
            with open(file, "r") as f:
                telescope = yaml.load(f)
        elif isinstance(file, dict):
            telescope = file
        elif isinstance(file, str):
            telescope = CONFIG.match_telescope_name(file)
            if telescope is None:
                warn(f"telescope {file} not found")

        elif file is None:
            return False
        else:
            raise ValueError("file must be path or dict")

        if telescope is not None:
            self.__dict__.update(telescope)

        if telescope is None:
            return False
        else:
            return True
    
    def is_new(self):
        return not self.name.lower() in CONFIG.telescopes_dict()

    @property
    def earth_location(self):
        if self.latlong[0] is None or self.latlong[1] is None:
            return None
        else:
            return EarthLocation(self.latlong[1], self.latlong[0], self.altitude)

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

    @staticmethod
    def from_name(name):
        telescope = Telescope()
        telescope.load(name)
        return telescope
