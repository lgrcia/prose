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

def str_to_astropy_unit(unit_string):
    return u.__dict__[unit_string]


# TODO: add exposure time unit
class Telescope:
    """Object containing telescope information.

    Once a new telescope is instantiated its dictionary is permanantly saved by prose and automatically used whenever the telescope name is encountered in a fits header. Saved telescopesare located in ``~/.prose`` as ``.telescope`` files (yaml format).


    Example
    -------

    .. jupyter-execute::

        from prose import Telescope

        telescope_dict = dict(
            # Name(s)
            # -------
            name = "Unknown",
            names = [],

            # Keywords
            # --------
            keyword_telescope = "TELESCOP",
            keyword_object = "OBJECT",
            keyword_image_type = "IMAGETYP",
            keyword_light_images = "light",
            keyword_dark_images = "dark",
            keyword_flat_images = "flat",
            keyword_bias_images = "bias",
            keyword_observation_date = "DATE-OBS",
            keyword_exposure_time = "EXPTIME",
            keyword_filter = "FILTER",
            keyword_airmass = "AIRMASS",
            keyword_fwhm = "FWHM",
            keyword_seeing = "SEEING",
            keyword_ra = "RA",
            keyword_dec = "DEC",
            keyword_jd = "JD",
            keyword_bjd = "BJD",
            keyword_flip = "PIERSIDE",
            keyword_observation_time = None,

            # Units, formats and scales
            # -------------------------
            ra_unit = "deg",
            dec_unit = "deg",
            jd_scale = "utc",
            bjd_scale = "utc",
            mjd = 0,
            
            # Specs
            # -----
            trimming = (0, 0), # in piwel along y/x
            read_noise = 9, # in ADU
            gain = 1, # in e-/ADU
            altitude = 2000, # in meters
            diameter = 100, # in meters
            pixel_scale = None, # in arcseconds
            latlong = [None, None], 
            saturation = 55000, # in ADU
            hdu = 0
        )

        telescope = Telescope(telescope_dict)

    """
    def __init__(self, telescope=None, verbose=True):
        """Object containing telescope information.

        Parameters
        ----------
        telescope : dict or str, optional
            telescope dict ot path of the .telescope file containing the dict in yaml format, by default None
        verbose : bool, optional
            whether to talk, by default True
        """

        self.verbose = verbose

        if telescope is not None:
            if isinstance(telescope, str):
                success = self.load(telescope)
                if success:
                    CONFIG.save_telescope_file(telescope)
                    CONFIG.build_telescopes_dict()
            elif isinstance(telescope, dict):
                self.__dict__.update(telescope)
                CONFIG.save_telescope_file(telescope)
            else:
                raise AssertionError("telescope must be a dict or a path str")
        else:
            self.__dict__.update(default)
                
    def __getattribute__(self, name):
        if name == "ra_unit":
            return str_to_astropy_unit(self.__dict__[name])
        elif name == "dec_unit":
            return str_to_astropy_unit(self.__dict__[name])
        elif name == "pixel_scale":
            return self.__dict__[name] * u.arcsec
        return super(Telescope, self).__getattribute__(name)

    def load(self, file, verbose=True):
        if isinstance(file, str) and path.exists(file) and not path.isdir(file):
            with open(file, "r") as f:
                telescope = yaml.load(f)
        elif isinstance(file, dict):
            telescope = file
        elif isinstance(file, str):
            telescope = CONFIG.match_telescope_name(file)
            if telescope is None:
                if self.verbose and verbose:
                    info(f"telescope {file} not found - using default")
                self.name = file
                return False

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
        from astropy.coordinates import EarthLocation
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
    def from_name(name, verbose=True, strict=False):
        telescope = Telescope(verbose=verbose)
        success = telescope.load(name)
        if strict and not success:
            return None
        else:
            return telescope

    @staticmethod
    def from_names(instrument_name, telescope_name, verbose=True):
        # we first check by instrument name
        telescope = Telescope.from_name(instrument_name, strict=True, verbose=False)
        # if not found we check telescope name
        if telescope is None:
            telescope = Telescope.from_name(telescope_name, verbose=verbose)
        
        return telescope

    def date(self, header):
        return dparser.parse(header.get(self.keyword_observation_date, ""))

    def image_type(self, header):
        return header.get(self.keyword_image_type, "").lower()