from os import path
import yaml
import numpy as np
from . import CONFIG
import astropy.units as u
from warnings import warn
from .console_utils import info
from .builtins import default

def str_to_astropy_unit(unit_string):
    return u.__dict__[unit_string]


# TODO: add exposure time unit
class Telescope:
    """Object containing telescope information.
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
            else:
                raise AssertionError("telescope must be a dict or a path str")
        else:
            self.__dict__.update(default)
                
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
                if self.verbose:
                    info(f"telescope {file} not found - using default")
                self.name = file

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
    def from_name(name, verbose=True):
        telescope = Telescope(verbose=verbose)
        telescope.load(name)
        return telescope

    # TODO: explain in documentation
    def date(self, header):
        _date = header.get(self.keyword_observation_date, None)
        if _date is None:
            _date = "2000-01-01T00:00:00.000"
        else:
            if self.keyword_observation_time is not None:
                _date = _date + "T" + header.get(self.keyword_observation_time, "00:00:00.000")

        return _date

    def image_type(self, header):
        return header.get(self.keyword_image_type, "").lower()
