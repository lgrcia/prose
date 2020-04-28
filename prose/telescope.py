from os import path
from astropy.coordinates import EarthLocation
import yaml
from prose import CONFIG

class Telescope:
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
        self.keyword_fwhm = "FWHM"
        self.keyword_ra = "RA"
        self.keyword_dec = "DEC"
        self.keyword_julian_date = "JD"
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

        if telescope_file is not None:
            success = self.load(telescope_file)
            if success and self.is_new():
                CONFIG.save_telescope_file(telescope_file)

    def load(self, file):
        if isinstance(file, str) and path.exists(file):
            with open(file, "r") as f:
                telescope = yaml.load(f)
        elif isinstance(file, dict):
            telescope = file
        elif file is None:
            return False
        else:
            raise ValueError("file must be path or dict")
        
        self.__dict__.update(telescope)

        if telescope is None:
            return False
        else:
            return True
    
    def is_new(self):
        return not self.name.lower() in CONFIG.telescopes_dict()

    @property
    def earth_location(self):
        return EarthLocation(*self.latlong, self.altitude)
