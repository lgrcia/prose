from os import path
from astropy.coordinates import EarthLocation
import yaml

speculoos_south = {
    "keyword_object": "OBJECT", # keyword holding object or target name
    "keyword_image_type": "IMAGETYP", # keyword holding image type (calibration, science... etc)
    "keyword_light_images": "light", # value of `keyword_image_type` for science images ...
    "keyword_dark_images": "dark", # for dark images ...
    "keyword_flat_images": "flat", # for flat images ...
    "keyword_bias_images": "bias", # for bias images
    "keyword_observation_date": "DATE-OBS", # keyword holding observation date (ISO)
    "keyword_exposure_time": "EXPTIME", # keyword holding exposure time
    "keyword_filter": "FILTER", # keyword holding filter name
    "keyword_observatory": "OBSERVAT", # keyword holding observatory name
    "keyword_julian_date": "JD", # keyword holding time in JD
    "keyword_ra": "RA", # keyword holding RA
    "keyword_dec": "DEC", # keyword holding DEC
    "keyword_flip": "PIERSIDE", # keyword holding meridan flip (WEST or EAST) 
    "keyword_fwhm": "FWHM",
    "name": "SSO",
    "trimming": [8, 22], # in piwel along y/x
    "read_noise": 10, # in ADU
    "gain": 1.02, # in e-/ADU
    "altitude": 2000, # in meters
    "diameter": 100, # in meters
    "pixel_scale": 0.33, # in arcseconds
    "latlong": [24.6275, 70.4044] # [latitude, longitude] in degree
}

callisto = speculoos_south.copy()
callisto.update({"name": "Callisto"})

io = speculoos_south.copy()
io.update({"name": "Io"})

ganymede = speculoos_south.copy()
ganymede.update({"name": "Ganymede"})

europa = speculoos_south.copy()
europa.update({"name": "Europa"})

saintex = speculoos_south.copy()
saintex.update({"name": "SaintEx"})

artemis = speculoos_south.copy()
artemis.update({"name": "Artemis"})

trappist = speculoos_south.copy()
trappist.update({
    "name": "Trappist",
    "trimming": [40, 40],
    "pixel_scale": 0.66})

trappistN = trappist.copy()
trappistN.update({"name": "NTM"})

built_in_telescopes = {
    "trappist": trappist,
    "artemis": artemis,
    "europa": europa,
    "saintex": saintex,
    "ganymede": ganymede,
    "io": io,
    "callisto": callisto,
    "ntm": trappistN
}


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
        self.trimming = (8, 22)
        self.read_noise = 9
        self.gain = 1
        self.altitude = 2000
        self.diameter = 100
        self.pixel_scale = None
        self.latlong = [None, None]

        if telescope_file is not None:
            self.load(telescope_file)

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

    @property
    def earth_location(self):
        return EarthLocation(*self.latlong, self.altitude)
