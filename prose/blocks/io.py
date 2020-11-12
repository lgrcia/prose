import numpy as np
from prose.blocks.base import Block
from astropy.time import Time
import xarray as xr


class SavePhot(Block):
    """Save photometric products into a FITS :code:`.phots` file. See :ref:`phots-structure` for more info

    Parameters
    ----------
    destination : str
        path of the file (must be a .phots file name)
    overwrite : bool, optional
        weather to overwrite file if exists, by default False
    """
    def __init__(self, destination, header=None, stack=None, **kwargs):
        super().__init__(**kwargs)
        self.destination = destination
        self.telescope = None
        self.images = []
        self.stack_path = None
        self.telescope = None
        self.fits_manager = None
        self.header = header
        self.stack = stack

    def run(self, image, **kwargs):
        self.images.append(image)

    def terminate(self):
        if self.header is not None:
            self.header["REDDATE"] = Time.now().to_value("fits")

        x = np.array([im.fluxes for im in self.images])
        errors = np.array([im.fluxes_errors for im in self.images])
        stars = self.images[0].stars_coords

        dims = ("apertures", "star", "time")

        attrs = self.header if isinstance(self.header, dict) else {}
        attrs.update(dict(target=-1, aperture=-1, telescope=self.telescope.name))

        x = xr.Dataset({
            "fluxes": xr.DataArray(x, dims=["time", "apertures", "star"]).transpose(*dims),
            "errors": xr.DataArray(errors, dims=["time",  "apertures", "star"]).transpose(*dims),
        }, attrs=attrs)

        for key in [
            "sky",
            "fwhm",
            "fwhmx",
            "fwhmy",
            "psf_angle",
            "dx",
            "dy",
            "airmass",
            self.telescope.keyword_exposure_time,
            self.telescope.keyword_julian_date,
            self.telescope.keyword_seeing,
            self.telescope.keyword_ra,
            self.telescope.keyword_dec,
        ]:
            _data = []
            if key in self.images[0].header:
                for image in self.images:
                    _data.append(image.header[key])

                x[key.lower()] = ('time', _data)

        for key in [
            "apertures_area",
            "annulus_area"
        ]:
            if key in self.images[0].__dict__:
                _data = []
                for image in self.images:
                    _data.append(image.__dict__[key])
                _data = np.array(_data)

                if len(_data.shape) == 2:
                    x[key.lower()] = (('time', 'apertures'), _data)
                elif len(_data.shape) == 1:
                    x[key.lower()] = ('time', _data)
                else:
                    raise AssertionError("")

        if self.stack is not None:
            x = x.assign_coords(stack=(('w', 'h'), self.stack))

        for key, value in self.header.items():
            if isinstance(value, str):
                x.attrs[key] = value
            elif isinstance(value, (float, np.ndarray, np.number)):
                x.attrs[key] = float(value)
            elif isinstance(value, (int, bool)):
                x.attrs[key] = int(value)

        x = x.assign_coords(time=x.jd)
        x = x.assign_coords(stars=(('star', 'n'), stars))
        x.to_netcdf(self.destination)


