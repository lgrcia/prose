import numpy as np
from ..core import Block
from astropy.time import Time
import xarray as xr
from .. import utils

# TODO remove SavePhot and add SaveReduced and Xarray

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
        self.images = []
        self.stack_path = None
        self.fits_manager = None
        self.header = header
        self.stack = stack

    def run(self, image, **kwargs):
        self.images.append(image.copy(data=False))

    def terminate(self):
        if self.header is not None:
            self.header["REDDATE"] = Time.now().to_value("fits")

        x = np.array([im.fluxes for im in self.images])
        errors = np.array([im.fluxes_errors for im in self.images])

        imref = self.images[0]
        telescope = imref.telescope

        stars = imref.stars_coords

        dims = ("apertures", "star", "time")

        attrs = self.header if isinstance(self.header, dict) else {}
        attrs.update(dict(
            target=-1,
            aperture=-1,
            telescope=telescope.name,
            filter=self.header[telescope.keyword_filter],
            exptime=self.header[telescope.keyword_exposure_time],
            name=self.header[telescope.keyword_object],
            date=str(utils.format_iso_date(self.header[telescope.keyword_observation_date])).replace("-", ""),
        ))

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
            telescope.keyword_exposure_time,
            telescope.keyword_jd,
            telescope.keyword_bjd,
            telescope.keyword_seeing,
            telescope.keyword_ra,
            telescope.keyword_dec,
            telescope.keyword_flip,
        ]:
            _data = []
            if key in self.images[0].header:
                for image in self.images:
                    _data.append(image.header[key])

                if key in [
                    telescope.keyword_jd,
                    telescope.keyword_bjd
                ]:
                    if key == telescope.keyword_jd:
                        x["jd_utc"] = ('time', Time(np.array(_data) + telescope.mjd,
                                                    format="jd", scale=telescope.jd_scale,
                                                    location=telescope.earth_location).utc.value)

                    elif key == telescope.keyword_bjd:
                        x["bjd_tdb"] = ('time', Time(np.array(_data) + telescope.mjd,
                                                     format="jd", scale=telescope.jd_scale,
                                            location=telescope.earth_location).tdb.value)
                else:
                    x[key.lower()] = ('time', _data)

        for key in [
            "apertures_area",
            "annulus_area",
            "apertures_radii",
            "annulus_rin",
            "annulus_rout"
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

        if "peaks" in self.images[0].__dict__:
            x["peaks"] = ("time", "stars", np.array([im.peaks for im in self.images]))

        if self.stack is not None:
            x = x.assign_coords(stack=(('w', 'h'), self.stack))

        x.attrs.update(utils.header_to_cdf4_dict(self.header))

        jd_kw = telescope.keyword_jd.lower()
        bjd_kw = telescope.keyword_bjd.lower()

        # Dealing with time
        if bjd_kw in x:
            x = x.assign_coords(time=x.bjd_tdb)
            x.attrs["time_format"] = "bjd_tdb"
        else:
            x = x.assign_coords(time=x.jd_utc)
            x.attrs["time_format"] = "jd_utc"

        x = x.assign_coords(stars=(('star', 'n'), stars))
        x.to_netcdf(self.destination)


