from .. import Sequence, blocks, Block, Telescope
import os
from os import path
from pathlib import Path
import xarray as xr
from astropy.io import fits
from .photometry import plot_function



class Calibration:
    """A calibration unit producing a reduced FITS folder

    The calibration encompass more than simple flat, dark and bias calibration. It contains two sequences whose ...
    TODO

    Parameters
    ----------
    destination : str, optional
        Destination of the newly created folder, by default beside the folder given to FitsManager
    reference : float, optional
        Reference image to use for alignment from 0 (first image) to 1 (last image), by default 1/2
    overwrite : bool, optional
        whether to overwrite existing products, by default False
    flats : list, optional
        list of flats images paths, by default None
    bias : list, optional
        list of bias images paths, by default None
    darks : list, optional
        list of darks images paths, by default None
    images : list, optional
        list of images paths to be calibrated, by default None
    psf : `Block`, optional
        a `Block` to be used to characterise the effective psf, by default blocks.Moffat2D
    """

    def __init__(
            self,
            reference=1/2,
            overwrite=False,
            flats=None,
            bias=None,
            darks=None,
            images=None,
            psf=blocks.Moffat2D,
            verbose=True,
            show=False,
    ):
        self.destination = None
        self.overwrite = overwrite
        self._reference = reference
        self.verbose = verbose
        self.show = show

        if show:
            self.show = blocks.LivePlot(plot_function, size=(10, 10))
        else:
            self.show = blocks.Pass()

        # set on prepare
        self.flats = flats
        self.bias = bias
        self.darks = darks
        self._images = images

        self.detection_s = None
        self.calibration_s = None

        assert psf is None or issubclass(psf, Block), "psf must be a subclass of Block"
        self.psf = psf
        
        # set reference file
        reference_id = int(self._reference * len(self._images))
        self.reference_fits = self._images[reference_id]
        self.calibration_block = blocks.Calibration(self.darks, self.flats, self.bias, name="calibration")

    def run(self, destination):
        """Run the calibration pipeline

        Parameters
        ----------
        destination : str
            Destination where to save the calibrated images folder
        """
        self.destination = destination

        self.make_destination()

        self.detection_s = Sequence([
            self.calibration_block,
            blocks.Trim(name="trimming"),
            blocks.SegmentedPeaks(n_stars=20, name="detection"),
            # blocks.KeepGoodStars(),
            blocks.ImageBuffer(name="buffer")
        ], self.reference_fits)

        self.detection_s.run(show_progress=False)

        ref_image = self.detection_s.buffer.image

        self.calibration_s = Sequence([
            self.calibration_block,
            blocks.Trim(name="trimming", skip_wcs=True),
            blocks.Flip(ref_image, name="flip"),
            blocks.SegmentedPeaks(n_stars=20, name="detection"),
            # blocks.KeepGoodStars(),
            blocks.Twirl(ref_image.stars_coords, n=15, name="twirl"),
            self.psf(name="fwhm"),
            blocks.SaveReduced(self.destination, overwrite=self.overwrite, name="save_reduced"),
            blocks.AffineTransform(stars=True, data=True),
            self.show,
            blocks.Stack(self.stack_path, header=ref_image.header, overwrite=self.overwrite, name="stack"),
            blocks.Video(self.gif_path, name="video", from_fits=True),
            blocks.XArray(
                ("time", "jd_utc"),
                ("time", "bjd_tdb"),
                ("time", "flip"),
                ("time", "fwhm"),
                ("time", "fwhmx"),
                ("time", "fwhmy"),
                ("time", "dx"),
                ("time", "dy"),
                ("time", "airmass"),
                ("time", "exposure")
            )
        ], self._images, name="Calibration")

        self.calibration_s.run(show_progress=self.verbose)

        # saving xarray
        calib_xarray = self.calibration_s.xarray.xarray
        stack_xarray = self.calibration_s.stack.xarray
        xarray = xr.merge([calib_xarray, stack_xarray], combine_attrs="no_conflicts")
        xarray = xarray.assign_coords(time=xarray.jd_utc)
        xarray.attrs["time_format"] = "jd_utc"
        xarray.to_netcdf(self.phot_path)

    @property
    def stack_path(self):
        return self.destination / "stack.fits"

    @property
    def phot_path(self):
        return self.destination / (self.destination.name + ".phot")

    @property
    def stack(self):
        return self.stack_path

    @property
    def images(self):
        return self.calibration_s.save_reduced.files

    @property
    def gif_path(self):
        prepend = "movie.gif"
        return path.join(self.destination, prepend)

    @property
    def processing_time(self):
        return self.calibration_s.processing_time + self.detection_s.processing_time

    def make_destination(self):
        self.destination = Path(self.destination)
        self.destination.mkdir(exist_ok=True)
        if not path.exists(self.destination):
            os.mkdir(self.destination)

    def __repr__(self):
        return f"{self.detection_s}\n{self.calibration_s}"

    @property
    def xarray(self):
        return self.calibration_s.xarray()

