from .. import Sequence, blocks, Block, Image
import os
from os import path
from pathlib import Path
from .. import utils
from ..blocks.vizualisation import LivePlot
from .photometry import plot_function
from astropy.time import Time
from .. import __version__


class Calibration:
    """A calibration unit producing a reduced FITS folder

    The calibration encompass more than simple flat, dark and bias calibration. It contains two sequences whose ...
    TODO

    Parameters
    ----------
    overwrite : bool, optional
        whether to overwrite existing products, by default False
    flats : list, optional
        list of flats images paths, by default None
    bias : list, optional
        list of bias images paths, by default None
    darks : list, optional
        list of darks images paths, by default None
    psf : `Block`, optional
        a `Block` to be used to characterise the effective psf, by default None, setting a default blocks.Moffat2D
    detection: `Block`, optional
        a `Block` to be used for stars detection, by default None, setting a default blocks.SegmentedPeaks
    show: bool, optional
        within a notebook, whether to show processed image during reduction, by default False 
    verbose: bool, optional
        whether to print processing info and loading bars, by default True
    twirl: bool, optional,
        whether to use the Twirl block for alignment (see blocks.registration.Twirl)
    n: int, optional,
        number of stars used for alignment, by default None leading to 12 if twirl else 50
    loader: Image class, optional
        class to load Images, by default Image
    """

    def __init__(
            self,
            overwrite=False,
            flats=None,
            bias=None,
            darks=None,
            psf_block=None,
            detection_block=None,
            verbose=True,
            show=False,
            twirl=True,
            n=None,
            loader=Image,
            cores=False,
            bad_pixels=False,
            save_masters=True,
            min_stars=3,
            **kwargs
    ):
        self.destination = None
        self.overwrite = overwrite
        self.verbose = verbose
        self.show = show
        self.n = n if n is not None else (12 if twirl else 50)
        self.twirl = twirl
        self.cores = cores
        self.xarray = None
        self.loader = loader
        self.show = show

        self.flats = flats
        self.bias = bias
        self.darks = darks
        self.loader = loader

        # The two sequences
        self.detection = None
        self.calibration = None

        # Checking psf block
        if psf_block is None:
            psf_block = blocks.psf.Moffat2D
        else:
            assert issubclass(psf_block, Block), "psf must be a subclass of Block"
            psf_block.name = "fwhm"

        self.psf = psf_block

        # checking detection block
        if detection_block is None:
            detection_block = blocks.SegmentedPeaks(n_stars=self.n, name="detection")
        else:
            assert isinstance(detection_block, Block), "detection must be a subclass of Block"
            detection_block.name = "detection"

        self.detection_block = detection_block
        self.bad_pixels = bad_pixels

        self.calibration_blocks = [
            blocks.Calibration(self.darks,  self.flats,  self.bias,  loader=loader, name="calibration"),
            blocks.CleanBadPixels(darks=self.darks, flats=self.flats) if self.bad_pixels else blocks.Pass(),
            blocks.Trim(name="trimming", skip_wcs=True),
        ]

        self.min_stars = min_stars

        if save_masters:
            pass

    def run(self, images, destination=None, reference=1/2, movie="gif"):
        """Run the calibration pipeline

        Parameters
        ----------
        images : list, optional
            List of images paths to be calibrated
        destination : str, optional
            Destination where to save the calibrated images folder, by default reference Image.label
        reference : float or str, optional
            Reference image to use for alignment:
                - if ``float``: from 0 (first image) to 1 (last image)
                - if ``str``: path of the reference image
            by default 1/2
        movie: str or False, optional
            - if str, produce a movie of the sequence with format 'movie'
            - if False movie not produced
        """

        # reference image
        if isinstance(reference, (int, float)):
            self.reference = self.loader(images[int(reference * len(images))])
        elif isinstance(reference, (str, Path)):
            self.reference = self.loader(reference)

        # Creating reduced image folder
        if destination is None:
            destination = self.reference.label
        self.destination = Path(destination)
        self.destination.mkdir(exist_ok=True)

        # Detection sequence
        # ------------------
        self.detection = Sequence([
            *self.calibration_blocks,
            self.detection_block,
            blocks.Cutouts(),
            blocks.MedianPSF(),
        ])

        self.detection.run(self.reference, show_progress=False, loader=self.loader)

        # Checking enough reference stars are found
        if self.twirl:
            n_reference_stars = len(self.reference.stars_coords)
            assert n_reference_stars >= 4, f"Only {n_reference_stars} stars detected (must be >= 4). See detection kwargs"

        # Calibration sequence
        # --------------------
        self.calibration = Sequence([
            *self.calibration_blocks,
            blocks.Flip(self.reference, name="flip"),
            self.detection_block,
            blocks.detection.LimitStars(min=self.min_stars),
            blocks.Twirl(self.reference.stars_coords, n=self.n, name="twirl") if self.twirl 
            else blocks.XYShift(self.reference.stars_coords),
            blocks.Cutouts(),
            blocks.MedianPSF(),
            self.psf(name="psf", reference=self.reference),
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
                ("time", "exposure"),
                ("time", "path")
            ),
            blocks.Cutout2D(self.reference) if not self.twirl else blocks.Pass(),
            blocks.SaveReduced(destination, overwrite=self.overwrite, name="save_reduced"),
            blocks.AffineTransform(stars=True, data=True) if self.twirl else blocks.Pass(),
            LivePlot(plot_function, size=(10, 10)) if self.show else blocks.Pass(),
            blocks.Stack(self.reference, name="stack"),
            blocks.RawVideo(self.destination / f"movie.{movie}", function=utils.z_scale, scale=0.5) if movie is not False else blocks.Pass(),
        ], name="Calibration")

        self.calibration.run(images, show_progress=self.verbose, loader=self.loader)

        # Saving outpout
        # first image serves as reference for info (not reference image because it
        # can be from another observation (we encountered this use case)
        self.stack = self.calibration.stack.image
        self.save()

    @property
    def stack_path(self):
        return self.destination / "stack.fits"

    @property
    def phot(self):
        return self.destination / (self.destination.name + ".phot")

    @property
    def images(self):
        return self.calibration.save_reduced.files

    @property
    def processing_time(self):
        return self.calibration.processing_time + self.detection.processing_time
        
    def __repr__(self):
        return f"{self.detection}\n{self.calibration}"

    def save(self):
        self.stack.writeto(self.stack_path)
        xarray = self.calibration.xarray.xarray
        xarray = utils.image_in_xarray(self.stack, xarray, stars=False)
        xarray.attrs["reduction"] = [b.__class__.__name__ for b in self.calibration.blocks]
        xarray.attrs["reduction_prose_version"] = __version__
        self.xarray = xarray
        xarray.to_netcdf(self.phot)

