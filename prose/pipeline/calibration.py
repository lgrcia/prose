from .. import Sequence, MultiProcessSequence, blocks, Block, Image
import os
from os import path
from pathlib import Path
from .. import utils
from .photometry import plot_function
from astropy.time import Time


class Calibration:
    """A calibration unit producing a reduced FITS folder

    The calibration encompass more than simple flat, dark and bias calibration. It contains two sequences whose ...
    TODO

    Parameters
    ----------
    reference : float or str, optional
        Reference image to use for alignment:
         
        - if ``float``: from 0 (first image) to 1 (last image)
        - if ``str``: path of the reference image
        
        by default 1/2
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
            reference=1/2,
            overwrite=False,
            flats=None,
            bias=None,
            darks=None,
            images=None,
            psf=None,
            detection=None,
            verbose=True,
            show=False,
            twirl=True,
            n=None,
            loader=Image,
            cores=False,
    ):
        self.destination = None
        self.overwrite = overwrite
        self._reference = reference
        self.verbose = verbose
        self.show = show
        self.n = n if n is not None else (12 if twirl else 50)
        self.twirl = twirl
        self.cores = cores
        self.xarray = None

        if show:
            self.show = blocks.LivePlot(plot_function, size=(10, 10))
        else:
            self.show = blocks.Pass()

        self.flats = flats
        self.bias = bias
        self.darks = darks
        self._images = images
        self.loader = loader

        self.detection_s = None
        self.calibration_s = None

        # Checking psf block
        if psf is None:
            psf = blocks.Moffat2D(name="fwhm")
        else:
            assert isinstance(psf, Block), "psf must be a subclass of Block"
            psf.name = "fwhm"

        self.psf = psf

        # checking detection block
        if detection is None:
            detection = blocks.SegmentedPeaks(n_stars=self.n, name="detection")
        else:
            assert isinstance(detection, Block), "psf must be a subclass of Block"
            detection.name = "detection"

        self.detection = detection

        # set reference file
        if isinstance(self._reference, (int, float)):
            reference_id = int(self._reference * len(self._images))
            self.reference_fits = self._images[reference_id]
        elif isinstance(self._reference, (str, Path)):
            self.reference_fits = self._reference

        self.calibration_block = blocks.Calibration(self.darks, self.flats, self.bias, loader=loader, name="calibration")

    def run(self, destination, gif=True):
        """Run the calibration pipeline

        Parameters
        ----------
        destination : str
            Destination where to save the calibrated images folder
        """
        self.destination = destination
        gif_block = blocks.Video(self.gif_path, name="video", from_fits=True) if gif else blocks.Pass()

        self.make_destination()

        self.detection_s = Sequence([
            self.calibration_block,
            blocks.Trim(name="trimming"),
            self.detection,
            blocks.ImageBuffer(name="buffer")
        ], self.reference_fits, loader=self.loader)

        self.detection_s.run(show_progress=False)

        ref_image = self.detection_s.buffer.image
        ref_stars = ref_image.stars_coords

        if self.twirl:
            assert len(ref_stars) >= 4, f"Only {len(ref_stars)} stars detected (must be >= 4). See detection kwargs"

        SequenceObject = MultiProcessSequence if self.cores else Sequence

        self.calibration_s = SequenceObject([
            self.calibration_block,
            blocks.Trim(name="trimming", skip_wcs=True),
            blocks.Flip(ref_image, name="flip"),
            self.detection,
            blocks.Twirl(ref_stars, n=self.n, name="twirl") if self.twirl else blocks.XYShift(ref_stars),
            self.psf,
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
            blocks.Cutout2D(ref_image) if not self.twirl else blocks.Pass(),
            blocks.SaveReduced(self.destination, overwrite=self.overwrite, name="save_reduced"),
            blocks.AffineTransform(stars=True, data=True) if self.twirl else blocks.Pass(),
            self.show,
            blocks.Stack(self.stack_path, header=ref_image.header, overwrite=self.overwrite, name="stack"),
            gif_block,
        ], self._images, name="Calibration", loader=self.loader)

        self.calibration_s.run(show_progress=self.verbose)

        # saving xarray
        xarray = self.calibration_s.xarray.xarray
        # first image serve as reference for info (not reference image because it
        # can be from another observation (we encountered this use case)
        reference = self.loader(self._images[0])
        reference_header = reference.header
        reference_telescope = reference.telescope
        xarray.attrs.update(utils.header_to_cdf4_dict(reference_header))
        xarray.attrs.update(dict(
            target=-1,
            aperture=-1,
            telescope=reference_telescope.name,
            filter=reference_header.get(reference_telescope.keyword_filter, ""),
            exptime=reference_header.get(reference_telescope.keyword_exposure_time, ""),
            name=reference_header.get(reference_telescope.keyword_object, ""),
        ))

        if reference_telescope.keyword_observation_date in reference_header:
            date = reference_header[reference_telescope.keyword_observation_date]
        else:
            date = Time(reference_header[reference_telescope.keyword_jd], format="jd").datetime

        xarray.attrs.update(dict(date=utils.format_iso_date(date).isoformat()))
        xarray.coords["stack"] = (('w', 'h'), self.calibration_s.stack.stack)

        xarray = xarray.assign_coords(time=xarray.jd_utc)
        xarray = xarray.sortby("time")
        xarray.attrs["time_format"] = "jd_utc"
        xarray.attrs["reduction"] = [b.__class__.__name__ for b in self.calibration_s.blocks]
        self.xarray = xarray
        xarray.to_netcdf(self.phot)

    @property
    def stack_path(self):
        return self.destination / "stack.fits"

    @property
    def phot(self):
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