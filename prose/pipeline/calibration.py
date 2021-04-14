from .. import Sequence, blocks, Block, Telescope
import os
from os import path
from astropy.io import fits


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
            verbose=True
    ):
        self.destination = None
        self.overwrite = overwrite
        self._reference = reference
        self.verbose = verbose

        # set on prepare
        self.flats = flats
        self.bias = bias
        self.darks = darks
        self._images = images

        self.reference_sequence = None
        self.calibration_sequence = None

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

        self.reference_sequence = Sequence([
            self.calibration_block,
            blocks.Trim(name="trimming"),
            blocks.SegmentedPeaks(n_stars=50, name="detection"),
            blocks.ImageBuffer(name="buffer")
        ], self.reference_fits)

        self.reference_sequence.run(show_progress=False)

        ref_image = self.reference_sequence.buffer.image

        self.calibration_sequence = Sequence([
            self.calibration_block,
            blocks.Trim(name="trimming", skip_wcs=True),
            blocks.Flip(ref_image, name="flip"),
            blocks.SegmentedPeaks(n_stars=50, name="detection"),
            blocks.XYShift(ref_image.stars_coords, name="shift"),
            blocks.Align(ref_image.data, name="alignment"),
            self.psf(name="fwhm"),
            blocks.Stack(self.stack_path, header=ref_image.header, overwrite=self.overwrite, name="stack"),
            blocks.SaveReduced(self.destination if destination is None else destination, overwrite=self.overwrite,
                               name="save"),
            blocks.Video(self.gif_path, name="video", from_fits=True)
        ], self._images, name="Calibration")

        self.calibration_sequence.run(show_progress=self.verbose)

    @property
    def stack_path(self):
        prepend = "stack.fits"
        return path.join(self.destination, prepend)

    @property
    def stack(self):
        return self.stack_path

    @property
    def images(self):
        return self.calibration_sequence.save.files

    @property
    def gif_path(self):
        prepend = "movie.gif"
        return path.join(self.destination, prepend)

    @property
    def processing_time(self):
        return self.calibration_sequence.processing_time + self.reference_sequence.processing_time

    def make_destination(self):
        if not path.exists(self.destination):
            os.mkdir(self.destination)

    def __repr__(self):
        return f"{self.reference_sequence}\n{self.calibration_sequence}"
