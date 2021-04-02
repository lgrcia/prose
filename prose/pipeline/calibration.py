from .. import Sequence, blocks, Block, Telescope
import os
from os import path
from astropy.io import fits


class Calibration:
    """A calibration unit producing a reduced FITS folder

    Parameters
    ----------
    destination : str, optional
        Destination of the newly created folder, by default beside the folder given to FitsManager
    reference : float, optional
        Reference image to use for alignment from 0 (first image) to 1 (last image), by default 1/2
    overwrite : bool, optional
        whether to overwrite existing products, by default False
    ignore_telescope: bool, optional
        whether to load a default telescope if telescope not recognised, by default False
    """

    def __init__(
            self,
            reference=1 / 2,
            overwrite=False,
            flats=None,
            bias=None,
            darks=None,
            images=None,
            psf=blocks.Moffat2D
    ):
        self.destination = None
        self.overwrite = overwrite
        self._reference = reference

        # set on prepare
        self.flats = flats
        self.bias = bias
        self.darks = darks
        self.images = images

        self.reference_unit = None
        self.calibration_unit = None

        assert psf is None or issubclass(psf, Block), "psf must be a subclass of Block"
        self.psf = psf

        # set Telescope
        self.telescope = Telescope.from_name(fits.getheader(self.images[0])["TELESCOP"])
        # set reference file
        reference_id = int(self._reference * len(self.images))
        self.reference_fits = self.images[reference_id]
        self.calibration_block = blocks.Calibration(self.darks, self.flats, self.bias, name="calibration")

    def run(self, destination):
        self.destination = destination

        self.make_destination()

        self.reference_unit = Sequence([
            self.calibration_block,
            blocks.Trim(name="trimming"),
            blocks.SegmentedPeaks(n_stars=50, name="detection"),
            blocks.ImageBuffer(name="buffer")
        ], self.reference_fits, telescope=self.telescope)

        self.reference_unit.run(show_progress=False)

        ref_image = self.reference_unit.buffer.image

        self.calibration_unit = Sequence([
            self.calibration_block,
            blocks.Trim(name="trimming", skip_wcs=True),
            blocks.Flip(ref_image, name="flip"),
            blocks.SegmentedPeaks(n_stars=50, name="detection"),
            blocks.XYShift(ref_image.stars_coords, name="shift"),
            blocks.Align(ref_image.data, name="alignment"),
            self.psf(name="fwhm"),
            blocks.Stack(self.stack_path, header=ref_image.header, overwrite=self.overwrite, name="stack"),
            blocks.SaveReduced(self.destination if destination is None else destination, overwrite=self.overwrite,
                               name="saving"),
            blocks.Video(self.gif_path, name="video", from_fits=True)
        ], self.images, telescope=self.telescope, name="Calibration")

        self.calibration_unit.run()

    @property
    def stack_path(self):
        prepend = "stack.fits"
        return path.join(self.destination, prepend)

    @property
    def gif_path(self):
        prepend = "movie.gif"
        return path.join(self.destination, prepend)

    def make_destination(self):
        if not path.exists(self.destination):
            os.mkdir(self.destination)

    def __repr__(self):
        return f"{self.reference_unit}\n{self.calibration_unit}"
