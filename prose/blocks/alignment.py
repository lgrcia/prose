import numpy as np
from astropy.wcs.utils import fit_wcs_from_points
from skimage.transform import warp

from prose.core import Block, Image

from .geometry import ComputeTransform, SetAffineTransform

__all__ = ["Align", "AlignReferenceSources", "AlignReferenceWCS"]


# TODO test inverse
class Align(Block):
    def __init__(self, reference, name=None):
        """Align image data to a reference

        Parameters
        ----------
        reference: :py:class:`~prose.Image`
            reference image to align images on
        name : str, optional
            name of the block, by default None
        """
        super().__init__(name)
        self.reference = reference
        self.compute_transform = ComputeTransform(self.reference)

    def run(self, image: Image):
        self.compute_transform.run(image)

        # if self.inverse:
        #    transform = transform.inverse
        try:
            image.data = warp(
                image.data,
                image.transform.inverse,
                cval=np.nanmedian(image.data),
                output_shape=image.shape,
            )
        except np.linalg.LinAlgError:
            image.discard = True


class AlignReferenceSources(Block):
    def __init__(self, reference: Image, name=None, verbose=False):
        """Set Image sources to reference Image sources, properly aligned

        |read| Image.sources

        |write| Image.sources

        Parameters
        ----------
        reference : Image
            reference image containing sources
        name : _type_, optional
            _description_, by default None
        verbose : bool, optional
            _description_, by default False
        """
        super().__init__(name, verbose)
        self.reference_sources = reference.sources
        self.compute_transform = ComputeTransform(reference)
        self._parallel_friendly = True

    def run(self, image: Image):
        self.compute_transform.run(image)
        if not image.discard:
            sources = self.reference_sources.copy()
            new_sources_coords = image.transform.inverse(sources.coords.copy())

            # check if alignment potentially failed
            if (
                np.abs(
                    np.std(new_sources_coords) - np.std(self.reference_sources.coords)
                )
                > 100
            ):
                image.discard = True
            else:
                sources.coords = new_sources_coords
                image.sources = sources


class AlignReferenceWCS(Block):
    def __init__(self, reference: Image, name=None, verbose=False, n=6):
        """Create WCS based on reference WCS. To use this block, Image sources must match the sources from the reference (e.g. using AlignReferenceSources), i.e. same sources should be found at a given index in both images.

        |read| Image.sources

        |write| Image.wcs


        Parameters
        ----------
        reference : Image
            reference image containing a valid WCS
        n : int, optional
            number of stars used to match WCS, by default 6
        """
        super().__init__(name, verbose)
        self.reference = reference
        assert reference.plate_solved, "reference must have valid WCS"
        self.n = n

    def run(self, image: Image):
        ref_skycoords = self.reference.wcs.pixel_to_world(
            *self.reference.sources.coords[0 : self.n].T
        )
        image.wcs = fit_wcs_from_points(
            image.sources.coords[0 : self.n].T, ref_skycoords
        )
