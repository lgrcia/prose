from typing import Optional, Union

import numpy as np
from scipy.spatial import KDTree
from skimage.transform import AffineTransform
from twirl import quads, triangles
from twirl.geometry import get_transform_matrix, pad
from twirl.match import count_cross_match

from prose.core import Block, Image

__all__ = ["Trim", "Cutouts", "Drizzle"]


def hashes(asterisms):
    # check type
    if not isinstance(asterisms, int):
        raise TypeError("asterisms must be an int")
    # check value
    if not asterisms in [3, 4]:
        raise ValueError("asterisms must be 3 or 4")
    else:
        if asterisms == 3:
            return triangles.hashes
        elif asterisms == 4:
            return quads.hashes
        else:
            raise ValueError("asterisms must be 3 or 4")


class Trim(Block):
    """Image trimming

    If trim is not specified, triming is taken from the "overscan" in image metadata

    |write| ``Image.header``

    |modify|

    Parameters
    ----------
    skip_wcs : bool, optional
        whether to skip applying trim to WCS (If None: wcs is skipped only
        if image is not plated solved), by default None
    trim : tuple, int or flot, optional
        (x, y) trim values, by default None which uses the ``trim`` value
        from the image telescope definition. If an int or a float is provided trim will be be applied to both axes.

    """

    def __init__(self, trim=None, skip_wcs=None, name=None, verbose=False):
        super().__init__(name, verbose)
        assert (skip_wcs is None) or isinstance(
            skip_wcs, bool
        ), "skip_wcs must be None or bool"
        self.skip_wcs = skip_wcs
        if isinstance(trim, (int, float)):
            trim = (trim, trim)
        self.trim = trim
        self._parallel_friendly = True

    def run(self, image):
        trim = self.trim if self.trim is not None else image.metadata["overscan"]
        center = image.shape[::-1] / 2
        shape = image.shape - 2 * np.array(trim)
        if self.skip_wcs is None:
            skip_wcs = not image.plate_solved
        else:
            skip_wcs = self.skip_wcs
        cutout = image.cutout(center, shape, wcs=skip_wcs)
        image.data = cutout.data
        image.sources = cutout.sources
        image.wcs = cutout.wcs


class Cutouts(Block):
    def __init__(
        self,
        shape: Union[int, tuple] = 50,
        wcs: bool = False,
        name: Optional[str] = None,
        sources: bool = False,
    ):
        """Create cutouts around all sources

        |read| :code:`Image.sources`

        |write| :code:`Image.cutouts`

        Parameters
        ----------
        shape : int or tuple, optional
            cutout shape, by default 50
        wcs : bool, optional
            whether to compute cutouts WCS, by default False
        name : str, optional
            name of the blocks, by default None
        sources: bool, optional
            whether to keep sources in cutouts, by default False
        """
        super().__init__(name=name)
        if isinstance(shape, int):
            shape = (shape, shape)
        self.shape = shape
        self.wcs = wcs
        self.sources = sources
        self._parallel_friendly = True

    def run(self, image: Image):
        image.cutouts = [
            image.cutout(coords, self.shape, wcs=self.wcs, sources=self.sources)
            for coords in image.sources.coords
        ]
        f = 0


class SetAffineTransform(Block):
    def __init__(self, name=None, verbose=False):
        super().__init__(name, verbose)
        self._parallel_friendly = True

    def run(self, image):
        rotation = image.__dict__.get("rotation", 0)
        translation = image.__dict__.get("translation", (0, 0))
        scale = image.__dict__.get("scale", 0)
        image.transform = AffineTransform(
            rotation=rotation, translation=translation, scale=scale
        )


class ComputeTransform(Block):
    """
    Compute transformation from a reference image

    |read| ``Image.sources`` on both reference and input image

    |write| ``Image.transform``

    Parameters
    ----------
    ref : Image
        image containing detected sources
    n : int, optional
        number of stars to consider to compute transformation, by default 10
    """

    def __init__(
        self,
        reference_image: Image,
        n: int = 20,
        discard: bool = True,
        asterisms: int = 3,
        name: Optional[str] = None,
        min_match: int = 10,
    ):
        super().__init__(name=name)
        self._coords_ref = reference_image.sources.coords[0:n]
        self.n = n
        self._asterisms = asterisms
        self._hashes = hashes(asterisms)
        self._hashes_ref, self._asterism_coords_ref = self._hashes(self._coords_ref)
        self.discard = discard
        self._parallel_friendly = True
        self._min_match = min_match

    def run(self, image):
        if len(image.sources.coords) >= self._asterisms + 2:
            result = self.solve(image.sources.coords.copy())
            if result is not None:
                image.transform = AffineTransform(result)
            else:
                image.discard = True
        else:
            image.discard = True

    def solve(self, coords, tolerance=2):
        hashes, asterism_coords = self._hashes(coords)
        distances = np.linalg.norm(
            hashes[:, None, :] - self._hashes_ref[None, :, :], axis=2
        )
        shortest_hash = np.argmin(distances, 1)
        ns = []

        for i, j in enumerate(shortest_hash):
            M = get_transform_matrix(asterism_coords[j], self._asterism_coords_ref[i])
            test = (M @ pad(coords).T)[0:2].T
            n = count_cross_match(self._coords_ref, test, tolerance)
            ns.append(n)
            if self._min_match is not None:
                if n >= self._min_match:
                    break

            i = np.argmax(ns)
            M = get_transform_matrix(
                coords[np.argmin(distances, 1)[i]],
                self._asterism_coords_ref[i],
            )

        return M


class Drizzle(Block):
    def __init__(self, reference, pixfrac=1.0, **kwargs):
        """Produce a dithered image. Requires :code:`drizzle` package.

        All images (including reference must be plate-solved). After :code:`terminate` is called
        (e.g. when a sequence is entirely ran), the dithered image can be found in self.image

        Parameters
        ----------
        reference : prose.Image
            Reference image on which the stacking is based
        pixfrac : float, optional
            fraction of pixel used in dithering, by default 1.
        """
        from drizzle import drizzle

        super().__init__(self, **kwargs)
        self.pixfrac = pixfrac
        reference.wcs.pixel_shape = reference.shape
        self.drizzle = drizzle.Drizzle(outwcs=reference.wcs, pixfrac=pixfrac)
        self.image = reference.copy()

    def run(self, image):
        WCS = image.wcs
        self.drizzle.add_image(image.data, image.wcs)

    def terminate(self):
        self.image.data = self.drizzle.outsci
