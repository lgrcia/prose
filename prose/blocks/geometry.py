from ..core import Block, Image
import numpy as np
from skimage.transform import AffineTransform
from scipy.spatial import KDTree
from twirl import utils as twirl_utils
from typing import Union

__all__ = ["Trim", "Cutouts"]


class Trim(Block):
    """Image trimming
    
    If trim is not specified, triming is taken from the "overscan" in image metadata

    |write| ``Image.header``

    |modify|

    Parameters
    ----------
    skip_wcs : bool, optional
        whether to skip applying trim to WCS, by default False
    trim : tuple, int or flot, optional
        (x, y) trim values, by default None which uses the ``trim`` value from the image telescope definition. If an int or a float is provided trim will be be applied to both axes.
        
    """

    def __init__(self, trim=None, skip_wcs=True, name=None, verbose=False):
        super().__init__(name, verbose)
        self.skip_wcs = skip_wcs
        if isinstance(trim, (int, float)):
            trim = (trim, trim)
        self.trim = trim
        self._parallel_friendly = True

    def run(self, image):
        trim = self.trim if self.trim is not None else image.metadata["overscan"]
        center = image.shape[::-1] / 2
        shape = image.shape - 2 * np.array(trim)
        cutout = image.cutout(center, shape)
        image.data = cutout.data
        image.sources = cutout.sources
        image.wcs = cutout.wcs


class Cutouts(Block):
    def __init__(self, shape:Union[int, tuple]=50, wcs:bool=False, name:str=None, sources: bool=False):
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
    Compute transformation fromm a reference image

    |read| ``Image.sources`` on both reference and input image

    |write| ``Image.transform``

    Parameters
    ----------
    ref : Image
        image containing detected sources
    n : int, optional
        number of stars to consider to compute transformation, by default 10
    """

    def __init__(self, reference_image: Image, n=10, discard=True, **kwargs):
        super().__init__(**kwargs)
        ref_coords = reference_image.sources.coords
        self.ref = ref_coords[0:n].copy()
        self.n = n
        self.quads_ref, self.stars_ref = twirl_utils.quads_stars(ref_coords, n=n)
        self.KDTree = KDTree(self.quads_ref)
        self.discard = discard
        self._parallel_friendly = True

    def run(self, image):
        if len(image.sources.coords) >= 5:
            result = self.solve(image.sources.coords)
            if result is not None:
                image.transform = AffineTransform(result)
            else:
                image.discard = True
        else:
            image.discard = True

    def solve(self, coords, tolerance=2):
        s = coords.copy()
        quads, stars = twirl_utils.quads_stars(s, n=self.n)
        _, indices = self.KDTree.query(quads)

        # We pick the two asterismrefs leading to the highest stars matching
        closeness = []
        for i, m in enumerate(indices):
            M = twirl_utils._find_transform(self.stars_ref[m], stars[i])
            new_ref = twirl_utils.affine_transform(M)(self.ref)
            closeness.append(
                twirl_utils._count_cross_match(s, new_ref, tolerance=tolerance)
            )

        i = np.argmax(closeness)
        m = indices[i]
        S1 = self.stars_ref[m]
        S2 = stars[i]
        M = twirl_utils._find_transform(S1, S2)
        new_ref = twirl_utils.affine_transform(M)(self.ref)

        matches = twirl_utils.cross_match(
            new_ref, s, tolerance=tolerance, return_ixds=True
        ).T
        if len(matches) == 0:
            return None
        else:
            i, j = matches

        return twirl_utils._find_transform(s[j], self.ref[i])


class Drizzle(Block):
    
    def __init__(self, reference, pixfrac=1., **kwargs):
        from drizzle import drizzle
        super().__init__(self, **kwargs)
        self.reference = reference
        self.pixfrac = pixfrac
        self.drizzle = drizzle.Drizzle(outwcs=reference.wcs, pixfrac=pixfrac)
        self.image = None

    def run(self, image):
        WCS = image.wcs
        self.drizzle.add_image(image.data, image.wcs)

    def terminate(self):
        data = self.drizzle.outsci
        header = self.reference.header.copy()
        header.update(self.drizzle.outwcs.to_header())
        self.image = Image(data=data, header=header)