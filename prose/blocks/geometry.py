from typing import Union

import numpy as np
from scipy.spatial import KDTree
from skimage.transform import AffineTransform
from twirl import utils as twirl_utils

from prose.core import Block, Image

__all__ = ["Trim", "Cutouts", "Drizzle"]


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
        name: str = None,
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



### EDIT ###

class ComputeTransformTwirl(Block):
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
    
    
class ComputeTransformXYShift(Block):
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
        self.ref_coords = ref_coords[0:n].copy()
        self.n = n
        self.quads_ref, self.stars_ref = twirl_utils.quads_stars(ref_coords, n=n)
        self.KDTree = KDTree(self.quads_ref)
        self.discard = discard
        self._parallel_friendly = True

    def run(self, image):
        
        if len(image.sources.coords) >= 5:
            if len(image.sources.coords) <= 2:
                shift = self.ref_coords[0] - image.sources.coords[0]
            else:
                shift = self.xyshift(image.sources.coords, self.ref_coords) # TODO implement tolerance parameter for more modularity
            #result = self.solve(image.sources.coords) #dx, dy from XYshift
            if shift is not None:
                image.transform = AffineTransform(translation=shift)
            else:
                image.discard = True
        else:
            image.discard = True
            


    def xyshift(self, im_stars_pos, ref_stars_pos, tolerance=1.5):
        """
        Compute shift between two set of coordinates (e.g. stars)

        Parameters
        ----------
        im_stars_pos : list or ndarray
            (x,y) coordinates of n points (shape should be (2, n))
        ref_stars_pos : list or ndarray
            [(x,y) coordinates of n points (shape should be (2, n)). Reference set
        tolerance : float, optional
            by default 1.5
        clean : bool, optional
            Merge coordinates if too close, by default False

        Returns
        -------
        ndarray
            (dx, dy) shift
        """
        assert len(im_stars_pos) > 2, f"{len(im_stars_pos)} star coordinates provided (should be > 2)"
        
        clean_ref = ref_stars_pos
        clean_im = im_stars_pos

        delta_x = np.array([clean_ref[:, 0] - v for v in clean_im[:, 0]]).flatten()
        delta_y = np.array([clean_ref[:, 1] - v for v in clean_im[:, 1]]).flatten()

        delta_x_compare = []
        for i, dxi in enumerate(delta_x):
            dcxi = dxi - delta_x
            dcxi[i] = np.inf
            delta_x_compare.append(dcxi)

        delta_y_compare = []
        for i, dyi in enumerate(delta_y):
            dcyi = dyi - delta_y
            dcyi[i] = np.inf
            delta_y_compare.append(dcyi)

        tests = [
            np.logical_and(np.abs(dxc) < tolerance, np.abs(dyc) < tolerance)
            for dxc, dyc in zip(delta_x_compare, delta_y_compare)
        ]
        num = np.array([np.count_nonzero(test) for test in tests])

        max_count_num_i = int(np.argmax(num))
        max_nums_ids = np.argwhere(num == num[max_count_num_i]).flatten()
        dxs = np.array([delta_x[np.where(tests[i])] for i in max_nums_ids])
        dys = np.array([delta_y[np.where(tests[i])] for i in max_nums_ids])

        return np.nan_to_num(np.array([np.mean(dxs), np.mean(dys)]))