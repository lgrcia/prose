from astropy.stats import gaussian_sigma_to_fwhm, sigma_clipped_stats
import numpy as np
from photutils.aperture import aperture_photometry
from .. import Image, Block

__all__ = ["AperturePhotometry", "AnnulusBackground"]


class AperturePhotometry(Block):
    def __init__(self, radii:np.ndarray=None, scale:bool=True, name=None):
        """Perform aperture photometry of each sources

        Parameters
        ----------
        radii : np.ndarray, optional
            apertures radii (definition vary depending of sources), by default None
        scale : bool, optional
            whether to scale radii with :code:`Image.fwhm` usually present in :code:`Image.epsf`, by default True
        name : str, optional
            name of the block, by default None
        """
        super().__init__(name=name)
        if radii is None:
            self._radii = np.linspace(0.3, 8, 40)
        else:
            self._radii = radii
        self.scale = scale

    def run(self, image: Image):
        if self.scale:
            radii = np.array(image.fwhm * self._radii)
        else:
            radii = np.array(self._radii)

        apertures = [image.sources.apertures(r) for r in radii]
        aperture_fluxes = np.array(
            [aperture_photometry(image.data, a)["aperture_sum"].data for a in apertures]
        ).T

        image.aperture = {"fluxes": aperture_fluxes, "radii": radii}


class _AnnulusPhotometry(Block):
    def __init__(self, name=None, rin=5, rout=8, scale=True):
        super().__init__(name=name)
        self.rin = rin
        self.rout = rout
        self.scale = scale


class AnnulusBackground(_AnnulusPhotometry):
    def __init__(self, rin: float=5, rout: float=8, sigma: float=3, scale=True, name: str=None,):
        """Estimate background around each source using an annulus aperture

        Parameters
        ----------

        rin : float, optional
            inner radius of the annulus, by default 5
        rout : float, optional
            outer radius of the annulus, by default 8
        sigma : float, optional
            sigma clipping applied to pixel within annulus before taking the median value, by default 3.
        scale : bool, optional
            wether to scale annulus to EPSF fwhm, by default True. If True, each image must contain an effective PSF and its model (e.g. using :py:class:`~prose.blocks.psf.MedianEPSF` and one of :py:class:`~prose.blocks.psf.Gaussian2D`)
        name : str, optional
            name of the block, by default None
        """
        super().__init__(name=name, rin=rin, rout=rout, scale=scale)
        self.sigma = sigma

    def run(self, image: Image):
        if self.scale:
            fwhm = image.epsf.params["sigma_x"] * gaussian_sigma_to_fwhm
            rin = float(fwhm * self.rin)
            rout = float(fwhm * self.rout)
        else:
            rin = self.rin
            rout = self.rout

        annulus = image.sources.annulus(rin, rout)
        annulus_masks = annulus.to_mask(method="center")

        bkg_median = []
        for mask in annulus_masks:
            annulus_data = mask.multiply(image.data)
            if annulus_data is not None:
                annulus_data_1d = annulus_data[mask.data > 0]
                _, median_sigma_clip, _ = sigma_clipped_stats(
                    annulus_data_1d, sigma=self.sigma
                )
                bkg_median.append(median_sigma_clip)
            else:
                bkg_median.append(0.0)

        image.computed["annulus"] = {
            "rin": rin,
            "rout": rin,
            "median": np.array(bkg_median),
            "sigma": self.sigma,
        }