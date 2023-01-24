from astropy.stats import gaussian_sigma_to_fwhm, sigma_clipped_stats
import numpy as np
from photutils.aperture import aperture_photometry
from .. import Image, Block

__all__ = ["AperturePhotometry", "AnnulusBackground"]


class AperturePhotometry(Block):
    def __init__(self, name=None, radii=None, scale=True):
        super().__init__(name=name)
        if radii is None:
            self._radii = np.linspace(0.3, 8, 40)
        else:
            self._radii = radii
        self.scale = scale

    def run(self, image: Image):
        if self.scale:
            fwhm = image.epsf_params["sigma_x"] * gaussian_sigma_to_fwhm
            radii = np.array(fwhm * self._radii)
        else:
            radii = np.array(self._radii)

        apertures = [image.sources.apertures(r) for r in radii]
        aperture_fluxes = np.array(
            [aperture_photometry(image.data, a)["aperture_sum"].data for a in apertures]
        )

        image.aperture = {"fluxes": aperture_fluxes, "radii": radii}


class _AnnulusPhotometry(Block):
    def __init__(self, name=None, rin=5, rout=8, scale=True):
        super().__init__(name=name)
        self.rin = rin
        self.rout = rout
        self.scale = scale


class AnnulusBackground(_AnnulusPhotometry):
    def __init__(self, name=None, rin=5, rout=8, sigma=3, scale=True):
        super().__init__(name=name, rin=rin, rout=rout, scale=scale)
        self.sigma = sigma

    def run(self, image: Image):
        if self.scale:
            fwhm = image.computed["epsf_params"]["sigma_x"] * gaussian_sigma_to_fwhm
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