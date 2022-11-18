from . import Image
import matplotlib.pyplot as plt
import numpy as np
import re
from astropy.time import Time
from astropy import units as u
from .fluxes import ApertureFluxes, pont2006
from . import viz
from .telescope import Telescope
from . import utils
import pandas as pd
from scipy.stats import binned_statistic
from .blocks.psf import Gaussian2D
from astropy.io.fits.verify import VerifyWarning
import warnings
import shutil
from pathlib import Path
from .utils import fast_binning, z_scale, clean_header
from .console_utils import info, warning, error
from . import blocks
from prose import Sequence
from matplotlib import gridspec
from .blocks import catalogs
from tqdm import tqdm
from .console_utils import TQDM_BAR_FORMAT

warnings.simplefilter('ignore', category=VerifyWarning)

def _progress(do=True):
    if do:
        return lambda x: tqdm(x, unit="observations", ncols=80, bar_format=TQDM_BAR_FORMAT)
    else:
        return lambda x: x


class Observation(ApertureFluxes):
    """
    Class to load and analyze photometry products
    """

    def __init__(self, photfile, time_verbose=False):
        """
        Parameters
        ----------
        photfile : str
            path of the `.phot` file to load
        time_verbose: bool, optional
            whether time conversion success should be verbose
        """
        super().__init__(photfile)

        utils.remove_sip(self.xarray.attrs)

        self.phot = photfile if isinstance(photfile, (Path, str)) else None
        self.telescope = Telescope.from_name(self.telescope)
        self.gaia_data = None
        self._meridian_flip = None

        if self.has_stack:
            self._stack = Image(data=self.x["stack"].values, header=clean_header(self.x.attrs))
            
            if "stars" in self.x:
                self.stack.stars_coords = self.x.stars.values

        has_bjd = hasattr(self.xarray, "bjd_tdb")
        if has_bjd:
            has_bjd = ~np.all(self.xarray.bjd_tdb.isnull().values)

        if not has_bjd:
            try:
                self.compute_bjd()
                if not time_verbose:
                    info("Time converted to BJD TDB")
            except:
                if not time_verbose:
                    info("Could not convert time to BJD TDB")

    @property
    def has_stack(self):
        return 'stack' in self.xarray is not None

    def assert_stack(self):
        assert self.has_stack, "This observation has no stack image"

    # Loaders and savers (files and data)
    # ------------------------------------
    def __copy__(self):
        copied = self.__class__(self.xarray.copy(), time_verbose=True)
        copied.phot = self.phot
        copied.telescope = self.telescope
        if self.has_stack:
            copied.stack.wcs = self.wcs

        return copied

    def copy(self):
        return self.__copy__()

    def to_csv(self, destination, sep=" ", old=True):
        """Export a typical csv of the observation's data

        Parameters
        ----------

        destination : str
            Path of the csv file to save
        sep : str, optional
            separation string within csv, by default " "
        """

        # TODO
        new_fields = {
            "bjd_tdb": "BJD-TDB",
            "diff_flux": "DIFF_FLUX",
            "diff_error": "ERROR",
            "x": "dx_MOVE",
            "y": "dy_MOVE",
            "fwhm": "FWHM",
            "fwhm_x": "FWHMx",
            "fwhm_y": "FWHMy",
            "sky": "SKYLEVEL",
            "airmass": "AIRMASS",
            "exposure": "EXPOSURE",
        }

        df = pd.DataFrame(
            {
                "BJD-TDB" if self.time_format == "bjd_tdb" else "JD-UTC": self.time,
                "DIFF_FLUX": self.diff_flux,
                "ERROR": self.diff_error,
                "dx_MOVE": self.dx,
                "dy_MOVE": self.dy,
                "FWHM": self.fwhm,
                "FWHMx": self.fwhm,
                "FWHMy": self.fwhm,
                "SKYLEVEL": self.sky,
                "AIRMASS": self.airmass,
                "EXPOSURE": self.exptime,
            }
        )

        df.to_csv(destination, sep=sep, index=False)

    # TODO: if None, phot if path was provided , if direct, autoname else filename
    def save(self, destination=None, verbose=True):
        """Save current observation

        Parameters
        ----------
        destination : str, optional
            path to phot file, by default None
            TODO: doc
        """
        destination = self.phot if destination is None else destination
        self.xarray.attrs.update(self.stack.header)
        self.xarray.to_netcdf(destination)
        if verbose:
            info(f"saved {str(Path(destination).absolute())}")

    # Convenience
    # -----------

    @property
    def stack(self):
        return self._stack

    @stack.setter
    def stack(self, new):
        self._stack = new

    @property
    def simbad_url(self):
        """
        [notebook feature] clickable simbad query url for specified target
        """
        from IPython.core.display import display, HTML

        display(HTML('<a href="{}">{}</a>'.format(self.simbad, self.simbad)))

    @property
    def simbad(self):
        """
        simbad query url for specified target
        """
        ra = str(self.stack.ra.to(u.hourangle))
        dec = str(self.stack.dec.to(u.deg))

        return f"http://simbad.u-strasbg.fr/simbad/sim-coo?Coord={ra}+{dec}&CooFrame=FK5&CooEpoch=2000&CooEqui=" \
               "2000&CooDefinedFrames=none&Radius=2&Radius.unit=arcmin&submit=submit+query&CoordList="

    # TODO add to core Image and replace by stack Image

    @property
    def label(self):
        """A conveniant name for the observation: {telescope}_{date}_{name}_{filter}

        Returns
        -------
        [type]
            [description]
        """
        return self.stack.label

    @property
    def meridian_flip(self):
        """Meridian flip time. Supposing EAST and WEST encode orientation
        """
        if self._meridian_flip is not None:
            return self._meridian_flip
        else:
            has_flip = hasattr(self.xarray, "flip")
            # if has_flip:
            #     try:
            #         self.flip = np.array(self.flip, dtype='str')
            #         np.all(np.isnan(self.flip))
            #     except TypeError:
            #         pass

            if has_flip:
                if "WEST" in self.flip:
                    flip = (self.flip.copy() == "WEST").astype(int)
                    diffs = np.abs(np.diff(flip))
                    if np.any(diffs):
                        self._meridian_flip = self.time[np.argmax(diffs).flatten()]
                    else:
                        self._meridian_flip = None

                    return self._meridian_flip
                else:
                    return None
            else:
                return None

    @property
    def date(self):
        return self.stack.date

    @property
    def night_date(self):
        return self.stack.night_date


    # WCS
    # ----

    # TODO replace by stack Image methods
    @property
    def wcs(self):
        return self.stack.wcs

    # Methods
    # -------

    def compute_bjd(self, version="prose"):
        """Compute BJD_tdb based on current time

        Once this is done self.time is BJD tdb and time format can be checked in self.time_format. Note that half the
        exposure time is added to the JD times before conversion. The precision of the returned time is not
        guaranteed, especially with "prose" method (~30ms). "eastman" option accuracy is 20ms. See
        http://astroutils.astronomy.ohio-state.edu/time/utc2bjd.html for more details.

        Parameters
        ----------
        version : str, optiona
            - "prose": uses an astropy method
            - "eastman": uses the web applet http://astroutils.astronomy.ohio-state.edu (Eastman et al. 2010) [requires
            an  internet connection]
            by default "prose"
        """
        assert self.telescope is not None
        assert self.stack.skycoord is not None

        exposure_days = self.xarray.exposure.values/60/60/24

        # For backward compatibility
        # --------------------------
        if "time_format" not in self.xarray.attrs:
            self.xarray.attrs["time_format"] = "jd_utc"
            self.xarray["jd_utc"] = ("time", self.time)
        if "jd_utc" not in self:
            self.xarray["jd_utc"] = ("time", self.jd)
            self.xarray.drop("jd")
        # -------------------------

        if version == "prose":
            time = Time(self.jd_utc + exposure_days/2, format="jd", scale="utc", location=self.telescope.earth_location).tdb
            light_travel_tbd = time.light_travel_time(self.stack.skycoord, location=self.telescope.earth_location)
            bjd_time = (time + light_travel_tbd).value

        elif version == "eastman":
            bjd_time = utils.jd_to_bjd(self.jd_utc + exposure_days/2, self.stack.skycoord.ra.deg, self.stack.skycoord.dec.deg)

        self.xarray = self.xarray.assign_coords(time=bjd_time)
        self.xarray["bjd_tdb"] = ("time", bjd_time)
        self.xarray.attrs["time_format"] = "bjd_tdb"


    # Plot
    # ----

    def plot_comps_lcs(self, n=15, ylim=None):
        """Plot comparison stars light curves along target star light curve

        Parameters
        ----------
        n : int, optional
            Number max of comparison to show, by default 5
        ylim : tuple, optional
            ylim of the plot, by default None and autoscale
        """
        idxs = [self.target, *self.xarray.comps.isel(apertures=self.aperture).values[0:n]]
        lcs = [self.xarray.diff_fluxes.isel(star=i, apertures=self.aperture).values for i in idxs]

        if ylim is None:
            ylim = (self.diff_flux.min() * 0.99, self.diff_flux.max() * 1.01)

        offset = ylim[1] - ylim[0]
        
        if len(plt.gcf().axes) == 0:
            plt.figure(figsize=(5, 10))

        for i, lc in enumerate(lcs):
            color = "grey" if i != 0 else "black"
            viz.plot(self.time, lc - lc.mean() - i * offset, bincolor=color)
            plt.annotate(idxs[i], (self.time.min() + 0.005, - i * offset + offset/3))

        plt.ylim(-len(lcs)*offset + offset/2, offset/2)
        plt.title("Comparison stars", loc="left")
        plt.grid(color="whitesmoke")
        plt.tight_layout()

    def _compute_psf_model(self, star=None, size=21, model=Gaussian2D):
        """Compute a PSF model stored in ``stack`` (see ``prose.blocks``) 

        Parameters
        ----------
        star : int, optional
            star psf, by default None which evaluates a model on a Median PSF based on all stars in the field
        size : int, optional
            cutout size, by default 21
        model : _type_, optional
            psf block to be used, by default Gaussian2D
        """
        self.stack = blocks.Cutouts(size=size)(self.stack)
        
        if star is None:
            Sequence([
                blocks.MedianPSF(),
                model()
            ]).run(self.stack, show_progress=False)
        else:
            assert star in self.stack.cutouts_idxs, "star seems out of frame"
            self.stack.psf = self.stack.cutouts[star].data.copy()
            self.stack.psf /= self.stack.psf.sum()
            self.stack = model()(self.stack)
    @property
    def mean_epsf(self):
        """
        Returns
        -------
        The mean of the image effective PSF in pixels
        """
        return np.mean(self.x.fwhm.values)

    @property
    def mean_target_psf(self):
        """
        Returns
        -------
        An estimation of the fwhm of the target psf using a Gaussian 2D model in pixels
        """
        self._compute_psf_model(star=self.target)
        return np.mean([self.stack.fwhmx, self.stack.fwhmy])

    @property
    def optimal_aperture(self):
        """
        Returns
        -------
        The optimal aperture radius in pixels
        """
        return np.mean(self.apertures_radii[self.aperture,:])

    def plot_psf_model(self, star=None, size=21, cmap="inferno", c="blueviolet", model=Gaussian2D, figsize=(5, 5), axes=None):
        """Plot a PSF model fit of the a PSF

        After this method is called, the model parameters are accessible through the Observation.stack Image attributes

        If star is None, the model is evaluated on a median PSF from all stars.

        Parameters
        ----------
        star : int, optional
            star for which to show psf model, default is None which shows the model of the Median PSF over the stack
        size : int, optional
            square size of extracted PSF, by default 21
        cmap : str, optional
            color map of psf image, by default "inferno"
        c : str, optional
            color of model plot line, by default "blueviolet"
        model : prose.blocks, optional
            a PsfFit block, by default Gaussian2D
        figsize : tuple, optional
            size of the pyplot figure, default (5, 5)
        Returns
        -------
        dict
            PSF fit info (theta, std_x, std_y, fwhm_x, fwhm_y)
        """

        # Example
        # -------
        # .. jupyter-execute::

        #     from prose import Observation
        #     from prose.tutorials import example_phot
        
        #     obs = Observation(example_phot)
        #     obs.plot_psf_model()
            

        self.assert_stack()
        
        # Extracting and computing PSF model
        # ---------------------------------    
        self._compute_psf_model(star=star, model=model, size=size)
        self.stack.plot_psf_model(cmap=cmap, c=c, figsize=figsize, axes=axes)


    def plot_systematics(self, fields=None, ylim=None, amplitude_factor = None):
        """Plot systematics measurements along target light curve

        Parameters
        ----------
        fields : list of str, optional
            list of systematic to include (must be in self), by default None
        ylim : tuple, optional
            plot ylim, by default (0.98, 1.02)
        """

        # Example
        # -------
        # .. jupyter-execute::

        #     from prose import Observation
        #     from prose.tutorials import example_phot
        
        #     obs = Observation(example_phot)
        #     obs.plot_systematics()

        if fields is None:
            fields = ["dx", "dy", "fwhm", "airmass", "sky"]

        flux = self.diff_flux.copy()
        flux /= np.nanmean(flux)
        _, amplitude = pont2006(self.time, self.diff_flux, plot=False)
        if amplitude_factor is None:
            amplitude *= 3
        else:
            amplitude *= amplitude_factor
        offset = 2.5*amplitude

        if len(plt.gcf().axes) == 0:
            plt.figure(figsize=(5 ,10))

        viz.plot(self.time, flux, bincolor="black")
        plt.annotate("diff. flux", (self.time.min() + 0.005, 1 + 1.5*amplitude))

        for i, field in enumerate(fields):
            if field in self:
                scaled_data = self.xarray[field].values.copy()
                off = (i+1)*offset
                scaled_data = scaled_data - np.mean(scaled_data)
                bx, by, be = utils.fast_binning(self.time, scaled_data, 0.005)
                scaled_data /= np.max([10*np.mean(be), (np.percentile(by, 95) - np.percentile(by, 5))])
                scaled_data = scaled_data*amplitude + 1 - off
                viz.plot(self.time, scaled_data, bincolor="grey")
                plt.annotate(field, (self.time.min() + 0.005, 1 - off + amplitude / 3))
            else:
                i -= 1

        if ylim is None:
            plt.ylim(1 - off - offset, 1 + offset)
        else:
            plt.ylim(ylim)
        plt.title("Systematics (scaled to diff. flux)", loc="left")
        plt.tight_layout()

    def plot_raw_diff(self):
        """Plot raw target flux and differantial flux 
        """


        # Example
        # -------
        # .. jupyter-execute::

        #     from prose import Observation
        #     from prose.tutorials import example_phot
        
        #     obs = Observation(example_phot)
        #     obs.plot_raw_diff()


        plt.subplot(211)
        plt.title("Differential lightcurve", loc="left")
        self.plot()
        plt.grid(color="whitesmoke")

        plt.subplot(212)
        plt.title("Raw flux", loc="left")
        flux = self.xarray.raw_fluxes.isel(star=self.target, apertures=self.aperture).values
        plt.plot(self.time, flux, ".", ms=3, label="target", c="C0")
        if 'alc' in self:
            plt.plot(self.time, self.xarray.alc.isel(apertures=self.aperture).values*np.median(flux), ".", ms=3, c="k", label="artifical star")

        plt.legend()
        plt.grid(color="whitesmoke")
        plt.xlim([np.min(self.time), np.max(self.time)])
        plt.tight_layout()

    def plot_precision(self, bins=0.005, aperture=None):
        """Plot observation precision estimate against theorethical error (background noise, photon noise and CCD equation)

        Parameters
        ----------
        bins : float, optional
            bin size used to estimate error, by default 0.005 (in days)
        aperture : int, optional
            chosen aperture, by default None
        """

        # Example
        # -------
        # .. jupyter-execute::

        #     from prose import Observation
        #     from prose.tutorials import example_phot
        
        #     obs = Observation(example_phot)
        #     obs.plot_precision()


        n_bin = int(bins / (np.mean(self.exptime) / (60 * 60 * 24)))

        assert len(self.time) > n_bin, "Your 'bins' size is less than the total exposure"

        x = self.xarray.isel(apertures=self.aperture if aperture is None else aperture).copy()

        fluxes = x.raw_fluxes.values
        errors = x.raw_errors.values

        mean_fluxes = np.mean(fluxes, axis=1)
        mean_errors = np.mean(errors, axis=1)

        error_estimate = [np.median(binned_statistic(self.time, f, statistic='std', bins=n_bin)[0]) for f in fluxes]

        area = x.apertures_area[0].values

        # ccd_equation = phot_prose.telescope.error(
        # prose_fluxes, tp_area, np.mean(self.sky), np.mean(self.exptime), np.mean(self.airmass))

        ccd_equation = (mean_errors / mean_fluxes)

        inv_snr_estimate = error_estimate / mean_fluxes

        positive_est = inv_snr_estimate > 0
        mean_fluxes = mean_fluxes[positive_est]
        inv_snr_estimate = inv_snr_estimate[positive_est]
        ccd_equation = ccd_equation[positive_est]
        sorted_fluxes_idxs = np.argsort(mean_fluxes)

        plt.plot(np.log(mean_fluxes), inv_snr_estimate, ".", alpha=0.5, ms=2, c="k",
                 label=f"flux rms ({0.005 * (60 * 24):.1f} min bins)")
        plt.plot(np.log(mean_fluxes)[sorted_fluxes_idxs], (np.sqrt(mean_fluxes) / mean_fluxes)[sorted_fluxes_idxs],
                 "--", c="k", label="photon noise", alpha=0.5)
        plt.plot(np.log(mean_fluxes)[sorted_fluxes_idxs],
                 (np.sqrt(np.mean(self.sky) * area) / mean_fluxes)[sorted_fluxes_idxs], c="k", label="background noise",
                 alpha=0.5)
        # plt.plot(np.log(prose_fluxes)[s], (prose_e/prose_fluxes)[s], label="CCD equation")
        plt.plot(np.log(mean_fluxes)[sorted_fluxes_idxs], ccd_equation[sorted_fluxes_idxs], label="CCD equation")
        plt.legend()
        plt.ylim(
            0.5 * np.percentile(inv_snr_estimate, 2),
            1.5 * np.percentile(inv_snr_estimate, 98))
        plt.xlim(np.min(np.log(mean_fluxes)), np.max(np.log(mean_fluxes)))
        plt.yscale("log")
        plt.xlabel("log(ADU)")
        plt.ylabel("$SNR^{-1}$")
        plt.title("Photometric precision (raw fluxes)", loc="left")

    def plot_meridian_flip(self):
        """Plot vertical line marking the meridian flip time if any
        """
        if self.meridian_flip is not None:
            plt.axvline(self.meridian_flip, c="k", alpha=0.15)
            _, ylim = plt.ylim()
            plt.text(self.meridian_flip, ylim, "meridian flip ", ha="right", rotation="vertical", va="top", color="0.7")

    def plot(self, star=None, meridian_flip=True, bins=0.005, color="k", std=True):
        """Plot observation light curve

        Parameters
        ----------
        star : [type], optional
            [description], by default None
        meridian_flip : bool, optional
            whether to show meridian flip, by default True
        bins : float, optional
            bin size in same unit as Observation.time, by default 0.005
        color : str, optional
            binned points color, by default "k"
        std : bool, optional
            whether to see standard deviation of bins as error bar, by default True, otherwise theoretical error bat is shown
        """

        super().plot(star=star, bins=bins, color=color, std=std)
        if meridian_flip:
            self.plot_meridian_flip()

    # TODO: plot_radial_psf

    def plot_radial_psf(self, star=None, n=40, zscale=False, aperture=None, rin=None, rout=None, axes=None):
        """Plot star cutout overalid with aperture and radial flux.

        Parameters
        ----------
        star : int or list like, optional
            if int: star to plot cutout on, if list like (tuple, np.ndarray) of size 2: coords of cutout, by default None
        n : int, optional
            cutout width and height, by default 40
        zscale : bool, optional
            whether to apply a zscale to cutout image, by default False
        aperture : float or int optional
            radius of aperture to display, by default None corresponds to best target aperture
            if int, corresponds to the number of the aperture in the xarray
            if float, corresponds to the aperture in pixels
        rin : [type], optional
            radius of inner annulus to display, by default None corresponds to inner radius saved
        rout : [type], optional
            radius of outer annulus to display, by default None corresponds to outer radius saved
        """

        # Example
        # -------
        # .. jupyter-execute::

        #     from prose import Observation
        #     from prose.tutorials import example_phot
        
        #     obs = Observation(example_phot)
        #     obs.plot_radial_psf()

        self.assert_stack()

        if isinstance(star, (tuple, list, np.ndarray)):
            x, y = star
        else:
            if star is None:
                star = self.target
            #assert isinstance(star, int), "star must be star coordinates or integer index"

            x, y = self.stars[star]

        Y, X = np.indices(self.stack.shape)
        cutout_mask = (np.abs(X - x) < n) & (np.abs(Y - y) < n)
        inside = np.argwhere((cutout_mask).flatten()).flatten()
        radii = (np.sqrt((X - x) ** 2 + (Y - y) ** 2)).flatten()[inside]
        idxs = np.argsort(radii)
        radii = radii[idxs]
        pixels = self.stack.data.flatten()[inside]
        pixels = pixels[idxs]

        binned_radii, binned_pixels, _ = fast_binning(radii, pixels, bins=1)

        if axes is None:
            fig = plt.figure(figsize=(9.5, 4))
            fig.patch.set_facecolor('xkcd:white')
            ax1 = plt.subplot(1, 5, (1, 3))
            ax2 = plt.subplot(1, 5, (4, 5))
        else:
            ax1, ax2 = axes

        ax1.plot(radii, pixels, "o", fillstyle='none', c="0.7", ms=4)
        ax1.plot(binned_radii, binned_pixels, c="k")
        ax1.set_xlabel("distance from center (pixels)")
        ax1.set_ylabel("ADUs")
        _, ylim = ax1.set_ylim()

        if isinstance(aperture, int) or aperture is None:
            if "apertures_radii" in self.xarray:
                apertures = self.apertures_radii[:, 0]
                if aperture is None:
                    ap = apertures[self.aperture]
                else:
                    ap = apertures[aperture]
        if isinstance(aperture, float):
            ap = aperture
        ax1.set_xlim(0)
        ax1.text(ap, ylim, "APERTURE", ha="right", rotation="vertical", va="top")
        ax1.axvline(ap, c="k", alpha=0.1)
        ax1.axvspan(0, ap, color="0.9", alpha=0.1)

        if "annulus_rin" in self:
            if rin is None:
                rin = self.annulus_rin.mean()
            if rout is None:
                rout = self.annulus_rout.mean()

        if rin is not None:
            ax1.axvline(rin, color="k", alpha=0.2)

        if rout is not None:
            ax1.axvline(rout, color="k", alpha=0.2)
            if rin is not None:
                ax1.axvspan(rin, rout, color="0.9", alpha=0.2)
                _ = ax1.text(rout, ylim, "ANNULUS", ha="right", rotation="vertical", va="top")

        n = np.max([np.max(radii), rout +2 if rout else 0])
        ax1.set_xlim(0, n)
        
        im = self.stack.data[int(y - n):int(y + n), int(x - n):int(x + n)]
        if zscale:
            im = z_scale(im)

        plt.imshow(im, cmap="Greys_r", aspect="auto", origin="lower")

        plt.axis("off")
        center = (n+0.5, n+0.5)
        if aperture is not None:
            ax2.add_patch(plt.Circle(center, aperture, ec='grey', fill=False, lw=2))
        if rin is not None:
            ax2.add_patch(plt.Circle(center, rin, ec='grey', fill=False, lw=2))
        if rout is not None:
            ax2.add_patch(plt.Circle(center, rout, ec='grey', fill=False, lw=2))
        if star is None:
            ax2.text(0.05, 0.05, f"{self.target}", fontsize=12, color="white", transform=ax2.transAxes)
        else:
            ax2.text(0.05, 0.05, f"{star}", fontsize=12, color="white", transform=ax2.transAxes)

        plt.tight_layout()

    def plot_systematics_signal(self, systematics, signal=None, ylim=None, offset=None, figsize=(6, 7)):
        """Plot a systematics and signal model over diff_flux. systeamtics + signal is plotted on top, signal alone on detrended
        data on bottom

        Parameters
        ----------
        systematics : np.ndarray
        signal : np.ndarray
        ylim : tuple, optional
            ylim of the plot, by default None, using the dispersion of y
        offset : tuple, optional
            offset between, by default None
        figsize : tuple, optional
            figure size as in in plt.figure, by default (6, 7)            
        """
        

        viz.plot_systematics_signal(self.time, self.diff_flux, systematics, signal, ylim=ylim, offset=offset,
                                figsize=figsize)

        self.plot_meridian_flip()
        plt.legend()
        plt.xlabel(self.xlabel)
        plt.ylabel("diff. flux")
        plt.tight_layout()
        viz.paper_style()

    @property
    def xlabel(self):
        """Plot xlabel (time) according to its units
        """
        return self.time_format.upper().replace("_", "-")

    def where(self, condition):
        """return filtered observation given a boolean mask of time

        Parameters
        ----------
        condition : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        new_obs = self.copy()
        new_obs.xarray = new_obs.xarray.sel(time=self.time[condition])
        return new_obs


    def show_stars(self, view=None, n=None, flip=False,
               comp_color="yellow", color=[0.51, 0.86, 1.], stars=None, legend=True, ax=None, **kwargs):
        """Show detected stars over stack image
        
        Parameters
        ----------
        view : str, optional
            "all" to see all stars OR "reference" to have target and comparison stars hilighted, by default None
        n : int, optional
            max number of stars to show, by default None,
        flip : bool, optional
            whether to flip image, by default False
        """

        # Example
        # -------
        # .. jupyter-execute::

        #     from prose import Observation
        #     from prose.tutorials import example_phot
        
        #     obs = Observation(example_phot)
        #     obs.show_stars()
            

        self.assert_stack()

        ax = self.stack.show(stars=False, ax=ax)
        
        if stars is None:
            stars = self.stars

        if n is not None:
            if view == "reference":
                raise AssertionError("'n_stars' kwargs is incompatible with 'reference' view that will display all stars")
        else:
            n = len(stars)

        stars = stars[0:n]

        if view is None:
            view = "reference" if 'comps' in self else "all"

        image_size = np.array(np.shape(self.stack))[::-1]

        if flip:
            stars = np.array(image_size) - stars

        if view == "all":
            viz.plot_marks(*stars.T, np.arange(len(stars)), color=color, ax=ax)

            if "stars" in self.xarray:
                others = np.arange(n, len(self.stars))
                others = np.setdiff1d(others, self.target)
                viz.plot_marks(*self.stars[others].T, alpha=0.4, color=color, ax=ax)

        elif view == "reference":
            x = self.xarray.isel(apertures=self.aperture)
            assert 'comps' in self, "No differential photometry"

            comps = x.comps.values

            others = np.setdiff1d(np.arange(len(stars)), x.comps.values)
            others = np.setdiff1d(others, self.target)

            _ = viz.plot_marks(*stars[self.target], self.target, color=color, ax=ax)
            _ = viz.plot_marks(*stars[comps].T, comps, color=comp_color, ax=ax)
            _ = viz.plot_marks(*stars[others].T, alpha=0.4, color=color, ax=ax)

            if legend:
                colors = [comp_color, color]
                texts = ["Comparison stars", "Target"]
                viz.circles_legend(colors, texts, ax=ax)

    def keep_good_stars(self, threshold=3., upper_threshold=35000., trim=10, keep=None, inplace=True):
        """Keep only  stars with a median flux higher than ``threshold*sky``. 
        
        This action will reorganize stars indexes (target id will be recomputed) and reset the differential fluxes to raw.

        Parameters
        ----------
        threshold : float
            threshold for which stars with flux/sky > threshold are kept, default is 3
        upper_threshold : float
            maximum value allowed for the peaks median in time, default is 35000
        trim : int, optional
            value in pixels above which stars are kept, default is 10 to avoid stars too close to the edge, default is 10
        keep : int or list, optional
            number of stars to exclude (starting from 0 if int), default is None (all stars kept)
        inplace : bool
            whether to replace current object or return a new one, by default True

        Returns
        -------
        _type_
            _description_
        """
        good_stars = np.argwhere((np.median(self.peaks, 1)/np.median(self.sky) > threshold) & (np.median(self.peaks, 1) < upper_threshold)).squeeze()
        mask = np.any(np.abs(self.stars[good_stars] - max(self.stack.shape) / 2) > (max(self.stack.shape) - 2 * trim) / 2, axis=1)
        bad_stars = np.argwhere(mask == True).flatten()

        final_stars = np.delete(good_stars, bad_stars)

        if isinstance(keep,int):
            final_stars = np.concatenate([final_stars,np.arange(0,keep+1)],axis=0)
            final_stars = np.unique(final_stars)
        if isinstance(keep,list):
            final_stars = np.concatenate([final_stars,keep ], axis=0)
            final_stars = np.unique(final_stars)

        if inplace:
            new_self = self
        else:
            new_self = self.copy()

        new_self.xarray = new_self.xarray.isel(star=final_stars)

        if self.target != -1:
            new_self.target = np.argwhere(final_stars == new_self.target).flatten()[0]

        if not inplace:
            return new_self

    def flip_correction(self, inplace=True):
        """Align all differential fluxes using a step model of the meridian flip

        Parameters
        ----------
        inplace : bool, optional
            wheter to replace the current Observation or return a new one, by default True
        """
        if inplace:
            new_self = self
        else:
            new_self = self.copy()

        new_diff_fluxes = np.zeros_like(self.diff_fluxes)
        X = self.step()

        for i in range(len(self.apertures)):
            for j in range(len(self.stars)):
                diff_flux = self.diff_fluxes[i, j]
                w = np.linalg.lstsq(X, diff_flux,rcond=-1)[0]
                new_diff_fluxes[i, j] = diff_flux - X @ w + 1.

        new_self.xarray['diff_fluxes'] = (new_self.xarray.diff_fluxes.dims, new_diff_fluxes)

        if not inplace:
            return new_self

    def _convert_flip(self,keyword):
        self.xarray['flip'] = ('time', (self.flip == keyword).astype(int))

    def folder_to_phot(self, confirm=True):
        """replace all the ``phot`` file parent folder content by the ``phot`` file

        Warning: this erases all the parent folder content

        Parameters
        ----------
        confirm : bool, optional
            whether to show a prompt to confirm, by default True
        """
        if confirm:
            confirm = str(input("Will erase all but .phot, enter 'y' to continue: "))
        else:
            confirm = True

        if confirm:
            _phot = Path(self.phot)
            folder = Path(_phot).parent
            shutil.move(str(_phot.absolute()), str(folder.parent.absolute()))
            shutil.rmtree(str(folder.absolute()))

    def lc_widget(self, width=500):
        """[notebook/jupyter feature] displays a widget to play with light curve aperture and binning

        Parameters
        ----------
        width : int, optional
            pixel width of the html widget, by default 500
        """

        # Example
        # -------

        #  .. jupyter-execute::

        #     from prose import Observation
        #     from prose.tutorials import example_phot

        #     obs = Observation(example_phot)
        #     obs.lc_widget()

        from IPython.core.display import display, HTML
        import json
        from pathlib import Path

        html = Path(__file__).parent.absolute() / "html/lightcurve_widget.html"
        widget = html.open("r").read()
        widget = widget.replace("__fluxes__", json.dumps(self.diff_fluxes[:, self.target].tolist()))
        widget = widget.replace("__time__", json.dumps((self.time - 2450000).tolist()))
        widget = widget.replace("__best__", json.dumps(int(self.aperture)))
        widget = widget.replace("__apertures__", json.dumps(self.apertures.tolist()))
        widget = widget.replace("__width__", json.dumps(width))
        i = "a" + str(int(np.random.rand()*100000))
        widget = widget.replace("__divid__", i)
        display(HTML(widget))

    def plot_summary(self, ylim=None, zscale=True):
        fig = plt.figure(figsize=(16,8))
        widths = [0.55, 0.35, 1, 0.7]
        heights = [0.8, 0.4, 0.5]
        specs = fig.add_gridspec(ncols=4, nrows=3, width_ratios=widths, height_ratios=heights)

        image = self.stack

        params = [
            ("Telescope", f"{image.telescope.name}"),
            ("Date", f"{image.night_date}"),
            ("Filter", f"{image.filter}"),
            ("Exposure", f"{image.exposure}"),
            ("RA", f"{image.skycoord.ra:.4f}"),
            ("DEC", f"{image.skycoord.dec:.4f}"),
            ("Dimenion", f"{image.shape[0]}x{image.shape[1]} pixels"),
        ]

        # stack_zoom
        ax_stack = fig.add_subplot(specs[0:2, 0:2])
        self.stack.show_cutout(star=int(self.target), ax=ax_stack, zscale=zscale)
        ax_stack.set_title("Target on stack", loc="left")

        # radial psf
        psf_ax1 = fig.add_subplot(specs[2,0])
        psf_ax2 = fig.add_subplot(specs[2,1])
        self.plot_radial_psf(star=int(self.target), axes=(psf_ax1, psf_ax2))

        # light curve
        ax_lc = fig.add_subplot(specs[0,2])
        self.plot()
        plt.xlabel(f"BJD")
        plt.ylabel("diff. flux")
        if ylim:
            plt.ylim(ylim)
        plt.title("Differential light curve", loc="left")

        # table
        ax_table = fig.add_subplot(specs[1::,2])
        table = plt.table(params, loc=8, fontsize=35, cellLoc="left")
        table.scale(1, 2)
        plt.axis("off")

        # systematics
        ax_syst = fig.add_subplot(specs[0:3, 3])
        self.plot_systematics()
        plt.xlabel(f"BJD")
        plt.ylabel("diff. flux")

        fig.suptitle(f"{image.object} from {image.telescope.name} in {image.filter} on {image.night_date}", fontsize=13)
        viz.paper_style([ax_lc, ax_syst])
        plt.tight_layout()

    def plate_solve(self):
        """Plate solve the current py::stack
        """
        self.stack = blocks.catalogs.PlateSolve()(self.stack)   

    def query_catalog(self, name, correct_pm=True):
        if not self.stack.plate_solved:
            self.plate_solve()
        if name == "gaia":
            self.stack = blocks.catalogs.GaiaCatalog(correct_pm=correct_pm)(self.stack)
        if name == "tess":
            self.stack = blocks.catalogs.TESSCatalog()(self.stack)
        else:
            error(f"No catalog named {name}")

    def set_catalog_target(self, catalog_name, designation, verbose=True):
        self.query_catalog(catalog_name, correct_pm=True)
        i = np.flatnonzero(self.stack.catalogs[catalog_name].id == designation)

        if len(i) == 0:
            if verbose: error("No target found")
            self.target = None
        else:
            i = i[0]
            catalog_xy = self.stack.catalogs[catalog_name][["x", "y"]].values[i]
            self.target = int(np.argmin(np.linalg.norm(self.stars - catalog_xy, axis=1)))
            if verbose: info(f"target is {self.target}")

    def set_gaia_target(self, gaia_id, verbose=False, raise_far=True):
        if not self.stack.plate_solved:
            if verbose:
                info("plate solving ...")
            self.plate_solve()

        if verbose:
            info("querying Gaia catalog ...")
        
        catalog_image = catalogs.GaiaCatalog()(self.stack)

        gaia_table = catalog_image.catalogs["gaia"]
        matched_gaia = gaia_table.id.values == f"Gaia DR2 {gaia_id}"
        target_xy = gaia_table[matched_gaia][["x", "y"]].values[0]
        distances = np.linalg.norm(self.stack.stars_coords - target_xy, axis=1)
        i = np.argmin(distances)

        if distances[i] > 10:
            _distance = f"{distances[i]:.0f} pix."
            if raise_far:
                error(f"matched gaia star too far from detected star ({_distance})")
                return None
            else:
                warning(f"matched gaia star is far from detected star ({_distance})")

        self.target = i
        if verbose:
            info(f"target = {i}")

