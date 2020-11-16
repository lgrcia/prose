from prose import io, Image
import matplotlib.pyplot as plt
import numpy as np
from os import path
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord, Distance
from prose.fluxes import Fluxes
import warnings
from prose import visualisation as viz
from astropy.io import fits
from prose.telescope import Telescope
from prose import utils, CONFIG
from astropy.table import Table
from astropy.wcs import WCS, utils as wcsutils
import pandas as pd
from scipy.stats import binned_statistic
from prose.blocks.psf import Gaussian2D, Moffat2D
import os
import shutil
from astropy.stats import sigma_clip


# TODO: add n_stars to show_stars

class Observation(Fluxes):
    """
    Class to load and analyze photometry products
    """

    def __init__(self, photfile):
        super().__init__(photfile)

        # Observation info
        # self.observation_date = None
        # self.n_images = None
        # self.filter = None
        # self.exposure = None
        # self.data = {}  # dict
        # self.all_data = {}

        self.phot = photfile
        self.telescope = Telescope.from_name(self.telescope)

        self.gaia_data = None
        self.tic_data = None
        # self.stars = None  # in pixels
        # self.target = {"id": None,
        #                "name": None,
        #                "radec": [None, None]} # ra dec are loaded as written in stack FITS header
        #
        # # Convenience
        # self.bjd_tdb = None
        # self.gaia_data = None
        self.wcs = WCS(self.xarray.attrs)
        # self.hdu = None

    def _check_stack(self):
        assert 'stack' in self.xarray is not None, "No stack found"

    # Loaders and savers (files and data)
    # ------------------------------------
    def __copy__(self):
        copied = Observation(self.xarray.copy())
        copied.phot = self.phot
        return copied

    def copy(self):
        return self.__copy__()

    def save_mcmc_file(self, destination):

        assert self.lcs is not None, "Lightcurve is missing"

        df = pd.DataFrame(
            {
                "BJD-TDB": self.time,
                "DIFF_FLUX": self.lc.flux,
                "ERROR": self.lc.error,
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

        df.to_csv(
            "{}/MCMC_{}.txt".format(destination, self.products_denominator), sep=" ", index=False
        )

    def save(self, filepath=None):
        self.xarray.to_netcdf(self.phot if filepath is None else filepath)

    def export_stack(self, destination, **kwargs):
        header = {name: value for name, value in self.xarray.attrs.items() if name.isupper()}
        data = self.stack

        hdul = fits.HDUList([fits.PrimaryHDU(data=data, header=fits.Header(header))])
        hdul.writeto(destination, **kwargs)

    def import_stack(self, fitsfile):
        data = fits.getdata(fitsfile)
        header = fits.getheader(fitsfile)

        self.wcs = WCS(header)
        self.xarray.attrs.update(utils.header_to_cdf4_dict(header))
        self.xarray["stack"] = (('w', 'h'), data)

    # Convenience
    # -----------
    @property
    def skycoord(self):
        return SkyCoord(self.RA, self.DEC, frame='icrs', unit=(self.telescope.ra_unit, self.telescope.dec_unit))

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
        return f"http://simbad.u-strasbg.fr/simbad/sim-coo?Coord={self.RA}+{self.DEC}&CooFrame=FK5&CooEpoch=2000&CooEqui=" \
               "2000&CooDefinedFrames=none&Radius=2&Radius.unit=arcmin&submit=submit+query&CoordList="

    @property
    def products_denominator(self):
        return "{}_{}_{}_{}".format(
            self.telescope.name,
            self.observation_date.strftime(format="%Y%m%d"),
            self.target["name"],
            self.filter,
        )

    # Methods
    # -------

    def _compute_bjd(self):
        assert self.telescope is not None
        assert self.skycoord is not None
        time = Time(self._time, format="jd", scale="utc", location=self.telescope.earth_location)
        light_travel_tbd = time.light_travel_time(self.skycoord, location=self.telescope.earth_location)
        self.bjd_tdb = (time + light_travel_tbd).value
        self.data["bjd tdb"] = self.bjd_tdb

        # Catalog queries
        # ---------------

    def query_gaia(self, n_stars=1000):
        from astroquery.gaia import Gaia

        header = self.xarray.attrs
        shape = self.stack.shape
        cone_radius = np.sqrt(2) * np.max(shape) * self.telescope.pixel_scale / 120

        coord = self.skycoord
        radius = u.Quantity(cone_radius, u.arcminute)
        gaia_query = Gaia.cone_search_async(coord, radius, verbose=False, )
        self.gaia_data = gaia_query.get_results()
        self.gaia_data.sort("phot_g_mean_flux", reverse=True)

        skycoords = SkyCoord(
            ra=self.gaia_data['ra'],
            dec=self.gaia_data['dec'],
            pm_ra_cosdec=self.gaia_data['pmra'],
            pm_dec=self.gaia_data['pmdec'],
            radial_velocity=self.gaia_data['radial_velocity'],
            obstime=Time(2015.0, format='decimalyear'))

        self.gaia_data["x"], self.gaia_data["y"] = np.array(wcsutils.skycoord_to_pixel(skycoords, self.wcs))

    def query_tic(self, n_stars=1000):
        from astroquery.mast import Catalogs

        header = self.xarray.attrs
        shape = self.stack.shape
        cone_radius = np.sqrt(2) * np.max(shape) * self.telescope.pixel_scale / 120

        coord = self.skycoord
        radius = u.Quantity(cone_radius, u.arcminute)
        self.tic_data = Catalogs.query_region(coord, radius, "TIC", verbose=False)
        self.tic_data.sort("Jmag")

        skycoords = SkyCoord(
            ra=self.tic_data['ra'],
            dec=self.tic_data['dec'], unit="deg")

        self.tic_data["x"], self.tic_data["y"] = np.array(wcsutils.skycoord_to_pixel(skycoords, self.wcs))

    # Plot
    # ----

    def show_stars(self, size=10, flip=False, view=None, zoom=False, n=None):
        """
        Show stack image and detected stars

        Parameters
        ----------
        size: float (optional)
            pyplot figure (size, size)
        n: int
            max number of stars to show
        flip: bool
            flip image
        view: 'all', 'reference'
            - ``reference`` : only highlight target and comparison stars
            - ``all`` : all stars are shown

        Example
        -------
        .. image:: /guide/examples_images/plot_stars.png
           :align: center
        """

        if self.target == -1:
            zoom = False

        self._check_stack()

        if n is not None:
            if view == "reference":
                raise AssertionError("'n_stars' kwargs is incompatible with 'reference' view that will display all stars")
            stars = self.stars[0:n]
        else:
            stars = self.stars

        if view is None:
            view = "reference" if 'comps' in self else "all"

        if view == "all":
            viz.fancy_show_stars(
                self.stack, stars,
                flip=flip, size=size, target=self.target,
                pixel_scale=self.telescope.pixel_scale, zoom=zoom)

        elif view == "reference":
            assert 'comps' in self, "No differential photometry"
            viz.fancy_show_stars(
                self.stack, stars,
                ref_stars=self.xarray.comps.isel(apertures=self.aperture).values, target=self.target,
                flip=flip, size=size, view="reference", pixel_scale=self.telescope.pixel_scale, zoom=zoom)

    @staticmethod
    def plot_ids(x, y, idxs, color, alpha, n=None, split=False, **kwargs):
        ax = plt.gcf().axes[0]
        xlim, ylim = ax.get_xlim(), ax.get_ylim()

        if n is not None:
            x = x[0:n]
            y = y[0:n]
            idxs = idxs[0:n]

        within = np.argwhere(np.logical_and.reduce([xlim[0] < x,  x < xlim[1],  ylim[1] < y,  y < ylim[0]])).flatten()
        x = x[within]
        y = y[within]
        idxs = idxs[within]

        ax.plot(x, y, "x", color=color, alpha=alpha, **kwargs)
        ax = plt.gca()

        for x, y, i in zip(x, y, idxs):
            if xlim[0] < x < xlim[1] and ylim[1] < y < ylim[0]:
                _id = str(i)
                if split:
                    _id = f"{_id[0:len(_id) // 2]}\n{_id[len(_id) // 2::]}"
                plt.annotate(_id,
                             xy=[x, y - 12],
                             color=color,
                             ha='center', fontsize=8, va='bottom')

    def show_gaia(self, color="lightblue", alpha=0.5, n=None, **kwargs):
        """
        Overlay Gaia objects on last axis

        Parameters
        ----------
        color : str, optional
            color of markers, by default "lightblue"
        alpha : float, optional
            opacity of markers, by default 0.5
        **kwargs : dict
            any kwargs compatible with pyplot.plot function
        """
        if self.gaia_data is None:
            self.query_gaia()

        self.plot_ids(
            self.gaia_data["x"].data,
            self.gaia_data["y"].data,
            self.gaia_data["source_id"].data,
            color=color, alpha=alpha, n=n, split=True, **kwargs)

    def show_tic(self, color="white", alpha=0.7, n=None, **kwargs):
        """
        Overlay TIC objects on last axis

        Parameters
        ----------
        color : str, optional
            color of markers, by default "lightblue"
        alpha : float, optional
            opacity of markers, by default 0.5
        **kwargs : dict
            any kwargs compatible with pyplot.plot function
        """
        if self.tic_data is None:
            self.query_tic()

        self.plot_ids(
            self.tic_data["x"].data,
            self.tic_data["y"].data,
            self.tic_data["ID"].data,
            color=color, alpha=alpha, n=n, **kwargs)

    def show_cutout(self, star=None, size=200):
        """
        Show a zoomed cutout around a detected star

        Parameters
        ----------
        star : [type], optional
            detected star id, by default None
        size : int, optional
            side size of square cutout in pixel, by default 200
        """
        if star == None:
            star = self.target
        viz.show_stars(self.stack, self.stars, highlight=star, size=6)
        ax = plt.gcf().axes[0]
        ax.set_xlim(np.array([-size / 2, size / 2]) + self.stars[star][0])
        ax.set_ylim(np.array([size / 2, -size / 2]) + self.stars[star][1])

    def plot_comps_lcs(self, n=5, ylim=(0.98, 1.02)):
        """
        Plot comparison stars light curves along target star light curve

        Example
        -------
        .. image: /examples_images/plot_comps.png
           :align: center
        """
        idxs = [self.target, *self.xarray.comps.isel(apertures=self.aperture).values[0:n]]
        lcs = [self.xarray.fluxes.isel(star=i, apertures=self.aperture).values for i in idxs]

        if ylim is None:
            ylim = (self.flux.min() * 0.99, self.flux.max() * 1.01)

        offset = ylim[1] - ylim[0]

        if len(plt.gcf().axes) == 0:
            plt.figure(figsize=(5, 10))

        for i, lc in enumerate(lcs):
            viz.plot_lc(self.time, lc - i * offset, errorbar_kwargs=dict(c="grey", ecolor="grey"))
            plt.annotate(idxs[i], (self.time.min() + 0.005, 1 - i * offset + offset / 3))

        plt.ylim(1 - (i + 0.5) * offset, ylim[1])
        plt.title("Comparison stars", loc="left")
        plt.grid(color="whitesmoke")
        plt.tight_layout()

    def plot_data(self, key):
        """
        Plot a data time-serie on top of the target light curve

        Parameters
        ----------
        key : str
            key of data time-serie in self.data dict

        Examples
        --------
        .. image:: /guide/examples_images/plot_systematic.png
           :align: center
        """
        self.plot()
        amp = (np.percentile(self.flux, 95) - np.percentile(self.flux, 5)) / 2
        plt.plot(self.time, amp * utils.rescale(self.xarray[key]) + 1,
                 label="normalized {}".format(key),
                 color="k"
                 )
        plt.legend()

    def plot_psf_fit(self, size=21, cmap="inferno", c="blueviolet", model=Gaussian2D):
        """
         Plot a 2D gaussian fit of the global psf (extracted from stack fits)

        Example
        -------
        .. image:: /guide/examples_images/plot_psf_fit.png
           :align: center
        """

        psf_fit = model()
        image = Image(data=self.stack, stars_coords=self.stars)
        psf_fit.run(image)

        if len(plt.gcf().get_axes()) == 0:
            plt.figure(figsize=(12, 4))
        viz.plot_marginal_model(psf_fit.epsf, psf_fit.optimized_model, cmap=cmap, c=c)

        return {"theta": image.theta,
                "std_x": image.psf_sigma_x,
                "std_y": image.psf_sigma_y,
                "fwhm_x": image.fwhmx,
                "fwhm_y": image.fwhmy }

    def plot_rms(self, bins=0.005):
        """
        Plot binned rms of lightcurves vs the CCD equation

        Example
        -------
        .. image:: /guide/examples_images/plot_rms.png
           :align: center
        """
        self._check_diff()
        viz.plot_rms(
            self.fluxes,
            self.lcs,
            bins=bins,
            target=self.target["id"],
            highlights=self.comparison_stars)

    def plot_systematics(self, fields=None, ylim=(0.98, 1.02)):
        if fields is None:
            fields = ["dx", "dy", "fwhm", "airmass", "sky"]

        flux = self.flux.copy()
        flux /= np.mean(flux)

        if ylim is None:
            ylim = (flux.min() * 0.99, flux.max() * 1.01)

        offset = ylim[1] - ylim[0]

        if len(plt.gcf().axes) == 0:
            plt.figure(figsize=(5 ,10))

        viz.plot_lc(self.time, flux, errorbar_kwargs=dict(c="C0", ecolor="C0"))

        for i, field in enumerate(fields):
            if field in self:
                scaled_data = sigma_clip(self.xarray[field].values)
                scaled_data = scaled_data - np.median(scaled_data)
                scaled_data = scaled_data / np.std(scaled_data)
                scaled_data *= np.std(flux)
                scaled_data += 1 - (i + 1) * offset
                viz.plot_lc(self.time, scaled_data, errorbar_kwargs=dict(c="grey", ecolor="grey"))
                plt.annotate(field, (self.time.min() + 0.005, 1 - (i + 1) * offset + offset / 3))
            else:
                i -= 1

        plt.ylim(1 - (i + 1.5) * offset, ylim[1])
        plt.title("Systematics", loc="left")
        plt.grid(color="whitesmoke")
        plt.tight_layout()

    def plot_raw_diff(self):

        plt.subplot(211)
        plt.title("Differential lightcurve", loc="left")
        self.plot()
        plt.grid(color="whitesmoke")

        plt.subplot(212)
        plt.title("Normalized flux", loc="left")
        flux = self.xarray.raw_fluxes.isel(star=self.target, apertures=self.aperture).values
        plt.plot(self.time, flux, ".", ms=3, label="target", c="C0")
        if 'alc' in self:
            plt.plot(self.time, self.xarray.alc.isel(apertures=self.aperture).values*np.median(flux), ".", ms=3, c="k", label="artifical star")

        plt.legend()
        plt.grid(color="whitesmoke")
        plt.xlim([np.min(self.time), np.max(self.time)])
        plt.tight_layout()

    def save_report(self, destination, fields=None, std=None, ylim=(0.98, 1.02), remove_temp=True):

        if fields is None:
            fields = ["dx", "dy", "fwhm", "airmass", "sky"]

        file_name = destination

        temp_folder = path.join(path.dirname(destination), "temp")

        if path.isdir("temp"):
            shutil.rmtree(temp_folder)

        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)

        os.mkdir(temp_folder)

        self.show_stars(10, view="reference")
        star_plot = path.join(temp_folder, "starplot.png")
        fig = plt.gcf()
        fig.patch.set_alpha(0)
        plt.savefig(star_plot)
        plt.close()

        plt.figure(figsize=(6, 10))
        self.plot_raw_diff()
        if ylim is not None:
            plt.gcf().axes[0].set_ylim(ylim)
        lc_report_plot = path.join(temp_folder, "lcreport.png")
        fig = plt.gcf()
        fig.patch.set_alpha(0)
        plt.savefig(lc_report_plot)
        plt.close()

        plt.figure(figsize=(6, 10))
        self.plot_systematics(fields=fields)
        syst_plot = path.join(temp_folder, "systplot.png")
        fig = plt.gcf()
        fig.patch.set_alpha(0)
        plt.savefig(syst_plot)
        plt.close()

        if 'comps' in self:
            plt.figure(figsize=(6, 10))
            self.plot_comps_lcs()
            lc_comps_plot = path.join(temp_folder, "lccompreport.png")
            fig = plt.gcf()
            fig.patch.set_alpha(0)
            plt.savefig(lc_comps_plot)
            plt.close()

        plt.figure(figsize=(10, 3.5))
        psf_p = self.plot_psf_fit()

        psf_fit = path.join(temp_folder, "psf_fit.png")
        plt.savefig(psf_fit, dpi=60)
        plt.close()
        theta = psf_p["theta"]
        std_x = psf_p["std_x"]
        std_y = psf_p["std_y"]

        marg_x = 10
        marg_y = 8

        pdf = viz.prose_FPDF(orientation='L', unit='mm', format='A4')
        pdf.add_page()

        pdf.set_draw_color(200, 200, 200)

        pdf.set_font("helvetica", size=12)
        pdf.set_text_color(50, 50, 50)
        pdf.text(marg_x, 10, txt="{}".format(self.name))

        pdf.set_font("helvetica", size=6)
        pdf.set_text_color(74, 144, 255)
        pdf.text(marg_x, 17, txt="simbad")
        pdf.link(marg_x, 15, 8, 3, self.simbad)

        pdf.set_text_color(150, 150, 150)
        pdf.set_font("Helvetica", size=7)
        pdf.text(marg_x, 14, txt="{} · {} · {}".format(
            self.date, self.telescope.name, self.filter))

        pdf.image(star_plot, x=78, y=17, h=93.5)
        pdf.image(lc_report_plot, x=172, y=17, h=95)
        pdf.image(syst_plot, x=227, y=17, h=95)

        if 'comps' in self:
            pdf.image(lc_comps_plot, x=227, y=110, h=95)

        datetimes = Time(self.time, format='jd', scale='utc').to_datetime()
        min_datetime = datetimes.min()
        max_datetime = datetimes.max()

        obs_duration_hours = (max_datetime - min_datetime).seconds // 3600
        obs_duration_mins = ((max_datetime - min_datetime).seconds // 60) % 60

        obs_duration = f"{min_datetime.strftime('%H:%M')} - {max_datetime.strftime('%H:%M')} " \
            f"[{obs_duration_hours}h{obs_duration_mins if obs_duration_mins!=0 else ''}]"

        max_psf = np.max([std_x, std_y])
        min_psf = np.min([std_x, std_y])
        ellipticity = (max_psf ** 2 - min_psf ** 2) / max_psf ** 2

        viz.draw_table(pdf, [
            ["Time", obs_duration],
            ["RA - DEC", f"{self.RA} {self.DEC}"],
            ["images", len(self.time)],
            ["GAIA id", None],
            ["mean std · fwhm",
             f"{np.mean(self.fwhm) / (2 * np.sqrt(2 * np.log(2))):.2f} · {np.mean(self.fwhm):.2f} pixels"],
            ["Telescope", self.telescope.name],
            ["Filter", self.filter],
            ["exposure", f"{np.mean(self.exptime)} s"],
        ], (5, 20))

        viz.draw_table(pdf, [
            ["PSF std · fwhm (x)", f"{std_x:.2f} · {2 * np.sqrt(2 * np.log(2)) * std_x:.2f} pixels"],
            ["PSF std · fwhm (y)", f"{std_y:.2f} · {2 * np.sqrt(2 * np.log(2)) * std_y:.2f} pixels"],
            ["PSF ellipicity", f"{ellipticity:.2f}"],
        ], (5, 78))

        pdf.image(psf_fit, x=5.5, y=55, w=65)

        pdf.output(file_name)

        if path.isdir("temp") and remove_temp:
            shutil.rmtree(temp_folder)

        print("report saved at {}".format(path.abspath(file_name)))

    def plot_precision(self, bins=0.005, aperture=None):

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
