from prose import io, Image
import matplotlib.pyplot as plt
import numpy as np
from os import path
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord
from prose.lightcurves import LightCurves, Broeg2005, differential_photometry
import warnings
from prose import visualisation as viz
from astropy.io import fits
from prose.telescope import Telescope
from prose import utils, CONFIG
from astropy.table import Table
from astropy.wcs import WCS, utils as wcsutils
import pandas as pd
from scipy.stats import binned_statistic
from prose._blocks.psf import Gaussian2D, Moffat2D


# TODO: add n_stars to show_stars

class PhotProducts:
    """
    Class to load and analyze photometry products
    """

    def __init__(self, folder_or_file=None, sort_stars=False, keyword=None):
        """
        Parameters
        ----------
        folder_or_file : str path, optional
            Path of folder or file. If folder, should contain at least a ``.phots`` file. If file, should be a ``.phots`` file. By default None
        sort_stars : bool, optional
            wether to sort stars by luminosity on loading, by default True
        """
        # Photometry
        self.fluxes = None
        self._time = None

        # Differential photometry
        self.lcs = None
        self.artificial_lcs = None
        self._comparison_stars = None  # ndarray

        # Files
        self.phot_file = None
        self.stack_fits = None
        self.folder = None

        # Observation info
        self.observation_date = None
        self.n_images = None
        self.filter = None
        self.exposure = None
        self.data = {}  # dict
        self.all_data = {}
        self.telescope = None  # Telescope object
        self.stars = None  # in pixels
        self.target = {"id": None,
                       "name": None,
                       "radec": [None, None]} # ra dec are loaded as written in stack FITS header

        # Convenience
        self.bjd_tdb = None
        self.gaia_data = None
        self.wcs = None
        self.hdu = None

        if folder_or_file is not None:
            isdir = self._check_folder_or_file(folder_or_file)
            if isdir:
                self.folder = folder_or_file
                self.load_folder(folder_or_file, sort_stars=sort_stars, keyword=keyword)
            else:
                self.phot_file = folder_or_file
                self.load_phot(folder_or_file)

    # Properties
    # ----------
    @property
    def time(self):
        if self.bjd_tdb is not None:
            return self.bjd_tdb
        else:
            warnings.warn("Time is JD-UTC")
            return self._time

    @property
    def lc(self):
        self._check_lcs()
        return self.lcs[self.target["id"]]

    @property
    def comparison_stars(self):
        if self._comparison_stars is not None:
            return self._comparison_stars[self.lcs.best_aperture_id]

    @property
    def target_id(self):
        return self.target["id"]

    @target_id.setter
    def target_id(self, id):
        self.target["id"] = id

    # Loaders and savers (files and data)
    # ------------------------
    def _check_folder_or_file(self, folder_or_file):
        """
        Check if provided path is a phots file or folder, else raise error

        Parameters
        ----------
        folder_or_file : str path
            path of file or folder

        Returns
        -------
        bool
            True if folder, False if path
        """
        if path.isfile(folder_or_file):
            if "phot" in path.splitext(folder_or_file)[1]:
                return False
            else:
                raise ValueError("File should be a .phots file")
        elif path.isdir(folder_or_file):
            return True
        else:
            raise NotADirectoryError("path doesn't exist")

    def load_folder(self, folder_path, sort_stars=False, keyword=None):
        """
        Load data from folder containing at least a phots file

        Parameters
        ----------
        folder : [type]
            [description]
        """
        if not path.isdir(folder_path):
            raise FileNotFoundError("Folder does not exist")

        # Loading unique phot file
        phot_files = io.get_files(".phot*", folder_path, single_list_removal=False)
        if len(phot_files) > 1:
            if keyword is not None:
                assert isinstance(keyword, str), "kwargs 'keyword' should be a string"
                for phot_file in phot_files:
                    if keyword in phot_file:
                        self.phot_file = phot_file
                if self.phot_file is None:
                    raise ValueError("Cannot find a phot file matching keyword '{}'".format(keyword))
            else:
                raise ValueError(
                    "Several phot files present in folder, should contain one (or specify kwarg 'keyword')")
        elif len(phot_files) == 0:
            raise ValueError("Cannot find a phot file in this folder, should contain one")
        else:
            self.phot_file = phot_files[0]

        # Loading unique stack file
        stack_fits = io.get_files("_stack.f*ts", folder_path, single_list_removal=False)
        if len(stack_fits) > 1:
            raise ValueError("Several stack files present in folder, should contain one")
        elif len(stack_fits) == 1:
            self.stack_fits = stack_fits[0]

        self.gif = io.get_files("gif", folder_path, none_for_empty=True)
        self.load_phot(self.phot_file, sort_stars=sort_stars)
        if self.stack_fits is not None:
            self.wcs = WCS(self.stack_fits)

    def load_phot(self, phots_path, sort_stars=False):
        self.load_phot_fits(phots_path, sort_stars=sort_stars)

    def _data_as_attributes(self):
        """
        Load the self.data dict as self attributes
        """
        if isinstance(self.data, pd.DataFrame):
            data = self.data.to_dict(orient="list")
        else:
            data = self.data

        self.__dict__.update(data)

    def save(self, destination=None):
        """
        Save data into a ``.phots`` file at specified destination. File name is : ``Telescope_date(YYYYmmdd)_target_filter``. For more info check :doc:`/notes/phots-structure`

        Parameters
        ----------
        destination : str path, optional
            path of destination where to save file, by default None
        """
        self.save_phot_fits(destination=destination)

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

    # Convenience
    # -----------
    @property
    def skycoord(self):
        ra, dec = self.target["radec"]
        return SkyCoord(ra, dec, frame='icrs', unit=(self.telescope.ra_unit, self.telescope.dec_unit))

    @property
    def simbad_url(self):
        """
        [notebook feature] clickable simbad query url for specified ``target_id``
        """
        from IPython.core.display import display, HTML

        display(HTML('<a href="{}">{}</a>'.format(self.simbad, self.simbad)))

    @property
    def simbad(self):
        """
        simbad query url for specified ``target_id``
        """
        ra, dec = self.target["radec"]
        return "http://simbad.u-strasbg.fr/simbad/sim-coo?Coord={}+{}&CooFrame=FK5&CooEpoch=2000&CooEqui=" \
               "2000&CooDefinedFrames=none&Radius=2&Radius.unit=arcmin&submit=submit+query&CoordList=".format(
            ra, dec)

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
    def _check_lcs(self):
        if self.lcs is None:
            raise ValueError("No differential light curve")
        else:
            pass

    def _compute_bjd(self):
        assert self.telescope is not None
        assert self.skycoord is not None
        time = Time(self._time, format="jd", scale="utc", location=self.telescope.earth_location)
        light_travel_tbd = time.light_travel_time(self.skycoord, location=self.telescope.earth_location)
        self.bjd_tdb = (time + light_travel_tbd).value
        self.data["bjd tdb"] = self.bjd_tdb

    def Broeg2005(self, **kwargs):
        """
        Differential photometry using the `Broeg (2005) <https://ui.adsabs.harvard.edu/abs/2005AN....326..134B/abstract>`_ algorithm

        Parameters
        ----------
        keep: None, int, float, string or None (optional, default is 'float')
            - if ``None``: use a weighted artificial comparison star based on all stars (weighted mean)
            - if ``float``: use a weighted artificial comparison star based on `keep` stars (weighted mean)
            - if ``int``: use a simple artificial comparison star based on `keep` stars (mean)
            - if ``'float'``: use a weighted artificial comparison star based on an optimal number of stars (weighted mean)
            - if ``'int'``: use a simple artificial comparison star based on an optimal number of stars (mean)
        max_iteration: int (optional, default is 50)
            maximum number of iteration to adjust weights
        tolerance: float (optional, default is 1e-8)
            mean difference between weights to reach
        n_comps: int (optional, default is 500)
            limit on the number of stars to keep (see keep kwargs)
        show_comps: bool (optional, default is False)
            show stars and weights used to build the artificial comparison star

        """
        assert self.target["id"] is not None, "target id is not defined"

        fluxes, fluxes_errors = self.fluxes.as_array()

        lcs, lcs_errors, art_lcs, comps = Broeg2005(
            fluxes, fluxes_errors, self.target_id, return_art_lc=True, return_comps=True, **kwargs
        )
        self.artificial_lcs = np.array(art_lcs)
        self.lcs = LightCurves(self.time, np.moveaxis(lcs, 0, 1), np.moveaxis(lcs_errors, 0, 1))
        best_aperture_id = self.lcs[self.target["id"]].best_aperture_id
        self.lcs.set_best_aperture_id(best_aperture_id)
        self._comparison_stars = np.array(comps)

    def DiffPhot(self, comps):
        """
        Manual differential photometry

        Parameters
        ----------
        comps : list
            list of detected stars ids to be used as comparison stars
        """
        fluxes, fluxes_errors = self.fluxes.as_array()
        lcs, lcs_errors, art_lcs = differential_photometry(fluxes, fluxes_errors, comps, return_art_lc=True)
        self.artificial_lcs = np.array(art_lcs)
        self.lcs = LightCurves(self.time, np.moveaxis(lcs, 0, 1), np.moveaxis(lcs_errors, 0, 1))
        best_aperture_id = self.lcs[self.target["id"]].best_aperture_id
        self.lcs.set_best_aperture_id(best_aperture_id)
        self._comparison_stars = np.array(comps)

    def set_target(self, target):
        """
        Setting target either by id or Gaia id

        Parameters
        ----------
        target : int or str
            if int, id among detected targets, else if str, Gaia id
        """
        pass

    def query_gaia(self, n_stars=1000):
        from astroquery.gaia import Gaia

        header = fits.getheader(self.stack_fits)
        shape = fits.getdata(self.stack_fits).shape
        cone_radius = np.sqrt(2) * np.max(shape) * self.telescope.pixel_scale / 120
        wcs = WCS(header)

        coord = self.skycoord
        radius = u.Quantity(cone_radius, u.arcminute)
        gaia_query = Gaia.cone_search_async(coord, radius, verbose=False, )
        self.gaia_data = gaia_query.get_results()

        skycoords = SkyCoord(
            ra=self.gaia_data['ra'],
            dec=self.gaia_data['dec'],
            pm_ra_cosdec=self.gaia_data['pmra'],
            pm_dec=self.gaia_data['pmdec'],
            radial_velocity=self.gaia_data['radial_velocity'])

        self.gaia_data["x"], self.gaia_data["y"] = np.array(wcsutils.skycoord_to_pixel(skycoords, self.wcs))

    # Plot
    # ----

    def show_stars(self, size=10, flip=False, view=None, zoom=True):
        """
        Show stack image and detected stars

        Parameters
        ----------
        size: float (optional)
            pyplot figure (size, size)
        n_stars: int
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
        if view is None:
            view = "reference" if self.comparison_stars is not None else "all"
        if view == "all":
            viz.fancy_show_stars(
                self.stack_fits, self.stars,
                flip=flip, size=size, target=self.target["id"],
                pixel_scale=self.telescope.pixel_scale)
        elif view == "reference":
            viz.fancy_show_stars(
                self.stack_fits, self.stars,
                ref_stars=self.comparison_stars, target=self.target["id"],
                flip=flip, size=size, view="reference", pixel_scale=self.telescope.pixel_scale, zoom=zoom)

    def show_gaia(self, color="lightblue", alpha=0.5, **kwargs):
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

        ax = plt.gcf().axes[0]
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.plot(self.gaia_data["x"], self.gaia_data["y"], "x", color=color, alpha=alpha, **kwargs)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

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
            star = self.target["id"]
        viz.show_stars(self.stack_fits, self.stars, highlight=star, size=6)
        ax = plt.gcf().axes[0]
        ax.set_xlim(np.array([-size / 2, size / 2]) + self.stars[star][0])
        ax.set_ylim(np.array([size / 2, -size / 2]) + self.stars[star][1])

    def plot_comps_lcs(self, n=5):
        """
        Plot comparison stars light curves along target star light curve

        Example
        -------
        .. image: /examples_images/plot_comps.png
           :align: center
        """
        self._check_lcs()
        idxs = [self.target["id"], *self.comparison_stars[0:n]]
        lcs = [self.lcs[i] for i in idxs]
        if len(plt.gcf().get_axes()) == 0:
            plt.figure(figsize=(5, 8))
        viz.plot_comparison_lcs(lcs, idxs)

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
        self.lc.plot()
        amp = (np.percentile(self.lc.flux, 95) - np.percentile(self.lc.flux, 5)) / 2
        plt.plot(self.lc.time, amp * utils.rescale(self.data[key]) + 1,
                 label="normalized {}".format(key),
                 color="k"
                 )
        plt.legend()

    def plot_psf_fit(self, size=21, cmap="inferno", c="blueviolet"):
        """
         Plot a 2D gaussian fit of the global psf (extracted from stack fits)

        Example
        -------
        .. image:: /guide/examples_images/plot_psf_fit.png
           :align: center
        """

        psf_fit = Moffat2D()
        image = Image(self.stack_fits, stars_coords=self.stars)
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
        self._check_lcs()
        viz.plot_rms(
            self.fluxes,
            self.lcs,
            bins=bins,
            target=self.target["id"],
            highlights=self.comparison_stars)

    def plot_systematics(self, fields=None):
        if fields is None:
            fields = ["dx", "dy", "fwhm", "airmass", "sky"]

        viz.plot_systematics(self.time, self.lc.flux, self.data, fields=[f for f in fields if f in self.data])

    def plot_raw_diff(self):
        viz.plot_lc_raw_diff(self)

    def save_report(self, destination=None, fields=None):
        if destination is None:
            destination = self.folder

        if fields is None:
            fields = ["dx", "dy", "fwhm", "airmass", "sky"]

        viz.save_report(self, destination, fields=fields)

    def plot_precision(self, bins=0.005, aperture=None):
        if aperture is None:
            assert self.lcs is not None, "You must set 'aperture' kwargs (light-curve not present to pick best aperture id)"
            aperture = self.lc.best_aperture_id

        n_bin = int(bins / (np.mean(self.exptime) / (60 * 60 * 24)))

        assert len(self.time) > n_bin, "Your 'bins' size is less than the total exposure"

        fluxes, errors = self.fluxes.as_array(aperture)

        mean_fluxes = np.mean(fluxes, axis=1)
        mean_errors = np.mean(errors, axis=1)

        error_estimate = [np.median(binned_statistic(self.time, f, statistic='std', bins=n_bin)[0]) for f in
                          fluxes]
        area = self.apertures_area[0][aperture]

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
                 label="flux rms ({:.1f} min bins)".format(0.005 * (60 * 24)))
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

    # Loaders and Savers implementations
    # ----------------------------------
    def save_phot_fits(self, destination=None):
        """
        Save data into a ``.phots`` file at specified destination. File name is : ``Telescope_date(YYYYmmdd)_target_filter``. For more info check :doc:`/notes/phots-structure`

        Parameters
        ----------
        destination : str path, optional
            path of destination where to save file, by default None
        """
        if destination is None:
            if self.folder is not None:
                destination = self.phot_file
            else:
                raise ValueError("destination must be specified")
        else:
            destination = path.join(
                destination, "{}.phots".format(self.products_denominator)
            )

        self.phot_file = destination

        header = fits.PrimaryHDU(header=fits.getheader(self.phot_file))

        header.header.update({
            "TARGETID": self.target["id"],
            "TELESCOP": self.telescope.name,
            "OBSERVAT": self.telescope.name,
            "FILTER": self.filter,
            "NIMAGES": self.n_images,
            "EXTEND": True
        })

        fluxes, fluxes_errors = self.fluxes.as_array()
        io.set_hdu(self.hdu, header)
        io.set_hdu(self.hdu, fits.ImageHDU(fluxes, name="photometry"))
        io.set_hdu(self.hdu, fits.ImageHDU(fluxes_errors, name="photometry errors"))
        io.set_hdu(self.hdu, fits.ImageHDU(self.stars, name="stars"))
        io.set_hdu(self.hdu, fits.BinTableHDU(Table.from_pandas(self.data), name="time series"))

        if self.lcs is not None:
            lcs, lcs_errors = self.lcs.as_array()
            io.set_hdu(self.hdu, fits.ImageHDU(lcs, name="lightcurves"))
            io.set_hdu(self.hdu, fits.ImageHDU(lcs_errors, name="lightcurves errors"))
            io.set_hdu(self.hdu, fits.ImageHDU(self._comparison_stars, name="comparison stars"))
            io.set_hdu(self.hdu, fits.ImageHDU(self.artificial_lcs, name="artificial lcs"))

        self.hdu.writeto(destination, overwrite=True)

    def load_phot_fits(self, phots_path, sort_stars=False):
        # DOING: check how time is laoded when loading and see if tdb tdb appears in data
        self.hdu = fits.open(phots_path)
        phot_dict = io.phot2dict(phots_path)

        header = phot_dict["header"]
        self.n_images = phot_dict.get("nimages", None)

        # Loading telescope, None if name doesn't match any
        telescope = Telescope()
        telescope_name = header.get(telescope.keyword_observatory, None)
        found = telescope.load(CONFIG.match_telescope_name(telescope_name))
        self.telescope = telescope if found else None
        if self.telescope is not None:
            ra = header.get(self.telescope.keyword_ra, None)
            dec = header.get(self.telescope.keyword_dec, None)
            self.target["radec"] = [ra, dec]

        # Loading info
        self.filter = header.get(self.telescope.keyword_filter, None)
        self.observation_date = utils.format_iso_date(
            header.get(self.telescope.keyword_observation_date, None))
        self.target["name"] = header.get(self.telescope.keyword_object, None)

        # Loading data
        self.data = Table(phot_dict.get("time series")).to_pandas()
        self._data_as_attributes()

        # Loading time and exposure (load data first to create self.data)
        self._time = Table(phot_dict.get("time series", None))["jd"]
        if self._time is not None:
            self._compute_bjd()
        else:
            raise ValueError("Could not find time JD within phot time series")

        self.exposure = header.get(self.telescope.keyword_exposure_time)
        if self.exposure is None:
            self.exposure = np.min(np.diff(self.time))
            warnings.warn("Exposure not found in headers, computed from time")

        # Loading fluxes and sort by flux if specified
        fluxes = phot_dict.get("photometry", None)
        assert fluxes is not None
        fluxes_error = phot_dict.get("photometry errors", None)
        star_mean_flux = np.mean(np.mean(fluxes, axis=0), axis=1)

        if sort_stars:
            sorted_stars = np.argsort(star_mean_flux)[::-1]
        else:
            sorted_stars = np.arange(0, np.shape(fluxes)[1])
        fluxes = fluxes[:, sorted_stars, :]
        if fluxes_error is not None: fluxes_error = fluxes_error[:, sorted_stars, :]

        # Loading stars, target, apertures
        self.stars = phot_dict.get("stars", None)[sorted_stars]
        self.apertures_area = phot_dict.get("apertures area", None)
        self.annulus_area = phot_dict.get("annulus area", None)
        self.annulus_sky = phot_dict.get("annulus sky", None)
        target_id = header.get("targetid", None)
        if target_id is not None:
            self.target["id"] = sorted_stars[target_id]

        # Loading light curves
        lcs = phot_dict.get("lightcurves", None)
        lcs_error = phot_dict.get("lightcurves errors", None)
        if lcs is not None: lcs = lcs[:, sorted_stars, :]
        if lcs_error is not None: lcs_error = lcs_error[:, sorted_stars, :]
        comparison_stars = phot_dict.get("comparison stars", None)
        self.artificial_lcs = phot_dict.get("artificial lcs", None)
        if comparison_stars is not None: self._comparison_stars = sorted_stars[comparison_stars]

        time = self.time
        a, s, f = fluxes.shape  # saved as (apertures, stars, fluxes) for conveniance

        # Photometry into LightCurve objects
        # Here is where we compute the fluxes errors
        if fluxes_error is None:
            print("Photometry error not present. TODO")
            fluxes_error = np.zeros_like(fluxes)
        self.fluxes = LightCurves(
            time, np.moveaxis(fluxes, 1, 0), np.moveaxis(fluxes_error, 1, 0))
        self.fluxes.apertures = self.apertures_area

        # Differential photometry into LightCurve objects
        if lcs is not None:
            self.lcs = LightCurves(time, np.moveaxis(lcs, 1, 0), np.moveaxis(lcs_error, 1, 0))
            best_aperture_id = self.lcs[self.target["id"]].best_aperture_id
            self.lcs.set_best_aperture_id(best_aperture_id)
            self.fluxes.set_best_aperture_id(best_aperture_id)

    # Reporting
    # ---------


