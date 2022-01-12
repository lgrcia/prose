from . import Image
import matplotlib.pyplot as plt
import numpy as np
import re
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord
from .fluxes import ApertureFluxes
from . import viz
from astropy.io import fits
from .telescope import Telescope
from . import utils
from astroquery.mast import Catalogs
from astropy.wcs import WCS, utils as wcsutils
import pandas as pd
from scipy.stats import binned_statistic
from .blocks.psf import Gaussian2D, Moffat2D,cutouts
from .console_utils import INFO_LABEL
from astropy.stats import sigma_clipped_stats
from astropy.io.fits.verify import VerifyWarning
from datetime import datetime
import warnings
from .blocks.registration import distances
import requests
import shutil
from pathlib import Path
from . import twirl
import io
from .utils import fast_binning, z_scale
from .console_utils import info

warnings.simplefilter('ignore', category=VerifyWarning)


class Observation(ApertureFluxes):
    """
    Class to load and analyze photometry products

    Parameters
    ----------
    photfile : str
        path of the `.phot` file to load

    """

    def __init__(self, photfile, ignore_time=False):
        super().__init__(photfile)

        utils.remove_sip(self.xarray.attrs)

        self.phot = photfile
        self.telescope = Telescope.from_name(self.telescope)

        self.gaia_data = None
        self.tic_data = None
        self.wcs = WCS(utils.remove_arrays(self.xarray.attrs))
        self._meridian_flip = None

        has_bjd = hasattr(self.xarray, "bjd_tdb")
        if has_bjd:
            has_bjd = ~np.all(self.xarray.bjd_tdb.isnull().values)

        if not has_bjd:
            try:
                self.compute_bjd()
                if not ignore_time:
                    print(f"{INFO_LABEL} Time converted to BJD TDB")
            except:
                if not ignore_time:
                    print(f"{INFO_LABEL} Could not convert time to BJD TDB")

    def _check_stack(self):
        assert 'stack' in self.xarray is not None, "No stack found"

    # Loaders and savers (files and data)
    # ------------------------------------
    def __copy__(self):
        copied = Observation(self.xarray.copy(), ignore_time=True)
        copied.phot = self.phot
        copied.telescope = self.telescope
        copied.gaia_data = self.gaia_data
        copied.tic_data = self.tic_data
        copied.wcs = self.wcs

        return copied

    def copy(self):
        return self.__copy__()

    def to_csv(self, destination, sep=" "):
        """Export a typical csv of the observation's data

        Parameters
        ----------

        destination : str
            Path of the csv file to save
        sep : str, optional
            separation string within csv, by default " "
        """
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

    def save(self, destination=None):
        """Save current observation

        Parameters
        ----------
        destination : str, optional
            path to phot file, by default None
        """
        self.xarray.to_netcdf(self.phot if destination is None else destination)
        info(f"saved {self.phot}")

    def export_stack(self, destination, **kwargs):
        """Export stack to FITS file

        Parameters
        ----------
        destination : str
            path of FITS to export
        """
        header = {name: value for name, value in self.xarray.attrs.items() if name.isupper()}
        data = self.stack

        hdul = fits.HDUList([fits.PrimaryHDU(data=data, header=fits.Header(header))])
        hdul.writeto(destination, **kwargs)

    def import_stack(self, fitsfile):
        """Import FITS as stack to current obs (including WCS) - do not forget to save to keep it

        Parameters
        ----------
        fitsfile : str
            path of FITS stack to import
        """
        data = fits.getdata(fitsfile)
        header = fits.getheader(fitsfile)

        self.wcs = WCS(header)
        self.xarray.attrs.update(utils.header_to_cdf4_dict(header))
        self.xarray["stack"] = (('w', 'h'), data)

    # Convenience
    # -----------

    @property
    def skycoord(self):
        """astropy SkyCoord object for the target
        """
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
    def denominator(self):
        """A conveniant name for the observation: {telescope}_{date}_{name}_{filter}

        Returns
        -------
        [type]
            [description]
        """
        return f"{self.telescope.name}_{self.date}_{self.name}_{self.filter}"

    @property
    def meridian_flip(self):
        """Meridian flip time. Supposing EAST and WEST encode orientation
        """
        if self._meridian_flip is not None:
            return self._meridian_flip
        else:
            has_flip = hasattr(self.xarray, "flip")
            if has_flip:
                try:
                    np.all(np.isnan(self.flip))
                    return None
                except TypeError:
                    pass

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

    # TESS specific methods
    # --------------------

    @property
    def tic_id(self):
        """TIC id from digits found in target name
        """
        try:
            nb = re.findall('\d*\.?\d+', self.name)
            df = pd.read_csv("https://exofop.ipac.caltech.edu/tess/download_toi?toi=%s&output=csv" % nb[0])
            tic = df["TIC ID"][0]
            return f"{tic}"
        except KeyError:
            print('TIC ID not found')
            return None

    @property
    def gaia_from_toi(self):
        """Gaia id from TOI id if TOI is in target name
        """
        if self.tic_id is not None:
            tic_id = ("TIC " + self.tic_id)
            catalog_data = Catalogs.query_object(tic_id, radius=.001, catalog="TIC")
            return f"{catalog_data['GAIA'][0]}"
        else:
            return None

    @property
    def tfop_prefix(self):
        return f"TIC{self.tic_id}_{self.date}_{self.telescope.name}_{self.filter}"

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
        assert self.skycoord is not None

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
            light_travel_tbd = time.light_travel_time(self.skycoord, location=self.telescope.earth_location)
            bjd_time = (time + light_travel_tbd).value

        elif version == "eastman":
            bjd_time = utils.jd_to_bjd(self.jd_utc + exposure_days/2, self.skycoord.ra.deg, self.skycoord.dec.deg)

        self.xarray = self.xarray.assign_coords(time=bjd_time)
        self.xarray["bjd_tdb"] = ("time", bjd_time)
        self.xarray.attrs["time_format"] = "bjd_tdb"

        # Catalog queries
        # ---------------

    def query_gaia(self, limit=-1, cone_radius=None):
        """Query gaia catalog for stars in the field
        """
        from astroquery.gaia import Gaia

        Gaia.ROW_LIMIT = limit

        header = self.xarray.attrs
        shape = self.stack.shape
        if cone_radius is None:
            cone_radius = np.sqrt(2) * np.max(shape) * self.telescope.pixel_scale / 120

        coord = self.skycoord
        radius = u.Quantity(cone_radius, u.arcminute)
        gaia_query = Gaia.cone_search_async(coord, radius, verbose=False, )
        self.gaia_data = gaia_query.get_results()
        self.gaia_data.sort("phot_g_mean_flux", reverse=True)

        delta_years = (utils.datetime_to_years(datetime.strptime(self.date, "%Y%m%d")) - \
                    self.gaia_data["ref_epoch"].data.data) * u.year

        dra = delta_years * self.gaia_data["pmra"].to(u.deg / u.year)
        ddec = delta_years * self.gaia_data["pmdec"].to(u.deg / u.year)

        skycoords = SkyCoord(
            ra=self.gaia_data['ra'].quantity + dra,
            dec=self.gaia_data['dec'].quantity + ddec,
            pm_ra_cosdec=self.gaia_data['pmra'],
            pm_dec=self.gaia_data['pmdec'],
            radial_velocity=self.gaia_data['radial_velocity'],
            obstime=Time(2015.0, format='decimalyear'))

        gaias = np.array(wcsutils.skycoord_to_pixel(skycoords, self.wcs)).T
        gaias[np.any(np.isnan(gaias), 1), :] = [0, 0]
        self.gaia_data["x"], self.gaia_data["y"] = gaias.T
        inside = np.all((np.array([0, 0]) < gaias) & (gaias < np.array(self.stack.shape)), 1)
        self.gaia_data = self.gaia_data[np.argwhere(inside).squeeze()]
        
        w, h = self.stack.shape
        if np.abs(np.mean(self.gaia_data["x"])) > w or np.abs(np.mean(self.gaia_data["y"])) > h:
            warnings.warn("Catalog stars seem out of the field. Check that your stack is solved and that telescope "
                        "'ra_unit' and 'dec_unit' are well set")

    def query_tic(self,cone_radius=None):
        """Query TIC catalog (through MAST) for stars in the field
        """
        from astroquery.mast import Catalogs

        header = self.xarray.attrs
        shape = self.stack.shape
        if cone_radius is None:
            cone_radius = np.sqrt(2) * np.max(shape) * self.telescope.pixel_scale / 120

        coord = self.skycoord
        radius = u.Quantity(cone_radius, u.arcminute)
        self.tic_data = Catalogs.query_region(coord, radius, "TIC", verbose=False)
        self.tic_data.sort("Jmag")

        skycoords = SkyCoord(
            ra=self.tic_data['ra'],
            dec=self.tic_data['dec'], unit="deg")

        self.tic_data["x"], self.tic_data["y"] = np.array(wcsutils.skycoord_to_pixel(skycoords, self.wcs))

        w, h = self.stack.shape
        if np.abs(np.mean(self.tic_data["x"])) > w or np.abs(np.mean(self.tic_data["y"])) > h:
            warnings.warn("Catalog stars seem out of the field. Check that your stack is solved and that telescope "
                          "'ra_unit' and 'dec_unit' are well set")

    @property
    def gaia_target(self):
        return None

    @gaia_target.setter
    def gaia_target(self, gaia_id):
        """Set target with a gaia id

        Parameters
        ----------
        gaia_id : int
            gaia id
        """
        if self.gaia_data is None:
            self.query_gaia()
        _ = self.gaia_data.to_pandas()[["source_id", "x", "y"]].to_numpy()
        ids = _[:, 0]
        positions = _[:, 1:3]
        gaia_i = np.argmin(np.abs(gaia_id - ids))
        self.target = np.argmin(np.power(positions[gaia_i, :] - self.stars[:, ::-1], 2).sum(1))

    # Plot
    # ----

    def show(self, size=10, flip=False, zoom=False, contrast=0.05, wcs=False, cmap="Greys_r", sigclip=None,vmin=None,vmax=None):
        """Show stack image

        Parameters
        ----------
        size : int, optional
           size of the square figure, by default 10
        flip : bool, optional
            , by default False
        zoom : bool, optional
            whether to include a zoom inlay in the image, by default False
        contrast : float, optional
            contrast for the Zscale of image, by default 0.05
        wcs : bool, optional
            whether to show grid ans axes to world coordinate
        """
        if self.target == -1:
            zoom = False

        self._check_stack()

        fig = plt.figure(figsize=(size, size))
        fig.patch.set_facecolor('white')

        image = self.stack.copy()

        if flip:
            image = image[::-1, ::-1]

        if sigclip is not None:
            mean, median, std = sigma_clipped_stats(image)
            image[image - median < 2 * std] = median

        if wcs:
            ax = plt.subplot(projection=self.wcs, label='overlays')
        else:
            ax = fig.add_subplot(111)
        if all([vmin, vmax]) is False:
            _ = ax.imshow(utils.z_scale(image,c=contrast), cmap=cmap, origin="lower")
        else:
            _ = ax.imshow(image, cmap=cmap, origin="lower",vmin=vmin,vmax=vmax)

        if wcs:
            ax.coords.grid(True, color='white', ls='solid', alpha=0.3)
            ax.coords[0].set_axislabel('Galactic Longitude')
            ax.coords[1].set_axislabel('Galactic Latitude')

            overlay = ax.get_coords_overlay('fk5')
            overlay.grid(color='white', ls='--', alpha=0.3)
            overlay[0].set_axislabel('Right Ascension (J2000)')
            overlay[1].set_axislabel('Declination (J2000)')

    def _check_show(self, **kwargs):

        axes = plt.gcf().axes

        if len(axes) == 0:
            self.show(**kwargs)

    def show_stars(self, size=10, view=None, n=None, flip=False,
                   comp_color="yellow", color=[0.51, 0.86, 1.], stars=None, legend=True, **kwargs):
        """Show detected stars over stack image


        Parameters
        ----------
        size : int, optional
            size of the square figure, by default 10
        flip : bool, optional
            whether to flip image, by default False
        view : str, optional
            "all" to see all stars OR "reference" to have target and comparison stars hilighted, by default None
        n : int, optional
            max number of stars to show, by default None,

        Raises
        ------
        AssertionError
            [description]
        """

        self._check_show(flip=flip, size=size, **kwargs)

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
            viz.plot_marks(*stars.T, np.arange(len(stars)), color=color)

            if "stars" in self.xarray:
                others = np.arange(n, len(self.stars))
                others = np.setdiff1d(others, self.target)
                viz.plot_marks(*self.stars[others].T, alpha=0.4, color=color)

        elif view == "reference":
            x = self.xarray.isel(apertures=self.aperture)
            assert 'comps' in self, "No differential photometry"

            comps = x.comps.values

            others = np.setdiff1d(np.arange(len(stars)), x.comps.values)
            others = np.setdiff1d(others, self.target)

            _ = viz.plot_marks(*stars[self.target], self.target, color=color)
            _ = viz.plot_marks(*stars[comps].T, comps, color=comp_color)
            _ = viz.plot_marks(*stars[others].T, alpha=0.4, color=color)

            if legend:
                colors = [comp_color, color]
                texts = ["Comparison stars", "Target"]
                viz.circles_legend(colors, texts)

    def show_gaia(self, color="yellow", alpha=1, n=None, idxs=True, limit=-1, fontsize=8, align=False):
        """Overlay Gaia objects on stack image


        Parameters
        ----------
        color : str, optional
            color of marks and font, by default "lightblue"
        alpha : int, optional
            opacity of marks and font, by default 1
        n : int, optional
            max number of stars to show, by default None, by default None for all stars
        idxs : bool, optional
            wether to show gaia ids, by default True
        """

        self._check_show()

        if self.gaia_data is None:
            self.query_gaia(limit=limit)

        gaias = np.vstack([self.gaia_data["x"].data, self.gaia_data["y"].data]).T
        defined = ~np.any(np.isnan(gaias), 1)
        gaias = gaias[defined]
        labels = self.gaia_data["source_id"].data.astype(str)[defined]

        if align:
            X = twirl.find_transform(gaias[0:30], self.stars, n=15)
            gaias = twirl.affine_transform(X)(gaias)

        labels = [f"{_id[0:len(_id) // 2]}\n{_id[len(_id) // 2::]}" for _id in labels]
        _ = viz.plot_marks(*gaias.T, labels if idxs else None, color=color, alpha=alpha, n=n, position="top",
                           fontsize=fontsize)

    def show_tic(self, color="white", alpha=1, n=None, idxs=True, align=True):
        """Overlay TIC objects on stack image


        Parameters
        ----------
        color : str, optional
            color of marks and font, by default "lightblue"
        alpha : int, optional
            opacity of marks and font, by default 1
        n : int, optional
            max number of stars to show, by default None, by default None for all stars
        idxs : bool, optional
            wether to show TIC ids, by default True
        """

        self._check_show()

        if self.tic_data is None:
            self.query_tic()

        x = self.tic_data["x"].data
        y = self.tic_data["y"].data
        tics = np.vstack([x, y]).T
        ID = self.tic_data["ID"].data

        if align:
            X = twirl.find_transform(tics[0:30], self.stars, n=15)
            tics = twirl.affine_transform(X)(tics)

        _ = viz.plot_marks(*tics.T, ID if idxs else None, color=color, alpha=alpha, n=n, position="top", fontsize=9, offset=10)

    def show_cutout(self, star=None, size=200, marks=True,**kwargs):
        """
        Show a zoomed cutout around a detected star or coordinates

        Parameters
        ----------
        star : [type], optional
            detected star id or (x, y) coordinate, by default None
        size : int, optional
            side size of square cutout in pixel, by default 200
        """

        if star is None:
            x, y = self.stars[self.target]
        elif isinstance(star, int):
            x, y = self.stars[star]
        elif isinstance(star, (tuple, list, np.ndarray)):
            x, y = star
        else:
            raise ValueError("star type not understood")

        self.show(**kwargs)
        plt.xlim(np.array([-size / 2, size / 2]) + x)
        plt.ylim(np.array([-size / 2, size / 2]) + y)
        if marks:
            idxs = np.argwhere(np.max(np.abs(self.stars - [x, y]), axis=1) < size).squeeze()
            viz.plot_marks(*self.stars[idxs].T, label=idxs)

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

    def plot_psf_fit(self, size=21, cmap="inferno", c="blueviolet", model=Gaussian2D):
        """Plot a 2D gaussian fit of the global psf (extracted from stack fits)

        Parameters
        ----------
        size : int, optional
            square size of extracted PSF, by default 21
        cmap : str, optional
            color map of psf image, by default "inferno"
        c : str, optional
            color of model plot line, by default "blueviolet"
        model : prose.blocks, optional
            a PsfFit block, by default Gaussian2D

        Returns
        -------
        dict
            PSF fit info (theta, std_x, std_y, fwhm_x, fwhm_y)
        """

        psf_fit = model()
        image = Image(data=self.stack, stars_coords=self.stars, header=self.xarray.attrs)
        psf_fit.run(image)

        if len(plt.gcf().get_axes()) == 0:
            plt.figure(figsize=(12, 4))
        viz.plot_marginal_model(psf_fit.epsf, psf_fit.optimized_model, cmap=cmap, c=c)

        return {"theta": image.theta,
                "std_x": image.psf_sigma_x,
                "std_y": image.psf_sigma_y,
                "fwhm_x": image.fwhmx,
                "fwhm_y": image.fwhmy }

    def plot_star_psf(self,star=None,cutout_size=21,print_values=True,plot=True):
        if star is None:
            star = self.target
        cutout = cutouts(self.stack, [self.stars[star]], size=cutout_size)
        psf_fit = Moffat2D(cutout_size=cutout_size)
        params = ['fwhmx =', 'fwhmy =', 'theta =']
        values = []
        for i in range(len(params)):
            if print_values is True:
                print(params[i], psf_fit(cutout.data[0])[i])
            values.append(psf_fit(cutout.data[0])[i])
            if plot is True:
                viz.plot_marginal_model(psf_fit.epsf, psf_fit.optimized_model)
        return values

    def plot_rms(self, bins=0.005):
        """Plot binned rms of lightcurves vs the CCD equation

        Parameters
        ----------
        bins : float, optional
            bin size used to compute error, by default 0.005 (in days)
        """
        self._check_diff()
        viz.plot_rms(
            self.diff_fluxes,
            self.lcs,
            bins=bins,
            target=self.target["id"],
            highlights=self.comparison_stars)

    def plot_systematics(self, fields=None, ylim=(0.98, 1.02)):
        """Plot systematics measurements along target light curve

        Parameters
        ----------
        fields : list of str, optional
            list of systematic to include (must be in self), by default None
        ylim : tuple, optional
            plot ylim, by default (0.98, 1.02)
        """
        if fields is None:
            fields = ["dx", "dy", "fwhm", "airmass", "sky"]

        flux = self.diff_flux.copy()
        flux /= np.nanmean(flux)

        if ylim is None:
            ylim = (flux.nanmin() * 0.99, flux.nanmax() * 1.01)

        offset = ylim[1] - ylim[0]

        if len(plt.gcf().axes) == 0:
            plt.figure(figsize=(5 ,10))

        viz.plot(self.time, flux, bincolor="black")

        for i, field in enumerate(fields):
            if field in self:
                scaled_data = self.xarray[field].values.copy()
                scaled_data = np.nan_to_num(scaled_data, -1)
                scaled_data[scaled_data - np.nanmean(scaled_data) > 5*np.nanstd(scaled_data)] = -1
                scaled_data = scaled_data - np.median(scaled_data)
                scaled_data = scaled_data / np.std(scaled_data)
                scaled_data *= np.std(flux)
                scaled_data += 1 - (i + 1) * offset
                viz.plot(self.time, scaled_data, bincolor="grey")
                plt.annotate(field, (self.time.min() + 0.005, 1 - (i + 1) * offset + offset / 3))
            else:
                i -= 1

        plt.ylim(1 - (i + 1.5) * offset, ylim[1])
        plt.title("Systematics", loc="left")
        plt.grid(color="whitesmoke")
        plt.tight_layout()

    def plot_raw_diff(self):
        """Plot raw target flux and differantial flux 
        """

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

    def plot_precision(self, bins=0.005, aperture=None):
        """Plot observation precision estimate against theorethical error (background noise, photon noise and CCD equation)

        Parameters
        ----------
        bins : float, optional
            bin size used to estimate error, by default 0.005 (in days)
        aperture : int, optional
            chosen aperture, by default None
        """

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
        """Plot meridian flip line over existing axe
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

    def plot_psf(self, star=None, n=40, zscale=False, aperture=None, rin=None, rout=None):
        """Plot star cutout overalid with aperture and radial flux.

        Parameters
        ----------
        star : int or list like, optional
            if int: star to plot cutout on, if list like (tuple, np.ndarray) of size 2: coords of cutout, by default None
        n : int, optional
            cutout width and height, by default 40
        zscale : bool, optional
            whether to apply a zscale to cutout image, by default False
        aperture : float, optional
            radius of aperture to display, by default None corresponds to best target aperture
        rin : [type], optional
            radius of inner annulus to display, by default None corresponds to inner radius saved
        rout : [type], optional
            radius of outer annulus to display, by default None corresponds to outer radius saved
        """
        n /= np.sqrt(2)

        if isinstance(star, (tuple, list, np.ndarray)):
            x, y = star
        else:
            if star is None:
                star = self.target
            assert isinstance(star, int), "star must be star coordinates or integer index"
        
            x, y = self.stars[star]

        Y, X = np.indices(self.stack.shape)
        cutout_mask = (np.abs(X - x + 0.5) < n) & (np.abs(Y - y + 0.5) < n)
        inside = np.argwhere((cutout_mask).flatten()).flatten()
        radii = (np.sqrt((X - x) ** 2 + (Y - y) ** 2)).flatten()[inside]
        idxs = np.argsort(radii)
        radii = radii[idxs]
        pixels = self.stack.flatten()[inside]
        pixels = pixels[idxs]

        binned_radii, binned_pixels, _ = fast_binning(radii, pixels, bins=1)

        fig = plt.figure(figsize=(9.5, 4))
        fig.patch.set_facecolor('xkcd:white')
        _ = plt.subplot(1, 5, (1, 3))

        plt.plot(radii, pixels, "o", fillstyle='none', c="0.7", ms=4)
        plt.plot(binned_radii, binned_pixels, c="k")
        plt.xlabel("distance from center (pixels)")
        plt.ylabel("ADUs")
        _, ylim = plt.ylim()

        if "apertures_radii" in self and self.aperture != -1:
            apertures = self.apertures_radii[:, 0]
            aperture = apertures[self.aperture]
        
            if "annulus_rin" in self:
                if rin is None:
                    rin = self.annulus_rin.mean()
                if rout is None:
                    rout = self.annulus_rout.mean() 

        if aperture is not None:
            plt.xlim(0)
            plt.text(aperture, ylim, "APERTURE", ha="right", rotation="vertical", va="top")
            plt.axvline(aperture, c="k", alpha=0.1)
            plt.axvspan(0, aperture, color="0.9", alpha=0.1)

        if rin is not None:
            plt.axvline(rin, color="k", alpha=0.2)

        if rout is not None:
            plt.axvline(rout, color="k", alpha=0.2)
            if rin is not None:
                plt.axvspan(rin, rout, color="0.9", alpha=0.2)
                _ = plt.text(rout, ylim, "ANNULUS", ha="right", rotation="vertical", va="top")

        n = np.max([np.max(radii), rout +2 if rout else 0])
        plt.xlim(0, n)

        ax2 = plt.subplot(1, 5, (4, 5))
        im = self.stack[int(y - n):int(y + n), int(x - n):int(x + n)]
        if zscale:
            im = z_scale(im)

        plt.imshow(im, cmap="Greys_r", aspect="auto", origin="lower")

        plt.axis("off")
        if aperture is not None:
            ax2.add_patch(plt.Circle((n, n), aperture, ec='grey', fill=False, lw=2))
        if rin is not None:
            ax2.add_patch(plt.Circle((n, n), rin, ec='grey', fill=False, lw=2))
        if rout is not None:
            ax2.add_patch(plt.Circle((n, n), rout, ec='grey', fill=False, lw=2))
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
        self.xlabel()
        plt.ylabel("diff. flux")
        plt.tight_layout()
        viz.paper_style()

    def xlabel(self):
        """Plot xlabel (time) according to its units
        """
        plt.xlabel(self.time_format.upper().replace("_", "-"))

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

    def keep_good_stars(self, lower_threshold=3., upper_threshold=35000., trim=10, keep=None, inplace=True):
        """Keep only  stars with a median flux higher than `threshold`*sky. 
        
        This action will reorganize stars indexes (target id will be recomputed) and reset the differential fluxes to raw.

        Parameters
        ----------
        lower_threshold : float
            threshold for which stars with flux/sky > threshold are kept, default is 3
        trim : float
            value in pixels above which stars are kept, default is 10 to avoid stars too close to the edge
        keep : int or list
            number of stars to exclude (starting from 0 if int).
        inplace: bool
            whether to replace current object or return a new one
        """
        good_stars = np.argwhere((np.median(self.peaks, 1)/np.median(self.sky) > lower_threshold) & (np.median(self.peaks, 1) < upper_threshold)).squeeze()
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

    def plot_flip(self):
        plt.axvline(self.meridian_flip, ls="--", c="k", alpha=0.5)

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

    def set_tic_target(self):

        self.query_tic()
        try:
            # TOI to TIC
            toi = re.split("-|\.", self.name)[1]
            b = requests.get(f"https://exofop.ipac.caltech.edu/tess/download_toi?toi={toi}&output=csv").content
            TIC = pd.read_csv(io.BytesIO(b))["TIC ID"][0]

            # getting all TICs
            tics = self.tic_data["ID"].data
            tics.fill_value = 0
            tics = tics.data.astype(int)

            # Finding the one
            i = np.argwhere(tics == TIC).flatten()
            if len(i) == 0:
                raise AssertionError(f"TIC {TIC} not found")
            else:
                i = i[0]
            row = self.tic_data[i]

            # setting the closest to target
            self.target = np.argmin(distances(self.stars.T, [row['x'], row['y']]))

        except KeyError:
            print('TIC ID not found')

    def convert_flip(self,keyword):
        self.xarray['flip'] = ('time', (self.flip == keyword).astype(int))

    def folder_to_phot(self, confirm=True):
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

    def radec_to_pixel(self, radecs, unit=(u.deg, u.deg)):
        ra, dec = radecs.T
        skycoords = SkyCoord(ra=ra, dec=dec, unit=unit)
        return np.array(wcsutils.skycoord_to_pixel(skycoords, self.wcs)).T

    def plot_circle(self, center=None, arcmin=2.5):
        if center is None:
            x, y = self.stars[self.target]
        elif isinstance(center, int):
            x, y = self.stars[center]
        elif isinstance(center, (tuple, list, np.ndarray)):
            x, y = center
        else:
            raise ValueError("center type not understood")

        search_radius = 60 * arcmin / self.telescope.pixel_scale
        circle = plt.Circle((x, y), search_radius, fill=None, ec="white", alpha=0.6)

        ax = plt.gca()
        ax.add_artist(circle)
        plt.annotate(f"radius {arcmin}'", xy=[x, y + search_radius + 15], color="white",
                     ha='center', fontsize=12, va='bottom', alpha=0.6)

    def mask_transits(self, epoch, period, duration):
        xmin, xmax = self.time.min(), self.time.max()
        n_p = np.round((xmax - epoch) / period)
        ingress, egress = n_p * period + epoch - duration / 2, n_p * period + epoch + duration / 2
        mask = (self.time < ingress) | (self.time > egress)
        return self.mask(mask)
