from prose import io
import numpy as np
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord
from prose.lightcurves import LightCurves, Broeg2005
import warnings
from prose import visualisation as viz


class Photometry:
    """
    Class to load and analyze photometry products
    """
    
    def __init__(self, folder=None, sort_stars=True):
        # Photometry
        self.apertures = None # ndarray
        self.light_curves = None
        self._time = None

        # Differential photometry
        self.differential_light_curves = None
        self.artificial_lc = None
        self.comparison_stars = None  # ndarray

        # Files
        self.phot_file = None
        self.stack_fits = None
        self.folder = folder

        # Observation info
        self.observation_date = None
        self.n_images = None
        self.filter = None
        self.exposure = None
        self.data = {}  # dict
        self.telescope = None  # Telescope object
        self.stars_coords = None  # in pixels
        self.target = {"id": None,
                       "name": None,
                       "radec": [None, None]}

        # Convenience
        self.bjd_tdb = None

        if folder is not None:
            self.load_folder(folder, sort_stars=sort_stars)

    # Properties
    # ----------
    @property
    def target_flux(self):
        return self.differential_light_curves[self.target["i"]].flux

    @property
    def target_error(self):
        return self.differential_light_curves[self.target["i"]].error
        
    @property
    def time(self):
        if self.bjd_tdb is not None:
            return self.bjd_tdb
        else:
            warnings.warn("Time is JD-UTC")
            return self._time
    
    @property
    def lc(self):
        return self.differential_light_curves[self.target["id"]]

    # Loaders and savers (files and data)
    # ------------------------
    
    def load_folder(self, folder_path, sort_stars=True):
        """
        Load data from folder containing at least a phots file
        
        Parameters
        ----------
        folder : [type]
            [description]
        """
        self.phot_file = io.get_files("phot*", folder_path, none_for_empty=True)
        self.stack_fits = io.get_files("_stack.f*ts", folder_path, none_for_empty=True)
        self.gif = io.get_files("gif", folder_path, none_for_empty=True)

        if self.phot_file is not None:
            self.load_phot(self.phot_file, sort_stars=sort_stars)

    def load_phot(self, phots_path, sort_stars=True):
        io.load_phot_fits(self, phots_path, sort_stars=True)

    def _data_as_attributes(self):
        """
        Load the self.data dict as self attributes
        """
        self.__dict__.update(self.data)

    def save(self, destination=None):
        """
        Save data into a ``.phots`` file at specified destination. File name is : ``Telescope_date(YYYYmmdd)_target_filter``. For more info check :doc:`/notes/phots-structure`
        
        Parameters
        ----------
        destination : str path, optional
            path of destination where to save file, by default None
        """
        io.save_phot_fits(self, destination=destination)

    # Convenience
    # -----------
    @property
    def skycoord(self):
        ra, dec = self.target["radec"]
        return SkyCoord(ra, dec, frame='icrs', unit=(u.hourangle, u.deg))

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
    def _compute_bjd(self):
        assert self.telescope is not None
        assert self.skycoord is not None
        time = Time(self._time, format="jd", scale="utc", location=self.telescope.earth_location)
        light_travel_tbd = time.light_travel_time(self.skycoord, location=self.telescope.earth_location)
        self.bjd_tdb = (time + light_travel_tbd).value

    def error(self, signal, npix, scinfac=0.09, method="scinti"):
        assert self.data.get("sky", None) is not None, "sky not found to compute flux error"

        _signal = signal.copy()
        _squarred_error = _signal + npix * (
            self.data["sky"] + self.telescope.read_noise ** 2 + (self.telescope.gain / 2) ** 2
        )

        if method == "scinti":
            assert self.data.get("airmass", None) is not None, "airmass not found to compute flux error"
            scintillation = (
                scinfac
                * np.power(self.telescope.diameter, -0.6666)
                * np.power(self.data["airmass"], 1.75)
                * np.exp(-self.telescope.altitude / 8000.0)
            ) / np.sqrt(2 * self.exposure)

            _squarred_error += np.power(signal * scintillation, 2)

        return np.sqrt(_squarred_error)
    
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

        fluxes, fluxes_errors = self.light_curves.as_array()

        lcs, lcs_errors, art_lcs, comps = Broeg2005(
            fluxes, fluxes_errors, self.target["id"], return_art_lc=True, return_comps=True, **kwargs
        )

        self.artificial_lc = np.array(art_lcs)
        self.differential_light_curves = LightCurves(self.time, np.moveaxis(lcs,0,1), np.moveaxis(lcs_errors,0,1))
        best_aperture_id = self.differential_light_curves[self.target["id"]]._best_aperture_id
        self.differential_light_curves.set_best_aperture_id(best_aperture_id)
        self.comparison_stars = np.array(comps)

    # Plot
    # ----

    def plot_stars(self):
        viz.plot_stars(self.stack_fits, self.stars_coords)


    

