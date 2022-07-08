from astroquery.mast import Catalogs
from astropy.coordinates import SkyCoord
from astropy.wcs import utils as wcsutils
from astropy import units as u
from prose import Observation
import re
import pandas as pd
import numpy as np
from prose.blocks.registration import distances


class TFOPObservation(Observation):
    """
    Subclass of Observation specific to TFOP observations
    """

    def __init__(self, photfile, name=None):
        """
        Parameters
        ----------
        photfile : str
            path of the `.phot` file to load
        """
        super().__init__(photfile)
        if name is None:
            name = self.name
        self.tic_data = None
        self.priors_dataframe = self.find_priors(name)

    # TESS specific methods
    # --------------------

    def find_priors(self, name):

        try:
            nb = re.findall('\d*\.?\d+', name) #TODO add the possibility to do this with TIC ID rather than TOI number (also in obs)
            priors_dataframe = pd.read_csv("https://exofop.ipac.caltech.edu/tess/download_toi?toi=%s&output=csv" % nb[0])
        except KeyError:
            print('TOI not found')
        return priors_dataframe

    @property
    def tic_id(self):
        """TIC id from digits found in target name
        """
        tic = self.priors_dataframe["TIC ID"][0]
        return f"{tic}"

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
        if any(["TIC" in self.name, "TOI" in self.name]):
            try:
                return f"TIC{self.tic_id}-{self.name.split('.')[1]}_{self.date.date()}_{self.telescope.name}_{self.filter}"
            except IndexError:
                return f"TIC{self.tic_id}-01_{self.date.date()}_{self.telescope.name}_{self.filter}"

    # Catalog queries
    # ---------------

    # TODO replace using stack Image and create TIC catalog block
    def query_tic(self, cone_radius=None):
        """Query TIC catalog (through MAST) for stars in the field
        """
        header = self.xarray.attrs
        shape = self.stack.shape
        if cone_radius is None:
            cone_radius = np.sqrt(2) * np.max(shape) * self.telescope.pixel_scale / 120

        coord = self.stack.skycoord
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

    def set_tic_target(self):

        self.query_tic()
        try:
            # getting all TICs
            tics = self.tic_data["ID"].data
            tics.fill_value = 0
            tics = tics.data.astype(int)

            # Finding the one
            i = np.argwhere(tics == np.int64(self.tic_id)).flatten()
            if len(i) == 0:
                raise AssertionError(f"TIC {self.tic_id} not found")
            else:
                i = i[0]
            row = self.tic_data[i]

            # setting the closest to target
            self.target = np.argmin(distances(self.stars.T, [row['x'], row['y']]))

        except KeyError:
            print('TIC ID not found')


