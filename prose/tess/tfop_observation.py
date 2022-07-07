from astroquery.mast import Catalogs
from astropy import units as u
from prose import Observation
import re
import pandas as pd
import numpy as np
from ..console_utils import info, error, warning
from ..blocks import catalogs

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
        return f"TIC{self.tic_id}_{self.stack.night_date}_{self.telescope.name}_{self.filter}"

    # Catalog queries
    # ---------------
    def query_tic(self, cone_radius=None):
        """Query TIC catalog (through MAST) for stars in the field
        """
        self.stack = catalogs.TESSCatalog(mode="crossmatch")(self.stack)

    def set_tic_target(self, verbose=True):

        # using Gaia is easier to correct for proper motion... (discuss with LG)
        self.set_gaia_target(self.gaia_from_toi, verbose=verbose)


