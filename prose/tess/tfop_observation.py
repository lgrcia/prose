from astroquery.mast import Catalogs
from astropy import units as u
from prose import Observation
import re
import pandas as pd
import requests as req
import numpy as np
from ..console_utils import info, error, warning
from ..blocks import catalogs
from datetime import datetime

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
        self.exofop_priors = self.find_exofop_priors(name)
        self.ttf_priors = self.find_ttf_priors()
        self.toi = self.name.split('-')[1]
        try :
            self.planet = self.toi.split('.')[1]
        except IndexError:
            self.planet = '01'
        self.samples = None
        self.posteriors = None
        self.summary = None
        self.detrends = None

    # TESS specific methods
    # --------------------

    def find_exofop_priors(self, name):

        try:
            nb = re.findall('\d*\.?\d+', name) #TODO add the possibility to do this with TIC ID rather than TOI number (also in obs)
            return pd.read_csv("https://exofop.ipac.caltech.edu/tess/download_toi?toi=%s&output=csv" % nb[0])
        except KeyError:
            print('TOI not found')

    def find_ttf_priors(self):
        date = self.stack.night_date
        ttf_url = self.telescope.TTF_link.format(date=f'{date.strftime("%m-%d-%Y")}', tic_id=self.tic_id).replace(
            'print_html=1', 'print_html=2')
        r = req.get(ttf_url,
                    auth=('tess_nda_observer', 'F1nd_TE$S_PlaNets!'))
        ttf_priors = [{k:v for k,v in zip([l[:25]+l[-11:]
                                for l in [j.split(',')
                                for j in r.text[r.text.find('# ')+2:].splitlines()]][0], i)}
                            for i in [l[:25]+l[-11:]
                            for l in [j.split(',')
                            for j in r.text[r.text.find('# ')+2:].splitlines()]][1:]]
        for i, j in zip(ttf_priors, r.text.splitlines()[1:]):
            i['Comments'] = j[j.find('"') + 1:-j[::-1].find('"') - 1]

        ttf_priors[0]['Comments'] = ttf_priors[0]['Comments'].replace('σ', 'sigma')
        ttf_priors[0]['Comments'] = ttf_priors[0]['Comments'].replace('Δ', 'Delta')

        return ttf_priors[0]

    @property
    def tic_id(self):
        """TIC id from digits found in target name
        """
        tic = self.exofop_priors["TIC ID"][0]
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
        date = self.stack.night_date.strftime("%Y%m%d")
        return f"TIC{self.tic_id}-{self.planet}_{date}_{self.telescope.name}_{self.filter}"

    # Catalog queries
    # ---------------
    def query_tic(self, cone_radius=None):
        """Query TIC catalog (through MAST) for stars in the field
        """
        self.stack = catalogs.TESSCatalog(mode="crossmatch")(self.stack)

    def set_tic_target(self, verbose=True):

        # using Gaia is easier to correct for proper motion... (discuss with LG)
        self.set_gaia_target(self.gaia_from_toi, verbose=verbose)


    def auto_modeling(self,detrends=None,limb_darkening_coefs=None):

        import pymc3 as pm
        import exoplanet as xo
        import pymc3_ext as pmx
        from prose.utils import earth2sun
        from prose import viz
        import arviz as az
        self.detrends = detrends

        X = self.polynomial(**detrends).T
        c = np.linalg.lstsq(X.T, self.diff_flux)[0]

        with pm.Model() as model:
            # Systematics
            # -----------------
            w = pm.Flat('w', shape=len(X), testval=np.array(c))
            systematics = pm.Deterministic('systematics', w @ X)

            # Stellar parameters
            # -----------------
            u = xo.distributions.QuadLimbDark("u", testval=np.array(limb_darkening_coefs))
            star = xo.LimbDarkLightCurve(u[0], u[1])
            m_s = pm.Normal('m_s',self.exofop_priors["Stellar Mass (M_Sun)"].values[0],self.exofop_priors["Stellar Mass (M_Sun) err"].values[0])
            r_s = pm.Normal('r_s', self.exofop_priors["Stellar Radius (R_Sun)"].values[0],self.exofop_priors["Stellar Radius (R_Sun) err"].values[0])

            # Orbital parameters
            # -----------------
            t0 = pm.Normal('t0', 2450000 + float(self.ttf_priors['jd_mid']), 0.05)
            p = pm.Normal('P', self.exofop_priors['Period (days)'].values[0], self.exofop_priors["Period (days) err"].values[0])
            b = pm.Uniform("b", 0, 1)
            depth = pm.Uniform("depth", 0, self.exofop_priors['Depth (ppm)'].values[0]*2 *1e-6, testval=self.exofop_priors['Depth (ppm)'].values[0]*1e-6)
            ror = pm.Deterministic("ror", star.get_ror_from_approx_transit_depth(depth, b))
            r_p = pm.Deterministic("r_p", ror * r_s)  # In solar radius
            r = pm.Deterministic('r', r_p * 1 / earth2sun)

            # Keplerian orbit
            # ---------------
            orbit = xo.orbits.KeplerianOrbit(period=p, t0=t0, r_star=r_s, b=b, m_star=m_s)

            # starry light-curve
            # ---------------
            light_curves = star.get_light_curve(orbit=orbit, r=r_p, t=self.time)
            transit = pm.Deterministic("transit", pm.math.sum(light_curves, axis=-1))

            # Let's track some parameters :
            pm.Deterministic("a", orbit.a)
            pm.Deterministic('i', orbit.incl * 180 / np.pi)
            pm.Deterministic('a/r_s', orbit.a / orbit.r_star)

            # Systematics and final model
            # ---------------------------
            # residuals = pm.Deterministic("residuals", obs.diff_flux - transit)
            mu = pm.Deterministic("mu", transit + systematics)

            # Likelihood function
            # -----------------------------
            pm.Normal("obs", mu=mu, sd=self.diff_error, observed=self.diff_flux)

            # Maximum a posteriori
            # --------------------
            self.opt = pmx.optimize(start=model.test_point)

        viz.plot_systematics_signal(self.time, self.diff_flux, self.opt['systematics'], self.opt['transit'])
        viz.paper_style()

        np.random.seed(42)

        with model:
            trace = pm.sample(
                tune=2000,
                draws=3000,
                start=self.opt,
                cores=3,
                chains=2,
                init="adapt_full",
                target_accept=0.9,
                return_inferencedata=False
            )
        variables = ["P", "r", 't0', 'b', 'u', 'r_s', 'm_s', 'ror', 'depth', 'a', 'a/r_s', 'i']

        self.samples = pm.trace_to_dataframe(trace, varnames=variables)

        with model:
            self.summary = az.summary(
                trace, var_names=variables, round_to=4
            )
        # This should go in the results table of transitmodel
        #for i in self.summary.index:
        #    print(i, '=', self.summary['mean'][i], '+/-', self.summary['sd'][i])

        self.posteriors = {}
        for i in self.summary.index:
            self.posteriors[i] = self.summary['mean'][i]
            self.posteriors[i + '_e'] = self.summary['sd'][i]
