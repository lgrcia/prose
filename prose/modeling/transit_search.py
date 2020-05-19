from prose import utils
import numpy as np
from functools import reduce
from itertools import combinations, product
import george
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from collections import OrderedDict
from scipy import interpolate
from scipy.interpolate import interp1d
from prose.modeling.models import *

# TODO: convert to model
def protopapas2005(t, t0, duration, depth, c, period=1):
    _t = period * np.sin(np.pi * (t - t0) / period) / (np.pi * duration)
    return (1 - depth) + (depth / 2) * (
        2 - np.tanh(c * (_t + 1 / 2)) + np.tanh(c * (_t - 1 / 2))
    )

class LinearTransitFinder:
    def __init__(
        self,
        phot,
        flux=None,
        error=None,
        data=None,
        orders=None,
        n=2,
        optimize=True,
        bins=0.005,
        c=20,
        n_transits=1
    ):
        
        self.c = c
        self.n_transits = n_transits
        
        if orders is None:
            self.orders = OrderedDict({
                "fwhm": n,
                "sky": n,
                "dx": n,
                "dy": n,
                "airmass": n
            })
            
        else:
            self.orders = OrderedDict(orders)

        if not isinstance(phot, (list, np.ndarray)):
            assert phot.lc is not None, "Lightcurve is missing"
            self.original_flux = phot.flux
            self.original_time = phot.jd
            self.original_original_data = OrderedDict(phot.data[fields].to_dict(orient="list"))
            self.original_error = phot.lc_error
        else:
            self.original_flux = flux
            self.original_time = phot
            self.original_original_data = OrderedDict(data)
            self.original_error = error
            
        if bins is not None:
            self.original_data = OrderedDict({
                key: utils.binning(np.array(self.original_time), np.array(value), bins)[1]
                for key, value in self.original_original_data.items()})
            self.time, self.flux, self.error = utils.binning(self.original_time, self.original_flux, bins, error=self.original_error, std=False)
        else:
            self.flux = self.original_flux  
            self.time = self.original_time
            self.original_data = self.original_original_data
            self.error = self.original_error
        
        self.rescaled_data = OrderedDict(
            {field: utils.rescale(dat) for field, dat in self.original_data.items()}
        )
        
        self.current_mean = self.flux
        
        self.durations = None
        self.stime = None
        self.lls = None
        self.depths = None
        
        self.transits = []
        
        if optimize:
            self.optimize()

    def X(self,include_transit=True):
        X = [np.ones(len(self.time))]
        for field, order in self.orders.items():
            for o in range(1, order + 1):
                X.append(np.power(self.rescaled_data[field].astype("float"), o))

        if include_transit:
            X.append(np.ones(len(self.time)))

        return np.array(X).transpose()

    def svd(self, X, y=None):
        if y is None:
            y = self.current_mean
        U, S, V = np.linalg.svd(X, full_matrices=False)
        S = np.diag(S)
        S[S == 0] = 1.0e10
        c = reduce(np.dot,[V.T,(1./S)*(1./S),V])
        return reduce(np.dot, [U.T, y.T, 1.0 / S, V])
    
    def ll(self, x, transit, min_depth=0):
        T0, period, depth, duration = transit
        x[:, -1] = protopapas2005(self.time, T0, duration, depth, self.c, period=period)-1
        A = self.svd(x)
        m, _depth = np.dot(A, x.transpose()), A[-1]
        if _depth > min_depth:
            return np.sum(np.log(1/(2*np.pi*self.error**2)) - (1/(2*self.error))*np.power(self.current_mean-m, 2)), _depth
        else:
            return 0, 0
    
    def optimize(self, bins=1, n_durations=20):
        dt = np.mean(np.diff(self.time))*bins
        self.durations = np.linspace(20, 100, n_durations)/60/24
        self.stime = np.arange(self.time.min()-np.max(self.durations)*0.7, self.time.max()+np.max(self.durations)*0.7, dt)
        min_depth = np.median(self.error)*0.75
        
        for _ in range(self.n_transits):
            if len(self.transits) == 0:
                transits = np.zeros_like(self.time)
            else:
                transits = np.sum([protopapas2005(self.time, *params, self.c, period=1000)-1 for params in self.transits], axis=None if len(self.transits)==0 else 0)
            
            self.current_mean = self.flux - transits
            self.lls = np.zeros((len(self.stime), len(self.durations)))
            self.depths = np.zeros((len(self.stime), len(self.durations)))

            x = self.X()
            for i, t in enumerate(self.stime):
                for j, duration in enumerate(self.durations):
                    self.lls[i,j], self.depths[i, j] = self.ll(x, [t, 2000, 1, duration], min_depth=min_depth)
                    
            self.transits.append(self.best_params())
                
    def best_params(self):
        max_i, max_j = np.unravel_index(np.argmax(self.lls, axis=None), self.lls.shape)
        return self.stime[max_i], self.durations[max_j], self.depths[max_i, max_j]
    
    def search_params(self):
        max_i, max_j = np.unravel_index(np.argmax(self.lls, axis=None), self.lls.shape)
        return self.stime[max_i], self.durations[max_j], self.depths[max_i, max_j], self.stime[max_i]-self.time.min(), self.time.max()-self.stime[max_i]
    
    def plot_model(self, axe=None):
        transits = np.sum([protopapas2005(self.time, *params, self.c, period=1000)-1 for params in self.transits], axis=None if len(self.transits)==0 else 0)
        x = self.X(False)
        A = self.svd(x, y=self.flux-transits)
        if axe is None:
            axe = plt.subplot(111)
        axe.plot(self.original_time, self.original_flux, ".", c="gainsboro", zorder=0, label="raw data")
        axe.plot(self.time, np.dot(A, x.transpose()) + transits, c="k", zorder=1, label=" transit + systematics)")
        for (T0, duration, depth) in self.transits:
            axe.plot(self.time, protopapas2005(self.time, T0, duration, depth, self.c, period=1000), "--", c="k", zorder=2, label="transit")
            
    def get_model(self):
        transits = np.sum([protopapas2005(self.time, *params, self.c, period=1000)-1 for params in self.transits], axis=None if len(self.transits)==0 else 0)
        x = self.X(False)
        A = self.svd(x, y=self.flux-transits)
        return np.dot(A, x.transpose())


class DifferentialEvolutionSearch():
    def __init__(
        self,
        phot,
        flux=None,
        error=None,
        data=None,
        orders=None,
        n=2,
        optimize=True,
        bins=None,
        c=20,
        n_transits=1
    ):
        
        if orders is None:
            self.orders = OrderedDict({
                "fwhm": n,
                "sky": n,
                "dx": n,
                "dy": n,
                "airmass": n
            })
            
        else:
            self.orders = OrderedDict(orders)

        if not isinstance(phot, (list, np.ndarray)):
            assert phot.lc is not None, "Lightcurve is missing"
            self.original_flux = phot.flux
            self.original_time = phot.jd
            self.original_original_data = OrderedDict(phot.data[fields].to_dict(orient="list"))
            self.original_error = phot.lc_error
        else:
            self.original_flux = flux
            self.original_time = phot
            self.original_original_data = OrderedDict(data)
            self.original_error = error
            
        if bins is not None:
            self.original_data = OrderedDict({
                key: utils.binning(np.array(self.original_time), np.array(value), bins)[1]
                for key, value in self.original_original_data.items()})
            self.time, self.flux, self.error = utils.binning(self.original_time, self.original_flux, bins, error=self.original_error, std=False)
        else:
            self.flux = self.original_flux  
            self.time = self.original_time
            self.original_data = self.original_original_data
            self.error = self.original_error
        
        self.rescaled_data = OrderedDict(
            {field: utils.rescale(dat) for field, dat in self.original_data.items()}
        )

        self.optimized_transit = None
        self.x = self.X()
        self.U, self.Sinv, self.V = None, None, None
        self.build_svd()
        
        if optimize:
            self.optimize()
            
    def X(self):
        X = [np.ones_like(self.time)]
        for field, order in self.orders.items():
            for o in range(1, order + 1):
                X.append(np.power(self.rescaled_data[field].astype("float"), o))
        return np.array(X).transpose()

    def build_svd(self):
        U, S, V = np.linalg.svd(self.x, full_matrices=False)
        S = np.diag(S)
        S[S == 0] = 1.0e10
        self.U = U
        self.Sinv = 1.0 / S
        self.V = V
        
    def optim(self, y):
        return np.dot(reduce(np.dot, [self.U.T, y.T, self.Sinv, self.V]), self.x.T)

    def model(self, p):
        return protopapas2005(self.time, *p, 20)

    def nll(self, p):
        m = self.model(p)
        return - np.sum(np.log(1/(2*np.pi*self.error**2)) - np.power(self.flux-self.optim(m),2)/(2*self.error)**2)
    
    def optimize(self):
        amp_flux = np.percentile(self.flux, 95) - np.percentile(self.flux, 5)

        s = differential_evolution(self.nll, [
                (self.time.min()-120/60/40, self.time.max()+120/60/40),
                (20/60/24, 120/60/24),
                (0, 1.5*amp_flux),
                (10, 50)
        ], workers=-1, tol=1e-3)
        self.optimized_transit = s.x

    def get_model(self, time=None, transit=True):
        m = self.optim(self.flux-self.model(self.optimized_transit)) + (self.model(self.optimized_transit) if transit else 0)
        if time is None:
            return m
        else:
            return interp1d(self.time, m, fill_value='extrapolate')(time)
        
    def get_transit_model(self, time=None):
        m = self.model(self.optimized_transit)
        if time is None:
            return m
        else:
            return interp1d(self.time, m, fill_value='extrapolate')(time)