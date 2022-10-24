from .observation import Observation
from .fluxes import scargle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tabulate import tabulate
from tabulate import tabulate
from . import viz
from . import models
from .console_utils import info, progress


def phase_coverage(t, p):
    if p==0:
        return 1
    else:
        ph = ((t + 0.5*p)%p - (0.5*p))
        sph_in = np.sort(ph)
        sph_out = sph_in[::-1]

        sph_in_diff = np.abs(np.diff(sph_in))
        sph_out_diff = np.abs(np.diff(sph_out))

        df = np.min(np.diff(t))

        spaces_in = np.sort(sph_in[np.hstack([*np.argwhere(sph_in_diff > 4*df).T, len(sph_in)-1])])
        spaces_out = np.sort(sph_out[np.hstack([*np.argwhere(sph_out_diff > 4*df).T, len(sph_in)-1])])

        return np.sum(spaces_in - spaces_out)/p


def plot_slice(offset, width, color='#016CA050', R=1, alpha=0.5):
    n = plt.pie([width*100, 100-width*100], colors=[color, 'none'], counterclock=False, startangle=90-offset*360, radius=R)
    for wedge in n[0]:
        wedge.set_alpha(alpha)


class Observations:
    def __init__(self, files_or_obs, verbose=True):
        """Object to hold multiple observations

        Parameters
        ----------
        files_or_obs : list
            A list containing phot file paths or Observation object instances
        """

        def progress(x):
            return tqdm(x) if verbose else x

        self.observations = []
        for fo in progress(files_or_obs):
            if isinstance(fo, Observation):
                self.observations.append(fo)
            else:
                assert isinstance(fo, str), "files_or_obs must be a list of Observation or paths"
                self.observations.append(Observation(fo))
        idxs = np.argsort([np.min(o.time) for o in self.observations])
        self.observations = [self.observations[i] for i in idxs]
                
    def __getitem__(self, key):
        return self.observations[key]
    
    def __repr__(self):
        rows = [[i, o.date, o.telescope.name, o.name, o.filter, len(o.time)] for i, o in enumerate(self)]
        table_string = tabulate(rows, tablefmt="fancy_grid", headers=["index", "date", "telescope", "target", "filter", "images"])
        return table_string
    
    def __getattr__(self, attr):
        return np.hstack([o.__getattr__(attr) for o in self.observations])
    
    
    @property
    def diff_flux(self):
        return np.hstack([o.diff_flux for o in self.observations])

    @property
    def raw_flux(self):
        return np.hstack([o.raw_flux for o in self.observations])

    @property
    def diff_error(self):
        return np.hstack([o.diff_error for o in self.observations])

    @property
    def raw_error(self):
        return np.hstack([o.raw_error for o in self.observations])

    @property
    def stacked_trends(self):
        return np.hstack([o.trend for o in self.observations])

    def plot_each(self, ylim=None, w=4, bins=0.005, color="k", std=True):
        # TODO: add viz.plot_lcs kwargs
        """Plot all observations in a grid plot

        Parameters
        ----------
        ylim : tuple, optional
            plots ylim, by default None
        w: int, optional
            grid width in number of plots, default is 4
        """

        viz.multiplot(
            [[o.time, o.diff_flux] for o in self.observations],
            labels=[f"{i}: {o.date}" for i, o in enumerate(self.observations)],
            ylim=ylim, w=w, bincolor=color, std=std, bins=bins
        )

        viz.paper_style()

    def plot(self):
        viz.plot(self.time, self.diff_flux)

    def polynomial_trend(self, verbose=True, **kwargs):
        if verbose:
            info(f"Polynomial trends:")
        for i, o in enumerate(self.observations):
            o.polynomial_trend(**kwargs, verbose=False)
            if verbose:
                viz.print_tex(rf"{i}: " + o.xarray.attrs["trend"][1:-1])

    def plot_detrended(self, w=4, bins=0.005, color="k", std=True, ylim=None, each=True, label=True):

        if each:
            data = [[o.time, o.detrended_diff_flux] for o in self.observations]
        else:
            data = []
            for o in self.observations:
                time_min = o.time.min()
                time_max = o.time.max()
                idxs = (time_min <= self.time) & (time_max >= self.time)
                data.append([o.time, o.diff_flux - self.trend[idxs]])

        viz.multiplot(
            data,
            labels=[f"{i}: {o.date}" for i, o in enumerate(self.observations)],
            ylim=ylim, w=w, bincolor=color, std=std, bins=bins
        )

        viz.paper_style()

        if label:
            if each:
                label = "all independantly detrended"
            else:
                label = "all globally detrended"

            viz.corner_text(label, loc=(0.05, 0.95), ax=plt.gcf().axes[0], c="C0" if each else "C3", va="top")
        
    def plot_folded(self, period, t0=0, bins=0.005, phase=True):
        """Phase folded differential flux (in time) 

        Parameters
        ----------
        period : float
            period to phase fold on
        t0 : int, optional
            zeroth phase, by default 0
        bins : float, optional
            bin size in self.time unit, by default 0.005
        phase : bool, optional
            whether to plot differential flux against phase, by default True otherwise x axis is phase*period (true time)
        """
        time_fold = (self.time - t0 + 0.5 * period) % period - 0.5 * period
        viz.plot(time_fold * (1 if phase else period), self.diff_flux, bins=bins*period)
        
    def show_phase_diagram(self, period, t0=0, R=1.5, c="C0", alpha=0.5):
        """Show a diagram of the periodic orbit coverage from all observations

        Parameters
        ----------
        period : float
            orbital period
        t0 : int, optional
            zeroth time, by default 0
        R : float, optional
            figure radius of the orbit (for vizualisation), by default 1.5
        c : str, optional
            phase portions color, by default "C0"
        alpha : float, optional
            phase portions opacity, by default 0.5

        Returns
        -------
        float
            fractional coverage value
        """
        times = np.array([np.array([np.min(o.time), np.max(o.time)]) for o in self.observations])

        length = [(tmax-tmin)/period for tmin, tmax in times]
        offseted_times =  np.array(
            [
                ((time - (np.round((time - t0) / period) * period)) - t0)
                / period
                for time in times
            ]
        )

        for l, t in zip(length, offseted_times):
            l = np.min([l, 1])
            plot_slice(t[0], l, R=R, color=c, alpha=alpha)

        orbit = plt.Circle((0,0), R-.15, facecolor='none', edgecolor='black', alpha=0.5)
        my_circle = plt.Circle((0,0), R - 0.3, color='white')
        planet = plt.Circle((-np.cos(2*np.pi*(-t0/period) + np.pi/2)*(R-.15),(R-.15)*np.sin(2*np.pi*(t0/period) + np.pi/2)), 0.07, facecolor='white', edgecolor='black')
        star = plt.Circle((0,0), 0.3, facecolor='white', edgecolor='black')
        p=plt.gcf()
        p.gca().add_artist(orbit)
        p.gca().add_artist(my_circle)
        p.gca().add_artist(my_circle)
        p.gca().add_artist(star)
        p.gca().add_artist(planet)
        plt.autoscale()

        return self.phase_coverage(period)[0]
    
    def phase_coverage(self, periods):
        """Computes and returns the fractional coverage of a periodic orbit by the observations

        Parameters
        ----------
        periods : float
            orbital period

        Returns
        -------
        np.array
            fractional coverages
        """
        return np.array([phase_coverage(self.time, p) for p in np.atleast_1d(periods)])

    @property
    def X(self):
        Xs = []

        for i, o in enumerate(self.observations):
            _X = o.X
            _n = _X.shape[1]
            Xs.append(np.vstack(
                [_X if j == i else np.zeros((len(_o.time), _n)) for j, _o in enumerate(self.observations)]))

        return np.hstack(Xs).T

    def scargle(self, periods, X=True, n=1):

        if X is not None:
            if isinstance(X, bool):
                if X:
                    X = self.X
                else:
                    X = None

        if X is None:
            X = np.atleast_2d(np.ones_like(self.time))

        return scargle(
            self.time,
            self.diff_flux,
            self.diff_error,
            periods,
            X=X,
            n=n,
        )

    def plot_folded_scargle(self, period, X=True, n=1):

        if X is not None:
            if isinstance(X, bool):
                if X:
                    X = self.X
                else:
                    X = None

        if X is None:
            X = np.atleast_2d(np.ones_like(self.time))

        def variability(p):
            return models.harmonics(self.time, p, n)

        var = variability(period)
        x = models.design_matrix([X.T, *var])
        w = np.linalg.lstsq(x, self.diff_flux, rcond=None)[0]
        corrected_flux = self.diff_flux - x.T[0:-len(var)].T @ w.T[0:-len(var)].T

        ft = (self.time + 0.5 * period) % period - 0.5 * period
        i = np.argsort(ft)

        _variability = models.harmonics(ft[i], period, n)

        plt.plot(ft, corrected_flux, ".", c="0.7")
        x = np.hstack(_variability)
        w = np.linalg.lstsq(x, corrected_flux[i], rcond=None)[0]
        plt.plot(ft[i], x @ w, "k")

    def save(self):
        for o in self.observations:
            o.save()

    def mask_transits(self, epoch, period, duration):
        return Observations([o.mask_transits(epoch, period, duration) for o in self.observations], verbose=False)

    def set_catalog_target(self, catalog, designation, verbose=True):
        for obs in progress(verbose, unit="observations")(self.observations):
            obs.query_catalog(catalog)
            obs.set_catalog_target(catalog, designation, verbose=False)

    def broeg2005(self, inplace=True, cut=True, nans=False, verbose=True):
        for obs in progress(verbose, unit="observations")(self.observations):
            obs.broeg2005(inplace=inplace, cut=cut, nans=nans)

