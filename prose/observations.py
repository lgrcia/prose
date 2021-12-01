from .observation import Observation
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tabulate import tabulate
from tabulate import tabulate
from prose import viz

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
    def __init__(self, files_or_obs):
        """Object to hold multiple observations

        Parameters
        ----------
        files_or_obs : list
            A list containing phot file paths or Observation object instances
        """

        self.observations = []
        for fo in tqdm(files_or_obs):
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
        rows = [[i, o.date, o.telescope.name, o.name, o.filter, len(o.time)] for i, o in enumerate(obs)]
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
    
    def plot(self, ylim=None, w=4):
        """Plot all observations in a grid plot

        Parameters
        ----------
        ylim : tuple, optional
            plots ylim, by default None
        w: int, optional
            grid width in number of plots, default is 4
        """
        viz.plot_lcs(
            [[o.time, o.diff_flux] for o in self.observations], 
             labels=[f"{i}: {o.date}" for i, o in enumerate(self.observations)],
             ylim=ylim, w=w)
        
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