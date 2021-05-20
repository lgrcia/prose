from prose.blocks.registration import distances
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from prose import utils
from prose.utils import binning
import matplotlib.patches as mpatches
from .. import viz
import os
from os import path
import shutil
from prose import Observation
from astropy.time import Time
from astropy import units as u
from prose.models import transit


def protopapas2005(t, t0, duration, depth, c, period=1):
    _t = period * np.sin(np.pi * (t - t0) / period) / (np.pi * duration)
    return (1 - depth) + (depth / 2) * (
            2 - np.tanh(c * (_t + 1 / 2)) + np.tanh(c * (_t - 1 / 2))
    )


def template_transit(t, t0, duration):
    return protopapas2005(t, t0, duration, 1, 50, 100)


class NEB(Observation):
    """Tool to detect and diagnose near eclipsing binaries in a field

    Parameters
    ----------
    obs : prose.Observation
        observation on which to apply the tool
    radius : float, optional
       radius around the target in which to analyse other stars fluxes, by default 2.5 (in arcminutes)
    """

    def __init__(self, obs, radius=2.5):
        
        super(NEB, self).__init__(obs.xarray.copy())
        
        self.radius = radius
        target_distance = np.array(distances(obs.stars.T, obs.stars[obs.target]))
        self.nearby_ids = np.argwhere(target_distance * obs.telescope.pixel_scale / 60 < self.radius).flatten()

        self.nearby_ids = self.nearby_ids[np.argsort(np.array(distances(obs.stars[self.nearby_ids].T,
                                                                        obs.stars[obs.target])))]

        self.time = self.time
        self.epoch = None
        self.duration = None
        self.period = None
        self.depth = None
        self.expected_depth = None
        self.rms_ppt = None
        self.depth_rms = None

        self.transits = np.ones((len(self.nearby_ids), len(self.time)))
        self.disposition = np.empty(len(self.nearby_ids))
        self.cleared = np.ones(len(self.nearby_ids))
        self.likely_cleared = np.ones(len(self.nearby_ids))
        self.cleared_too_faint = np.ones(len(self.nearby_ids))
        self.flux_too_low = np.ones(len(self.nearby_ids))
        self.not_cleared = np.ones(len(self.nearby_ids))

        self.cmap =['r', 'g']

    def mask_lc(self, star, sigma=3.):
        return np.abs(self.diff_fluxes[self.aperture, star] - np.median(
            self.diff_fluxes[self.aperture, star])) < sigma * np.std(self.diff_fluxes[self.aperture, star])

    def get_transit(self, value, star, dmag_buffer=-0.5, bins=0.0027, sigma=3.):
        """Set transit parameters and run analysis of other stars to detect matching signals

        Parameters
        ----------
        value : dict
            dict containing:

            - epoch
            - duration 
            - period
            - depth (in ppt)
            in same time unit as observation
        star: float
        dmag_buffer: float
        bins: float
        sigma: float
        """
        self.epoch = value["epoch"]
        self.duration = value["duration"]
        self.period = value["period"]
        self.depth = value["depth"]

        flux = self.raw_fluxes[self.aperture, star][self.mask_lc(star,sigma)]
        flux_target = self.raw_fluxes[self.aperture, self.target][self.mask_lc(star, sigma)]
        dmag = np.nanmean(-2.5 * np.log10(flux / flux_target))
        bins, binned_flux = binning(self.time[self.mask_lc(star,sigma)], self.diff_fluxes[self.aperture, star][self.mask_lc(star,sigma)], bins)
        variance = np.var(binned_flux)
        rms_bin = np.sqrt(variance)
        if star == self.target:
            corr_dmag = dmag
        else:
            corr_dmag = dmag + dmag_buffer
        x = np.power(10, -corr_dmag / 2.50)
        self.expected_depth = self.depth / x
        self.rms_ppt = rms_bin * 1000
        self.depth_rms = self.expected_depth / self.rms_ppt

        return transit(self.time, self.epoch, self.duration, depth=self.expected_depth*1e-3, c=50,
                                period=self.period)

    def evaluate_score(self, value, **kwargs):
        for i_star in np.arange(len(self.nearby_ids)):
            self.transits[i_star] = self.get_transit(value, star=i_star, **kwargs).flatten()
            if (self.depth_rms >= 3) & (self.depth_rms <= 5):
                self.disposition[i_star] = 0
                continue
            elif self.depth_rms > 5:
                self.disposition[i_star] = 1
                continue
            elif self.expected_depth >= 1000:
                self.disposition[i_star] = 2
                continue
            elif any(self.raw_fluxes[self.aperture, i_star] / self.apertures_area[self.aperture].mean() <= 2):
                self.disposition[i_star] = 3
                continue
            else:
                self.disposition[i_star] = 4

    def plot_lc(self, star):
        viz.plot(self.time[self.mask_lc(star)], self.diff_fluxes[self.aperture, self.nearby_ids[star]][self.mask_lc(star)], std=True)
        plt.plot(self.time, self.transits[star]+1, label="expected transit")
        self.plot_meridian_flip()
        plt.legend()

    def show_stars(self, size=10):

        self._check_show(size=size)

        search_radius = 60*self.radius/self.telescope.pixel_scale
        target_coord = self.stars[self.target]
        circle = mpatches.Circle(
            target_coord,
            search_radius,
            fill=None,
            ec="white",
            alpha=0.6)

        ax = plt.gca()
        ax.add_artist(circle)
        plt.annotate("radius {}'".format(self.radius),
                     xy=[target_coord[0], target_coord[1] + search_radius + 15],
                     color="white",
                     ha='center', fontsize=12, va='bottom', alpha=0.6)

        viz.plot_marks(*self.stars.T, alpha=0.4)

        self.likely_cleared = np.argwhere(self.disposition == 0).squeeze()
        self.cleared = np.argwhere(self.disposition == 1).squeeze()
        self.cleared_too_faint = np.argwhere(self.disposition == 2).squeeze()
        self.flux_too_low = np.argwhere(self.disposition == 3).squeeze()
        self.not_cleared = np.argwhere(self.disposition == 4).squeeze()

        viz.plot_marks(*self.stars[self.target],self.target, position="top")
        viz.plot_marks(*self.stars[self.cleared].T, self.cleared, color="white", position="top")
        viz.plot_marks(*self.stars[self.cleared_too_faint].T, self.cleared_too_faint, color="white", position="top")
        viz.plot_marks(*self.stars[self.likely_cleared].T, self.likely_cleared,  color="goldenrod", position="top")
        viz.plot_marks(*self.stars[self.not_cleared].T, self.not_cleared, color="indianred", position="top")
        viz.plot_marks(*self.stars[self.flux_too_low].T, self.flux_too_low, color="indianred", position="top")

        # plt.tight_layout()
        # ax = plt.gca()
        #
        # if self.telescope.pixel_scale is not None:
        #     ob = viz.AnchoredHScaleBar(size=60 / self.telescope.pixel_scale,
        #                                label="1'", loc=4, frameon=False, extent=0,
        #                                pad=0.6, sep=4, linekw=dict(color="white", linewidth=0.8))
        #     ax.add_artist(ob)

        ylim = np.array([target_coord[1] + search_radius + 100, target_coord[1] - search_radius - 100])
        xlim = np.array([target_coord[0] - search_radius - 100, target_coord[0] + search_radius + 100])
        xlim.sort()
        ylim.sort()

        plt.ylim(ylim)
        plt.xlim(xlim)
        plt.tight_layout()

    def color(self, i, white=False):
        if self.nearby_ids[i] == self.target:
            return 'k'
        elif any(self.not_cleared == i) or any(self.flux_too_low == i):
            return "firebrick"
        elif any(self.likely_cleared == i):
            return "goldenrod"
        else:
            if white:
                return "grey"
            else:
                return "yellowgreen" #np.array([131, 220, 255]) / 255 #np.array([78, 144, 67])/255

    def plot_suspects(self):
        """Plot fluxes on which a suspect NEB signal has been identified
        """
        self.plot(idxs=np.unique(np.hstack([self.not_cleared,
                                            self.flux_too_low,
                                            self.likely_cleared
                                            ])), force_width=False)

    def plot(self, idxs=None, **kwargs):
        """Plot all fluxes and model fit used for NEB detection

        Parameters
        ----------
        idxs : list of int, optional
            list of star indexes to plot, by default None and plot fluxes of all stars
        """
        if idxs is None:
            idxs = np.arange(len(self.nearby_ids))

        nearby_ids = self.nearby_ids[idxs]
        viz.plot_lcs(
            [(self.time[self.mask_lc(i)], self.diff_fluxes[self.aperture, i][self.mask_lc(i)]) for i in nearby_ids],
            **kwargs
        )
        axes = plt.gcf().get_axes()
        for i, axe in enumerate(axes):
            if i < len(nearby_ids):
                if nearby_ids[i] == self.target:
                    color = "k"
                else:
                    color = self.color(idxs[i], white=True)
                axe.plot(self.time, self.transits[nearby_ids[i]]+1, c=color)