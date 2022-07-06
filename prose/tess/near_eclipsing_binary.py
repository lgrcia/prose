from prose.blocks.registration import distances
import numpy as np
import matplotlib.pyplot as plt
from prose.utils import binning, sigma_clip
import matplotlib.patches as mpatches
from prose import viz
from prose import Observation, utils
from prose.models import transit
from prose import blocks
import pandas as pd
from astropy import units as u
from prose.pipeline import AperturePhotometry
from prose import load


def protopapas2005(t, t0, duration, depth, c, period=1):
    _t = period * np.sin(np.pi * (t - t0) / period) / (np.pi * duration)
    return (1 - depth) + (depth / 2) * (
            2 - np.tanh(c * (_t + 1 / 2)) + np.tanh(c * (_t - 1 / 2))
    )


def template_transit(t, t0, duration):
    return protopapas2005(t, t0, duration, 1, 50, 100)


class NEB(Observation):
    """Tool to detect and diagnose near eclipsing binaries in a field (using AIJ methods)

    Parameters
    ----------
    obs : prose.Observation
        observation on which to apply the tool
    radius : float, optional
       radius around the target in which to analyse other stars fluxes, by default 2.5 (in arcminutes)
    """

    def __init__(self, obs, radius=2.5, nearby_ids=None, flip_corr=False, photometry=False, radec_file='', **kwargs):

        super(NEB, self).__init__(obs.xarray.copy())

        if photometry is True: #TODO test this new way of redoing the photometry for the NEB check
            self.new_photometry(radec_file, **kwargs)
            self = new_self.copy()

        self.radius = radius
        if nearby_ids is None:
            target_distance = np.linalg.norm(self.stars - self.stars[self.target], axis=1)
            self.nearby_ids = np.argwhere(target_distance * self.telescope.pixel_scale / 60 < self.radius).flatten()
        else:
            self.nearby_ids = nearby_ids

        if flip_corr is True:
            self.flip_correction() #Does this modify only the neb xarray or the obs xarray as well ?

        # transit params
        self.epoch = None
        self.duration = None
        self.period = None
        self.depth = None

        # computed
        self.dmags = {}
        self.expected_depths = {}  # not ppt
        self.rmss_ppt = {}
        self.depths_rms = {}
        self.transits = {}
        self.disposition = {}

        # indexes
        self.cleared = None
        self.likely_cleared = None
        self.cleared_too_faint = None
        self.flux_too_low = None
        self.not_cleared = None
        self.suspects = None

    def new_photometry(self, radec_file, **kwargs):
        base_stars, _ = blocks.DAOFindStars(n_stars=100, min_separation=10)(self.stack)
        radecs = pd.read_csv(radec_file, names=["ra", "dec"], usecols=[0, 1]).iloc[:, [0, 1]].values
        close_stars = self.radec_to_pixel(radecs, unit=(u.hourangle, u.deg))
        target_distance = np.linalg.norm(base_stars - close_stars[0], axis=1)
        base_stars = base_stars[np.argwhere(target_distance * self.telescope.pixel_scale / 60 > 2.6).flatten()]
        full_stars = np.vstack([close_stars, base_stars])
        photometry = AperturePhotometry(stars=full_stars, overwrite=True, **kwargs)
        photometry.run(self.phot)
        new_self = load(self.phot)
        return(new_self)


    def set_transit(self, epoch, duration, depth, period):
        self.epoch = epoch
        self.duration = duration
        self.depth = depth
        self.period = period

    @property
    def transit_params(self):
        return self.epoch, self.duration, self.period, self.depth

    def evaluate_score(self, epoch, duration, depth, period, dmag_buffer=-0.5, bins=0.0027, sigma=3.):
        """Find the expected transit and star disposition"""

        self.set_transit(epoch, duration, depth, period)

        flux_target = self.raw_fluxes[self.aperture, self.target].copy()

        self.cleared = []
        self.likely_cleared = []
        self.cleared_too_faint = []
        self.flux_too_low = []
        self.not_cleared = []
        self.suspects = []

        for i in self.nearby_ids:
            raw_flux = self.raw_fluxes[self.aperture, i].copy()
            diff_flux = self.diff_fluxes[self.aperture, i].copy()
            mask = sigma_clip(diff_flux, return_mask=True, sigma=sigma)

            # computing depth relative rms
            dmag = np.nanmean(-2.5 * np.log10(raw_flux[mask] / flux_target[mask])) + (dmag_buffer if i != self.target else 0)
            binned_std = np.std(binning(self.time[mask], diff_flux[mask], bins)[1])
            expected_depth = (self.depth / (np.power(10, -dmag / 2.50)))/1e3
            depth_rms = expected_depth / binned_std

            self.transits[i] = transit(
                self.time, self.epoch, self.duration, depth=expected_depth, c=50, period=self.period
            ).flatten()
            self.dmags[i] = dmag
            self.expected_depths[i] = expected_depth
            self.rmss_ppt[i] = binned_std * 1000
            self.depths_rms[i] = depth_rms

            # check score
            if (depth_rms >= 3) & (depth_rms <= 5):
                self.disposition[i] = 0
                self.likely_cleared.append(i)
            elif depth_rms > 5:
                self.disposition[i] = 1
                self.cleared.append(i)
            elif expected_depth >= 1000:
                self.disposition[i] = 2
                self.cleared_too_faint.append(i)
            elif any(self.raw_fluxes[self.aperture, i] / self.apertures_area[self.aperture].mean() <= 2):
                self.disposition[i] = 3
                self.flux_too_low.append(i)
            else:
                self.disposition[i] = 4
                self.not_cleared.append(i)

        self.likely_cleared = np.array(self.likely_cleared)
        self.cleared = np.array(self.cleared)
        self.cleared_too_faint = np.array(self.cleared_too_faint)
        self.flux_too_low = np.array(self.flux_too_low)
        self.not_cleared = np.array(self.not_cleared)
        self.suspects = np.unique(np.hstack([self.not_cleared, self.flux_too_low, self.likely_cleared])).astype(int)

    def plot_lc(self, star, sigma=3.):
        """
        Plotting the light curve of designated star with the expected transit. Use evaluate_score() first.
                Parameters
                ----------
                star : int
        """
        diff_flux = self.diff_fluxes[self.aperture, star].copy()
        mask = sigma_clip(diff_flux, return_mask=True, sigma=sigma)
        viz.plot(self.time[mask],
                 diff_flux[mask], std=True)
        plt.plot(self.time, self.transits[star] + 1, label="expected transit")
        self.plot_meridian_flip()
        plt.legend()

    def show_neb_stars(self, size=10, legend=True, **kwargs):
        """
        Visualization of the star dispositions on the zoomed stack image.
                Parameters
                ----------
                size : int
        """

        self._check_show(size=size, **kwargs)

        search_radius = 60 * self.radius / self.telescope.pixel_scale
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
        viz.plot_marks(*self.stars[self.target], self.target, position="top")

        if len(self.cleared) > 0:
            viz.plot_marks(*self.stars[self.cleared].T, self.cleared, color="white", position="top")
        if len(self.cleared_too_faint) > 0:
            viz.plot_marks(*self.stars[self.cleared_too_faint].T, self.cleared_too_faint, color="white", position="top")
        if len(self.likely_cleared) > 0:
            viz.plot_marks(*self.stars[self.likely_cleared].T, self.likely_cleared, color="goldenrod", position="top")
        if len(self.not_cleared) > 0:
            viz.plot_marks(*self.stars[self.not_cleared].T, self.not_cleared, color="indianred", position="top")
        if len(self.flux_too_low) > 0:
            viz.plot_marks(*self.stars[self.flux_too_low].T, self.flux_too_low, color="indianred", position="top")

        ylim = np.array([target_coord[1] + search_radius + 100, target_coord[1] - search_radius - 100])
        xlim = np.array([target_coord[0] - search_radius - 100, target_coord[0] + search_radius + 100])
        xlim.sort()
        ylim.sort()

        plt.ylim(ylim)
        plt.xlim(xlim)
        if legend:
            colors = ["gainsboro", "goldenrod", "indianred"]
            texts = ["Cleared/Cleared too faint", "Likely cleared", "Not cleared/Flux too low"]
            viz.circles_legend(colors, texts)
        plt.tight_layout()

    def color(self, i, white=False):
        if i == self.target:
            return 'k'
        elif np.any(self.not_cleared == i) or np.any(self.flux_too_low == i):
            return "firebrick"
        elif np.any(self.likely_cleared == i):
            return "goldenrod"
        else:
            if white:
                return "darkgrey"
            else:
                return "yellowgreen"  # np.array([131, 220, 255]) / 255 #np.array([78, 144, 67])/255

    def plot_suspects(self):
        """Plot fluxes on which a suspect NEB signal has been identified
        """
        self.plot_lcs(idxs=self.suspects, force_width=False)

    def plot_lcs(self, idxs=None, **kwargs):
        """Plot all fluxes and model fit used for NEB detection

        Parameters
        ----------
        idxs : list of int, optional
            list of star indexes to plot, by default None and plot fluxes of all stars
        """
        if idxs is None:
            idxs = self.nearby_ids

        labels = idxs.astype(str)
        for k in np.arange(len(idxs)):
            if idxs[k] == self.target:
                labels[k] = labels[k] + ' (target)'

        viz.multiplot([sigma_clip(self.diff_fluxes[self.aperture, i], x=self.time) for i in idxs],
                      labels=labels,
                      **kwargs)

        axes = plt.gcf().get_axes()
        for i, axe in enumerate(axes):
            if i < len(idxs):
                if idxs[i] == self.target:
                    axe.plot(self.time, self.transits[idxs[i]] + 1, c='k', label='expected transit')
                    axe.legend()
                else:
                    color = self.color(idxs[i], white=True)
                    axe.plot(self.time, self.transits[idxs[i]] + 1, c=color)
                if self.meridian_flip is not None:
                    axe.vlines(self.meridian_flip, axe.get_ylim()[0], axe.get_ylim()[1], colors="gray", linestyle='--')
                    if idxs[i] == self.target:
                        axe.text(self.meridian_flip, axe.get_ylim()[1], "MF", ha="right", rotation="vertical", va="top",
                                 color="gray")