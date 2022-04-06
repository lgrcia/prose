from ..reports.core import LatexTemplate
from ..blocks.registration import distances
from .. import utils
import numpy as np
import matplotlib.pyplot as plt
from os import path
import shutil
from .. import viz
from .. import Observation


def protopapas2005(t, t0, duration, depth, c, period=1):
    _t = period * np.sin(np.pi * (t - t0) / period) / (np.pi * duration)
    return (1 - depth) + (depth / 2) * (
            2 - np.tanh(c * (_t + 1 / 2)) + np.tanh(c * (_t - 1 / 2))
    )


def template_transit(t, t0, duration):
    return protopapas2005(t, t0, duration, 1, 50, 100)


class NEB(LatexTemplate, Observation):

    def __init__(self, obs, radius=2.5, style="paper", template_name="neb.tex"):
        Observation.__init__(self, obs.xarray)
        LatexTemplate.__init__(self, template_name, style=style)

        self.radius = radius
        target_distance = np.array(distances(obs.stars.T, obs.stars[obs.target]))
        self.nearby_ids = np.argwhere(target_distance * obs.telescope.pixel_scale / 60 < self.radius).flatten()

        self.nearby_ids = self.nearby_ids[
            np.argsort(np.array(distances(obs.stars[self.nearby_ids].T, obs.stars[obs.target])))]

        self.epoch = None
        self.duration = None
        self.period = None

        self.score = np.ones(len(self.nearby_ids)) * -1
        self.depths = np.ones(len(self.nearby_ids)) * -1
        self.X = None
        self.XXT_inv = None
        self.ws = None

        self.cmap = ['r', 'g']
        self.dpi = 150

    @property
    def transit(self):
        return self.epoch, self.period, self.duration

    @transit.setter
    def transit(self, value):
        """Set transit parameters and run analysis of other stars to detect matching signals

        Parameters
        ----------
        value : dict
            dict containing:

            - epoch
            - duration
            - period

            in same time unit as observation
        """
        self.epoch = value["epoch"]
        self.duration = value["duration"]
        self.period = value["period"]

        self.X = np.hstack([
            utils.rescale(self.time)[:, None] ** np.arange(0, 1 + 1),
            template_transit(self.time, self.epoch, self.duration)[:, None]
        ])
        self.XXT_inv = np.linalg.inv(self.X.T @ self.X)
        self.ws = np.ones((len(self.nearby_ids), self.X.shape[1]))
        self.evaluate_score()

    def evaluate_transit(self, lc, error):
        w = (self.XXT_inv @ self.X.T) @ lc
        dw = np.var(lc) * len(lc) * self.XXT_inv
        return w, dw

    def evaluate_score(self):
        target_score = None
        for i, i_star in enumerate(self.nearby_ids):
            x = self.xarray.isel(star=i_star, apertures=self.aperture)
            flux = x.diff_fluxes.values
            error = x.diff_errors.values
            w, dw = self.evaluate_transit(flux, error)
            self.ws[i] = w
            self.depths[i], self.score[i] = w[-1], np.abs(w[-1] / np.sqrt(np.diag(dw))[-1])
            if i_star == self.target:
                target_score = self.score[i]

        self.score[self.score < 0] = 0
        self.score = np.abs(self.score)
        self.score /= target_score
        self.suspects = self.score > 5 * np.std(self.score)
        self.potentials = self.score > 3.5 * np.std(self.score)

    def plot_lci(self, i):
        viz.plot(self.time, self.fluxes[self.aperture, self.nearby_ids[i]].flux, std=True)
        plt.plot(self.time, self.X @ self.ws[i], label="model")
        plt.legend()

    def plot_neb_stars(self, size=10):

        self._check_show(size=size)

        search_radius = 60 * self.radius / self.telescope.pixel_scale
        target_coord = self.stars[self.target]
        circle = plt.Circle(
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

        clean = self.nearby_ids[
            np.argwhere(np.logical_and(np.logical_not(self.potentials), np.logical_not(self.suspects))).flatten()]
        clean = np.setdiff1d(clean, self.target)
        suspects = self.nearby_ids[np.argwhere(self.suspects).flatten()]
        potentials = np.setdiff1d(self.nearby_ids[np.argwhere(self.potentials).flatten()], suspects)

        viz.plot_marks(*self.stars[self.target], self.target, position="top")
        viz.plot_marks(*self.stars[clean].T, clean, color="white", position="top")
        viz.plot_marks(*self.stars[potentials].T, potentials, color="goldenrod", position="top")
        viz.plot_marks(*self.stars[suspects].T, suspects, color="indianred", position="top")

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
        elif self.suspects[i]:
            return "firebrick"
        elif self.potentials[i]:
            return "goldenrod"
        else:
            if white:
                return "grey"
            else:
                return "yellowgreen"  # np.array([131, 220, 255]) / 255 #np.array([78, 144, 67])/255

    def plot_suspects(self):
        """Plot fluxes on which a suspect NEB signal has been identified
        """
        self.plot(
            idxs=np.unique(np.hstack([np.argwhere(self.suspects).flatten(), np.argwhere(self.potentials).flatten()])),
            force_width=False)

    def plot_all(self, idxs=None, **kwargs):
        """Plot all fluxes and model fit used for NEB detection

        Parameters
        ----------
        idxs : list of int, optional
            list of star indexes to plot, by default None and plot fluxes of all stars
        """
        if idxs is None:
            idxs = np.arange(len(self.nearby_ids))

        nearby_ids = self.nearby_ids[idxs]
        viz.multiplot(
            [(self.time, self.diff_fluxes[self.aperture, i]) for i in nearby_ids],
            labels=nearby_ids,
            colors=[self.color(idxs[i], white=True) for i in range(len(nearby_ids))],
            **kwargs
        )
        axes = plt.gcf().get_axes()
        for i, axe in enumerate(axes):
            if i < len(nearby_ids):
                if nearby_ids[i] == self.target:
                    color = "k"
                else:
                    color = self.color(idxs[i], white=True)
                axe.plot(self.time, self.X @ self.ws[idxs[i]], c=color)

    def make(self, destination):

        assert self.X is not None, "NEB.transit must be set first"
        self.evaluate_score()

        self.make_report_folder(destination)
        self.make_figures(self.figure_destination)

        shutil.copyfile(path.join(template_folder, "prose-report.cls"), path.join(destination, "prose-report.cls"))
        repdest = path.join(destination, self.template_name)
        open(repdest, "w").write(self.template.render(
            lcs=self.lcs,
            target=self.name
        ))

    def make_figures(self, destination):
        self.lcs = []
        a = np.arange(len(self.nearby_ids))
        if len(self.nearby_ids) > 30:
            split = [np.arange(0, 30), *np.array([a[i:i + 7 * 8] for i in range(30, len(a), 7 * 8)])]
        else:
            split = [np.arange(0, len(self.nearby_ids))]

        for i, idxs in enumerate(split):
            lcs_path = path.join(destination, "lcs{}.png".format(i))
            self.lcs.append(lcs_path)
            if i == 0:
                self.plot_all(np.arange(0, np.min([30, len(self.nearby_ids)])), W=5)
            else:
                self.plot_all(idxs, W=8)

            self.style()
            plt.savefig(lcs_path, dpi=self.dpi)
            plt.close()

        self.plot_neb_stars(size=8)
        plt.savefig(path.join(self.figure_destination, "nebstars"), dpi=self.dpi)
        plt.close()