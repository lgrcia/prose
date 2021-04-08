import numpy as np
import matplotlib.pyplot as plt
from .. import Observation
from ..fluxes import pont2006
from ..utils import binning
import os
from os import path
import shutil
from .. import viz
from .core import LatexTemplate, template_folder


class TransitModel(Observation, LatexTemplate):

    def __init__(self, obs, transit, trend=None, expected=None, rms_bin=0.005, style="paper", template_name="transitmodel.tex"):
        """Transit modeling report

        Parameters
        ----------
        obs : Observation
            observation to model
        transit : 1D array
            transit model
        trend : 1D array, optional
            trend model, by default None
        expected : tuple, optional
            tuple of (t0, duration) of expected transit, by default None
        rms_bin : float, optional
            [description], by default 0.005
        style : str, optional
            [description], by default "paper"
        template_name : str, optional
            [description], by default "transitmodel.tex"
        """
        Observation.__init__(self, obs.xarray)
        LatexTemplate.__init__(self, template_name, style=style)

        # Some paths
        # ----------
        self.destination = None
        self.report_name = None
        self.figure_destination = None

        self.transit_model = transit
        self.trend_model = trend if trend is not None else np.zeros_like(self.time)
        intransit = self.transit_model < -1e-6
        self.ingress = self.time[intransit][0]
        self.egress = self.time[intransit][-1]
        self.t14 = (self.egress - self.ingress) * 24 * 60
        self.snr = self.snr()
        self.rms_bin = rms_bin
        self.rms = self.rms_binned()

        self.expected = expected
        self.obstable = [
            ["Parameters", "Model", "TESS"],
            ["[u1,u2]", None, "-"],
            ["[R*]", None, None],
            ["[M*]", None, None],
            ["P", None, None],
            ["Rp", None, None],
            ["Tc", None, None],
            ["b", None, None],
            ["Duration", f"{self.t14:.2f} min", None],
            ["(Rp/R*)\u00b2", None, None],
            ["Apparent depth", f"{np.abs(min(self.transit_model)):.2e}", None],
            ["a/R*", None, None],
            ["i", None, None],
            ["SNR", f"{self.snr:.2f}", None],
            ["RMS per bin (%s min)" % f"{self.rms[1]:.1f}", f"{self.rms[0]:.2e}", None],
        ]

    def plot_ingress_egress(self):
        plt.axvline(self.t0 + self.duration / 2, c='C4', alpha=0.4, label='predicted ingress/egress')
        plt.axvline(self.t0 - self.duration / 2, c='C4', alpha=0.4)
        plt.axvline(self.ingress, ls='--', c='C4', alpha=0.4, label='observed ingress/egress')
        plt.axvline(self.egress, ls='--', c='C4', alpha=0.4)
        _, ylim = plt.ylim()

    def make_figures(self, destination):
        self.plot_lc_model()
        plt.savefig(path.join(destination, "model.png"), dpi=self.dpi)
        plt.close()

    def make(self, destination):
        self.make_report_folder(destination)
        self.make_figures(self.figure_destination)

        shutil.copyfile(path.join(template_folder, "prose-report.cls"), path.join(destination, "prose-report.cls"))
        tex_destination = path.join(self.destination, f"{self.report_name}.tex")
        open(tex_destination, "w").write(self.template.render(
            obstable=self.obstable
        ))

    def plot_lc_model(self):
        fig = plt.figure(figsize=(6, 7 if self.trend_model is not None else 4))
        fig.patch.set_facecolor('white')
        viz.plot(self.time, self.diff_flux)
        plt.plot(self.time, self.trend_model + self.transit_model, c="C0",
                 label="systematics + transit model")
        plt.plot(self.time, self.transit_model + 1. - 0.03, label="transit model", c="k")
        plt.text(plt.xlim()[1] + 0.002, 1, "RAW", rotation=270)
        viz.plot(self.time, self.diff_flux - self.trend_model + 1. - 0.03)
        plt.text(plt.xlim()[1] + 0.002, 1 - 0.03, "DETRENDED", rotation=270)
        plt.ylim(0.95, 1.03)
        ymin = plt.ylim()[0]

        # plot expected and observed transits
        std = 2 * np.std(self.diff_flux)
        a = 0.5
        t0, duration = self.expected
        viz.plot_section(1 + std, "predicted", t0, duration, c="k")
        plt.vlines(t0 - duration / 2, ymin, ymin + 0.002, color="k", alpha=a)
        plt.vlines(t0 + duration / 2, ymin, ymin + 0.002, color="k", alpha=a)
        duration = self.egress - self.ingress
        viz.plot_section(1 + std + 0.005, "observed", self.ingress + duration / 2, duration, c="C0")
        plt.vlines(self.ingress, ymin, ymin + 0.002, color="C0", alpha=a)
        plt.vlines(self.egress, ymin, ymin + 0.002, color="C0", alpha=a)

        self.plot_meridian_flip()
        plt.legend()
        plt.xlabel(f"BJD-TDB")
        plt.ylabel("diff. flux")
        plt.tight_layout()
        self.style()

    def snr(self):
        lc = self.diff_flux - self.transit_model
        wn, rn = pont2006(self.time, lc, plot=False)
        texp = np.mean(self.exptime)
        _duration = (self.egress - self.ingress) * 24 * 60 * 60
        n = int(_duration / texp)
        depth = np.abs(min(self.transit_model))
        return depth / (np.sqrt(((wn ** 2) / n) + (rn ** 2)))

    def rms_binned(self):
        bins, flux, std = binning(self.time, self.diff_flux - self.trend_model + 1, bins=self.rms_bin, std=True)
        return np.mean(std), self.rms_bin * 24 * 60

    def compile(self):
        cwd = os.getcwd()
        os.chdir(self.destination)
        os.system(f"pdflatex {self.report_name}")
        os.chdir(cwd)
