import numpy as np
import matplotlib.pyplot as plt
from .. import Observation
from ..fluxes import pont2006
from ..utils import binning
from os import path
from .. import viz
from .core import LatexTemplate
import pandas as pd
import collections


class TransitModel(Observation, LatexTemplate):

    def __init__(self, obs, transit, trend=None, expected=None, rms_bin=5/24/60, style="paper", template_name="transitmodel.tex"):
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
        self.residuals = self.diff_flux - self.transit_model - self.trend_model
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
            ["[R*]", "R$_{\odot}$", "R$_{\odot}$"],
            ["[M*]", "M$_{\odot}$", "M$_{\odot}$"],
            ["P", "d", "d"],
            ["Rp", "R$_{\oplus}$",  "R$_{\oplus}$"],
            ["Tc", None , None],
            ["b", None, None],
            ["Duration", f"{self.t14:.2f} min", None],
            ["(Rp/R*)\u00b2", None, None],
            ["Apparent depth (min. flux)", f"{np.abs(min(self.transit_model)):.2e}", None],
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
        open(self.tex_destination, "w").write(self.template.render(
            obstable=self.obstable
        ))
        self.to_csv_report()

    def plot_lc_model(self):
        fig = plt.figure(figsize=(6, 7 if self.trend_model is not None else 4))
        fig.patch.set_facecolor('white')
        viz.plot(self.time, self.diff_flux,label='data',binlabel='binned data (7.2 min)')
        plt.plot(self.time, self.trend_model + self.transit_model, c="C0",
                 label="systematics + transit model")
        plt.plot(self.time, self.transit_model + 1. - 0.03, label="transit model", c="k")
        plt.text(plt.xlim()[1] + 0.002, 1, "RAW", rotation=270, ha="center")
        viz.plot(self.time, self.diff_flux - self.trend_model + 1. - 0.03)
        plt.text(plt.xlim()[1] + 0.002, 1 - 0.03, "DETRENDED", rotation=270, ha="center")
        plt.ylim(0.95, 1.03)

        # plot expected and observed transits
        std = 2 * np.std(self.diff_flux)
        t0, duration = self.expected
        viz.plot_section(1 + std, "expected", t0, duration, c="k")
        duration = self.egress - self.ingress
        viz.plot_section(1 + std + 0.005, "observed", self.ingress + duration / 2, duration, c="C0")

        self.plot_meridian_flip()
        plt.legend()
        plt.xlabel(f"BJD-TDB")
        plt.ylabel("diff. flux")
        plt.tight_layout()
        self.style()

    def to_csv_report(self):
        """
        This one adds de-trended light-curve
        """
        destination = path.join(self.destination, "..", 'measurements' + ".txt")

        comparison_stars = self.comps[self.aperture]
        list_diff = ["DIFF_FLUX_C%s" % i for i in comparison_stars]
        list_err = ["DIFF_ERROR_C%s" % i for i in comparison_stars]
        list_columns = [None] * (len(list_diff) + len(list_err))
        list_columns[::2] = list_diff
        list_columns[1::2] = list_err
        list_diff_array = [self.diff_fluxes[self.aperture, i] for i in comparison_stars]
        list_err_array = [self.diff_errors[self.aperture, i] for i in comparison_stars]
        list_columns_array = [None] * (len(list_diff_array) + len(list_err_array))
        list_columns_array[::2] = list_diff_array
        list_columns_array[1::2] = list_err_array

        df = pd.DataFrame(collections.OrderedDict(
            {
                "BJD-TDB" if self.time_format == "bjd_tdb" else "JD-UTC": self.time,
                "DIFF_FLUX_T%s" % self.target: self.diff_flux,
                "DIFF_FLUX_T%s_DETRENDED" % self.target: self.diff_flux - self.trend_model + 1,
                "DIFF_ERROR_T%s" % self.target: self.diff_error,
                **dict(zip(list_columns, list_columns_array)),
                "dx": self.dx,
                "dy": self.dy,
                "FWHM": self.fwhm,
                "SKYLEVEL": self.sky,
                "AIRMASS": self.airmass,
                "EXPOSURE": self.exptime,
            })
        )
        df.to_csv(destination, sep="\t", index=False)

    def snr(self):
        lc = self.diff_flux - self.transit_model - self.trend_model
        wn, rn = pont2006(self.time, lc, plot=False)
        texp = np.mean(self.exptime)
        _duration = (self.egress - self.ingress) * 24 * 60 * 60
        n = int(_duration / texp)
        depth = np.abs(min(self.transit_model))
        return depth / (np.sqrt(((wn ** 2) / n) + (rn ** 2)))

    def rms_binned(self):
        bins, flux, std = binning(self.time, self.residuals, bins=self.rms_bin, std=True)
        return np.std(flux), self.rms_bin * 24 * 60