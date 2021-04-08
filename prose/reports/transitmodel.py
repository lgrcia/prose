import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import collections
from ..utils import fast_binning, z_scale
from ..console_utils import INFO_LABEL
from .. import Observation
from ..fluxes import pont2006
from ..utils import binning
import os
from os import path
import shutil
from astropy.time import Time
import jinja2
from .. import viz
from .core import LatexTemplate, template_folder

template_folder = path.abspath(path.join(path.dirname(__file__), "..", "..", "latex"))


class TransitModel(Observation, LatexTemplate):

    def __init__(self, obs, t0=None, duration=None, rms_bin=0.005, style="paper", template_name="transitmodel.tex"):
        Observation.__init__(self, obs.xarray)
        LatexTemplate.__init__(self, template_name, style=style)

        # Some paths
        # ----------
        self.destination = None
        self.report_name = None
        self.figure_destination = None
        self.measurement_destination = None

        self.ingress = self.time[np.nonzero(self.transit_model)[0][0]]
        self.egress = self.time[np.nonzero(self.transit_model)[0][-1]]
        self.t14 = (self.egress - self.ingress) * 24 * 60
        self.duration = duration
        self.t0 = t0
        self.snr = self.SNR()
        self.rms = self.RMS_binned(rms_bin)
        self.dpi = 100
        self.obstable = [
            ["Parameters", "Model","TESS"]
            ["[u1,u2]",None,"-"],
            ["[R*]", None,None],
            ["[M*]", None,None],
            ["P", None,None],
            ["Rp", None, None],
            ["Tc", None, None],
            ["b", None, None],
            ["Duration", f"{self.t14:.2f} min", None],
            ["(Rp/R*)\u00b2", None,None],
            ["Apparent depth", f"{np.abs(min(self.transit_model)):.2e}", None],
            ["a/R*", None, None],
            ["i", None, None],
            ["SNR", f"{self.snr:.2f}",None],
            ["RMS per bin (%s min)" % f"{self.rms[1]:.1f}", f"{self.rms[0]:.2e}",None],
        ]

    def plot_meridian_flip(self):
        if self.meridian_flip is not None:
            plt.axvline(self.meridian_flip, c="k", alpha=0.15)
            _, ylim = plt.ylim()
            plt.text(self.meridian_flip, ylim, "meridian flip ", ha="right", rotation="vertical", va="top", color="0.7")

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

    def to_csv_report(self, destination, sep=" "):
        """Export a typical csv of the observation's data

        Parameters
        ----------
        destination : str
            Path of the csv file to save
        sep : str, optional
            separation string within csv, by default " "
        """
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
                "DIFF_FLUX_T1": self.diff_flux,
                "DIFF_FLUX_T1_DETRENDED": self.diff_flux - self.trend_model + 1,
                "DIFF_ERROR_T1": self.diff_error,
                **dict(zip(list_columns, list_columns_array)),
                "dx": self.dx,
                "dy": self.dy,
                "FWHM": self.fwhm,
                "SKYLEVEL": self.sky,
                "AIRMASS": self.airmass,
                "EXPOSURE": self.exptime,
            })
        )
        df.to_csv(destination, sep=sep, index=False)

    def make(self, destination):
        self.make_report_folder(destination)
        self.make_figures(self.figure_destination)
        self.to_csv_report(self.measurement_destination)

        shutil.copyfile(path.join(template_folder, "prose-report.cls"), path.join(destination, "prose-report.cls"))
        tex_destination = path.join(self.destination, f"{self.report_name}.tex")
        open(tex_destination, "w").write(self.template.render(
            obstable=self.obstable
        ))

    def plot_lc_model(self):
        fig = plt.figure(figsize=(6, 7 if self.trend_model is not None else 4))
        fig.patch.set_facecolor('xkcd:white')
        viz.plot_lc(self.time, self.diff_flux)
        plt.plot(self.time, self.trend_model + self.transit_model, c="C0", alpha=0.5,
                 label="systematics + transit model")
        plt.plot(self.time, self.transit_model + 1. - 0.03, label="transit model", c="k")
        viz.plot_lc(self.time, self.diff_flux - self.trend_model + 1. - 0.03, plot_kwargs=dict(label=None),
                    errorbar_kwargs=dict(label=None))
        plt.ylim(0.95, 1.03)
        self.plot_ingress_egress()
        ax = plt.gca()
        plt.text(0.95, 0.05, "detrended", fontsize=12, horizontalalignment='right', verticalalignment='bottom',
                 transform=ax.transAxes)
        plt.text(0.95, 0.85, "raw", fontsize=12, horizontalalignment='right', verticalalignment='top',
                 transform=ax.transAxes, color="grey")
        self.plot_meridian_flip()
        plt.legend()
        plt.xlabel(f"BJD-TDB")
        plt.ylabel("diff. flux")
        plt.tight_layout()
        self.style()

    def SNR(self):
        lc = self.diff_flux - self.transit_model
        wn, rn = pont2006(self.time, lc, plot=False)
        texp = np.mean(self.exptime)
        _duration = (self.egress - self.ingress) * 24 * 60 * 60
        n = int(_duration / texp)
        depth = np.abs(min(self.transit_model))
        return depth / (np.sqrt(((wn ** 2) / n) + (rn ** 2)))

    def RMS_binned(self, rms_bin):
        bins, flux, std = binning(self.time, self.diff_flux - self.trend_model + 1, bins=rms_bin, std=True)
        return np.mean(std), rms_bin * 24 * 60

    def compile(self):
        cwd = os.getcwd()
        os.chdir(self.destination)
        os.system(f"pdflatex {self.report_name}")
        os.chdir(cwd)
