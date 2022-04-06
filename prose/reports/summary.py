import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt
from ..utils import fast_binning, z_scale
from ..console_utils import INFO_LABEL
from .. import Observation
import os
from os import path
from astropy.time import Time
from .. import viz
from .core import LatexTemplate
import astropy.units as u


class Summary(Observation, LatexTemplate):
    def __init__(self, obs, style="paper", template_name="summary.tex"):
        Observation.__init__(self, obs.xarray)
        LatexTemplate.__init__(self, template_name, style=style)

        datetimes = Time(self.time, format='jd', scale='utc').to_datetime()
        min_datetime = datetimes.min()
        max_datetime = datetimes.max()

        obs_duration_hours = (max_datetime - min_datetime).seconds // 3600
        obs_duration_mins = ((max_datetime - min_datetime).seconds // 60) % 60

        obs_duration = f"{min_datetime.strftime('%H:%M')} - {max_datetime.strftime('%H:%M')} " \
            f"[{obs_duration_hours}h{obs_duration_mins if obs_duration_mins != 0 else ''}]"

        # TODO: adapt to use PSF model block here (se we don't use the plot_... method from Observation)

        mean_fwhm = np.mean(self.x.fwhm.values)
        self._compute_psf_model(star=self.target)
        mean_target_fwhm = np.mean([self.stack.fwhmx, self.stack.fwhmy])
        optimal_aperture = np.mean(self.apertures_radii[self.aperture,:])

        self.obstable = [
            ["Time", obs_duration],
            ["RA - DEC", f"{self.RA} {self.DEC}"],
            ["Images", len(self.time)],
            ["Mean std · fwhm (epsf)",
             f"{mean_fwhm / (2 * np.sqrt(2 * np.log(2))):.2f} · {mean_fwhm:.2f} pixels"],
            ["Fwhm (target)", f"{mean_target_fwhm:.2f} pixels · {(mean_target_fwhm*self.telescope.pixel_scale.to(u.arcsec)):.2f}"],
            ["Optimum aperture", f"{optimal_aperture:.2f} pixels · "
                                 f"{(optimal_aperture*self.telescope.pixel_scale.to(u.arcsec)):.2f}"],
            ["Telescope", self.telescope.name],
            ["Filter", self.filter],
            ["Exposure", f"{np.mean(self.exptime)} s"],
        ]

        self.description = f"{self.date.strftime('%Y %m %d')} $\cdot$ {self.telescope.name} $\cdot$ {self.filter}"
        self._trend = None
        self._transit = None
        self.dpi = 100

        # Some paths
        # ----------
        self.header = "Observation report"

    def plot_psf_summary(self):
        self.plot_radial_psf()
        self.style()

    def plot_stars(self, size=8,**kwargs):
        self.show_stars(size=size,**kwargs)
        plt.tight_layout()

    def plot_syst(self, size=(6, 8)):
        fig = plt.figure(figsize=size)
        fig.patch.set_facecolor('xkcd:white')
        ax = fig.add_subplot(111)

        self.plot_systematics()
        self.plot_meridian_flip()
        _ = plt.gcf().axes[0].set_title("", loc="left")
        plt.xlabel(f"BJD")
        plt.ylabel("diff. flux")
        plt.tight_layout()
        self.style()

    def plot_comps(self, size=(6, 8)):
        fig = plt.figure(figsize=size)
        fig.patch.set_facecolor('xkcd:white')
        ax = fig.add_subplot(111)

        self.plot_comps_lcs()
        _ = plt.gcf().axes[0].set_title("", loc="left")
        self.plot_meridian_flip()
        plt.xlabel(f"BJD")
        plt.ylabel("diff. flux")
        plt.tight_layout()
        self.style()

    def plot_raw(self, size=(6, 4)):
        fig = plt.figure(figsize=size)
        fig.patch.set_facecolor('xkcd:white')
        raw_f = self.raw_fluxes[self.aperture, self.target]
        plt.plot(self.time, raw_f / np.mean(raw_f), c="k", label="target")
        plt.plot(self.time, self.alc[self.aperture], c="C0", label="artificial")
        self.plot_meridian_flip()
        plt.legend()
        plt.tight_layout()
        plt.xlabel(f"BJD")
        plt.ylabel("norm. flux")
        self.style()

    def make_figures(self, destination):
        self.plot_psf_summary()
        plt.savefig(path.join(destination, "psf.png"), dpi=self.dpi)
        plt.close()
        self.plot_comps()
        plt.savefig(path.join(destination, "comparison.png"), dpi=self.dpi)
        plt.close()
        self.plot_raw()
        plt.savefig(path.join(destination, "raw.png"), dpi=self.dpi)
        plt.close()
        self.plot_syst()
        plt.savefig(path.join(destination, "systematics.png"), dpi=self.dpi)
        plt.close()
        self.plot_stars()
        plt.savefig(path.join(destination, "stars.png"), dpi=self.dpi)
        plt.close()
        self.plot_lc()
        plt.savefig(path.join(destination, "lightcurve.png"), dpi=self.dpi)
        plt.close()

    def make(self, destination):

        self.make_report_folder(destination)
        self.make_figures(self.figure_destination)
        open(self.tex_destination, "w").write(self.template.render(
            obstable=self.obstable,
            target=self.name,
            description=self.description,
            header=self.header
        ))

    def set_trend(self, trend):
        self._trend = trend

    def set_transit(self, transit):
        self._transit = transit

    def plot_lc(self):
        fig = plt.figure(figsize=(6, 7 if self._trend is not None else 4))
        fig.patch.set_facecolor('xkcd:white')
        if self._trend is not None:
            plt.plot(self.time, self.diff_flux - 0.03, ".", color="gainsboro", alpha=0.3)
            plt.plot(self.time, self._trend - 0.03, c="k", alpha=0.2, label="systematics model")
            viz.plot(self.time, self.diff_flux - self._trend + 1.,label='data',binlabel='binned data (7.2 min)')
            plt.text(plt.xlim()[1], 0.965, "RAW", rotation=270)
            plt.text(plt.xlim()[1], 0.995, "DETRENDED", rotation=270)
            plt.ylim(0.95, 1.02)
        else:
            plt.ylim(0.98, 1.02)
            viz.plot(self.time, self.diff_flux,label='data',binlabel='binned data (7.2 min)')
            self.plot_meridian_flip()
        if self._transit is not None:
            plt.plot(self.time, self._transit + 1., label="transit", c="k")

        plt.legend()
        plt.xlabel(f"BJD")
        plt.ylabel("diff. flux")
        plt.tight_layout()
        self.style()

    def compile(self):
        cwd = os.getcwd()
        os.chdir(self.destination)
        os.system(f"pdflatex {self.report_name}")
        os.chdir(cwd)


class TESSSummary(Summary):

    def __init__(self, obs, style="paper", expected=None, template_name="summary.tex"):
        Summary.__init__(self, obs, style=style, template_name=template_name)
        self.obstable.insert(0, ("TIC id", self.tic_id))
        self.obstable.insert(4, ("GAIA id", self.gaia_from_toi))
        self.header = "TESS follow-up"
        self.expected = expected

    def to_csv_report(self):
        """Export a typical csv of the observation's data
        """
        destination = path.join(self.destination, "..", "measurements.txt")

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
                "DIFF_FLUX_T%s" % self.target : self.diff_flux,
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

    def make(self, destination):
        super().make(destination)
        self.to_csv_report()

    def plot_lc(self):
        super().plot_lc()
        if self.expected is not None:
            t0, duration = self.expected
            std = 2 * np.std(self.diff_flux)
            viz.plot_section(1 + std, "expected transit", t0, duration, c="k")

