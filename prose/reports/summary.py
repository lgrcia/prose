import numpy as np
import matplotlib.pyplot as plt
from .. import Observation
from ..tess import TFOPObservation
import os
from os import path
from astropy.time import Time
from .. import viz
from .core import LatexTemplate
import astropy.units as u


class Summary(TFOPObservation, Observation, LatexTemplate):
    def __init__(self, obs,mean_target_fwhm,optimal_aperture,mean_fwhm, name=None, style="paper", template_name="summary.tex",tess=False):
        #if tess is True :
        #    TFOPObservation.__init__(self, photfile,name=name)
        #else :
        #    Observation.__init__(self,photfile)
        LatexTemplate.__init__(self, template_name, style=style)
        self.obs=obs

        datetimes = Time(self.obs.time, format='jd', scale='utc').to_datetime()
        min_datetime = datetimes.min()
        max_datetime = datetimes.max()

        obs_duration_hours = (max_datetime - min_datetime).seconds // 3600
        obs_duration_mins = ((max_datetime - min_datetime).seconds // 60) % 60

        self.obs_duration = f"{min_datetime.strftime('%H:%M')} - {max_datetime.strftime('%H:%M')} " \
            f"[{obs_duration_hours}h{obs_duration_mins if obs_duration_mins != 0 else ''}]"

        # TODO: adapt to use PSF model block here (se we don't use the plot_... method from Observation)

        self.mean_fwhm = mean_fwhm
        #self._compute_psf_model(star=self.target)
        self.mean_target_fwhm = mean_target_fwhm
        self.optimal_aperture = optimal_aperture

        self.obstable = [
            ["Time", self.obs_duration],
            ["RA - DEC", f"{self.obs.RA} {self.obs.DEC}"],
            ["Images", len(self.obs.time)],
            ["Mean std · fwhm (epsf)",
             f"{self.mean_fwhm / (2 * np.sqrt(2 * np.log(2))):.2f} · {self.mean_fwhm:.2f} pixels"],
            ["Fwhm (target)", f"{self.mean_target_fwhm:.2f} pixels · {(self.mean_target_fwhm*0.34*u.arcsec):.2f}"],
            ["Optimum aperture", f"{self.optimal_aperture:.2f} pixels · "
                                 f"{(self.optimal_aperture*0.34*u.arcsec):.2f}"],
            ["Telescope", self.obs.telescope.name],
            ["Filter", self.obs.filter],
            ["Exposure", f"{np.mean(self.obs.exptime)} s"],
        ]

        self.description = f"{self.obs.night_date.strftime('%Y %m %d')} $\cdot$ {self.obs.telescope.name} $\cdot$ {self.obs.filter}"
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

    def make(self, destination):

        self.make_report_folder(destination)
        #self.make_figures(self.figure_destination)
        open(self.tex_destination, "w").write(self.template.render(
            obstable=self.obstable,
            target=self.obs.name,
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
