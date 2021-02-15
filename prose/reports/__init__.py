import numpy as np
import matplotlib.pyplot as plt
from ..utils import fast_binning, z_scale
from ..console_utils import INFO_LABEL
from .. import Observation
import os
from os import path
import shutil
from astropy.time import Time
import jinja2
from .. import viz


template_folder = path.abspath(path.join(path.dirname(__file__), "..", "..", "latex"))


class ObservationReport(Observation):

    def __init__(self, obs, style="paper", template_name="ObservationReport.tex"):
        super().__init__(obs.xarray)
        self._style = style

        datetimes = Time(self.time, format='jd', scale='utc').to_datetime()
        min_datetime = datetimes.min()
        max_datetime = datetimes.max()

        obs_duration_hours = (max_datetime - min_datetime).seconds // 3600
        obs_duration_mins = ((max_datetime - min_datetime).seconds // 60) % 60

        obs_duration = f"{min_datetime.strftime('%H:%M')} - {max_datetime.strftime('%H:%M')} " \
            f"[{obs_duration_hours}h{obs_duration_mins if obs_duration_mins != 0 else ''}]"

        self.obstable = [
            ["Time", obs_duration],
            ["RA - DEC", f"{self.RA} {self.DEC}"],
            ["images", len(self.time)],
            ["GAIA id", None],
            ["mean std · fwhm",
             f"{np.mean(self.fwhm) / (2 * np.sqrt(2 * np.log(2))):.2f} · {np.mean(self.fwhm):.2f} pixels"],
            ["Telescope", self.telescope.name],
            ["Filter", self.filter],
            ["exposure", f"{np.mean(self.exptime)} s"],
        ]

        self.description = f"{self.date[0:4]} {self.date[4:6]} {self.date[6::]} $\cdot$ {self.telescope.name} $\cdot$ {self.filter}"
        self.template_name = template_name
        self.template = None
        self.load_template()
        self._trend = None
        self._transit = None
        self.dpi = 100

    def load_template(self):
        latex_jinja_env = jinja2.Environment(
            block_start_string='\BLOCK{',
            block_end_string='}',
            variable_start_string='\VAR{',
            variable_end_string='}',
            comment_start_string='\#{',
            comment_end_string='}',
            line_statement_prefix='%%',
            line_comment_prefix='%#',
            trim_blocks=True,
            autoescape=False,
            loader=jinja2.FileSystemLoader(template_folder)
        )
        self.template = latex_jinja_env.get_template(self.template_name)

    def style(self):
        if self._style == "paper":
            viz.paper_style()
        elif self._style == "bokeh":
            viz.bokeh_style()

    def plot_psf(self, n=40, zscale=False):
        n /= np.sqrt(2)
        x, y = self.stars[self.target]
        Y, X = np.indices(self.stack.shape)
        cutout_mask = (np.abs(X - x + 0.5) < n) & (np.abs(Y - y + 0.5) < n)
        inside = np.argwhere((cutout_mask).flatten()).flatten()
        radii = (np.sqrt((X - x) ** 2 + (Y - y) ** 2)).flatten()[inside]
        idxs = np.argsort(radii)
        radii = radii[idxs]
        pixels = self.stack.flatten()[inside]
        pixels = pixels[idxs]

        binned_radii, binned_pixels, _ = fast_binning(radii, pixels, bins=1)

        plt.figure(figsize=(9.5, 4))
        ax = plt.subplot(1, 5, (1, 3))

        plt.plot(radii, pixels, "o", fillstyle='none', c="0.7", ms=4)
        plt.plot(binned_radii, binned_pixels, c="k")

        apertures = self.apertures_radii[0]
        a = self.aperture
        if "annulus_rin" in self:
            rin = self.annulus_rin.mean()
            rout = self.annulus_rout.mean()
        else:
            print(
                f"{INFO_LABEL} You are probably using a last version phot file, aperture_rin/out has been, set to default AperturePhotometry value")
            rin = self.fwhm.mean() * 5
            rout = self.fwhm.mean() * 8

        plt.xlabel("distance from center (pixels)")
        plt.ylabel("ADUs")
        _, ylim = plt.ylim()
        plt.xlim(0)
        plt.text(apertures[a], ylim, "APERTURE", ha="right", rotation="vertical", va="top")
        plt.axvspan(0, apertures[a], color="0.9", alpha=0.1)

        plt.axvspan(rin, rout, color="0.9", alpha=0.1)
        plt.axvline(rin, color="k", alpha=0.1)
        plt.axvline(rout, color="k", alpha=0.1)
        plt.axvline(apertures[a], c="k", alpha=0.1)
        _ = plt.text(rout, ylim, "ANNULUS", ha="right", rotation="vertical", va="top")

        ax2 = plt.subplot(1, 5, (4, 5))
        im = self.stack[int(y - n):int(y + n), int(x - n):int(x + n)]
        if zscale:
            im = z_scale(im)
        plt.imshow(im, cmap="Greys_r", aspect="auto", origin="lower")
        plt.axis("off")
        ax2.add_patch(plt.Circle((n, n), apertures[a], ec='grey', fill=False, lw=2))
        ax2.add_patch(plt.Circle((n, n), rin, ec='grey', fill=False, lw=2))
        ax2.add_patch(plt.Circle((n, n), rout, ec='grey', fill=False, lw=2))
        plt.tight_layout()
        self.style()

    def plot_stars(self, size=8):
        self.show_stars(size=size)
        plt.tight_layout()

    def plot_syst(self, size=(6, 8)):
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111)

        self.plot_systematics()
        _ = plt.gcf().axes[0].set_title("", loc="left")
        plt.xlabel(f"BJD")
        plt.ylabel("diff. flux")
        plt.tight_layout()
        self.style()

    def plot_comps(self, size=(6, 8)):
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111)

        self.plot_comps_lcs()
        _ = plt.gcf().axes[0].set_title("", loc="left")
        plt.xlabel(f"BJD")
        plt.ylabel("diff. flux")
        plt.tight_layout()
        self.style()

    def plot_raw(self, size=(6, 4)):
        plt.figure(figsize=size)
        raw_f = self.raw_fluxes[self.aperture, self.target]
        plt.plot(self.time, raw_f / np.mean(raw_f), c="k", label="target")
        plt.plot(self.time, self.alc[self.aperture], c="C0", label="artificial")
        plt.legend()
        plt.tight_layout()
        plt.xlabel(f"BJD")
        plt.ylabel("norm. flux")
        self.style()

    def make(self, destination):
        if not path.exists(destination):
            os.mkdir(destination)
        figdest = path.join(destination, "figures")
        if not path.exists(figdest):
            os.mkdir(figdest)

        self.plot_psf();
        plt.savefig(path.join(figdest, "psf.png"), dpi=self.dpi);
        plt.close()
        self.plot_comps();
        plt.savefig(path.join(figdest, "comparison.png"), dpi=self.dpi);
        plt.close()
        self.plot_raw();
        plt.savefig(path.join(figdest, "raw.png"), dpi=self.dpi);
        plt.close()
        self.plot_syst();
        plt.savefig(path.join(figdest, "systematics.png"), dpi=self.dpi);
        plt.close()
        self.plot_stars();
        plt.savefig(path.join(figdest, "stars.png"), dpi=self.dpi);
        plt.close()
        self.plot_lc();
        plt.savefig(path.join(figdest, "lightcurve.png"), dpi=self.dpi);
        plt.close()

        shutil.copyfile(path.join(template_folder, "prose-report.cls"), path.join(destination, "prose-report.cls"))
        repdest = path.join(destination, "report.tex")
        open(repdest, "w").write(self.template.render(
            obstable=self.obstable,
            target=self.name,
            description=self.description,
        ))

    def set_trend(self, trend):
        self._trend = trend

    def set_transit(self, transit):
        self._transit = transit

    def plot_lc(self):
        plt.figure(figsize=(6, 7 if self._trend is not None else 4))
        if self._trend is not None:
            plt.plot(self.time, self.flux - 0.03, ".", color="gainsboro", alpha=0.3)
            plt.plot(self.time, self._trend - 0.03, c="k", alpha=0.2, label="systematics model")
            viz.plot_lc(self.time, self.flux - self._trend + 1.)
            plt.ylim(0.95, 1.02)
        else:
            self.plot()
            plt.ylim(0.98, 1.02)
        if self._transit is not None:
            plt.plot(self.time, self._transit + 1., label="transit", c="k")
        plt.legend()
        plt.xlabel(f"BJD")
        plt.ylabel("diff. flux")
        plt.tight_layout()
        self.style()