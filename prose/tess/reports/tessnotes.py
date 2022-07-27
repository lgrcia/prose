from prose.reports.core import LatexTemplate
import astropy.units as u
import numpy as np

class TESSNotes( LatexTemplate):

    def __init__(self, obs, t_model, style="paper", template_name="tess-notes.tex"):

        LatexTemplate.__init__(self, template_name, style=style)
        self.obs = obs
        self.t_model = t_model
        self.notes_table = [
            ["Aperture radius", f"{(self.obs.optimal_aperture * self.obs.telescope.pixel_scale.to(u.arcsec)):.2f}"],
            ["Typical FWHM", f"{(self.obs.mean_target_psf*self.obs.telescope.pixel_scale.to(u.arcsec)):.2f}"],
            ["Predicted Tc", f"{(self.obs.ttf_priors['jd_mid'])} BJD-TDB"],
            ["Measured Tc", f"{self.t_model.posteriors['t0']-2450000:.4f} BJD-TDB"],
            #["List of NEBcheck stars NOT cleared", "See NEBcheck"],
            ["Transit depth on target (min. flux)", f"{np.abs(min(self.t_model.transit_model)) * 1e3:.1f} ppt"],
            ["Duration of the transit", f"{self.t_model.t14:.2f} min"],
            ["RMS per bin (%s min)" % f"{self.t_model.rms[1]:.1f}", f"{self.t_model.rms[0]:.2e}"],
            #["Meridian flip",self.obs.meridian_flip],
            ["Detrending parameters", f"{self.obs.detrends}"],
            ["Comments in TTF before the observation:", f"{(self.obs.ttf_priors['Comments'])}"]
        ]
    def make(self, destination):
        self.make_report_folder(destination)
        open(self.tex_destination, "w").write(self.template.render(
            notes_table=self.notes_table
        ))