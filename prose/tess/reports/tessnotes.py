from prose.reports.core import LatexTemplate
import astropy.units as u
import numpy as np
#from .. import TESSSummary
#from .. import TransitModel

class TESSNotes( LatexTemplate):

    def __init__(self,obs_t0,comments,obs,mean_target_fwhm,optimal_aperture,mean_fwhm,posteriors,style="paper", template_name="tess-notes.tex"):
        #TESSSummary.__init__(self, photfile, name=name, template_name=template_name, style=style)
        #TransitModel.__init__(self,photfile, transit=transit, trend=trend, expected=expected, posteriors=posteriors, rms_bin=rms_bin,)
        LatexTemplate.__init__(self, template_name, style=style)
        comments=comments.replace('σ', 'sigma')
        comments=comments.replace('Δ', 'delta')
        self.obs=obs
        self.mean_fwhm = mean_fwhm
        #self._compute_psf_model(star=self.target)
        self.mean_target_fwhm = mean_target_fwhm
        self.optimal_aperture = optimal_aperture
        self.posteriors=posteriors
        self.notes_table = [
            ["Aperture radius", f"{(self.optimal_aperture*0.34*u.arcsec):.2f}"],
            ["Typical FWHM", f"{(self.mean_target_fwhm*0.34*u.arcsec):.2f}"],
            ["Predicted Tc", f"{(obs_t0)} BJD-TDB"],
            ["Measured Tc", f"{self.posteriors['t0']}"],
            #["List of NEBcheck stars NOT cleared", "See NEBcheck"],
            ["Transit depth on target (min. flux)", f"{np.abs(min(self.obs.transit_model)):.2e}"],
            ["Duration of the transit", f"{self.obs.t14:.2f} min"],
            ["RMS per bin (%s min)" % f"{self.obs.rms[1]:.1f}", f"{self.obs.rms[0]:.2e}"],
            #["Meridian flip",self.meridian_flip],
            #["Detrending parameters", "-"],
            ["Comments in TTF before the observation:", comments]
        ]
    def make(self, destination):
        self.make_report_folder(destination)
        open(self.tex_destination, "w").write(self.template.render(notes=self.notes_table))