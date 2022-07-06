from prose.reports.core import LatexTemplate
#from .. import TESSSummary
#from .. import TransitModel

class TESSNotes( LatexTemplate):

    def __init__(self, style="paper", template_name="tess-notes.tex"):
        #TESSSummary.__init__(self, photfile, name=name, template_name=template_name, style=style)
        #TransitModel.__init__(self,photfile, transit=transit, trend=trend, expected=expected, posteriors=posteriors, rms_bin=rms_bin,)
        LatexTemplate.__init__(self, template_name, style=style)

        #self.notes_table = [
        #    ["Aperture radius", f"{(self.optimal_aperture * self.telescope.pixel_scale.to(u.arcsec)):.2f}"],
        #    ["Typical FWHM", f"{(self.mean_target_fwhm*self.telescope.pixel_scale.to(u.arcsec)):.2f}"],
        #    ["Predicted Tc", f"{(self.expected[0])} BJD-TDB"],
        #    ["Measured Tc", f"{self.posteriors['t0']}"],
        #    ["List of NEBcheck stars NOT cleared", "See NEBcheck"],
        #    ["Transit depth on target (min. flux)", f"{np.abs(min(self.transit_model)):.2e}"],
        #    ["Duration of the transit", f"{self.t14:.2f} min"],
        #    ["RMS per bin (%s min)" % f"{self.rms[1]:.1f}", f"{self.rms[0]:.2e}"],
        #    ["Meridian flip",self.meridian_flip],
        #    ["Detrending parameters", "-"],
        #    ["Comments in TTF before the observation:", "-"]
        #]
    def make(self, destination):
        self.make_report_folder(destination)
        open(self.tex_destination, "w").write(self.template.render())