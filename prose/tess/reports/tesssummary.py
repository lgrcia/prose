import numpy as np
import pandas as pd
import collections
from os import path
from prose import viz
from prose.reports import Summary


class TESSSummary(Summary):

    def __init__(self, photfile, name=None, style="paper", expected=None, template_name="summary.tex"):
        Summary.__init__(self, photfile, name=name,  style=style, template_name=template_name,tess=True)
        self.obstable.insert(0, ("TIC id", self.tic_id))
        self.obstable.insert(4, ("GAIA id", self.gaia_from_toi))
        self.header = "TESS follow-up"
        self.expected = expected
        self.measurements = self.to_csv_report()

    def to_csv_report(self):
        """Export a typical csv of the observation's data
        """
        destination = path.join(self.destination, "../..", "measurements.txt")

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
        return df

    def make(self, destination):
        super().make(destination)
        self.to_csv_report()

    def plot_lc(self):
        super().plot_lc()
        if self.expected is not None:
            t0, duration = self.expected
            std = 2 * np.std(self.diff_flux)
            viz.plot_section(1 + std, "expected transit", t0, duration, c="k")

