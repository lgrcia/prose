from .core import *
from ..blocks.registration import distances
from .. import utils
import numpy as np
import matplotlib.pyplot as plt
from os import path
import shutil
from .. import viz
from .. import Observation
from ..diagnostics.near_eclipsing_binary import NEB
from .core import LatexTemplate


class NEBCheck(LatexTemplate, NEB):

    def __init__(self, obs, value, radius=2.5, style="paper", template_name="neb.tex"):
        """NEB check report page

            Parameters
            ----------
            observation : prose.Observation
                observation on which to perform the NEB check
            value : dict
                dict containing:

                - epoch
                - duration
                - period
                - depth (in ppt)
                in same time unit as observation
            radius : float, optional
               radius around the target in which to analyse other stars fluxes, by default 2.5 (in arcminutes)
            style : str, optional
                [description], by default "paper"
            template_name : str, optional
                [description], by default "neb.tex"
            """
        Observation.__init__(self, obs.xarray)
        LatexTemplate.__init__(self, template_name, style=style)
        self.radius = radius
        self.value = value
        target_distance = np.array(distances(obs.stars.T, obs.stars[obs.target]))
        self.nearby_ids = np.argwhere(target_distance * obs.telescope.pixel_scale / 60 < self.radius).flatten()

        self.nearby_ids = self.nearby_ids[np.argsort(np.array(distances(obs.stars[self.nearby_ids].T,
                                                                        obs.stars[obs.target])))]
        self.disposition_string= self.disposition.astype("str")
        self.dpi=100
        self.neb_table = None


    def plot_suspect_lcs(self, **kwargs):
        self.evaluate_score(self.value, **kwargs)
        self.plot_suspects()
        fig.patch.set_facecolor('white')
        plt.tight_layout()

    def plot_stars(self,size):
        self.show_stars(size=size)

    def neb_table(self):
        destination = path.join(self.destination, "..", self.denominator + ".csv")

        self.disposition_string[self.disposition_string == '0.0'] = "Likely cleared"
        self.disposition_string[self.disposition_string == '1.0'] = "Cleared"
        self.disposition_string[self.disposition_string == '2.0'] = "Cleared too faint"
        self.disposition_string[self.disposition_string == '3.0'] = "Flux too low"
        self.disposition_string[self.disposition_string == '4.0'] = "Not cleared"

        df = pd.DataFrame(collections.OrderedDict(
            {
                "Star": self.nearby_ids,
                "RMS (ppt)": self.rms_ppt,
                "Expected depth (ppt)": self.expected_depth,
                "Disposition": self.disposition,
            }
        self.neb_table = df.values.tolist()

    def make_figures(self, destination):
        self.plot_suspect_lcs()
        plt.savefig(path.join(destination, "suspects.png"), dpi=self.dpi)
        plt.close()
        self.plot_stars()
        plt.savefig(path.join(destination, "stars.png"), dpi=self.dpi)
        plt.close()

    def make(self, destination):
        self.make_report_folder(destination)
        self.make_figures(self.figure_destination)
        open(self.tex_destination, "w").write(self.template.render(
            obstable=self.neb_table
        ))
        self.to_csv_report()








