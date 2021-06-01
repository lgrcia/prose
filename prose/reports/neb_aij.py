from .core import *
from ..blocks.registration import distances
from .. import utils
import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt
from os import path
from pathlib import Path
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
        #Observation.__init__(self, obs.xarray)
        LatexTemplate.__init__(self, template_name, style=style)
        self.radius = radius
        NEB.__init__(self, obs, radius=self.radius)
        self.value = value

        self.disposition_string = None
        self.dpi = 100
        self.evaluate_score(self.value)
        self.obstable = None
        self.lcs = []

    def plot_neb_lcs(self, destination, indexes, disposition, transparent=True, w=True):
        """Plot all the light curves of a given list of stars with the expected transits.

                    Parameters
                    ----------
                    destination : str
                        Path to save the images
                    indexes : list
                        List of indexes corresponding to star numbers
                    disposition: str
                       Can be "suspects" for only the not completely cleared light curves or "all" for cleared and not
                       cleared
                    transparent : bool
                        Whether to save the images with transparent background or not
                    w : bool
                        Number of columns to plot
                    """
        if w is True:
            if len(indexes) > 24:
                split = [indexes[:24], *np.array([indexes[i:i + 6 * 6] for i in range(24, len(indexes), 6 * 6)])]
            else:
                split = indexes
        else:
            if len(indexes) > 24:
                split = [indexes[:24], *np.array([indexes[i:i + 6 * 4] for i in range(24, len(indexes), 6 * 4)])]
            else:
                split = indexes
        path_destination = path.join(destination, disposition)
        Path(path_destination).mkdir(parents=True, exist_ok=True)
        for i, idxs in enumerate(split):
            lcs_path = path.join(destination, disposition, "lcs{}.png".format(i))
            if i > 0:
                self.lcs.append(lcs_path)
            if w is True:
                if i == 0:
                    self.plot(idxs)
                else:
                    self.plot(idxs, w=6)
            else:
                self.plot(idxs)
            viz.paper_style()
            fig = plt.gcf()
            fig.patch.set_facecolor('white')
            plt.tight_layout()
            plt.savefig(lcs_path, dpi=self.dpi, transparent=transparent)
            plt.close()

    def plot_stars(self,size=8):
        """
        Visualization of the star dispositions on the zoomed stack image.
                Parameters
                ----------
                size : int
        """
        self.show_stars(size=size)

    def plot_dmag_rms(self):
        """
        Plot of delta magnitude as a function of RMS.
        """
        fig = plt.figure(figsize=(6, 4))
        fig.patch.set_facecolor('white')
        dmag = np.arange(min(self.dmags), max(self.dmags), 0.01)
        for i in dmag:
            if i == 0:
                corr_dmag = dmag
            else:
                corr_dmag = dmag - 0.5
        x = np.power(10, - corr_dmag / 2.50)
        expected_depth = self.depth / x
        depth_factor3 = expected_depth / 3
        depth_factor5 = expected_depth / 5
        plt.plot(dmag, depth_factor3, '.', color='plum', alpha=0.5, label='Likely cleared boundary')
        plt.plot(dmag, depth_factor5, '.', color='mediumturquoise', alpha=0.5, label="cleared boundary")
        plt.plot(self.dmags, self.rmss_ppt, ".",color="0.4")
        for i,j,k in zip(self.dmags, self.rmss_ppt,self.nearby_ids):
            plt.annotate('%s' %k, xy=(i,j))
        plt.xlabel('Dmag')
        plt.ylabel('RMS (ppt)')
        plt.grid(color="whitesmoke")
        plt.legend()
        plt.tight_layout()
        self.style()

    def make_tables(self, destination):
        """
        Create a . txt table with the information on each star that was checked : star number, gaia id, tic id, RA/DEC, distance
        to target, dmag, rms, expected transit depth, ratio of rms to expected transit depth and disposition
                Parameters
                ----------
                destination : str
                    Path to store the table.
        """
        self.disposition_string = self.disposition.astype("str")
        for i, j in zip(['0.0', '1.0', '2.0', '3.0', '4.0'],
                        ["Likely cleared", "Cleared", "Cleared too faint", "Flux too low", "Not cleared"]):
            self.disposition_string[self.disposition_string == i] = j

        self.query_tic(cone_radius=2.5)
        tic_stars = np.vstack((np.array(self.tic_data['x']), np.array(self.tic_data['y']))).T
        idxs = []
        for i in self.stars[self.nearby_ids]:
            distance = np.linalg.norm((i - tic_stars), axis=1)
            _id = np.argmin(distance)
            if distance[_id] < 3 and not _id in idxs:
                idxs.append(np.argmin(distance))
            else:
                idxs.append(np.nan)
        list_tic = []
        list_gaia = []
        list_ra = []
        list_dec = []
        list_dist = []
        for j in idxs:
            if j is not np.nan:
                list_tic.append(self.tic_data['ID'][j])
                list_gaia.append(self.tic_data['GAIA'][j])
                list_ra.append(self.tic_data['ra'][j])
                list_dec.append(self.tic_data['dec'][j])
                list_dist.append(self.tic_data['dstArcSec'][j])

        df = pd.DataFrame(collections.OrderedDict(
            {
                "Star number": self.nearby_ids,
                "GAIA ID": list_gaia,
                "TIC ID": list_tic,
                "RA (deg)": list_ra,
                "DEC (deg)": list_dec,
                "Distance to target (arcsec)": list_dist,
                "Dmag": self.dmags,
                "RMS (ppt)": self.rmss_ppt,
                "Expected depth (ppt)": self.expected_depths,
                "RMS/expected depth":self.depths_rms,
                "Disposition": self.disposition_string,
            }))
        for c in ["Distance to target (arcsec)", "Dmag", "RMS (ppt)", "Expected depth (ppt)", "RMS/expected depth"]:
            df[c] = df[c].round(decimals=3)
        destination_path = Path(destination)
        df.to_csv(path.join(destination_path,"neb_table.txt"), sep="\t", index=False)
        self.obstable = [["Cleared", "Likely Cleared", "Cleared too faint", "Not cleared", "Flux too low"],
                         [len(self.cleared),len(self.likely_cleared),len(self.cleared_too_faint),len(self.not_cleared),
                          len(self.flux_too_low)]
                         ]
        return self.obstable

    def make_figures(self, destination, transparent=True, disposition='suspects', w=True):
        """
        Create the figures needed for the report : light curve plots, zoomed stack image, dmag vs rms plot.
                Parameters
                ----------
                destination : str
                        Path to save the images
                disposition: str
                   Can be "suspects" for only the not completely cleared light curves or "all" for cleared and not
                   cleared
                transparent : bool
                    Whether to save the images with transparent background or not
                w : bool
                    Number of columns to plot
        """
        if disposition == 'suspects':
            self.plot_neb_lcs(destination, indexes=self.suspects, disposition="suspects",transparent=transparent,w=w)
        elif disposition == 'all':
            self.plot_neb_lcs(destination, indexes=self.nearby_ids, disposition="all",transparent=transparent,w=w)
        else:
            self.plot_neb_lcs(destination, indexes=self.suspects, disposition="suspects", transparent=transparent, w=w)
            self.plot_neb_lcs(destination, indexes=self.nearby_ids, disposition="all", transparent=transparent, w=w)
        self.plot_stars()
        plt.savefig(path.join(destination, "neb_stars.png"), dpi=self.dpi,transparent=transparent)
        plt.close()
        self.plot_dmag_rms()
        plt.savefig(path.join(destination, "dmag_rms.png"), dpi=self.dpi, transparent=transparent)
        plt.close()

    def make(self, destination):
        """
        Automatically build the NEB check report
                Parameters
                ----------
                destination : str
                        Path to save the report
        """
        self.make_report_folder(destination)
        self.make_figures(self.figure_destination)
        open(self.tex_destination, "w").write(self.template.render(
            obstable=self.make_tables(destination), lcs=self.lcs
        ))








