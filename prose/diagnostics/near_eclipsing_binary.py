from prose.blocks.registration import distances
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from prose import utils
import matplotlib.patches as mpatches
import prose.visualisation as viz
import os
from os import path
import shutil
from astropy.time import Time
from astropy import units as u


def protopapas2005(t, t0, duration, depth, c, period=1):
    _t = period * np.sin(np.pi * (t - t0) / period) / (np.pi * duration)
    return (1 - depth) + (depth / 2) * (
            2 - np.tanh(c * (_t + 1 / 2)) + np.tanh(c * (_t - 1 / 2))
    )


def template_transit(t, t0, duration):
    return protopapas2005(t, t0, duration, 1, 50, 100)


class NearEclipsingBinary:

    def __init__(self, phot, t0, duration, radius=2.5):
        self.phot = phot
        self.radius = radius
        target_distance = np.array(distances(phot.stars.T, phot.stars[phot.target_id]))
        self.nearby_ids = np.argwhere(target_distance * phot.telescope.pixel_scale / 60 < self.radius).flatten()

        self.nearby_ids = self.nearby_ids[np.argsort(np.array(distances(phot.stars[self.nearby_ids].T, phot.stars[phot.target_id])))]

        self.time = self.phot.time
        self.t0 = t0
        self.duration = duration
        self.X = np.hstack([
            utils.rescale(self.time)[:, None] ** np.arange(0, 3),
            template_transit(self.time, self.t0, self.duration)[:, None]
        ])
        self.XXT_inv = np.linalg.inv(self.X.T @ self.X)

        self.score = np.ones(len(self.nearby_ids)) * -1
        self.depths = np.ones(len(self.nearby_ids)) * -1
        self.ws = np.ones((len(self.nearby_ids), self.X.shape[1]))

        self.evaluate_score()
        self.cmap =['r', 'g']

    def evaluate_transit(self, lc, error):
        w = (self.XXT_inv @ self.X.T) @ lc
        dw = np.var(lc)*len(lc) * self.XXT_inv
        return w, dw

    def evaluate_score(self):
        target_score = None
        for i, i_star in enumerate(self.nearby_ids):
            lc = self.phot.lcs[i_star].flux
            error = self.phot.lcs[i_star].error
            w, dw = self.evaluate_transit(lc, error)
            self.ws[i] = w
            self.depths[i], self.score[i] = w[-1], w[-1]/np.sqrt(np.diag(dw))[-1]
            if i_star == self.phot.target_id:
                target_score = self.score[i]

        self.score[self.score < 0] = 0
        self.score = np.abs(self.score)
        self.score /= target_score

    def plot_lc(self, i):
        viz.plot_lc(self.time, self.phot.lcs[self.nearby_ids[i]].flux, std=True)
        plt.plot(self.time, self.X @ self.ws[i], label="model")
        plt.legend()

    def show(self, size=15):
        marker_size = 12
        marker_color = np.array([131, 220, 255]) / 255
        image = fits.getdata(self.phot.stack_fits)
        image_size = np.array(np.shape(image))[::-1]

        fig = plt.figure(figsize=(size, size))
        image = utils.z_scale(image, c=0.05)

        search_radius = 60*self.radius/self.phot.telescope.pixel_scale
        target_coord = self.phot.stars[self.phot.target_id]
        circle = mpatches.Circle(
            target_coord,
            search_radius,
            fill=None,
            ec="white",
            alpha=0.6)

        plt.imshow(image, cmap="Greys_r")
        plt.title("Stack image", loc="left")

        ax = plt.gca()
        ax.add_artist(circle)
        plt.annotate("radius {}'".format(self.radius),
                     xy=[target_coord[0], target_coord[1] - search_radius - 15],
                     color="white",
                     ha='center', fontsize=12, va='bottom', alpha=0.6)

        for coord in self.phot.stars:
            circle = mpatches.Circle(coord, marker_size, fill=None, ec=marker_color, alpha=0.4)
            ax = plt.gca()
            ax.add_artist(circle)

        for i, coord in enumerate(self.phot.stars[self.nearby_ids]):
            if self.nearby_ids[i] == self.phot.target_id:
                color = marker_color
            else:
                color = self.color(i)
            circle = mpatches.Circle(coord, marker_size, fill=None, ec=color)
            ax = plt.gca()
            ax.add_artist(circle)
            plt.annotate(str(self.nearby_ids[i]),
                         xy=[coord[0], coord[1] - marker_size - 6],
                         color=color,
                         ha='center', fontsize=12, va='bottom')

        plt.tight_layout()
        ax = plt.gca()

        if self.phot.telescope.pixel_scale is not None:
            ob = viz.AnchoredHScaleBar(size=60 / self.phot.telescope.pixel_scale,
                                       label="1'", loc=4, frameon=False, extent=0,
                                       pad=0.6, sep=4, linekw=dict(color="white", linewidth=0.8))
            ax.add_artist(ob)

        plt.ylim(target_coord[1] + search_radius + 100, target_coord[1] - search_radius - 100)
        plt.xlim(target_coord[0] - search_radius - 100, target_coord[0] + search_radius + 100)

    def color(self, i, white=False):
        if self.nearby_ids[i] == self.phot.target_id:
            return 'k'
        elif self.score[i] >= 0.8:
            return "firebrick"
        elif self.score[i] >= 0.5:
            return "goldenrod"
        else:
            if white:
                return "olivedrab"
            else:
                return "yellowgreen" #np.array([131, 220, 255]) / 255 #np.array([78, 144, 67])/255

    def plot(self, idxs=None, **kwargs):
        if idxs is None:
            idxs = np.arange(len(self.nearby_ids))

        nearby_ids = self.nearby_ids[idxs]
        viz.plot_lcs(
            [(self.time, self.phot.lcs[i].flux) for i in nearby_ids],
            indexes=nearby_ids,
            colors=[self.color(idxs[i], white=True) for i in range(len(nearby_ids))],
            **kwargs
        )
        axes = plt.gcf().get_axes()
        for i, axe in enumerate(axes):
            if i < len(nearby_ids):
                if nearby_ids[i] == self.phot.target_id:
                    color = "k"
                else:
                    color = self.color(idxs[i], white=True)
                axe.plot(self.time, self.X @ self.ws[idxs[i]], c=color)

    def save_report(self, destination=None, remove_temp=True):

        if destination is None:
            destination = self.phot.folder

        def draw_table(table, table_start, marg=5, table_cell=(20, 4)):

            pdf.set_draw_color(200, 200, 200)

            for i, datum in enumerate(table):
                pdf.set_font("helvetica", size=6)
                pdf.set_fill_color(249, 249, 249)

                pdf.rect(table_start[0] + 5, table_start[1] + 1.2 + i * table_cell[1],
                         table_cell[0] * 3, table_cell[1], "FD" if i % 2 == 0 else "D")

                pdf.set_text_color(100, 100, 100)

                value = datum[1]
                if value is None:
                    value = "--"
                else:
                    value = str(value)

                pdf.text(
                    table_start[0] + marg + 2,
                    table_start[1] + marg + i * table_cell[1] - 1.2, datum[0])

                pdf.set_text_color(50, 50, 50)
                pdf.text(
                    table_start[0] + marg + 2 + table_cell[0]*1.2,
                    table_start[1] + marg + i * table_cell[1] - 1.2, value)

        if path.isdir(destination):
            file_name = "{}_NEB_{}arcmin.pdf".format(self.phot.products_denominator, self.radius)
        else:
            file_name = path.basename(destination.strip(".html").strip(".pdf"))

        temp_folder = path.join(path.dirname(destination), "temp")

        if path.isdir("temp"):
            shutil.rmtree(temp_folder)

        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)

        os.mkdir(temp_folder)

        star_plot = path.join(temp_folder, "starplot.png")
        self.show(10)
        fig = plt.gcf()
        fig.patch.set_alpha(0)
        plt.savefig(star_plot)
        plt.close()

        lcs = []
        a = np.arange(len(self.nearby_ids))
        if len(self.nearby_ids) > 30:
            split = [np.arange(0, 30), *np.array([a[i:i + 7*8] for i in range(30, len(a), 7*8)])]
        else:
            split = [np.arange(0, len(self.nearby_ids))]

        for i, idxs in enumerate(split):
            lcs_path = path.join(temp_folder, "lcs{}.png".format(i))
            lcs.append(lcs_path)
            if i == 0:
                self.plot(np.arange(0, np.min([30, len(self.nearby_ids)])), W=5)
            else:
                self.plot(idxs, W=8)
            viz.paper_style()
            fig = plt.gcf()
            fig.patch.set_alpha(0)
            plt.savefig(lcs_path)
            plt.close()

        lcs = np.array(lcs)

        plt.figure(figsize=(10, 3.5))
        psf_p = self.phot.plot_psf_fit(cmap="viridis", c="C0")

        psf_fit = path.join(temp_folder, "psf_fit.png")
        plt.savefig(psf_fit, dpi=60)
        plt.close()
        theta = psf_p["theta"]
        std_x = psf_p["std_x"]
        std_y = psf_p["std_y"]

        marg_x = 10
        marg_y = 8

        pdf = viz.prose_FPDF(orientation='L', unit='mm', format='A4')
        pdf.add_page()

        pdf.set_draw_color(200, 200, 200)

        pdf.set_font("helvetica", size=12)
        pdf.set_text_color(50, 50, 50)
        pdf.text(marg_x, 10, txt="{}".format(self.phot.target["name"]))

        pdf.set_font("helvetica", size=6)
        pdf.set_text_color(50, 50, 50)
        pdf.text(240, 15, txt="Nearby Eclipsing Binary diagnostic")

        pdf.set_font("helvetica", size=6)
        pdf.set_text_color(74, 144, 255)
        pdf.text(marg_x, 17, txt="simbad")
        pdf.link(marg_x, 15, 8, 3, self.phot.simbad)

        pdf.set_text_color(150, 150, 150)
        pdf.set_font("Helvetica", size=7)
        pdf.text(marg_x, 14, txt="{} · {} · {}".format(
            self.phot.observation_date, self.phot.telescope.name, self.phot.filter))

        datetimes = Time(self.phot.jd, format='jd', scale='utc').to_datetime()
        min_datetime = datetimes.min()
        max_datetime = datetimes.max()

        obs_duration = "{} - {} [{}h{}]".format(
            min_datetime.strftime("%H:%M"),
            max_datetime.strftime("%H:%M"),
            (max_datetime - min_datetime).seconds // 3600,
            ((max_datetime - min_datetime).seconds // 60) % 60)

        max_psf = np.max([std_x, std_y])
        min_psf = np.min([std_x, std_y])
        ellipticity = (max_psf ** 2 - min_psf ** 2) / max_psf ** 2

        draw_table([
            ["Time", obs_duration],
            ["RA Dec", "{:.4f} {:.4f}".format(
                self.phot.skycoord.ra,
                self.phot.skycoord.dec)],
            ["images", len(self.time)],
            ["GAIA id", None],
            ["mean fwhm", "{:.2f} pixels ({:.2f}\")".format(np.mean(self.phot.fwhm),
            np.mean(self.phot.fwhm)*self.phot.telescope.pixel_scale)],
            ["Telescope", self.phot.telescope.name],
            ["Filter", self.phot.filter],
            ["exposure", "{} s".format(np.mean(self.phot.data.exptime))],
        ], (5 + 12, 20 + 100))

        draw_table([
            ["stack PSF fwhm (x)", "{:.2f} pixels ({:.2f}\")".format(psf_p["fwhm_x"],
            psf_p["fwhm_x"]*self.phot.telescope.pixel_scale)],
            ["stack PSF fwhm (y)", "{:.2f} pixels ({:.2f}\")".format(psf_p["fwhm_y"],
            psf_p["fwhm_y"]*self.phot.telescope.pixel_scale)],
            ["stack PSF model", "Moffat2D"],
            ["stack PSF ellipicity", "{:.2f}".format(ellipticity)],
            ["diff. flux std", "{:.3f} ppt (5 min bins)".format(
                np.mean(utils.binning(self.phot.time, self.phot.lc.flux, 5/(24*60), std=True)[2])*1e3)]
        ], (5 + 12, 78 + 100))

        pdf.image(psf_fit, x=5.5 + 12, y=55 + 100, w=65)
        pdf.image(star_plot, x=5, y=20, h=93.5)
        pdf.image(lcs[0], x=100, y=22, w=185)

        for lcs_path in lcs[1::]:
            pdf.add_page()
            pdf.image(lcs_path, x=5, y=22, w=280)

        pdf_path = path.join(destination, "{}.pdf".format(file_name.strip(".html").strip(".pdf")))
        pdf.output(pdf_path)

        if path.isdir("temp") and remove_temp:
            shutil.rmtree(temp_folder)

        print("report saved at {}".format(pdf_path))

