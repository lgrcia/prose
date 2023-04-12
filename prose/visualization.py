import matplotlib
import numpy as np
from . import utils
import matplotlib.offsetbox
from matplotlib.lines import Line2D
from matplotlib import patches
from astropy.io import fits
from mpl_toolkits import axes_grid1
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes
import matplotlib.pyplot as plt
from skimage.transform import resize
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.ticker import AutoMinorLocator
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches


def plot(
    time,
    flux,
    error=None,
    bins=0.005,
    std=True,
    color="gainsboro",
    bincolor="k",
    alpha=0.6,
    binalpha=0.8,
    label=None,
    binlabel=None,
):
    plt.plot(time, flux, ".", color=color, alpha=alpha, zorder=0, label=label)

    if bins is not None:
        blc = binning(time, flux, bins=bins, error=error, std=std)
        plt.errorbar(
            *blc, fmt=".", zorder=1, color=bincolor, alpha=binalpha, label=binlabel
        )


def plot_marginal_model(data, model, cmap="inferno", c="blueviolet"):
    x, y = np.indices(data.shape)

    plt.subplot(131)
    plt.imshow(utils.z_scale(data), alpha=1, cmap=cmap)
    plt.contour(model, colors="w", alpha=0.7)
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.title("PSF", loc="left")

    plt.subplot(132)
    plt.plot(y[0], np.mean(data, axis=0), c=c)
    plt.plot(y[0], np.mean(model, axis=0), "--", c="k")
    plt.xlabel("x (pixels)")
    plt.ylim(data.min() * 0.98, np.mean(data, axis=0).max() * 1.02)
    plt.title("PSF x-axis projected", loc="left")
    plt.grid(color="whitesmoke")

    plt.subplot(133)
    plt.plot(y[0], np.mean(data, axis=1), c=c)
    plt.plot(y[0], np.mean(model, axis=1), "--", c="k")
    plt.xlabel("y")
    plt.ylim(data.min() * 0.98, np.mean(data, axis=1).max() * 1.02)
    plt.title("PSF y-axis projected", loc="left")
    plt.grid(color="whitesmoke")
    plt.tight_layout()


def plot_all_cuts(cuts, W=10, cmap="magma", stars=None, stars_in=None):
    H = np.ceil(len(cuts) / W).astype(int)

    fig, axes = plt.subplots(
        H,
        W,
        figsize=(W * 2, H * 2),
        subplot_kw={"xticks": [], "yticks": []},
        gridspec_kw=dict(hspace=0.1, wspace=0.1),
    )
    for i, ax in enumerate(axes.flat):
        if i < len(cuts):
            ax.imshow(utils.z_scale(cuts[i]), cmap=cmap)
            ax.annotate(
                str(i),
                (0, 0),
                xycoords="axes fraction",
                xytext=(5, 5),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=12,
                color="w",
            )

            if stars is not None:
                for j, s in enumerate(stars_in[i][0]):
                    ax.plot(*stars_in[i][1][j], "x", c="C0")
        else:
            ax.axis("off")


class AnchoredHScaleBar(matplotlib.offsetbox.AnchoredOffsetbox):
    def __init__(
        self,
        size=1,
        extent=0.03,
        label="",
        loc=2,
        ax=None,
        pad=0.4,
        borderpad=0.5,
        ppad=0,
        sep=2,
        prop=None,
        frameon=True,
        linekw={},
        **kwargs,
    ):
        if not ax:
            ax = plt.gca()

        trans = ax.get_xaxis_transform()
        size_bar = matplotlib.offsetbox.AuxTransformBox(trans)
        line = Line2D([0, size], [0, 0], **linekw)
        vline1 = Line2D([0, 0], [-extent / 2.0, extent / 2.0], **linekw)
        vline2 = Line2D([size, size], [-extent / 2.0, extent / 2.0], **linekw)
        size_bar.add_artist(line)
        size_bar.add_artist(vline1)
        size_bar.add_artist(vline2)
        txt = matplotlib.offsetbox.TextArea(
            label, minimumdescent=False, textprops=dict(color=linekw.get("color", "k"))
        )
        self.vpac = matplotlib.offsetbox.VPacker(
            children=[size_bar, txt], align="center", pad=ppad, sep=sep
        )
        matplotlib.offsetbox.AnchoredOffsetbox.__init__(
            self,
            loc,
            pad=pad,
            borderpad=borderpad,
            child=self.vpac,
            prop=prop,
            frameon=frameon,
            **kwargs,
        )


def multiplot(
    data,
    bins=0.005,
    std=True,
    color="gainsboro",
    bincolor="k",
    alpha=0.6,
    binalpha=0.8,
    w=4,
    show=None,
    hide=None,
    ylim=None,
    size=None,
    labels=None,
    force_width=True,
):
    """Plot multiple x, y with some shared axis

    Parameters
    ----------
    data : list
        a list of (x, y)
    w : int, optional
        number of plots per row, by default 4
    show : list, optional
        indexes of light curves to show, if None all are shown, by default None
    hide : list, optional
        indexes of light curves to hide, if None none are hidden, by default None
    ylim : tuple, optional
        common ylim, by default None which auto set ylim for individual plots
    size : tuple, optional
        size of individual plots like in plt.figure, by default None for (4, 3)
    labels : list of str, optional
        list lower corner text to add in individual plots, by default None
    bins : float, optional
        bin size, by default 0.005
    force_width : bool, optional
        whether to occupy all width, by default True
    """

    if size is None:
        size = (4, 3)

    if isinstance(data[0], dict):
        data = [[d["time"], d["lc"]] for d in data]

    if show is None:
        show = np.arange(0, len(data))
    if hide is None:
        hide = []

    indexes = np.setdiff1d(show, hide)

    H = np.ceil(len(indexes) / w).astype(int)
    if not force_width:
        w = np.min([len(indexes), w])
    fig, axes = plt.subplots(H, w, figsize=(w * size[0], H * size[1]))
    fig.patch.set_facecolor("white")
    max_duration = np.max([jd.max() - jd.min() for jd, _ in [data[i] for i in indexes]])

    for _i, ax in enumerate(axes.flat if len(indexes) > 1 else [axes]):
        if _i < len(indexes):
            i = indexes[_i]
            plt.sca(ax)
            jd, lc = data[i]
            center = jd.min() + (jd.max() - jd.min()) / 2

            plot(
                jd,
                lc,
                bins=bins,
                std=std,
                bincolor=bincolor,
                color=color,
                alpha=alpha,
                binalpha=binalpha,
            )
            if ylim is not None:
                plt.ylim(ylim)

            plt.xlim(center - (max_duration / 2), center + (max_duration / 2))
            # ax.get_legend().remove()
            if i % w != 0 and ylim is not None:
                ax.tick_params(labelleft=False)
            ax.set_title(None)
            if labels is not None:
                text = str(labels[i])
            else:
                text = str(i)
            ax.annotate(
                text,
                xy=(0.05, 0.05),
                xycoords="axes fraction",
                textcoords="axes fraction",
                ha="left",
                va="bottom",
                fontsize=12,
                color="k",
            )
        else:
            ax.axis("off")

    plt.tight_layout()


def bokeh_style(xminor=True, yminor=True, axes=None):
    if axes is None:
        axes = plt.gcf().axes
    elif not isinstance(axes, list):
        axes = [axes]

    for axe in axes:
        axe.set_axisbelow(True)
        axe.set_titleweight = 500
        axe.tick_params(gridOn=True, grid_color="whitesmoke")
        if xminor:
            axe.xaxis.set_minor_locator(AutoMinorLocator())
        if yminor:
            axe.yaxis.set_minor_locator(AutoMinorLocator())
        axe.tick_params(direction="inout", which="both")
        if hasattr(axe, "spines"):
            axe.spines["bottom"].set_color("#545454")
            axe.spines["left"].set_color("#545454")
            axe.spines["top"].set_color("#DBDBDB")
            axe.spines["right"].set_color("#DBDBDB")
            axe.spines["top"].set_linewidth(1)
            axe.spines["right"].set_linewidth(1)


def paper_style(axes=None):
    if axes is None:
        axes = plt.gcf().axes
    elif not isinstance(axes, list):
        axes = [axes]

    for axe in axes:
        axe.set_axisbelow(True)
        axe.set_titleweight = 500
        axe.tick_params(gridOn=True, grid_color="whitesmoke")
        axe.xaxis.set_minor_locator(AutoMinorLocator())
        axe.yaxis.set_minor_locator(AutoMinorLocator())
        axe.xaxis.set_ticks_position("both")
        axe.yaxis.set_ticks_position("both")
        axe.tick_params(which="both", direction="in")


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0 / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def show_stars(
    image,
    stars=None,
    highlight=None,
    size=15,
    options={},
    flip=None,
    color=None,
    contrast=0.05,
):
    if color is None:
        color = np.array([131, 220, 255]) / 255

    _options = {"aperture_color": "seagreen", "aperture_ls": "--"}
    _options.update(options)

    if isinstance(image, str):
        image = fits.getdata(image)

    image_size = np.array(np.shape(image))[::-1]

    fig = plt.figure(figsize=(size, size))

    if flip:
        image = utils.z_scale(image, c=contrast)[::-1, ::-1]
        if stars is not None:
            stars = np.array(image_size) - stars
    else:
        image = utils.z_scale(image, c=contrast)

    ax = fig.add_subplot(111)
    ax.imshow(image, cmap="Greys_r")
    plt.title("Stack image", loc="left")

    size_factor = size / 7
    fontsize = min(size_factor, 1) * 15
    label_yoffset = min(size_factor, 1) * 15

    if stars is not None:
        if highlight is not None:
            plt.plot(
                stars[highlight, 0],
                stars[highlight, 1],
                "o",
                markersize=14 * size_factor,
                markeredgecolor=color,
                markerfacecolor="none",
                label="target",
            )
            plt.annotate(
                highlight,
                xy=[stars[highlight][0], stars[highlight][1] + label_yoffset],
                color=color,
                fontsize=fontsize,
                ha="center",
                va="top",
            )

        else:
            highlight = -1

        other_stars = np.arange(len(stars))

        other_stars = np.setdiff1d(other_stars, highlight)

        plt.plot(
            stars[other_stars, 0],
            stars[other_stars, 1],
            "o",
            markersize=14 * size_factor,
            markeredgecolor=color,
            markerfacecolor="none",
            alpha=0.4 if highlight >= 0 else 1,
        )

        plt.tight_layout()


def plot_marks(
    x,
    y,
    label=None,
    position="bottom",
    offset=7,
    fontsize=12,
    color=[0.51, 0.86, 1.0],
    ms=12,
    n=None,
    inside=True,
    alpha=1,
    ax=None,
):
    y_offset = ms + offset

    if position == "top":
        y_offset *= -1

    if not isinstance(x, (list, np.ndarray, tuple)):
        x = np.array([x])
        y = np.array([y])
        if label is True:
            label = np.array([0])
        elif label is not None:
            label = np.array([label])
    else:
        if label is True:
            label = np.arange(len(x))
        elif label is not None:
            label = np.array(label)

    if ax is None:
        ax = plt.gcf().axes[0]

    if inside:
        ax = ax
        xlim, ylim = np.array(ax.get_xlim()), np.array(ax.get_ylim())
        xlim.sort()
        ylim.sort()
        within = np.argwhere(
            np.logical_and.reduce([xlim[0] < x, x < xlim[1], ylim[0] < y, y < ylim[1]])
        ).flatten()
        x = x[within]
        y = y[within]
        if label is not None:
            print
            label = label[within]

    if n is not None:
        x = x[0:n]
        y = y[0:n]
        if label is not None:
            label = label[0:n]

    if label is None:
        label = [None for _ in range(len(x))]

    for _x, _y, _label in zip(x, y, label):
        circle = mpatches.Circle((_x, _y), ms, fill=None, ec=color, alpha=alpha)
        ax = plt.gca()
        ax.add_artist(circle)
        f = 5
        if _label is not None:
            plt.annotate(
                _label,
                xy=[_x, _y - y_offset],
                color=color,
                ha="center",
                fontsize=fontsize,
                alpha=alpha,
                va="top" if position == "bottom" else "bottom",
            )


def fancy_show_stars(
    image,
    stars,
    ref_stars=None,
    target=None,
    size=15,
    pixel_scale=None,
    contrast=0.05,
    aperture=None,
    marker_color=np.array([131, 220, 255]) / 255,
    proper_motion=False,
    n_stars=None,
    flip=False,
    view="all",
    zoom=True,
    options={},
):
    _options = {"aperture_color": "seagreen", "aperture_ls": "--"}
    _options.update(options)

    marker_size = 9

    if isinstance(image, str):
        image = fits.getdata(image)

    image_size = np.array(np.shape(image))[::-1]

    fig = plt.figure(figsize=(size, size))

    if flip:
        image = utils.z_scale(image, c=contrast)[::-1, ::-1]
        stars = np.array(image_size) - stars
    else:
        image = utils.z_scale(image, c=contrast)

    ax = fig.add_subplot(111)
    ax.imshow(image, cmap="Greys_r")
    plt.title("Stack image", loc="left")

    size_factor = size / 15
    fontsize = min(size_factor, 1) * 15
    label_yoffset = min(size_factor, 1) * 30

    if view == "all":
        for i, coord in enumerate(stars):
            circle = mpatches.Circle(
                coord,
                marker_size,
                fill=None,
                ec=marker_color,
            )
            ax = plt.gca()
            ax.add_artist(circle)
            plt.annotate(
                str(i),
                xy=[coord[0], coord[1] + marker_size + 8],
                color=marker_color,
                ha="center",
                fontsize=12,
                va="top",
            )

    if ref_stars is not None:
        circle = mpatches.Circle(
            stars[target, :], marker_size, fill=None, ec=marker_color, label="target"
        )
        ax = plt.gca()
        ax.add_artist(circle)
        plt.annotate(
            target,
            xy=[stars[target][0], stars[target][1] + marker_size + 8],
            color=marker_color,
            ha="center",
            fontsize=12,
            va="top",
        )

        plt.imshow(image, cmap="Greys_r")

        for i in ref_stars:
            circle = mpatches.Circle(
                stars[i, :], marker_size, fill=None, ec="yellow", label="comparison"
            )
            ax.add_artist(circle)
            plt.annotate(
                str(i),
                xy=[stars[i][0], stars[i][1] + marker_size + 8],
                color="yellow",
                fontsize=12,
                ha="center",
                va="top",
            )

        other_stars = np.arange(len(stars))

        other_stars = np.setdiff1d(other_stars, target)
        other_stars = np.setdiff1d(other_stars, ref_stars)

        for i in other_stars:
            circle = mpatches.Circle(
                stars[i, :],
                marker_size,
                fill=None,
                ec=marker_color,
                label="comparison",
                alpha=0.4,
            )
            ax.add_artist(circle)

    plt.tight_layout()

    if pixel_scale is not None:
        ob = AnchoredHScaleBar(
            size=60 / pixel_scale,
            label="1'",
            loc=4,
            frameon=False,
            extent=0,
            pad=0.6,
            sep=4,
            linekw=dict(color="white", linewidth=0.8),
        )
        ax.add_artist(ob)

    if target is not None and zoom:
        with plt.rc_context(
            {"axes.edgecolor": "white", "xtick.color": "white", "ytick.color": "white"}
        ):
            x, y = stars[target]
            rect = patches.Rectangle(
                (x - 80, y - 80),
                160,
                160,
                linewidth=1,
                edgecolor="white",
                facecolor="none",
                alpha=0.3,
            )

            ax.add_patch(rect)
            axins = zoomed_inset_axes(ax, 2.5, loc=1)
            axins.imshow(image, cmap="Greys_r", origin="upper")
            if aperture is not None:
                ap = aperture / 2
                aperture = patches.Circle(
                    (x, y),
                    ap,
                    linewidth=1,
                    ls=_options["aperture_ls"],
                    edgecolor=_options["aperture_color"],
                    facecolor="none",
                    alpha=1,
                )
                axins.add_patch(aperture)
            axins.set_xlim([x - 80, x + 80])
            axins.set_ylim([y + 80, y - 80])

            if pixel_scale is not None:
                obin = AnchoredHScaleBar(
                    size=15 / pixel_scale,
                    label='15"',
                    loc=4,
                    frameon=False,
                    extent=0,
                    pad=0.6,
                    sep=4,
                    linekw=dict(color="white", linewidth=0.8),
                )
                axins.add_artist(obin)

    return fig


def plot_rms(fluxes_lcs, diff_lcs, target=None, highlights=None, bins=0.005):
    fluxes_lcs.set_best_aperture_id(diff_lcs.best_aperture_id)
    lcs = diff_lcs.fluxes
    errors = diff_lcs.errors
    fluxes = fluxes_lcs.fluxes

    time = diff_lcs[0].time.copy()

    fluxes_median = np.median(fluxes, axis=1)
    stds_median = np.array(
        [
            np.median(utils.binning(time, lc, bins, error=error, std=True)[2])
            for lc, error in zip(lcs, errors)
        ]
    )
    stds_median /= fluxes_median
    errors_median = np.array(
        [
            np.median(utils.binning(time, lc, bins, error=error, std=False)[2])
            for lc, error in zip(lcs, errors)
        ]
    )
    errors_median /= fluxes_median

    plt.grid(color="whitesmoke", zorder=0)

    if highlights is not None:
        for c in highlights:
            comp_flux_median = fluxes_median[c]
            comp_std_median = stds_median[c]
            plt.plot(comp_flux_median, comp_std_median, ".", c="gold", zorder=5)
        plt.plot(
            comp_flux_median,
            comp_std_median,
            ".",
            c="gold",
            label="comparison stars",
            zorder=5,
        )

    if target is not None:
        target_flux_median = fluxes_median[target]
        target_std_median = stds_median[target]
        plt.plot(
            target_flux_median,
            target_std_median,
            ".",
            c="C0",
            label="target",
            zorder=6,
            ms=10,
        )

    idxs = np.argsort(fluxes_median)

    stds_median = stds_median[idxs]
    errors_median = errors_median[idxs]
    fluxes_median = fluxes_median[idxs]

    plt.title("Light curves binned rms", loc="left")
    plt.plot(fluxes_median, stds_median, ".", c="darkgrey", zorder=4, label="others")
    plt.plot(
        fluxes_median,
        errors_median,
        c="k",
        lw=1,
        zorder=7,
        label="CCD equation",
        alpha=0.8,
    )
    plt.legend()
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("ADUs")
    plt.ylabel("diff. flux rms ({} JD bin)".format(bins))

    fluxes_median = fluxes_median[fluxes_median > 0]
    plt.xlim(fluxes_median.min(), fluxes_median.max())


def gif_image_array(image, factor=0.25):
    return (
        utils.z_scale(
            resize(
                image.astype(float),
                (np.array(np.shape(image)) * factor).astype(int),
                anti_aliasing=False,
            )
        )
        * 255
    ).astype("uint8")


def fancy_gif_image_array(image, median_psf, factor=0.25):
    fig = plt.figure(frameon=False)
    canvas = FigureCanvas(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    gif_im = utils.z_scale(
        resize(
            image,
            np.array(np.shape(image)).astype(int) * factor,
            anti_aliasing=True,
        )
    )
    ax.imshow(gif_im, cmap="Greys_r")
    axins = inset_axes(ax, width=1, height=1, loc=3)
    axins.axis("off")
    axins.imshow(median_psf)
    canvas.draw()
    width, height = (fig.get_size_inches() * fig.get_dpi()).astype(int)
    return np.fromstring(canvas.tostring_rgb(), dtype="uint8").reshape(height, width, 3)


def array_to_tex(a, fmt="{: 0.3f}", dim=True):
    """Display a LaTeX matrix in notebook

    Parameters
    ----------
    a : np.ndarray
        matric to display
    fmt : str, optional
        format of matrix element, by default '{: 0.3f}'
    dim : bool, optional
        show matrix dim, by default True

    Raises
    ------
    ValueError
        [description]
    """
    from IPython.display import display, Math

    if isinstance(a, tuple):
        a = np.array(a)

    shape = a.shape
    if len(shape) > 2:
        raise ValueError("bmatrix can at most display two dimensions")

    with np.printoptions(formatter={"float": fmt.format}):
        lines = str(a).replace("[", "").replace("]", "").splitlines()
        rv = [r"\begin{bmatrix}"]
        rv += ["  " + " & ".join(l.split()) + r"\\" for l in lines]
        rv += [r"\end{bmatrix}"]
    if dim:
        if len(shape) == 2:
            teX_math = "M_{" + str(shape[0]) + " \\times " + str(shape[1]) + "}="
        elif len(shape) == 1:
            teX_math = "a_{" + str(shape[0]) + "}="
    else:
        teX_math = ""

    teX_math += "\n".join(rv)

    display(Math(r"{}".format(teX_math)))


def print_tex(tex):
    from IPython.display import display, Math

    display(Math(r"{}".format(tex)))


def plot_expected_transit(time, epoch, period, duration, depth=None, color="gainsboro"):
    tmax = time.max()
    t_epoch = tmax - epoch
    t0 = t_epoch % period
    egress = tmax - t0 + duration / 2
    ingress = tmax - t0 - duration / 2

    plt.axvspan(ingress, egress, color=color, alpha=0.02, zorder=-1)
    plt.axvline(ingress, color=color, alpha=0.3, zorder=-1, label="expected transit")
    plt.axvline(egress, color=color, alpha=0.3, zorder=-1)

    if depth is not None:
        model = transit(time, epoch, duration, depth, period=period).flatten()
        plt.plot(time, model + 1.0, c="grey")


def rename_tab(name):
    """Rename a notebook tab

    Parameters
    ----------
    name : str
        name to be used
    """
    from IPython.display import display, Javascript

    return Javascript('document.title="{}"'.format(name))


def plot_section(y, s, t0, duration, c="C0", y0=1, offset=0.002):
    plt.hlines(
        y,
        t0 - duration / 2,
        t0 + duration / 2,
        clip_on=False,
        zorder=100,
        lw=2,
        alpha=0.5,
        color=c,
    )
    plt.text(t0, y + offset, s, ha="center", color=c, fontsize=11, alpha=0.8)
    plt.vlines(t0 - duration / 2, y, y0, alpha=0.15, color=c)
    plt.vlines(t0 + duration / 2, y, y0, alpha=0.15, color=c)
    ymin = plt.ylim()[0]
    plt.vlines(t0 - duration / 2, ymin, ymin + 0.002, color=c, alpha=0.6)
    plt.vlines(t0 + duration / 2, ymin, ymin + 0.002, color=c, alpha=0.6)
    plt.tight_layout()


# Debugging helpers

from astroquery.mast import Catalogs
import astropy.units as u
import numpy as np
from astropy.wcs import WCS, utils as wcsutils
from prose.telescope import Telescope
from astropy.coordinates import SkyCoord


def _show_tics(data, header=None, telescope_kw="TELESCOP", r=12 * u.arcminute):
    if header is None:
        header = fits.getheader(data)
        data = fits.getdata(data)

    telescope = Telescope.from_name(header[telescope_kw])
    ra = header["RA"]
    dec = header["DEC"]
    skycoord = SkyCoord(
        ra, dec, frame="icrs", unit=(telescope.ra_unit, telescope.dec_unit)
    )

    coord = skycoord
    tic_data = Catalogs.query_region(coord, r, "TIC", verbose=False)
    tic_data.sort("Jmag")

    skycoords = SkyCoord(ra=tic_data["ra"], dec=tic_data["dec"], unit="deg")

    x, y = np.array(wcsutils.skycoord_to_pixel(skycoords, WCS(header)))
    _ = show_stars(data, contrast=0.5)
    plot_marks(x, y)


def plot_transit_window(epoch, period, duration, color="C0"):
    """Plot transit window over all axes

    Parameters
    ----------
    epoch : float]
        transit epoch (mid-point)
    period : float
        transit period
    duration : float
        transit duration
    color : str, optional
        color of the window plotted, by default "C0"
    """
    axes = plt.gcf().axes

    for ax in axes:
        if ax.has_data():
            xmin, xmax = ax.get_xlim()
            n_p = np.round((xmax - epoch) / period)
            if (
                (xmin - epoch - duration / 2) / period
                < n_p
                < (xmax - epoch + duration / 2) / period
            ):
                ax.axvline(n_p * period + epoch - duration / 2, color=color, alpha=0.4)
                ax.axvline(n_p * period + epoch + duration / 2, color=color, alpha=0.4)
                ax.axvspan(
                    n_p * period + epoch - duration / 2,
                    n_p * period + epoch + duration / 2,
                    facecolor=color,
                    alpha=0.05,
                )


def polynomial_trend_latex(**kwargs):
    """LateX math for a polynomial trend

    Returns
    -------
    kwargs:
        dict of variable name and order (example: dict(x=2, y=4))
    """
    monomials = [
        f"{name}" + (f"^{order}" if order > 1 else "")
        for name, order in kwargs.items()
        if order > 0
    ]
    return rf"${' + '.join(monomials)}$"


def corner_text(
    text, loc=(0.05, 0.05), va="bottom", ha="left", fontsize=12, c="k", ax=None
):
    """Plot a text on the corner of axe

    Parameters
    ----------
    text : str
        text in plt.txt
    loc : tuple, optional
        (x, y) of plt.txt, by default (0.05, 0.05)
    va : str, optional
        as in plt.txt, by default "bottom"
    ha : str, optional
        as in plt.txt, by default 'left'
    fontsize : int, optional
         as in plt.txt, by default 12
    """
    if ax is None:
        ax = plt.gca()
    ax.text(
        *loc, text, fontsize=fontsize, ha=ha, va=va, transform=ax.transAxes, color=c
    )


def plot_systematics_signal(
    x,
    y,
    systematics,
    signal=None,
    ylim=None,
    offset=None,
    figsize=(6, 7),
    signal_label=None,
):
    """Plot a systematics and signal model over data. systematics + signal is plotted on top, signal alone on detrended
    data on bottom

    Parameters
    ----------
    x : np.ndarray
    y : np.ndarray
    systematics : np.ndarray
    signal : np.ndarray
    ylim : tuple, optional
        ylim of the plot, by default None, using the dispersion of y
    offset : tuple, optional
        offset between, by default None
    figsize : tuple, optional
        figure size as in in plt.figure, by default (6, 7)
    signal_label : str, optional
        label of signal, by default None
    """
    if ylim is not None:
        amplitude = ylim[1] - ylim[0]
        offset = amplitude
        ylim = (ylim[0] - amplitude, ylim[1])
    else:
        amplitude = np.percentile(y, 95) - np.percentile(y, 5)
        amplitude *= 1.5
        offset = amplitude
    if ylim is None:
        ylim = (1 - offset - 0.9 * amplitude, 1 + 0.9 * amplitude)
    if signal is None:
        signal = np.zeros_like(x)
        has_signal = False
    else:
        has_signal = True

    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("white")
    plot(x, y, label="data", binlabel="binned data (7.2 min)")
    plt.plot(
        x,
        systematics + signal,
        c="C0",
        label=f"systematics {'+ signal' if has_signal else ''} model",
    )
    if has_signal:
        plt.plot(x, signal + 1.0 - offset, label=signal_label, c="k")
    plt.text(plt.xlim()[1] + 0.005, 1, "RAW", rotation=270, va="center")
    plot(x, y - systematics + 1.0 - offset)
    plt.text(plt.xlim()[1] + 0.005, 1 - offset, "DETRENDED", rotation=270, va="center")
    plt.ylim(ylim)


class HandlerEllipse(HandlerPatch):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Ellipse(
            xy=center, width=height + xdescent, height=height + ydescent
        )
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


def circles_legend(colors, texts, ax=None):
    if ax is None:
        ax = plt.gca()
    c = [
        mpatches.Circle((0.5, 0.5), radius=0.25, fill=None, ec=colors[i])
        for i in range(len(texts))
    ]
    ax.legend(
        c,
        texts,
        bbox_to_anchor=(1, 1.05),
        loc="upper right",
        ncol=3,
        handler_map={mpatches.Circle: HandlerEllipse()},
        frameon=False,
    )


def plot_signal(x, y, label=None, **kwargs):
    axes = plt.gcf().axes
    for i, ax in enumerate(axes):
        xmin, xmax = ax.get_xlim()
        idxs = (xmin <= x) & (x <= xmax)
        ax.plot(x[idxs], y[idxs], label=label if i == 0 else None, **kwargs)
        if label is not None and i == 0:
            ax.legend()
