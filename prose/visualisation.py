import matplotlib
from .utils import binning
import numpy as np
from . import utils
import matplotlib.offsetbox
from matplotlib.lines import Line2D
from matplotlib import patches
from astropy.io import fits
from mpl_toolkits import axes_grid1
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes,  inset_axes
import matplotlib.pyplot as plt
from skimage.transform import resize
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.ticker import AutoMinorLocator
import matplotlib.patches as mpatches
from .models import transit


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
        plt.errorbar(*blc, fmt=".", zorder=1, color=bincolor, alpha=binalpha, label=binlabel)


def plot_lc_report(
    time,
    flux,
    data,
    bins=0.005,
    std=True,
    error=None,
    raw_markersize=5,
    raw_label="raw data",
    raw_color="gainsboro",
    binned_markersize=6,
    binned_color="C0",
    error_capsize=2,
    error_elinewidth=0.75,
    fields=["fwhm", "sky", "airmass", "dx", "dy"]):

    def plot_systematics(x, y, color):
        if isinstance(y, list):
            y = np.array(y)

        blc = utils.binning(time, y, bins=bins, error=error, std=True)

        plt.plot(
            x, y, ".",
            zorder=0,
            c=raw_color,
            markersize=raw_markersize,
        )
        plt.errorbar(
            *blc,
            c=color,
            fmt=".",
            capsize=error_capsize,
            elinewidth=error_elinewidth,
            markersize=binned_markersize,
        )

    _fields = []

    for i, field in enumerate(fields):
        if field in data:
            _fields.append(field)

    fig = plt.figure(figsize=(7, 2*(len(_fields)+1)))

    plt.subplot(len(_fields) + 1, 1, 1)
    plot(time, flux, binned_color=binned_color, std=std, bins=bins)
    plt.grid(color="whitesmoke")

    for i, field in enumerate(_fields):
        ax = fig.add_subplot(len(_fields)+1, 1, i+2)
        plot_systematics(time, data[field], "C0")
        plt.grid(color="whitesmoke")
        plt.annotate(
            field, (0, 1),
            xycoords="axes fraction", xytext=(-5, -5),
            textcoords="offset points",
            ha="left", va="top", fontsize=11,
        )
        if i != len(_fields):
            # Turn off tick labels
            ax.xaxis.set_visible(False)

    plt.tight_layout()
    plt.show()


def show_residuals(cut, model, imshow_kwargs={}, plot_kwargs={}):
    imshow_kwargs = dict(cmap = "inferno",**imshow_kwargs)
    plot_kwargs = dict(color="blueviolet",**plot_kwargs)

    x, y = np.indices(cut.shape)
    
    plt.subplot(131)
    plt.imshow(utils.z_scale(cut), alpha=1, **imshow_kwargs)
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.title("PSF", loc="left")

    plt.subplot(132)
    plt.imshow(utils.z_scale(model), alpha=1, **imshow_kwargs)
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.title("PSF model", loc="left")
    
    residuals = cut-model
    plt.subplot(133)
    im =plt.imshow(residuals, alpha=1, **imshow_kwargs)
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.title("Residuals", loc="left")
    viz.add_colorbar(im)


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
            ax.axis('off')


class AnchoredHScaleBar(matplotlib.offsetbox.AnchoredOffsetbox):
    """ size: length of bar in data units
        extent : height of bar ends in axes units """

    def __init__(self, size=1, extent=0.03, label="", loc=2, ax=None,
                 pad=0.4, borderpad=0.5, ppad=0, sep=2, prop=None,
                 frameon=True, linekw={}, **kwargs):

        if not ax:
            ax = plt.gca()

        trans = ax.get_xaxis_transform()
        size_bar = matplotlib.offsetbox.AuxTransformBox(trans)
        line = Line2D([0, size], [0, 0], **linekw)
        vline1 = Line2D([0, 0], [-extent / 2., extent / 2.], **linekw)
        vline2 = Line2D([size, size], [-extent / 2., extent / 2.], **linekw)
        size_bar.add_artist(line)
        size_bar.add_artist(vline1)
        size_bar.add_artist(vline2)
        txt = matplotlib.offsetbox.TextArea(label, minimumdescent=False, textprops=dict(color=linekw.get("color", "k")))
        self.vpac = matplotlib.offsetbox.VPacker(children=[size_bar, txt],
                                                 align="center", pad=ppad, sep=sep)
        matplotlib.offsetbox.AnchoredOffsetbox.__init__(self, loc, pad=pad,
                                                        borderpad=borderpad, child=self.vpac, prop=prop,
                                                        frameon=frameon,
                                                        **kwargs)


def plot_lcs(data, planets={}, W=4, show=None, hide=None, ylim=None, size=[4,3], indexes=None, colors=None, bins=0.005, force_width=True):
    """
    A global plot for multiple lightcurves
    
    Parameters
    ----------
    data : list
        a list of (time, flux)
    planets : dict, optional
        dict of planet to display as transit windows, format is:
            {"name": [t0, period, duration], ...}
        by default {}
    W : int, optional
        number of plots per row, by default 4
    show : list, optional
        indexes of light curves to show, if None all are shown, by default None
    hide : [type], optional
        indexes of light curves to hide, if None, None are hidden, by default None, by default None
    options : dict, optional
        plotting options, by default {}
    """
    
    if isinstance(data[0], dict):
        data = [[d["time"], d["lc"]] for d in data]
    
    if show is None:
        show = np.arange(0, len(data))
    if hide is None:
        hide = []
    
    idxs = np.setdiff1d(show, hide)

    H = np.ceil(len(idxs) / W).astype(int)
    if not force_width:
        W = np.min([len(idxs), W])
    fig, axes = plt.subplots(H,W,figsize=(W * size[0], H * size[1]))
    
    max_duration = np.max([jd.max() - jd.min() for jd, lc in [data[i] for i in idxs]])

    planet_colors = ["C0", "C1", "C2", "C3", "C4"]
    
    for _i, ax in enumerate(axes.flat if len(idxs) > 1 else [axes]):
        if _i < len(idxs):
            i = idxs[_i]
            plt.sca(ax)
            jd, lc = data[i]
            center = jd.min() + (jd.max() - jd.min())/2

            for pi, (planet, (t0, period, duration)) in enumerate(planets.items()):
                n_p = int((jd.max()-t0)/period)
                if (jd.min()-t0)/period < n_p < (jd.max()-t0)/period:
                    plt.plot(np.ones(2)*(n_p*period + t0 - duration/2), [0, 4], planet_colors[pi], alpha=0.4)
                    plt.plot(np.ones(2)*(n_p*period + t0 + duration/2), [0, 4], planet_colors[pi], alpha=0.4)
                    p1 = patches.Rectangle(
                        (n_p*period + t0 - duration/2, 0),
                        duration,
                        4,
                        facecolor=planet_colors[pi],
                        alpha=0.05,
                    )
                    ax.add_patch(p1)

            plot(jd, lc, bins=bins)
            if ylim is not None:
                plt.ylim(ylim)

            plt.xlim(center - (max_duration/2), center + (max_duration/2))
            # ax.get_legend().remove()
            if i%W != 0 and ylim is not None:
                ax.tick_params(labelleft=False)
            ax.set_title(None)
            if indexes is not None:
                text = str(indexes[i])
            else:
                text = str(i)
            if colors is not None:
                color = colors[i]
            else:
                color = "k"
            ax.annotate(
                    text,
                    xy=(0.05, 0.05),
                    xycoords="axes fraction",
                    textcoords="axes fraction",
                    ha="left",
                    va="bottom",
                    fontsize=10,
                    color=color,
                )
        else:
            ax.axis('off')

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
        if hasattr(axe, 'spines'):
            axe.spines['bottom'].set_color('#545454')
            axe.spines['left'].set_color('#545454')
            axe.spines['top'].set_color('#DBDBDB')
            axe.spines['right'].set_color('#DBDBDB')
            axe.spines['top'].set_linewidth(1)
            axe.spines['right'].set_linewidth(1)


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


class AnchoredHScaleBar(matplotlib.offsetbox.AnchoredOffsetbox):
    """ size: length of bar in data units
        extent : height of bar ends in axes units """

    def __init__(self, size=1, extent=0.03, label="", loc=2, ax=None,
                 pad=0.4, borderpad=0.5, ppad=0, sep=2, prop=None,
                 frameon=True, linekw={}, **kwargs):

        if not ax:
            ax = plt.gca()

        trans = ax.get_xaxis_transform()
        size_bar = matplotlib.offsetbox.AuxTransformBox(trans)
        line = Line2D([0, size], [0, 0], **linekw)
        vline1 = Line2D([0, 0], [-extent / 2., extent / 2.], **linekw)
        vline2 = Line2D([size, size], [-extent / 2., extent / 2.], **linekw)
        size_bar.add_artist(line)
        size_bar.add_artist(vline1)
        size_bar.add_artist(vline2)
        txt = matplotlib.offsetbox.TextArea(label, minimumdescent=False, textprops=dict(color=linekw.get("color", "k")))
        self.vpac = matplotlib.offsetbox.VPacker(children=[size_bar, txt],
                                                 align="center", pad=ppad, sep=sep)
        matplotlib.offsetbox.AnchoredOffsetbox.__init__(self, loc, pad=pad,
                                                        borderpad=borderpad, child=self.vpac, prop=prop,
                                                        frameon=frameon,
                                                        **kwargs)


def show_stars(image, stars=None, highlight=None, size=15, options={}, flip=None, color=None, contrast=0.05):

    if color is None:
        color = np.array([131, 220, 255]) / 255

    _options = {
        "aperture_color": "seagreen",
        "aperture_ls": "--"
    }
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

    size_factor = size/7
    fontsize = min(size_factor, 1)*15
    label_yoffset = min(size_factor, 1)*15

    if stars is not None:
        if highlight is not None:

            plt.plot(
                stars[highlight, 0],
                stars[highlight, 1],
                "o",
                markersize=14*size_factor,
                markeredgecolor=color,
                markerfacecolor="none",
                label="target",
            )
            plt.annotate(
                highlight, xy=[stars[highlight][0], stars[highlight][1] + label_yoffset],
                color=color, fontsize=fontsize, ha='center', va='top'
            )

        else:
            highlight = -1

        other_stars = np.arange(len(stars))

        other_stars = np.setdiff1d(other_stars, highlight)

        plt.plot(
            stars[other_stars, 0],
            stars[other_stars, 1],
            "o",
            markersize=14*size_factor,
            markeredgecolor=color,
            markerfacecolor="none",
            alpha=0.4 if highlight >= 0 else 1
        )

        plt.tight_layout()


class AnchoredHScaleBar(matplotlib.offsetbox.AnchoredOffsetbox):
    """ size: length of bar in data units
        extent : height of bar ends in axes units """

    def __init__(self, size=1, extent=0.03, label="", loc=2, ax=None,
                 pad=0.4, borderpad=0.5, ppad=0, sep=2, prop=None,
                 frameon=True, linekw={}, **kwargs):
        if not ax:
            ax = plt.gca()

        trans = ax.get_xaxis_transform()
        size_bar = matplotlib.offsetbox.AuxTransformBox(trans)
        line = Line2D([0, size], [0, 0], **linekw)
        vline1 = Line2D([0, 0], [-extent / 2., extent / 2.], **linekw)
        vline2 = Line2D([size, size], [-extent / 2., extent / 2.], **linekw)
        size_bar.add_artist(line)
        size_bar.add_artist(vline1)
        size_bar.add_artist(vline2)
        txt = matplotlib.offsetbox.TextArea(label, minimumdescent=False, textprops=dict(color=linekw.get("color", "k")))
        self.vpac = matplotlib.offsetbox.VPacker(children=[size_bar, txt],
                                                 align="center", pad=ppad, sep=sep)
        matplotlib.offsetbox.AnchoredOffsetbox.__init__(self, loc, pad=pad,
                                                        borderpad=borderpad, child=self.vpac, prop=prop,
                                                        frameon=frameon,
                                                        **kwargs)


def plot_marks(x, y, label=None, position="bottom", offset=7, fontsize=12, color=[0.51, 0.86, 1.], ms=12, n=None, inside=True, alpha=1):
    y_offset = ms + offset

    if position == "top":
        y_offset *= -1

    if not isinstance(x, (list, np.ndarray, tuple)):
        x = np.array([x])
        y = np.array([y])
        if label is not None:
            label = np.array([label])

    if inside:
        ax = plt.gcf().axes[0]
        xlim, ylim = np.array(ax.get_xlim()), np.array(ax.get_ylim())
        xlim.sort()
        ylim.sort()
        within = np.argwhere(np.logical_and.reduce([xlim[0] < x,  x < xlim[1],  ylim[0] < y,  y < ylim[1]])).flatten()
        x = x[within]
        y = y[within]
        if label is not None:
            label = np.array(label)[within]

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
            plt.annotate(_label, xy=[_x, _y - y_offset], color=color, ha='center', fontsize=fontsize, alpha=alpha,
                         va="top" if position is "bottom" else "bottom")


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
    """
    Plot stack image and detected stars

    Parameters
    ----------
    size: float (optional)
        pyplot figure (size, size)
    image: int (optional)
        index of image to plot in light files. Default is None, which show stack image
    contrast: float
        contrast within [0, 1] (zscale is applied here)
    marker_color: [r, g, b]
    proper_motion: bool
        whether to display proper motion on the image
    n_stars: int
        max number of stars to show
    flip: bool
        flip image
    view: 'all', 'reference'
        - ``reference`` : only highlight target and comparison stars
        - ``all`` : all stars are shown
    zoom: bool
        whether to include a zoom view
    options: dict
        style options:
            - to do

    Examples
    --------

    .. code-block:: python3

        from specphot.observations import Photometry

        phot = Photometry("your_path")
        phot.plot_stars(view="reference")

    .. image:: /user_guide/gallery/plot_stars.png
       :align: center
    """
    _options = {
        "aperture_color": "seagreen",
        "aperture_ls": "--"
    }
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
            circle = mpatches.Circle(coord, marker_size, fill=None, ec=marker_color, )
            ax = plt.gca()
            ax.add_artist(circle)
            plt.annotate(str(i),
                         xy=[coord[0], coord[1] + marker_size + 8],
                         color=marker_color,
                         ha='center', fontsize=12, va='top')

    if ref_stars is not None:
        circle = mpatches.Circle(stars[target, :], marker_size, fill=None, ec=marker_color, label="target")
        ax = plt.gca()
        ax.add_artist(circle)
        plt.annotate(target, xy=[stars[target][0], stars[target][1] + marker_size + 8],
                     color=marker_color, ha='center', fontsize=12, va='top')

        plt.imshow(image, cmap="Greys_r")

        for i in ref_stars:
            circle = mpatches.Circle(stars[i, :], marker_size, fill=None, ec="yellow", label="comparison")
            ax.add_artist(circle)
            plt.annotate(str(i), xy=[stars[i][0], stars[i][1] + marker_size+8], color="yellow",
                         fontsize=12,
                         ha='center',
                         va='top')

        other_stars = np.arange(len(stars))

        other_stars = np.setdiff1d(other_stars, target)
        other_stars = np.setdiff1d(other_stars, ref_stars)

        for i in other_stars:
            circle = mpatches.Circle(stars[i, :], marker_size, fill=None, ec=marker_color, label="comparison",
                                     alpha=0.4)
            ax.add_artist(circle)

    plt.tight_layout()

    if pixel_scale is not None:
        ob = AnchoredHScaleBar(size=60 / pixel_scale, label="1'", loc=4, frameon=False, extent=0,
                               pad=0.6, sep=4, linekw=dict(color="white", linewidth=0.8))
        ax.add_artist(ob)

    if target is not None and zoom:
        with plt.rc_context({
            'axes.edgecolor': "white",
            'xtick.color': "white",
            'ytick.color': "white"
        }):
            x, y = stars[target]
            rect = patches.Rectangle(
                (x - 80, y - 80),
                160, 160, linewidth=1,
                edgecolor='white',
                facecolor='none',
                alpha=0.3)

            ax.add_patch(rect)
            axins = zoomed_inset_axes(ax, 2.5, loc=1)
            axins.imshow(image, cmap="Greys_r", origin="upper")
            if aperture is not None:
                ap = aperture / 2
                aperture = patches.Circle(
                    (x, y),
                    ap, linewidth=1,
                    ls=_options["aperture_ls"],
                    edgecolor=_options["aperture_color"],
                    facecolor='none',
                    alpha=1)
                axins.add_patch(aperture)
            axins.set_xlim([x - 80, x + 80])
            axins.set_ylim([y + 80, y - 80])

            if pixel_scale is not None:
                obin = AnchoredHScaleBar(size=15 / pixel_scale, label="15\"", loc=4,
                                         frameon=False, extent=0, pad=0.6, sep=4,
                                         linekw=dict(color="white", linewidth=0.8))
                axins.add_artist(obin)

    return fig


def plot_comparison_lcs(lcs, idxs, bins=0.005, offset_factor=2.5, std=False):
    """Plot comparison stars light curves along target star light curve
    """
    time = lcs[0].time

    ax = plt.subplot(111)
    plt.title("Comparison diff. light curves", loc="left")
    plt.grid(color="whitesmoke")
    amp = np.percentile(lcs[0].diff_flux, 95) - np.percentile(lcs[0].diff_flux, 5)
    for i, lc in enumerate(lcs):
        lc.plot(offset=-offset_factor * amp * i, std=std)
        plt.annotate(
            "{}".format(idxs[i]), 
            (lc.time.min() + 0.005, 1.01 - offset_factor * amp * i)
            )

    plt.xlim(min(lcs[0].time), max(lcs[0].time))
    legends = ax.get_legend()
    if legends:
        legends.remove()
    plt.tight_layout()


def plot_rms(fluxes_lcs, diff_lcs, target=None, highlights=None, bins=0.005):
    fluxes_lcs.set_best_aperture_id(diff_lcs.best_aperture_id)
    lcs = diff_lcs.fluxes
    errors = diff_lcs.errors
    fluxes = fluxes_lcs.fluxes

    time = diff_lcs[0].time.copy()

    fluxes_median = np.median(fluxes, axis=1)
    stds_median = np.array([np.median(utils.binning(time, lc, bins, error=error, std=True)[2]) for lc, error in zip(lcs, errors)])
    stds_median /= fluxes_median
    errors_median = np.array([np.median(utils.binning(time, lc, bins, error=error, std=False)[2]) for lc, error in zip(lcs, errors)])
    errors_median /= fluxes_median

    plt.grid(color="whitesmoke", zorder=0)

    if highlights is not None:
        for c in highlights:
            comp_flux_median = fluxes_median[c]
            comp_std_median = stds_median[c]
            plt.plot(comp_flux_median, comp_std_median, ".", c="gold", zorder=5)
        plt.plot(comp_flux_median, comp_std_median, ".", c="gold", label="comparison stars", zorder=5)

    if target is not None:
        target_flux_median = fluxes_median[target]
        target_std_median = stds_median[target]
        plt.plot(target_flux_median, target_std_median, ".", c="C0", label="target",zorder=6, ms=10)

    idxs = np.argsort(fluxes_median)

    stds_median = stds_median[idxs]
    errors_median = errors_median[idxs]
    fluxes_median = fluxes_median[idxs]

    plt.title("Light curves binned rms", loc="left")
    plt.plot(fluxes_median, stds_median, ".", c="darkgrey", zorder=4, label="others")
    plt.plot(fluxes_median, errors_median, c="k", lw=1, zorder=7, label="CCD equation", alpha=0.8)
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("ADUs")
    plt.ylabel("diff. flux rms ({} JD bin)".format(bins))

    fluxes_median = fluxes_median[fluxes_median>0]
    plt.xlim(fluxes_median.min(), fluxes_median.max())


def gif_image_array(image, factor=0.25):
    return (utils.z_scale(
        resize(
            image.astype(float),
            (np.array(np.shape(image)) * factor).astype(int),
            anti_aliasing=False,
        )) * 255).astype("uint8")


def fancy_gif_image_array(image, median_psf, factor=0.25):

    fig = plt.figure(frameon=False)
    canvas = FigureCanvas(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    gif_im = utils.z_scale(
        resize(
            image,
            np.array(np.shape(image)).astype(int) * factor,
            anti_aliasing=True,
        ))
    ax.imshow(gif_im, cmap="Greys_r")
    axins = inset_axes(ax, width=1, height=1, loc=3)
    axins.axis('off')
    axins.imshow(median_psf)
    canvas.draw()
    width, height = (fig.get_size_inches() * fig.get_dpi()).astype(int)
    return np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)

def TeX(a, fmt='{: 0.3f}', dim=True):
    from IPython.display import display, Math
    if isinstance(a, tuple):
        a = np.array(a)
        
    shape = a.shape
    if len(shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
        
    with np.printoptions(formatter={'float': fmt.format}):
        lines = str(a).replace('[', '').replace(']', '').splitlines()
        rv = [r'\begin{bmatrix}']
        rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
        rv +=  [r'\end{bmatrix}']
    if dim:
        if len(shape) == 2:
            teX_math = "M_{" + str(shape[0]) + " \\times "+ str(shape[1]) + "}="
        elif len(shape) == 1:
            teX_math = "a_{" + str(shape[0]) + "}="
    else:
        teX_math = ""
        
    teX_math += '\n'.join(rv)
    
    display(Math(r"{}".format(teX_math)))

def plot_systematics(time, lc, data, fields=None, bins=0.005, offset_factor=1):
    amp = np.percentile(lc, 95) - np.percentile(lc, 5)
    rescale_data = []

    for field in fields:
        _data = amp*utils.rescale(data[field])
        w_std = np.mean(utils.binning(time, _data, 0.005, std=True)[2])
        _data /= w_std*10
        rescale_data.append(_data)

    max_amp = np.max([np.percentile(d, 95) - np.percentile(d, 5) for d in rescale_data])

    plot(time, lc)
    for i, rd in enumerate(rescale_data):
        plot_data = (rd/max_amp)*amp + 1 - offset_factor * amp * (i+1)
        blc = utils.binning(time, plot_data, bins=bins, std=True)

        plt.plot(
                time, plot_data, ".",
                zorder=0,
                c="gainsboro",
                markersize=5,
            )
        plt.errorbar(
                *blc,
                c="gray",
                fmt=".",
                capsize=2,
                elinewidth=0.75,
                markersize=6,
            )
        plt.annotate(
            "{}".format(fields[i]), 
            (time.min() + 0.005, np.median(plot_data)), fontsize=13)
        
    plt.title("Systematics", loc="left")
    plt.grid(color="whitesmoke")
    plt.tight_layout()


def plot_lc_raw_diff(self, binning=0.005, detrended=False, options={}, std=False):

    _options = {
        "raw_markersize": 5,
        "raw_label": "raw data",
        "raw_color": "gainsboro",
        "binned_markersize": 6,
        "binned_color": "C0",
        "binned_capsize": 2,
        "binned_elinewidth": 0.75,
        "offset": 2,
        "flux_color": "C0",
        "artificial_flux_color": "k",
        "flux_ms": 3,
        "flux_style": "."
    }

    _options.update(options)
    options = _options

    kernel_size = 2*int(binning/np.median(np.diff(self.jd)))
    kernel_size = kernel_size - kernel_size%2 + 1
    
    amp = np.percentile(self.lc.diff_flux, 95) - np.percentile(self.lc.diff_flux, 5)
    
    plt.subplot(211)
    plt.title("Differential lightcurve", loc="left")
    self.lc.plot(
        bins=binning, std=std
    )
    plt.grid(color="whitesmoke")

    plt.subplot(212)
    plt.title("Normalized flux", loc="left")
    flux = self.fluxes[self.target["id"]].diff_flux
    flux /= np.median(flux)
    plt.plot(self.time, flux, options["flux_style"], ms=options["flux_ms"], label="target", c=options["flux_color"])
    if self.artificial_lcs is not None:
        plt.plot(self.time, self.artificial_lcs[self.fluxes[0].best_aperture_id], options["flux_style"], ms=options["flux_ms"], c=options["artificial_flux_color"], label="artifical star")
    plt.legend()
    plt.grid(color="whitesmoke")
    plt.xlim([np.min(self.time), np.max(self.time)])
    plt.tight_layout()


def draw_table(pdf, table, table_start, marg=5, table_cell=(20,4)):

    pdf.set_draw_color(200,200,200)

    for i, datum in enumerate(table):
        pdf.set_font("helvetica", size=6)
        pdf.set_fill_color(249,249,249)

        pdf.rect(table_start[0] + 5, table_start[1] + 1.2 + i*table_cell[1],
                table_cell[0]*3, table_cell[1], "FD" if i%2 == 0 else "D")

        pdf.set_text_color(100,100,100)

        value = datum[1]
        if value is None:
            value = "--"
        else:
            value = str(value)

        pdf.text(
            table_start[0] + marg + 2,
            table_start[1] + marg + i*table_cell[1] - 1.2, datum[0])

        pdf.set_text_color(50,50,50)
        pdf.text(
            table_start[0] + marg + 2 + table_cell[0],
            table_start[1] + marg + i*table_cell[1] - 1.2, value)

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
        plt.plot(time, model + 1., c="grey")

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
    plt.hlines(y , t0 - duration/2, t0+duration/2, clip_on=False, zorder=100, lw=2, alpha=0.5, color=c)
    plt.text(t0, y + offset, s, ha="center", color=c, fontsize=11, alpha=0.8)
    plt.vlines(t0 - duration/2, y, y0, alpha=0.15, color=c)
    plt.vlines(t0 + duration/2, y, y0, alpha=0.15, color=c)
    plt.tight_layout()

# Debugging helpers

from astroquery.mast import Catalogs
import astropy.units as u
import numpy as np
from astropy.wcs import WCS, utils as wcsutils
from prose.telescope import Telescope
from astropy.coordinates import SkyCoord


def _show_tics(data, header=None, telescope_kw="TELESCOP", r=12*u.arcminute):
    if header is None:
        header = fits.getheader(data)
        data = fits.getdata(data)

    telescope = Telescope.from_name(header[telescope_kw])
    ra = header["RA"]
    dec = header["DEC"]
    skycoord = SkyCoord(ra, dec, frame='icrs', unit=(telescope.ra_unit, telescope.dec_unit))

    coord = skycoord
    tic_data = Catalogs.query_region(coord, r, "TIC", verbose=False)
    tic_data.sort("Jmag")

    skycoords = SkyCoord(
        ra=tic_data['ra'],
        dec=tic_data['dec'], unit="deg")

    x, y = np.array(wcsutils.skycoord_to_pixel(skycoords, WCS(header)))
    _ = show_stars(data, contrast=0.5)
    plot_marks(x, y)


