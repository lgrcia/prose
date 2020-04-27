import matplotlib.pyplot as plt
from prose.utils import binning
import numpy as np
from prose import utils
import matplotlib.offsetbox
from matplotlib.lines import Line2D
from matplotlib import patches
from astropy.visualization import ZScaleInterval
from astropy.io import fits
from mpl_toolkits import axes_grid1


def z_scale(data, c=0.05):
    if type(data) == str:
        data = fits.getdata(data)
    interval = ZScaleInterval(contrast=c)
    return interval(data.copy())


def plot_lc(
    time,
    flux,
    error=None,
    bins=0.005,
    std=True,
    options={},
):
    _options = {
        "raw_markersize": 5,
        "raw_label": "raw data",
        "raw_color": "gainsboro",
        "binned_markersize": 6,
        "binned_color": "C0",
        "binned_capsize": 2,
        "binned_elinewidth": 0.75,
        "figsize": None
    }
    _options.update(options)

    if isinstance(_options["figsize"], tuple):
        plt.figure(figsize=_options["figsize"])

    plt.plot(
        time,
        flux,
        ".",
        c=_options["raw_color"],
        zorder=0,
        markersize=_options["raw_markersize"],
        label=_options["raw_label"],
    )

    if bins is not None:
        blc = binning(time, flux, bins=bins, error=error, std=std)
        plt.errorbar(
            *blc,
            fmt=".",
            capsize=_options["binned_capsize"],
            elinewidth=_options["binned_elinewidth"],
            c=_options["binned_color"],
            ecolor=_options["binned_color"],
            markersize=_options["binned_markersize"],
            label="binned data ({} JD)".format(bins),
        )

    plt.legend()


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
    plot_lc(time, flux, binned_color=binned_color, std=std, bins=bins)
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


def plot_gaussian_model(cut, params, gauss_model, options={}):
    _options = {
        "figsize": (12, 4),
        "color": "blueviolet",
        "cmap": "inferno"
    }
    _options.update(options)

    options = _options

    image, parameters = (cut, params)

    x, y = np.indices(cut.shape)

    plt.figure(figsize=_options["figsize"])

    model = gauss_model(*np.indices(cut.shape), *params)

    plt.subplot(131)
    plt.imshow(z_scale(image), alpha=1, cmap=options["cmap"])
    plt.contour(model, colors="w", alpha=0.7)
    # plt.plot(x[yo], np.ones(len(x)) * yo, "--", c="w")
    # plt.plot(np.ones(len(y)) * xo, x[xo], "--", c="w")
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    plt.title("PSF", loc="left")

    plt.subplot(132)
    plt.plot(y[0], np.mean(image, axis=0), c=options["color"])
    plt.plot(y[0], np.mean(model, axis=0), "--", c="k")
    plt.xlabel("x (pixels)")
    plt.ylim(cut.min() * 0.98, np.mean(image, axis=0).max() * 1.02)
    plt.title("PSF x-axis projected", loc="left")
    plt.grid(color="whitesmoke")

    plt.subplot(133)
    plt.plot(y[0], np.mean(image, axis=1), c=options["color"])
    plt.plot(y[0], np.mean(model, axis=1), "--", c="k")
    plt.xlabel("y")
    plt.ylim(cut.min() * 0.98, np.mean(image, axis=1).max() * 1.02)
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
            ax.imshow(z_scale(cuts[i]), cmap=cmap)
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


def plot_stars(image, stars):
    pass

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


def plot_lcs(data, planets={}, W=4, show=None, hide=None, options={}):
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
    
    _options = {
        "ylim": [0.98, 1.02]
    }
    
    _options.update(options)
    options = _options
    
    if show is None:
        show = np.arange(0, len(data))
    if hide is None:
        hide = []
    
    idxs = np.setdiff1d(show, hide)

    H = np.ceil(len(idxs) / W).astype(int)

    fig, axes = plt.subplots(
        H,
        W,
        figsize=(W * 5, H * 4),
        gridspec_kw=dict(hspace=0.6, wspace=0.08),
    )
    
    max_duration = np.max([jd.max() - jd.min() for jd, lc in [data[i] for i in idxs]])

    planet_colors = ["C0", "C1", "C2", "C3", "C4"]
    
    for _i, ax in enumerate(axes.flat):
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

            plot_lc(jd, lc, options={"figsize": None})
            plt.ylim(options["ylim"])

            plt.xlim(center - (max_duration/2), center + (max_duration/2))
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
 
            ax.get_legend().remove()
            if i%W != 0:
                plt.yticks([])
                plt.ylabel(None)
            ax.set_title(None)
            ax.annotate(
                    "observation {}".format(i),
                    (1, 0),
                    xycoords="axes fraction",
                    xytext=(7, 7),
                    textcoords="offset points",
                    ha="right",
                    va="bottom",
                    fontsize=10,
                    color="k",
                )
        else:
            ax.axis('off')

    plt.tight_layout()


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0 / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def plot_stars(
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

        .. code:: python3

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

        if isinstance(image, str):
            image =fits.getdata(image)
  
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

        if view == "all":

            plt.plot(
                stars[:, 0],
                stars[:, 1],
                "o",
                markersize=10,
                markeredgecolor=marker_color,
                markerfacecolor="none",
            )

            for i, coord in enumerate(stars):
                plt.annotate(str(i),
                             xy=[coord[0], coord[1] + 45],
                             color=marker_color,
                             ha='center')

        if ref_stars is not None:
            plt.plot(
                stars[target, 0],
                stars[target, 1],
                "o",
                markersize=12,
                markeredgecolor=marker_color,
                markerfacecolor="none",
                label="target",
            )
            plt.annotate(
                target, xy=stars[target] + 25, color=marker_color
            )

            plt.imshow(image, cmap="Greys_r")
            plt.plot(
                stars[ref_sars, 0],
                stars[ref_sars, 1],
                "o",
                markersize=12,
                markeredgecolor="yellow",
                markerfacecolor="none",
                label="comparison",
            )
            for i in ref_stars:
                plt.annotate(str(i), xy=stars[i] + 25, color="yellow")

            other_stars = np.arange(len(stars))

            other_stars = np.setdiff1d(other_stars, target)
            other_stars = np.setdiff1d(other_stars, ref_stars)

            plt.plot(
                stars[other_stars, 0],
                stars[other_stars, 1],
                "o",
                markersize=11,
                markeredgecolor=marker_color,
                markerfacecolor="none",
                alpha=0.4
            )

        plt.tight_layout()

        if pixel_scale is not None:
            ob = viz.AnchoredHScaleBar(size=60 / pixel_scale, label="1'", loc=4, frameon=False, extent=0,
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
                    obin = viz.AnchoredHScaleBar(size=15 / pixel_scale, label="15\"", loc=4,
                                                 frameon=False, extent=0, pad=0.6, sep=4,
                                                 linekw=dict(color="white", linewidth=0.8))
                    axins.add_artist(obin)