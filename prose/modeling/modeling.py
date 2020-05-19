from prose import utils
import numpy as np
from functools import reduce
from itertools import combinations, product
import george
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from collections import OrderedDict
from scipy import interpolate
from scipy.interpolate import interp1d
from prose.modeling.models import ConstantModel

# TODO: write a simple modeling framework


def svd_cmp(systematics, lc):
    U, S, V = np.linalg.svd(systematics, full_matrices=False)

    S = np.diag(S)
    S[S == 0] = 1.0e10

    return reduce(np.matmul, [U.T, lc.T, 1.0 / S, V])


def square(t, t0, duration, depth):
    return np.piecewise(
        t,
        [
            t < t0 - duration / 2,
            np.logical_and(t < t0 + duration / 2, t > t0 - duration / 2),
            t > t0 + duration / 2,
        ],
        [0, -depth, 0],
    )


def protopapas2005(t, t0, duration, depth, c, period=1):
    _t = period * np.sin(np.pi * (t - t0) / period) / (np.pi * duration)
    return (1 - depth) + (depth / 2) * (
        2 - np.tanh(c * (_t + 1 / 2)) + np.tanh(c * (_t - 1 / 2))
    )


def latex_float(f, expression="{0:.1e}"):
    float_str = expression.format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \cdot 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


def find_initial_parameters(t, lc, plot=False):
    dlc = []

    binned_lc = np.array(
        [utils.binning(t, lc, _b / (24 * 60), std=True) for _b in np.arange(4, 12, 0.5)]
    )
    for blc in binned_lc:
        f = interpolate.interp1d(
            blc[0][1::], np.diff(blc[1]) ** 3, fill_value="extrapolate"
        )
        dlc.append(f(t) / np.std(f(t)))

    dlc = np.median(np.array(dlc), axis=0)

    max_dlc = np.abs(np.max(dlc))
    min_dlc = np.abs(np.min(dlc))

    if min_dlc > max_dlc:
        argmin_dlc = np.argmin(dlc)
        argmax_dlc = np.argmax(dlc[argmin_dlc::]) + argmin_dlc

        if dlc[argmax_dlc] < 0.5 * np.std(dlc):
            egr_t = t[1] + (t[1] - t[argmax_dlc])
        else:
            egr_t = t[argmax_dlc]

        ing_t = t[argmin_dlc]

    else:
        argmax_dlc = np.argmax(dlc)
        if argmax_dlc > 0:
            argmin_dlc = np.argmin(dlc[0:argmax_dlc])
        else:
            argmin_dlc = np.argmin(dlc[0])

        if dlc[argmin_dlc] > 1.5 * np.std(dlc):
            ing_t = t[0] - (t[argmin_dlc] - t[0])
        else:
            ing_t = t[argmin_dlc]

        egr_t = t[argmax_dlc]

    ing_egr = [ing_t, egr_t]
    dif_ing_egr = ing_egr[1] - ing_egr[0]
    arg_mid = int((argmax_dlc - argmin_dlc) / 2) + argmin_dlc

    real_filter_size = 0.005 / np.mean(np.diff(t))
    real_filter_size = int(real_filter_size - real_filter_size % 2 + 1)
    filtered_lc = medfilt(lc, real_filter_size)

    NN = 6

    if argmax_dlc > len(lc) + NN:
        argmax_dlc += NN
    if argmin_dlc > NN:
        argmin_dlc -= NN

    if argmax_dlc > len(lc) + 5:
        D = np.abs(filtered_lc[arg_mid] - filtered_lc[argmax_dlc + 5])
    else:
        D = np.abs(filtered_lc[arg_mid] - filtered_lc[argmin_dlc - 5])

    p0 = [dif_ing_egr / 2 + ing_egr[0], dif_ing_egr, D]

    if plot:
        plt.figure(figsize=(5, 5))
        plt.subplot(211)
        plt.plot(t, lc, ".", c="gainsboro", zorder=1)
        binned_lc = utils.binning(t, lc, 7 / (24 * 60), std=True)
        plt.errorbar(*binned_lc, fmt=".", c="C0", zorder=3)
        plt.plot(t[argmin_dlc], filtered_lc[argmin_dlc], "x", c="r")
        plt.plot(t[argmax_dlc], filtered_lc[argmax_dlc], "x", c="r")
        plt.plot(t[arg_mid], filtered_lc[arg_mid], "x", c="r")
        # plt.plot(t, tran(t, *p0), c="k")
        plt.ylim([0.98, 1.02])
        plt.subplot(212)
        plt.plot(t, dlc)

    return p0


class SVD:
    def __init__(
        self,
        phot,
        lc=None,
        data=None,
        fields=None,
        n_max=3,
        optimize=True,
        orders=None,
        criterion="AIC",
        model = ConstantModel(value=0)
    ):
        if isinstance(fields, str):
            fields = {fields: None}

        if fields is None:
            fields = OrderedDict({
                "fwhm": None,
                "sky": None,
                "dx": None,
                "dy": None,
                "airmass": None
            })
            
        else:
            fields = OrderedDict(fields)

        orders = [value for key, value in fields.items()]
        fields = [key for key, value in fields.items()]

        if not np.all(np.isfinite(np.array(orders, dtype=float))):
            orders=None
        elif np.any(np.invert(np.isfinite(np.array(orders, dtype=float)))):
           raise ValueError("Orders should all be defined")

        self.fields = fields
        self.model = model

        if not isinstance(phot, (list, np.ndarray)):
            assert phot.lc is not None, "Lightcurve is missing"
            self.flux = phot.lc
            self.time = phot.jd
            self.original_data = OrderedDict(phot.data[fields].to_dict(orient="list"))
            self.error = phot.lc_error
        else:
            self.flux = lc
            self.time = phot
            self.original_data = OrderedDict(data)
            
        self.residuals = self.flux - self.model.get_value(self.time)

        self.rescaled_data = OrderedDict(
            {field: utils.rescale(self.original_data[field]) for field in self.fields}
        )
        self.n_max = n_max

        self.best_fields = None
        self.best_orders = None
        self.singular_values = None
        self.X = None
        self.orders_idx = None
        self.criterion = criterion
        self.criteria = []

        self.same_orders = False

        if orders is not None:
            self.best_fields = self.fields
            self.best_orders = orders
            self.build_svd()
        elif optimize:
            self.optimize_orders(criterion=criterion)
            self.build_svd()

    @property
    def orders_dict(self):
        return {field: order for field, order in zip(self.best_fields, self.best_orders)}

    def _X(self, fields, orders, return_idxs=False):
        idxs_dict = None
        X = [np.ones(len(self.time))]
        if return_idxs:
            idxs_dict = {field: [] for field in fields}
        for i, field in enumerate(fields):
            for o, order in enumerate(range(1, orders[i] + 1)):
                X.append(np.power(self.rescaled_data[field], order))
                if return_idxs:
                    idxs_dict[field].append(np.shape(X)[0] - 1)
        if return_idxs:
            return np.array(np.transpose(X)), idxs_dict
        else:
            return np.array(np.transpose(X))

    def optimize_orders(self, criterion="AIC"):

        if criterion == "BIC":
            cr_function = self.bic
        elif criterion == "AIC":
            cr_function = self.aic

        fields_comb = []
        self.criteria = []

        for i in np.arange(0, len(self.fields) + 1):
            for u in combinations(self.fields, i):
                if not self.same_orders:
                    for order in list(product(range(self.n_max + 1), repeat=len(u))):
                        fields_comb.append([list(u), np.array(list(order))])
                else:
                    for o_n in range(self.n_max + 1):
                        fields_comb.append(
                            [list(u), (np.ones(len(u)) * o_n).astype(int)]
                        )
        criteria = []

        for c in fields_comb:
            fields, orders = c
            X = self._X(fields, orders)
            singular_values = svd_cmp(X, self.residuals)
            model = np.dot(singular_values, X.T) + self.model.get_value(self.time)
            n_params = (np.sum(orders + 1)) * len(fields)
            cr = cr_function(model, n_params)
            criteria.append(cr)
            self.criteria.append((fields, orders, cr))

        self.best_fields, self.best_orders = fields_comb[np.argmin(np.array(criteria))]

    def build_svd(self):
        self.X, self.orders_idx = self._X(
            self.best_fields, self.best_orders, return_idxs=True
        )
        self.singular_values = svd_cmp(self.X, self.residuals)

    def bic(self, model, k):
        n = len(self.flux)
        l = np.mean((self.flux - model) ** 2)
        # return n * np.log(l) + k * np.log(n) + n * np.log(2 * np.pi) + n
        return n * np.log(l) + k * np.log(n)

    def aic(self, model, k):
        n = len(self.flux)
        l = np.mean((self.flux - model) ** 2)
        return n * np.log(l) + 2 * k

    def _get_single_model(self, which):
        assert which in self.original_data, "'{}' is not part of your model".format(
            which
        )
        singular_values = self.singular_values[
            np.array(self.orders_idx[which]).astype("int")
        ][::-1]
        return np.poly1d([*singular_values, 0])(self.rescaled_data[which])

    def get_model(self, which="all", model=True):
        assert isinstance(
            which, (str, list)
        ), "'which' argument should be a string or a list"

        if isinstance(which, str):
            if which == "all":
                if len(self.best_fields) > 0:
                    returned = np.sum(
                        [self._get_single_model(field) for field in self.best_fields],
                        axis=0,
                    )
                else:
                    returned = np.mean(self.flux) * np.ones(len(self.time))
            else:
                returned = self._get_single_model(which)

        elif isinstance(which, list):
            for key in which:
                assert key in self.best_fields, "'{}' is not part of your model".format(
                    which
                )
            returned = np.sum(
                [
                    self._get_single_model(field)
                    for field in self.best_fields
                    if field in which
                ],
                axis=0,
            )
        
        if model:
            return returned + self.model.get_value(self.time)
        else:
            return returned

    def plot_model(self, which="all", filter_size=0, time=True):
        """
        Show models from differential lightcurve

        Parameters
        ----------
        which: string (optional)
            which model to consider for plotting

        filter_size: float or None (optional, in days)
            Apply a median filter of size `medfilter` if not 0 or None

        time: bool (optional)
            Keep time model from plots. If False, time model will be subtracted from
            the plotted differential flux and from the plotted model

        Returns
        -------

        """

        diff_flux_label = "differential flux"
        mean = self.flux

        if which == "all":
            model_label = "systematics model"
        else:
            model_label = "{} model".format(which)

        diff_flux = mean.copy()
        model = self.get_model(which)

        model = model - np.mean(model) + np.mean(mean)
        diff_flux = diff_flux - np.mean(diff_flux) + np.mean(mean)

        plt.plot(
            self.time, diff_flux, ".", c="gainsboro", zorder=0, label=diff_flux_label,
        )

        if filter_size:
            real_filter_size = filter_size / np.mean(np.diff(self.time))
            real_filter_size = int(real_filter_size - real_filter_size % 2 + 1)
            y = medfilt(model, real_filter_size)
            model_label += " (filtered)"
        else:
            y = model

        plt.plot(self.time, y, lw=1, c="k", label=model_label)

        if len(self.best_fields) > 0:
            txt = "\n".join(
                [
                    "{} : {}".format(field, order)
                    for field, order in zip(self.best_fields, self.best_orders)
                ]
            )
        else:
            txt = "no trend found"

        plt.annotate(
            txt,
            (0, 0),
            xycoords="axes fraction",
            xytext=(5, 5),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=10,
        )

        plt.legend()

    def plot_models(self, hide_extrema=5):
        # TODO: add option to separate model plotting from models sets
        
        n_plots = len(self.best_fields) + 1
        W = 3
        
        H = np.ceil(n_plots / W).astype(int)

        fig, axes = plt.subplots(H, W, figsize=(5 * W, 4 * H))
        
        for i, field in enumerate(self.best_fields):
            ax = axes.flat[i]
            if i < n_plots:
                ax.set_xlabel(field)

                model = self.get_model(field, model=False)

                diff_flux = self.flux - self.get_model(
                    [key for key in self.best_fields if key != field]
                )

                diff_flux -= np.mean(diff_flux)

                x = np.array(self.original_data[field])

                idxs = np.argsort(x)
                ax.plot(
                    x, diff_flux, ".", alpha=0.3, label="data - other models", c="C0",
                )
                ax.plot(
                    x[idxs],
                    (model - np.mean(model))[idxs],
                    c="k",
                    label="model",
                )
                ax.set_ylabel("diff. flux")
                percentile_up = 100
                percentile_down = 0

                if hide_extrema is not None:
                    percentile_up -= hide_extrema
                    percentile_down += hide_extrema

                ax.set_xlim(
                    [np.percentile(x, percentile_down), np.percentile(x, percentile_up)]
                )
                ax.set_ylim(
                    [
                        np.percentile(diff_flux, percentile_down) - 0.01,
                        np.percentile(diff_flux, percentile_up) + 0.01,
                    ]
                )
                ax.annotate(
                    r"$Polynomial(order={})$".format(self.best_orders[i]),
                    (0, 0),
                    xycoords="axes fraction",
                    xytext=(5, 5),
                    textcoords="offset points",
                    ha="left",
                    va="bottom",
                    fontsize=10,
                )
                ax.legend()
            else:
                ax.axis('off')
        
        ax = axes[n_plots - 1]
        ax.plot(
            self.time,
            self.flux - self.get_model(model=False),
            ".",
            c="gainsboro",
        )
        model = self.model.get_value(self.time)
        model = model - np.mean(model) + np.mean(self.flux)
        ax.set_title("Other model")
        ax.plot(
            self.time, model, c="k",
        )
        mean = np.mean(model)
        #ax.set_ylim([mean - 0.02, mean + 0.02])
        ax.set_xlim([np.min(self.time), np.max(self.time)])

        ax.annotate(
            "Unknown",
            (0, 0),
            xycoords="axes fraction",
            xytext=(5, 5),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=10,
        )

        plt.tight_layout()

    def get_trend(self):
        return self.get_model()