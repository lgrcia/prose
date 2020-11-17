import numpy as np
from scipy import interpolate

try:
    import celerite
except ModuleNotFoundError:
    ModuleNotFoundError("prose.simulations requires celerite (pip install celerite)")

from datetime import datetime
import uuid
from prose import LightCurve, LightCurves
import matplotlib.pyplot as plt


def protopapas2005(t, t0, duration, depth, c=20, period=1):
    _t = period * np.sin(np.pi * (t - t0) / period) / (np.pi * duration)
    return (1 - depth) + (depth / 2) * (
            2 - np.tanh(c * (_t + 1 / 2)) + np.tanh(c * (_t - 1 / 2))
    )


def variability(time, period, amplitude=None, log_S0=np.log(10), log_Q=np.log(100)):
    kernel = celerite.terms.SHOTerm(log_S0=log_S0, log_Q=log_Q, log_omega0=np.log(2 * np.pi / period))
    gp = celerite.GP(kernel)
    gp.compute(time)
    if amplitude is not None:
        return rescale(gp.sample()) * amplitude / 2
    else:
        return gp.sample()


def slice_observation(time, y=None, min_sep=0.01, y_only=False):
    split_idx = np.where(np.diff(time) > min_sep)[0] + 1

    sliced_time = np.split(time, split_idx)

    if y is None:
        return sliced_time

    else:
        sliced_y = np.split(y, split_idx)
        if not y_only:
            return [np.array([sliced_time[i], sliced_y[i]]) for i in range(len(sliced_time))]
        else:
            return sliced_y


def generate_time(duration, sampling, origin=0):
    """
    Return a 1D array representing observation time

    Parameters
    ----------
    duration: float
        duration in hours`
    sampling: float
        exposure time in seconds
    origin: float
        time origin

    Returns
    -------
    1D array

    """
    return np.arange(origin, origin + duration / 24, sampling / (60 * 60 * 24))


def rescale(f):
    return (f - np.mean(f)) / np.std(f)


def generate_systematics(
        time,
        sigma_r,
        measure_hyperparam,
        model_hyperparam,
        measurement_error):
    model_range = np.linspace(-1, 1, 100)
    model_kernel = celerite.terms.SHOTerm(log_omega0=np.log(2 * np.pi / model_hyperparam), log_S0=np.log(1), log_Q=np.log(1e3))
    gp_model = celerite.GP(model_kernel)
    gp_model.compute(model_range)
    model = sigma_r * rescale(gp_model.sample(len(model_range))[0])

    func_model = interpolate.interp1d(model_range, model, fill_value="extrapolate")

    measure_kernel = celerite.terms.SHOTerm(log_omega0=np.log(2 * np.pi / 100), log_S0=np.log(1), log_Q=np.log(measure_hyperparam))
    gp = celerite.GP(measure_kernel)
    gp.compute(time)
    measure = rescale(gp.sample(len(time))[0])
    noisy_measure = measure + np.random.normal(scale=measurement_error, size=len(measure))

    lc = func_model(measure)

    return lc, noisy_measure, model, model_range


def simulate_lcs(
        time,
        sigma_w=3e-3,
        sigma_r=0,
        variability_amplitude=0,
        variability_period=None,
        variability_S0 = 10,
        variability_Q =  100,
        transits_params=[],
        systematics_names=["dx", "dy", "sky", "airmass", "fwhm"],
        systematics_models_kernel_params=2.5,
        systematics_measures_kernel_params=5e-3,
        systematics_measures_std=0.05,
        return_models=False,
        return_LightCurve=False):

    times = slice_observation(time)

    # White noise
    white_noise = np.random.normal(loc=0, scale=sigma_w, size=len(time))

    # Red Noise
    if isinstance(sigma_r, (int, float)):
        sigma_r = np.ones(len(systematics_names)) * sigma_r

    systematics = [{name: generate_systematics(
        time,
        s_r,
        systematics_models_kernel_params,
        systematics_measures_kernel_params,
        systematics_measures_std
    ) for name, s_r in zip(systematics_names, sigma_r)} for time in times]

    red_noises = [np.sum([s[0] for s in systematic.values()], axis=0) for systematic in systematics]
    red_noise = np.hstack(red_noises)

    # Variability
    if variability_period is not None:
        variability_ = variability(time, variability_period, variability_amplitude, log_S0=np.log(variability_S0), log_Q=np.log(variability_Q))
    else:
        variability_ = np.zeros_like(time)

    variabilities = slice_observation(time, variability_, y_only=True)

    # Transit
    if transits_params:
        transits = [protopapas2005(time, t["T0"], t["duration"], t["depth"], period=t["period"]) - 1 for t in
                    transits_params]
        all_transit = np.sum(transits, axis=0)

    else:
        all_transit = np.zeros_like(time)

    all_transits = slice_observation(time, all_transit, y_only=True)

    lc = 1 + red_noise + white_noise + variability_ + all_transit
    lcs = slice_observation(time, lc, y_only=True)

    # min_depth = np.min([t["depth"] for t in transits_params]) if transits else None
    # min_duration = np.min([t["duration"] for t in transits_params]) if transits else None
    #
    # # Red Noise
    # dt = np.median(np.diff(time))
    # transit_snr_r = []
    # for transit in transits:
    #     n = len(np.argwhere(transit))
    #     Ntr =
    #
    #     transit["depth"] / (np.mean(sigma_w))
    #
    # red_noise_snr = sigma_r / np.mean(sigma_w)
    # red_noise_relative_amplitude = sigma_r / min_depth
    # red_noise_relative_timescale = sigma_r / min_depth
    #
    # # Variability
    # variability_snr = variability_amplitude / np.mean(sigma_w) if variability_amplitude else None
    # variability_relative_amplitude = variability_amplitude / min_depth if variability_amplitude else None
    # variability_relative_timescale = variability_period / ((2 / 3) * min_duration) if variability_amplitude else None
    #
    # # Transit
    # transit_snr_w = [transit["depth"] / np.mean(sigma_w) for transit in transits] if transits else None
    # transit_snr_r = [transit["depth"] / np.mean(sigma_w) for transit in transits] if transits else None
    # transit_global_snr = [transit["depth"] / (np.mean(sigma_w) + sigma_r) for transit in
    #                       transits] if transits else None

    systematics_dict_list = [{name: value[0] for name, value in sys.items()} for sys in systematics]
    errors = [np.ones_like(time) * sigma_w for time in times]

    dataset = {
        "data": {
            "times": times,
            "lcs": lcs,
            "systematics": systematics_dict_list,
            "error": errors,
        },
        "models": {
            "variabilities": variabilities,
            "transits": transits,
            "red_noises": red_noises,
            "systematics": systematics,
            "white_noise": white_noise,
        },
        "info": {
            "observations": len(times),
            "total_duration": time.max() - time.min(),
            "sampling": np.median(np.diff(times[0])),
            "transits_params": transits_params,
            "mean_sigma_w": np.mean(sigma_w),
            "sigma_w": sigma_w,
            "sigma_r": sigma_r,
            "variability_amplitude": variability_amplitude,
            "variability_period": variability_period,
            # "red_noise_snr": red_noise_snr,
            # "red_noise_relative_amplitude": red_noise_relative_amplitude,
            # "variability_snr": variability_snr,
            # "variability_relative_amplitude": variability_relative_amplitude,
            # "variability_relative_timescale": variability_relative_timescale,
            # "transit_snr": transit_snr,
            # "transit_global_snr": transit_global_snr,
        }
    }

    if return_LightCurve:
        lightcurves = []
        for i in range(len(times)):
            lc = LightCurve(times[i], [lcs[i]], [errors[i]], data=systematics_dict_list[i])
            lc.info = dataset['info']
            lc.models = {
                "variabilities": variabilities[i],
                "transits": all_transit[i],
                "red_noises": red_noises[i],
                "systematics": systematics[i],
                "white_noise": white_noise[i],
            }
            lightcurves.append(lc)

        lightcurves = LightCurves(lightcurves)
        lightcurves.info = dataset['info']
        lightcurves.models = {
                "variabilities": np.hstack(variabilities),
                "transits": transits,
                "red_noises": np.hstack(red_noises),
                "systematics": np.hstack(systematics),
                "white_noise": np.hstack(white_noise),
            }

        return lightcurves

    else:
        return dataset


def load_observations(dataset, blind=True):
    return dataset


def save_dataset(dataset, version=0):
    obsid = "simlc_v{}_{}_{}.npy".format(version, datetime.now().strftime('%Y%m%d%H%M%S'), str(uuid.uuid4()))
    np.save(obsid, [dataset])
    return obsid


def get_model(dataset, i=None):
    models = dataset["models"]

    if i is not None:
        return {
            "variability": models["variabilities"][i],
            "red_noise": models["red_noises"][i],
            "transit": models["transits"][i],
            "red_noises": "",
            "systematics": models["systematics"][i],
        }
    else:
        return {
            "variability": np.hstack(models["variabilities"]),
            "red_noise": np.hstack(models["red_noises"]),
            "transit": np.hstack(models["transits"]),
            "red_noises": "",
        }


def get_data(dataset, i=None):
    data = dataset["data"]

    if i is not None:
        return {
            "lc": data["lcs"][i],
            "time": data["times"][i],
            "systematics": data["systematics"][i],
            "error": np.ones_like(data["times"][i]) * data["error"] if isinstance(data["error"], float) else data[
                "error"]
        }
    else:
        return {
            "lc": np.hstack(data["lcs"]),
            "time": np.hstack(data["times"]),
            "systematics": np.hstack(data["systematics"]),
            "error": np.ones_like(np.hstack(data["times"])) * data["error"] if isinstance(data["error"], float) else
            data["error"]
        }


def get_info(dataset):
    return dataset["info"]


def get_observations(dataset, asarray=False):
    observations = []
    for i in range(dataset["info"]["observations"]):
        if asarray:
            d = get_data(dataset, i)
            observations.append([d["time"], d["lc"], d["error"], d["systematics"]])
        else:
            observations.append(get_data(dataset, i))
    return observations


def plot_model(time, lc, measure, model, model_range, name="systematic"):
    plt.figure(figsize=(14,3))

    # plotting simulated measurements
    plt.subplot(131)
    plt.plot(time, measure, c="k")
    plt.xlabel("time"), plt.ylabel(name)

    # plotting light curve
    plt.subplot(132)
    plt.plot(measure, lc, ".", alpha=0.3)
    plt.plot(model_range, model, c="k")
    plt.xlim(np.percentile(measure, 5), np.percentile(measure, 95))
    plt.xlabel(name), plt.ylabel("red noise flux")

    # plotting light curve
    plt.subplot(133)
    plt.plot(time, 1 + lc, c="k")
    plt.ylim([0.98, 1.02])
    plt.ylabel("ligth curve") ,plt.xlabel("time")

    plt.tight_layout()