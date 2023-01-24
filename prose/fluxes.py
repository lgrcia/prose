import numpy as np

def binned_white_function(x, bins=12):
    # set binning idxs for white noise evaluation
    bins = np.min([x.shape[-1], bins])
    n = x.shape[-1] // bins
    idxs = np.arange(n * bins)

    def compute(f):
        return np.nanmean(np.nanstd(np.array(np.split(f.take(idxs, axis=-1), n, axis=-1)), axis=-1), axis=0)
    
    return compute

def weights(fluxes, tolerance=1e-3, max_iteration=200, bins=5):
    
    # normalize
    dfluxes = fluxes/np.expand_dims(np.mean(fluxes, -1), -1)
    binned_white = binned_white_function(fluxes, bins=bins)
    
    i = 0
    evolution = 1e25
    lcs = None
    weights = None
    last_weights = np.zeros(dfluxes.shape[0:len(dfluxes.shape) - 1])

    # Broeg 2004 algorithm to find weights of comp stars
    # --------------------------------------------------
    while evolution > tolerance and i < max_iteration:
        if i == 0:
            weights = 1 / binned_white(dfluxes)
        else:
            # This metric is preferred from std to optimize over white noise and not red noise
            std = binned_white(lcs)
            weights = 1 / std

        weights[~np.isfinite(weights)] = 0

        # Keep track of weights
        evolution = np.nanstd(np.abs(np.nanmean(weights, axis=-1) - np.nanmean(last_weights, axis=-1)))

        last_weights = weights
        lcs = diff(dfluxes, weights=weights)
        i += 1
        
    return weights

def diff(fluxes, weights):
    # not to divide flux by itself
    sub = np.expand_dims((~np.eye(fluxes.shape[-2]).astype(bool)).astype(int), 0)
    dfluxes = fluxes/np.expand_dims(np.mean(fluxes, -1), -1)
    weighted_fluxes = dfluxes * np.expand_dims(weights, -1)
    art_lc = (sub @ weighted_fluxes) / np.expand_dims(weights @ sub[0], -1)
    lcs = dfluxes / art_lc
    return lcs

def auto_diff_1d(fluxes, i):
    dfluxes = fluxes / np.expand_dims(np.mean(fluxes, -1), -1)
    w = weights(dfluxes)
    idxs = np.argsort(w)[::-1]
    white_noise = binned_white_function(dfluxes)
    last_white_noise = 1e10
    
    def best_weights(j):
        _w = w.copy()
        _w[idxs[j::]] = 0.
        _w[i] = 0.
        return _w
    
    for j in range(w.shape[-1]):
        _w = best_weights(j)
        _df = diff(dfluxes, _w)
        _white_noise = np.take(white_noise(_df), i, axis=-1)[0]
        if not np.isfinite(_white_noise):
            continue
        if _white_noise < last_white_noise:
            last_white_noise = _white_noise
        else:
            break
    
    w = best_weights(j-1)
    df = diff(dfluxes, w)
        
    return df.reshape(fluxes.shape), w

def auto_diff(fluxes, i):
    if len(fluxes.shape) == 3:
        auto_diffs = [auto_diff_1d(f, i) for f in fluxes]
        w = [a[1] for a in auto_diffs]
        fluxes = np.array([a[0] for a in auto_diffs])
        return fluxes, np.array(w)
    else:
        return auto_diff_1d(fluxes, i)