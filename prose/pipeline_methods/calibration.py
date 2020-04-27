def calibration(image, exp_time, master_bias, master_dark, master_flat):
    """
    Return calibration image corrected for bias, dark and flat

    Parameters
    ----------
    image : ndarray
        image to calibrate
    exp_time : ndarray
        exposure time in time_unit
    master_bias : ndarray
        master bias in ADU/time_unit
    master_dark : ndarray
        master dark in ADU/time_unit
    master_flat : ndarray
        master flat in ADU/time_unit

    Returns
    -------
    ndarray
        Calibrated frame
    """
    return (image - (master_dark * exp_time + master_bias)) / master_flat


def no_calibration(image, *args):
    """
    No calibration, return image

    Parameters
    ----------
    image : ndarray
        image

    Returns
    -------
    ndarray
        image
    """
    return image.astype('float64')