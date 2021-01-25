from .datasets import *
from astropy.time import Time
from astropy.io.fits import Header
import datetime
import os
from os import path
import shutil


def fits_image(data, header, destination):
    header = dict(
        TELESCOP=header.get("TELESCOP", "fake"),
        EXPTIME=header.get("EXPTIME", 1),
        FILTER=header.get("FILTER", ""),
        OBJECT=header.get("OBJECT", "prose"),
        IMAGETYP=header.get("IMAGETYP", "light"),
        AIRMASS=header.get("AIRMASS", 1),
        JD=header.get("JD", 0),
        RA=header.get("RA", 12.84412),
        DEC=header.get("DEC", -22.85886),
    )
    header['DATE-OBS'] = header.get("DATE-OBS", Time(datetime.datetime.now()).to_value("fits"))
    hdu = fits.PrimaryHDU(data, header=Header(header))
    hdu.writeto(destination, overwrite=True)


def disorganised_folder(destination):

    if path.exists(destination):
        shutil.rmtree(destination)

    os.mkdir(destination)


    # Telescope A with filter a
    for i in range(5):
        data = np.random.random((10, 10))
        fits_image(data, dict(JD=i, TELESCOP="A", FILTER="a"), path.join(destination, f"A-test{i}.fits"))

    # Telescope B
    for i in range(5):
        data = np.random.random((20, 10))
        fits_image(data, dict(JD=i, TELESCOP="B", FILTER="b"), path.join(destination, f"B-test{i}.fits"))

    # Telescope A with filter b
    for i in range(5):
        data = np.random.random((10, 10))
        fits_image(data, dict(JD=i, TELESCOP="A", FILTER="b"), path.join(destination, f"A-bis-test{i}.fits"))
    # some calibration files
    for i in range(2):
        data = np.random.random((10, 10))
        fits_image(
            data,
            dict(JD=i, TELESCOP="A", IMAGETYP="dark"), path.join(destination, f"A-bis-test_d{i}.fits"))
    for i in range(2):
        data = np.random.random((10, 10))
        fits_image(
            data,
            dict(JD=i, TELESCOP="A", FILTER="b", IMAGETYP="flat"), path.join(destination, f"A-bis-testf1_{i}.fits"))
    for i in range(2):
        data = np.random.random((10, 10))
        fits_image(
            data,
            dict(JD=i, TELESCOP="A", FILTER="c", IMAGETYP="flat"), path.join(destination, f"A-bis-testf2_{i}.fits"))