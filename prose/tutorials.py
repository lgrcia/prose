from astropy.time import Time
from astropy.io.fits import Header
import os
from os import path
import shutil
from . import Telescope, Image
from .simulations import fits_image, ObservationSimulation
from .archive import sdss_image
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
from .simulations import source_example

example_phot = "/Users/lgrcia/data/test_data_prose/Io_2021-11-28_TOI-4508.01_g'.phot"

def image_sample(*coords):
    # example: "05 38 44.851", "+04 32 47.68",
    skycoord = SkyCoord(*coords, unit=["hourangle", "deg"])
    fov = [12, 12]*u.arcmin
    return sdss_image(skycoord, fov)

def simulate_observation(time, dflux, destination, dx=3):
    n = len(time)

    # Creating the observation
    obs = ObservationSimulation(600, Telescope.from_name("A"))
    obs.set_psf((3.5, 3.5), 45, 4)
    obs.add_stars(300, time)
    obs.set_target(0, dflux)
    obs.positions += np.random.uniform(-dx, dx, (2, n))[np.newaxis, :]

    # Cleaning the field
    obs.remove_stars(np.argwhere(obs.fluxes.mean(1) < 20).flatten())
    obs.clean_around_target(50)
    obs.save_fits(destination, calibration=True)


def disorganised_folder(destination):

    if path.exists(destination):
        shutil.rmtree(destination)

    os.mkdir(destination)

    # Telescope A with filter a
    for i in range(5):
        data = np.random.random((10, 10))
        fits_image(data, dict(JD=i, TELESCOP="A", FILTER="a"), path.join(destination, f"A-test{i}.fits"))

        # Telescope A with filter a
    for i in range(5):
        data = np.random.random((10, 10))
        fits_image(data, dict(JD=i, TELESCOP="A", FILTER="ab"), path.join(destination, f"A-test{i}-ab.fits"))

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

    for i in range(2):
        data = np.random.random((10, 10))
        fits_image(
            data,
            dict(JD=i, TELESCOP="A", FILTER="c", IMAGETYP="dark", EXPTIME=8), path.join(destination, f"A-bis-testf2_{i}_8s_dark.fits"))


def moving_object(time, destination):

    # Creating the observation
    obs = ObservationSimulation(600, Telescope.from_name("A"))
    obs.set_psf((3.5, 3.5), 45, 4)
    obs.add_stars(300, time)
    obs.positions[0] = ((obs.time/obs.time.max())[:, None]*(np.array(obs.shape)-50)).T

    # constant fluxes
    for i, f in enumerate(obs.fluxes):
        obs.fluxes[i, :] = f[0]

    # Cleaning the field
    obs.remove_stars(np.argwhere(obs.fluxes.mean(1) < 20).flatten())
    obs.save_fits(destination, calibration=True)