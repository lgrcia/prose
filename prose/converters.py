import os
from os import path
from prose import io
import numpy as np
import re
import datetime
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from prose import telescope
from prose import CONFIG, FitsManager
import shutil

def get_phot_stack(folder, phot_extension, stack_extension="stack.fits"):
    return io.get_files(phot_extension, folder), io.get_files(stack_extension, folder)

def trapphot_to_prose(folder, destination=None):

    def get_param(param, string, head=None):
        return re.findall((head if head is not None else "") + param + r".*", string)[
            0
        ].split()[-1]


    def string_date2datetime(string):
        try:
            return datetime.datetime.strptime(string, "%Y-%m-%dT%H:%M:%S.%f")
        except ValueError:
            return datetime.datetime.strptime(string, "%Y-%m-%dT%H:%M:%S")



    def read_tp_photfile(photfile, ape=None, opt_filter=None, n_stars=None, box=[40, 2000]):
        with open(photfile, "r") as phot:
            r = phot.read()

        header_string = re.findall(
            r"(?sm)(?:(#\sTarget.*?)(?=#\sImage|\Z))", r, re.MULTILINE
        )[0]

        header_params = [
            ("Target", lambda string: str(string)),
            # ('Date', lambda string: datetime.strptime(string, '%Y%m%d')),
            ("Date", lambda string: datetime.datetime.strptime(string, "%d-%m-%Y")),
            ("N_good_images", lambda string: int(string)),
            ("N_tot_images", lambda string: int(string)),
            ("Mean_FWHM_pm", lambda string: float(string)),
            ("Mean_FWHM_pp", lambda string: float(string)),
            ("Mean_background", lambda string: float(string)),
            ("Mean_read_noise", lambda string: float(string)),
            ("Mean_gain", lambda string: float(string)),
            ("Nstars", lambda string: int(string)),
        ]

        try:
            phot_header = {
                param.upper(): f(get_param(param, header_string))
                for param, f in header_params
            }
        except ValueError:
            phot_header = {
                param.upper(): f(get_param(param, header_string))
                for param, f in [
                    ("Target", lambda string: str(string)),
                    ("Date", lambda string: datetime.datetime.strptime(string, "%Y%m%d")),
                    ("N_good_images", lambda string: int(string)),
                    ("N_tot_images", lambda string: int(string)),
                    ("Mean_FWHM_pm", lambda string: float(string)),
                    ("Mean_FWHM_pp", lambda string: float(string)),
                    ("Mean_background", lambda string: float(string)),
                    ("Mean_read_noise", lambda string: float(string)),
                    ("Mean_gain", lambda string: float(string)),
                    ("Nstars", lambda string: int(string)),
                ]
            }

        phot_header["MFWHM"] = phot_header["MEAN_FWHM_PM"]
        phot_header["DATE-OBS"] = str(phot_header["DATE"])

        star_pix_coord_string = re.findall(
            r"(?sm)(?<=#\sNstars:).*", header_string, re.MULTILINE
        )[0].split("\n")[1::]
        star_pix_coord_string = [p.split("  ") for p in star_pix_coord_string]

        stars = [
            {"x": float(p[0]), "y": float(p[1])}
            for i, p in enumerate(star_pix_coord_string)
            if len(p) == 2
        ]

        images_string = re.findall(
            r"(?sm)(?:(#\sImage.*?)(?=#\sImage|\Z))", r, re.MULTILINE
        )

        params = [
            ("Image", lambda string: str(string)),
            #("Date-obs", lambda string: string_date2datetime(string)),
            ("Exptime", lambda string: float(string)),
            ("Gain", lambda string: float(string)),
            ("Readnoise", lambda string: float(string)),
            #("Filter", lambda string: str(string)),
            ("Airmass", lambda string: float(string)),
            ("JD", lambda string: float(string)),
            ("FWHM", lambda string: float(string)),
            ("FWHM2", lambda string: float(string)),
            ("Sky", lambda string: float(string)),
            ("DX", lambda string: float(string)),
            ("DY", lambda string: float(string)),
        ]

        images = [
            {param.lower(): f(get_param(param, image_string)) for param, f in params}
            for image_string in images_string
        ]

        if ape is None:
            ape = 1

        for image in images:
            image["apertures area"] = np.pi * ape ** 2

        n = int(len(images) / 8)
        all_images = [images[(ape - 1) * n : ape * n] for ape in range(1, 9)]
        all_images_string = [images_string[(ape - 1) * n : ape * n] for ape in range(1, 9)]

        _x = np.array([s["x"] for s in stars])
        _y = np.array([s["y"] for s in stars])

        # Find stars that are too close to the border
        # within_box = np.where(
        #     np.logical_and.reduce((box[0] < _x, _x < box[1], box[0] < _y, _y < box[1]))
        # )[0]
        #
        # stars = [stars[s] for s in within_box]

        images_df = pd.DataFrame.from_dict(all_images[0])
        stars_df = pd.DataFrame.from_dict(stars)

        if opt_filter is not None:
            right_filter_dfid = images_df["filter"] == opt_filter
            images_df = images_df[right_filter_dfid]

        fluxes = np.zeros((8, len(stars), len(images_df)))  # stars_flux[i] = flux of star i
        for a in range(8):
            for i, images_string in enumerate(all_images_string[a]):
                str_flux = images_string.split("#")[-1].split("\n")[1:-1]
                fluxes[a, :, i] = np.array([float(f) for f in str_flux])  # [within_box]

        return images_df, stars_df, fluxes, phot_header

    phot_file, stack_fits = get_phot_stack(folder, ".phot", "stack.fits")

    data, stars, fluxes, header = read_tp_photfile(phot_file)

    n_images = fluxes.shape[-1]

    fluxes_error = np.array([[np.ones(n_images)*np.std(flux) for flux in fluxes_ap] for fluxes_ap in fluxes]) * np.sqrt(n_images)

    hdu_list = fits.HDUList()

    io.set_hdu(hdu_list, fits.PrimaryHDU(header=fits.getheader(stack_fits)))
    io.set_hdu(hdu_list, fits.ImageHDU(fluxes, name="photometry"))
    io.set_hdu(hdu_list, fits.ImageHDU(fluxes, name="photometry errors"))
    io.set_hdu(hdu_list, fits.ImageHDU(stars, name="stars"))
    io.set_hdu(hdu_list, fits.BinTableHDU(Table.from_pandas(data), name="time series"))

    fm = FitsManager(folder)
    fm.set_observation(0)

    if destination is None:
        new_folder = path.join(path.dirname(folder), fm.products_denominator)
    else:
        new_folder = destination

    if not path.exists(new_folder):
        os.mkdir(new_folder)
    shutil.copyfile(stack_fits, path.join(new_folder, "{}_stack.fits".format(fm.products_denominator)))

    hdu_list.writeto(path.join(new_folder, "{}.phots".format(fm.products_denominator)), overwrite=True)

    print("Conversion done to {}".format(new_folder))

    # TODO: if comp and target add diff lightcurve processing
