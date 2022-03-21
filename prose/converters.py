import os
import re
import datetime
import pandas as pd
import shutil
from prose import io, Telescope, FitsManager
from os import path
from astropy.io import fits
from . import utils
from astropy.table import Table
import numpy as np
from astropy.time import Time
import xarray as xr
from prose.blocks.registration import xyshift, closeness
from prose.blocks import SegmentedPeaks
from .io import get_files

def get_phot_stack(folder, phot_extension, stack_extension="stack.fits"):
    return get_files(phot_extension, folder, single_list_removal=True),  get_files(stack_extension, folder, single_list_removal=True)


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
        new_folder = path.join(path.dirname(folder), fm.label)
    else:
        new_folder = destination

    if not path.exists(new_folder):
        os.mkdir(new_folder)
    shutil.copyfile(stack_fits, path.join(new_folder, "{}_stack.fits".format(fm.label)))

    hdu_list.writeto(path.join(new_folder, "{}.phots".format(fm.label)), overwrite=True)

    print("Conversion done to {}".format(new_folder))

    # TODO: if comp and target add diff lightcurve processing


def old_to_new(folder_path, destination=None, keyword_observatory="OBSERVAT"):
    """Convert an old data products folder (contianing *.phots and *_stack.fits) into .phot

    Parameters
    ----------
    folder_path : str
        path of old data products folder
    destination : str, optional
        path of converted file (must contain extension .phot), by default None and a {folder_path}.phot is created
    keyword_observatory : str, optional
        fits header where telescope name can be found, by default "OBSERVAT"
    """
    
    # READING FITS
    # ============
    if not path.isdir(folder_path):
        raise FileNotFoundError("Folder does not exist")

    # Loading unique phot file
    phot_files = get_files("*.phot*", folder_path)

    if len(phot_files) == 0:
        raise ValueError("Cannot find a phot file in this folder, should contain one")
    else:
        phot_file = phot_files[0]

    # Loading unique stack file
    stack_fits = get_files("*_stack.f*ts", folder_path)
    
    if len(stack_fits) > 1:
        raise ValueError("Several stack files present in folder, should contain one")
    elif len(stack_fits) == 1:
        stack_fits = stack_fits[0]
    else:
        stack_fits = None

    phot_dict = io.phot2dict(phot_file)

    header = phot_dict["header"]
    n_images = phot_dict.get("nimages", None)

    # Loading telescope, None if name doesn't match any
    telescope_name = header.get(keyword_observatory, None)
    telescope = Telescope.from_name(telescope_name)

    # Loading data
    data = Table(phot_dict.get("time series")).to_pandas()

    # Loading time and exposure (load data first to create self.data)
    time = Table(phot_dict.get("time series", None))
    if "jd" in time:
        time = time["jd"]
    else:
        time = phot_dict.get("jd")

    assert time is not None, "time cannot be found in this phots"

    # Loading fluxes and sort by flux if specified
    raw_fluxes = phot_dict.get("photometry", None)
    raw_errors = phot_dict.get("photometry errors", None)

    # Loading stars, target, apertures
    stars = phot_dict.get("stars", None)
    apertures_area = phot_dict.get("apertures area", None)
    annulus_area = phot_dict.get("annulus area", None)
    annulus_sky = phot_dict.get("annulus sky", None)
    target_id = header.get("targetid", None)

    # Loading light curves
    fluxes = phot_dict.get("lightcurves", None)
    errors = phot_dict.get("lightcurves errors", None)
    comparison_stars = phot_dict.get("comparison stars", None)
    artificial_lcs = phot_dict.get("artificial lcs", None)

    # WRITING XARRAY
    # ==============

    header["REDDATE"] = Time.now().to_value("fits")

    dims = ("apertures", "star", "time")

    attrs = header if isinstance(header, dict) else {}
    attrs.update(dict(
        target=-1,
        aperture=-1,
        telescope=telescope.name,
        filter=header[telescope.keyword_filter],
        exptime=header[telescope.keyword_exposure_time],
        name=header[telescope.keyword_object],
        date=str(utils.format_iso_date(header[telescope.keyword_observation_date])).replace("-", ""),
    ))

    x = xr.Dataset({
        "fluxes" if fluxes is None else "raw_fluxes": xr.DataArray(raw_fluxes, dims=dims),
        "errors" if errors is None else "raw_error": xr.DataArray(raw_errors, dims=dims)
    }, attrs=attrs)

    for key in [
        "sky",
        "fwhm",
        "fwhmx",
        "fwhmy",
        "psf_angle",
        "dx",
        "dy",
        "airmass",
        telescope.keyword_exposure_time,
        telescope.keyword_jd,
        telescope.keyword_seeing,
        telescope.keyword_ra,
        telescope.keyword_dec,
    ]:
        if key in data:
            x[key.lower()] = ('time', data[key].values)

    for value, key in [
        (apertures_area, "apertures_area"),
        (annulus_area, "annulus_area")
    ]:
        if value is not None:
            if len(value.shape) == 2:
                x[key.lower()] = (('time', 'apertures'), value)
            elif len(value.shape) == 1:
                x[key.lower()] = ('time', value)
            else:
                raise AssertionError("")

    if stack_fits is not None:
        x = x.assign_coords(stack=(('w', 'h'), fits.getdata(stack_fits)))

    x.attrs.update(utils.header_to_cdf4_dict(header))

    x = x.assign_coords(time=time)
    x = x.assign_coords(stars=(('star', 'n'), stars))

    if destination is None:
        destination = f"{folder_path}.phot"

    x.to_netcdf(destination)


def AIJ_to_phot(aij_file, telescope, destination, stack):
    """Convert an AIJ data file to a phot file

    Parameters
    ----------
    aij_file : str
        path of the AIJ file
    telescope : str
        telescope name
    destination : str
        path of the destination .phot file (must include filename)
    stack : str
        path of the corresponding stack image
    """
    df = pd.read_csv(aij_file, sep="\t")

    # getting values
    stars_ids = [eval(key.split('rel_flux_T')[-1].split("_")[0]) for key in df.keys() if
                 re.match('rel_flux_T[0-9]*$', key) is not None]
    bjd_tdb = df["BJD_TDB"]
    fluxes = np.zeros((1, len(stars_ids), len(bjd_tdb)))
    errors = np.zeros_like(fluxes)
    sky = np.zeros((len(stars_ids), len(bjd_tdb)))
    fwhm = np.zeros_like(sky)
    x = np.zeros_like(sky)
    y = np.zeros_like(sky)

    for i, j in enumerate(stars_ids):
        fluxes[0, i, :] = df[f"rel_flux_T{j}"]
        errors[0, i, :] = df[f"rel_flux_err_T{j}"]
        x[i, :] = df[f"X(IJ)_T{j}"]
        y[i, :] = df[f"Y(IJ)_T{j}"]
        fwhm[i, :] = df[f"FWHM_T{j}"]
        sky[i, :] = df[f"Sky/Pixel_T{j}"]

    stars = np.vstack([np.median(x, 1), np.median(y, 1)]).T

    if stack is not None:
        # detecting stars and figuring out the rotation to apply to stack
        sp = SegmentedPeaks()
        stack = fits.getdata(stack)

        rotated_stack = [
            stack[:, :],
            stack[::-1, :],
            stack[:, ::-1],
            stack[::-1, ::-1],
        ]
        reference_stars = sp.single_detection(stack)[0]

        close = []

        for rstack in rotated_stack:
            rstars = sp.single_detection(rstack)[0][0:30]
            close.append(closeness(stars, rstars, tolerance=500))

        stack = rotated_stack[np.argmax(close)]
        rstars = sp.single_detection(stack)[0][0:30]
        stars += xyshift(stars[0:30], rstars)

    # Defining the xarray
    obsx = xr.Dataset(dict(
        fluxes=xr.DataArray(fluxes / fluxes.mean((0, 2))[None, :, None], dims=("apertures", "star", "time")),
        errors=xr.DataArray(errors, dims=("apertures", "star", "time")),
        fwhm=xr.DataArray(np.nanmedian(fwhm, 0), dims="time"),
        dy=xr.DataArray(np.nanmedian(y, 0) - y.mean(), dims=("time")),
        dx=xr.DataArray(np.nanmedian(x, 0) - x.mean(), dims=("time")),
        sky=xr.DataArray(np.nanmedian(sky, 0), dims="time"),
        airmass=xr.DataArray(df["AIRMASS"].values, dims="time"),
        bjd_tdb=xr.DataArray(df["BJD_TDB"].values, dims=("time")),
        jd_utc=xr.DataArray(df["JD_UTC"].values, dims=("time")),
        exptime=xr.DataArray(df["EXPTIME"].values, dims=("time")),
    ), attrs=dict(
        target=0,
        aperture=-1,
        telescope=telescope,
        exptime=np.unique(df["EXPTIME"].values)[0],
        time_format='bjd_tdb'
    ), coords=dict(
        time=("time", bjd_tdb),
        stars=(("star", "n"), stars)
    ))

    if stack is not None:
        obsx.coords["stack"] = (("w", "h"), stack)

    if not destination.endswith(".phot"):
        destination = f"{destination}.phot"

    obsx.to_netcdf(destination)