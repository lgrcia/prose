import glob
from os import path
import pandas as pd
import numpy as np
from ..telescope import Telescope
from datetime import timedelta
from astropy.io import fits
from tqdm import tqdm
import os
import zipfile


def phot2dict(filename):
    hdu = fits.open(filename)
    dictionary = {h.name.lower(): h.data for h in hdu}
    dictionary["header"] = hdu[0].header

    return dictionary


def get_files(
        ext,
        folder,
        depth=0,
        return_folders=False,
        single_list_removal=True,
        none_for_empty=False,
):
    """

    Return files of specific extension in the specified folder and sub-folders

    Parameters
    ----------
    folder : str
        Folder to be analyzed
    depth : int
        Number how sub-folder layer to look into.
        0 (default) will look into current folder
        1 will look into current folder and its sub-folders
        2 will look into current folder, its sub-folders and their sub-folders
        ... etc

    Returns
    -------
    list of fits files

    """
    files = []
    for depth in range(depth + 1):
        files += glob.iglob(
            path.join(folder, "*/" * depth + "*{}".format(ext)), recursive=False
        )

    files = [path.abspath(f) for f in files if path.isfile(f)]

    if return_folders:
        folders = [path.dirname(file) for file in files]
        if single_list_removal and len(folders) == 1:
            return folders[0]
        else:
            return folders
    else:
        if single_list_removal and len(files) == 1:
            return files[0]
        elif len(files) == 0 and none_for_empty:
            return None
        else:
            return files


def set_hdu(hdu_list, value):
    key = value.name
    if key in hdu_list:
        hdu_list[key] = value
    else:
        hdu_list.append(value)


def fits_to_df(files, telescope_kw="TELESCOP", verbose=True):
    assert len(files) > 0, "Files not provided"

    last_telescope = "_"
    telescope = None
    df_list = []

    def progress(x):
        return tqdm(x) if verbose else x

    for i in progress(files):
        header = fits.getheader(i)
        telescope_name = header.get(telescope_kw, "")
        if telescope_name != last_telescope:
            telescope = Telescope.from_name(telescope_name)

        df_list.append(dict(
            path=i,
            date=header.get(telescope.keyword_observation_date, ""),
            telescope=telescope.name,
            type=header.get(telescope.keyword_image_type, "").lower(),
            target=header.get(telescope.keyword_object, ""),
            filter=header.get(telescope.keyword_filter, ""),
            dimensions=(header["NAXIS1"], header["NAXIS2"]),
            flip=header.get(telescope.keyword_flip, ""),
            jd=header.get(telescope.keyword_jd, "") + telescope.mjd,
        ))

    df = pd.DataFrame(df_list)
    df.type.loc[df.type.str.lower().str.contains(telescope.keyword_light_images)] = "light"
    df.type.loc[df.type.str.lower().str.contains(telescope.keyword_dark_images)] = "dark"
    df.type.loc[df.type.str.lower().str.contains(telescope.keyword_bias_images)] = "bias"
    df.type.loc[df.type.str.lower().str.contains(telescope.keyword_flat_images)] = "flat"
    df.telescope.loc[df.telescope.str.lower().str.contains("unknown")] = np.nan
    df.date = pd.to_datetime(df.date) - timedelta(hours=15)
    df.date = df.date.apply(lambda x: x.strftime('%Y-%m-%d'))

    return df.replace("", np.nan)


def get_new_fits(current_df, folder, depth=3):
    dirs = np.array(os.listdir(folder))
    new_dirs = dirs[np.argwhere(pd.to_datetime(dirs, errors='coerce') > pd.to_datetime(current_df.date).max()).flatten()]
    return np.hstack([get_files("*.f*ts", path.join(folder, f), depth=depth) for f in new_dirs])


def convert_old_index(df):
    new_df = df[["date", "path", "telescope", "type", "target", "filter", "dimensions", "flip", "jd"]]
    new_df.dimensions = new_df.dimensions.apply(
        lambda x: tuple(np.array(x.split("x")).astype(int) if x != np.nan else x)
    )
    return new_df


def is_zip(filename):
    return zipfile.is_zipfile(filename) or ".Z" in filename
