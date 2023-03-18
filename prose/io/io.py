import glob
from os import path
import pandas as pd
import numpy as np
from ..telescope import Telescope
from datetime import timedelta
from astropy.io import fits
from ..console_utils import progress, warning
from astropy.time import Time
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
    single_list_removal=False,
    none_for_empty=False,
):
    """

    Return files of specific extension in the specified folder and sub-folders

    Parameters
    ----------
    extension: str
        wildcard pattern for file extension
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
            path.join(folder, "*/" * depth + f"*{ext}"), recursive=False
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


def fits_to_df(
    files,
    telescope_kw="TELESCOP",
    instrument_kw="INSTRUME",
    telescope=None,
    verbose=True,
    hdu=0,
    raise_oserror=False,
    verbose_os=False,
):

    assert len(files) > 0, "Files not provided"

    last_telescope = "_"
    telescopes_seen = []
    _telescope = None
    df_list = []

    for i in progress(verbose, desc="Parsing FITS")(files):
        try:
            header = fits.getheader(i, hdu)
        except OSError as err:
            if verbose_os:
                warning(f"OS error for file {i}")
            if raise_oserror:
                print(f"OS error: {err}")
                raise
            else:
                continue

        if telescope is None:
            telescope_name = header.get(telescope_kw, "")
            instrument_name = header.get(instrument_kw, "")

            telescope_id = f"{telescope_name}_{instrument_name}"
            if telescope_id not in telescopes_seen:
                telescopes_seen.append(telescope_id)
                verbose = True
            else:
                verbose = False

            if telescope_id != last_telescope or _telescope is None:
                _telescope = Telescope.from_names(
                    header.get(instrument_kw, ""), header.get(telescope_kw, "")
                )
                last_telescope = telescope_id
        else:
            _telescope = telescope

        df_list.append(
            dict(
                path=i,
                date=_telescope.header_date(header).isoformat(),
                telescope=_telescope.name,
                type=_telescope.image_type(header),
                target=header.get(_telescope.keyword_object, ""),
                filter=header.get(_telescope.keyword_filter, ""),
                dimensions=(header.get("NAXIS1", 1), header.get("NAXIS2", 1)),
                flip=header.get(_telescope.keyword_flip, ""),
                jd=header.get(_telescope.keyword_jd, ""),
                exposure=float(header.get(_telescope.keyword_exposure_time, -1)),
            )
        )

    df = pd.DataFrame(
        df_list,
        columns=(
            "path",
            "date",
            "telescope",
            "type",
            "target",
            "filter",
            "dimensions",
            "flip",
            "jd",
            "exposure",
        ),
    )

    if len(df) > 0 and _telescope is not None:
        df.type.loc[
            df.type.str.lower().str.contains(_telescope.keyword_light_images.lower())
        ] = "light"
        df.type.loc[
            df.type.str.lower().str.contains(_telescope.keyword_dark_images.lower())
        ] = "dark"
        df.type.loc[
            df.type.str.lower().str.contains(_telescope.keyword_bias_images.lower())
        ] = "bias"
        df.type.loc[
            df.type.str.lower().str.contains(_telescope.keyword_flat_images.lower())
        ] = "flat"
        df.telescope.loc[df.telescope.str.lower().str.contains("unknown")] = ""
        df.date = pd.to_datetime(df.date)
        df["filter"] = df["filter"].str.replace("'", "p")

        if (df.jd == "").all():  # jd empty then convert from date
            df.jd = Time(df.date, scale="utc").to_value("jd") + _telescope.mjd

        # We want dates that correspond to same observations but night might be over 2 days (before and after midnight)
        # So we remove 15 hours to be sure the date year-month-day are consistent with single observations
        df.date = (df.date - timedelta(hours=15)).apply(
            lambda x: x.strftime("%Y-%m-%d")
        )

    return df


def get_new_fits(current_df, folder, depth=3):
    dirs = np.array(os.listdir(folder))
    new_dirs = dirs[
        np.argwhere(
            pd.to_datetime(dirs, errors="coerce")
            > pd.to_datetime(current_df.date).max()
        ).flatten()
    ]
    return np.hstack(
        [get_files("*.f*ts", path.join(folder, f), depth=depth) for f in new_dirs]
    )


def convert_old_index(df):
    new_df = df[
        [
            "date",
            "path",
            "telescope",
            "type",
            "target",
            "filter",
            "dimensions",
            "flip",
            "jd",
        ]
    ]
    new_df.dimensions = new_df.dimensions.apply(
        lambda x: tuple(np.array(x.split("x")).astype(int) if x != np.nan else x)
    )
    return new_df


def is_zip(filename):
    return zipfile.is_zipfile(filename) or ".Z" in filename
