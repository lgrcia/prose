from os import path
import pandas as pd
import numpy as np
from prose import Telescope
from datetime import timedelta
from astropy.io import fits
from prose.io import get_files
from tqdm import tqdm
import os


def fits_to_df(files, telescope_kw="TELESCOP"):
    assert len(files) > 0, "Files not provided"

    last_telescope = "_"
    telescope = None
    df_list = []

    for i in tqdm(files):
        header = fits.getheader(i)
        telescope_name = header.get(telescope_kw, "")
        if telescope_name != last_telescope:
            telescope = Telescope.from_name(telescope_name)

        df_list.append(dict(
            path=i,
            date=header.get(telescope.keyword_observation_date, ""),
            telescope=telescope.name,
            type=header.get(telescope.keyword_image_type, ""),
            target=header.get(telescope.keyword_object, ""),
            filter=header.get(telescope.keyword_filter, ""),
            dimensions=(header["NAXIS1"], header["NAXIS2"]),
            flip=header.get(telescope.keyword_flip, ""),
            jd=header.get(telescope.keyword_jd, ""),
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
    new_dirs = dirs[np.argwhere(pd.to_datetime(dirs) > pd.to_datetime(current_df.date).max()).flatten()]
    return np.hstack([get_files("*.f*ts", path.join(folder, f), depth=depth) for f in new_dirs])


def convert_old_index(df):
    new_df = df[["date", "path", "telescope", "type", "target", "filter", "dimensions", "flip", "jd"]]
    new_df.dimensions = new_df.dimensions.apply(lambda x: tuple(np.array(x.split("x")).astype(int)))
    return new_df
