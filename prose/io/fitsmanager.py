import shutil
from os import path
import pandas as pd
import numpy as np
from tabulate import tabulate
from collections import OrderedDict
from ..telescope import Telescope
from .. import CONFIG
from datetime import timedelta
import os
from pathlib import Path
from tqdm import tqdm
from .io import get_files, fits_to_df
import re


def sub(s):
    return re.sub("[\W_]+", "", s).lower()


def clean(df):
    return df.astype(str).apply(lambda s: sub(s))


class FilesDataFrame:
    """
    TODO: should use https://pandas.pydata.org/pandas-docs/stable/development/extending.html
    """
    def __init__(self, files_df, verbose=True):
        self.files_df = None
        self.verbose = verbose

        self.files_df = files_df.reset_index(drop=True)
        self._original_files_df = self.files_df.copy()

        assert len(self._original_files_df) != 0, "No data found"
        self.telescope = None
        self.sort_by_date()

    def restore(self):
        self.files_df = self._original_files_df.copy()
        self.sort_by_date()

    #     def __getattribute__(self)

    @staticmethod
    def _get(files_df, return_conditions=False, **kwargs):
        conditions = pd.Series(np.ones(len(files_df)).astype(bool))

        for field, value in kwargs.items():
            if isinstance(value, str):
                conditions = conditions & clean(files_df[field]).str.contains(sub(value)).reset_index(drop=True)
            else:
                conditions = conditions & (files_df[field] == value).reset_index(drop=True)

        if return_conditions:
            return conditions.reset_index(drop=True)
        else:
            return files_df.reset_index(drop=True).loc[conditions]

    def get(self, return_conditions=False, **kwargs):
        """Filter the current dataframe by values of its columns

        Parameters
        ----------
        return_conditions : bool, optional
            Wether to return a bool DataArray matching the filters, by default False
        **kwargs: dict
            dict of column-value filters to be applied. Example: telescope="A", filter="I" ... etc

        Returns
        -------
        dataframe
        """
        return self._get(self.files_df, return_conditions=return_conditions, **kwargs)

    def keep(self, inplace=True, **kwargs):
        """Same as get but can replace the inplace dataframe with filtered values and sort by date

        Parameters
        ----------
        inplace : bool, optional
            weather to replace dataframe inplace, by default True
        """

        new_df = self.get(**kwargs)
        if inplace:
            self.files_df = new_df
        else:
            return self.__class__(new_df)

        self.sort_by_date()

    def sort_by_date(self):
        if "jd" in self.files_df:
            self.sort_by("jd")

    def sort_by(self, field, inplace=True):
        """Sort dataframe with a specific keyword

        Parameters
        ----------
        field : string
            keyword to sort by
        inplace : bool, optional
            weather to replace current datframe by a the filtered one, by default True. If False, filtered will be returned

        """
        if field in self.files_df:
            new_df = self.files_df.sort_values([field]).reset_index(drop=True)
            if inplace:
                self.files_df = new_df
            else:
                return new_df
        else:
            raise KeyError("'{}' is not in files_df".format(field))

    def description_table(self, *fields, original=False, unique=True, index=False, hide=None,
                          **kwargs):
        """Print a table description of dataframe

        Parameters
        ----------
        return_string : bool, optional
            weather to return str of the table, by default False. If False table is printed.
        original : bool, optional
            weather to describe the original dataframe, by default False. If false, this is applied on current.
        unique : bool, optional
            wether to show rows of unique values, by default True
        index : bool, optional
            wether to show indexes as first table header, by default False
        hide : [type], optional
            fields to hide, by default None. If one is specified, dataframe will be filtered for this values but they will not be shown
        **kwargs: dict
            filters to be applied. Example: telescope="A", filter="I" ... etc

        Returns
        -------
        [type]
            [description]
        """

        if original:
            files_df = self._original_files_df.copy()
        else:
            files_df = self.files_df.copy()

        files_df = files_df.fillna("")

        if len(kwargs) > 0:
            files_df = self._get(files_df, **kwargs)
            headers = list(fields) + list(kwargs.keys())

        else:
            headers = list(fields)

        if hide is not None:
            assert isinstance(hide, list), "hide must be a list of string"
            for h in hide:
                headers.remove(h)

        if unique:
            multi_index_obs = files_df.pivot_table(index=headers, aggfunc="size")
            if index:
                headers.insert(0, "index")
            single_index_obs = multi_index_obs.reset_index().rename(columns={0: 'quantity'}).reset_index(level=0)
            return single_index_obs[[*headers, "quantity"]]
        else:
            return files_df[headers]

    def move_to(self, dirname):
        if not path.exists(dirname):
            os.mkdir(dirname)

        for file in self.files_df["path"].values:
            shutil.move(file, dirname)

    def copy_to(self, dirname):
        if not path.exists(dirname):
            os.mkdir(dirname)

        for file in tqdm(self.files_df["path"].values):
            shutil.copy(file, dirname)

    def __repr__(self):
        return self.files_df.__repr__()

    def _repr_html_(self):
        return self.files_df._repr_html_()


class FitsManager(FilesDataFrame):

    def __init__(self, files_df_or_folder, verbose=True, image_kw="light", extension="*.f*ts*", hdu=0, reduced=False, **kwargs):
        if reduced:
            image_kw = "reduced"
        if isinstance(files_df_or_folder, pd.DataFrame):
            files_df = files_df_or_folder
            self.folder = None
        elif isinstance(files_df_or_folder, (str, Path)):
            assert path.exists(files_df_or_folder), "Folder does not exist"
            files = get_files(extension, files_df_or_folder, depth=kwargs.get("depth", 1), single_list_removal=False)
            files_df = fits_to_df(files, verbose=verbose, hdu=hdu)
            self.folder = files_df_or_folder
        else:
            raise AssertionError("input must be pd.DataFrame or folder path")

        super().__init__(files_df, verbose=verbose)
        self.image_kw = image_kw

    @property
    def observations(self):
        """
        Print a table of observations (observation is defined as a unique combinaison of date, telescope, target and filter)
        Only for reverse compatibility
        """
        headers = ["date", "telescope", "target", "filter"]
        table = self.description_table(*headers, index=True, type=self.image_kw, hide=["type"])
        rows = OrderedDict(table.to_dict(orient="list"))
        table_string = tabulate(rows, tablefmt="fancy_grid", headers="keys")
        print(table_string)
        return None

    @property
    def _observations(self):
        light_rows = self.files_df.loc[self.files_df["type"].str.contains(self.image_kw)].fillna("")
        observations = (
            light_rows.pivot_table(
                index=["date", "telescope", "target", "dimensions", "filter"],
                aggfunc="size",
            )
                .reset_index()
                .rename(columns={0: "quantity"})
                .reset_index(level=0)
        )

        return observations

    @property
    def calib(self):
        """Print a table of observations and calibration files (observation is defined as a unique combinaison of date, telescope, target and filter)

        """
        headers = ["date", "telescope", "target", "filter", "type"]
        table = self.description_table(*headers, index=False)
        rows = OrderedDict(table.to_dict(orient="list"))
        table_string = tabulate(rows, tablefmt="fancy_grid", headers="keys")
        print(table_string)
        return None

    def describe(self, calib=True):
        """
       display a table of available observations (defined as a unique date, telescope, target, and filter)
        Returns:

        """
        headers = ["date", "telescope", "target", "filter"]
        _observations = self._observations[headers]
        table = self.description_table(*headers, "type", index=True)

        for i, row in table.iterrows():
            match = np.all((_observations == row[headers]).values, 1)
            if np.any(match):
                table["index"][i] = np.flatnonzero(match)[0]
            else:
                table["index"][i] = ""

        if not calib:
            table = table[table["type"] == self.image_kw][["index", *headers]]

        stable = OrderedDict(table.to_dict(orient="list"))
        table_string = tabulate(stable, tablefmt="fancy_grid", headers="keys")
        print(table_string)

    def set_observation(self, i, future=0, past=None, same_telescope=False):
        """Set the unique observation to use by its id. Observation indexes are specified in `self.observations`

        Parameters
        ----------
        i : int
            index of the observation as displayed in `self.observations`
        """
        self.files_df = self.get_observation(i, future=future, past=past, same_telescope=same_telescope, return_df=True)
        assert self.unique_obs, "observation should be unique, please use set_observation"
        obs = self._observations.loc[0]
        ids_dict = {value["name"].lower(): key.lower() for key, value in CONFIG.telescopes_dict.items()}
        self.telescope = Telescope.from_name(ids_dict[obs.telescope.lower()])
        self.sort_by_date()

    def get_observation(self, i, future=0, past=None, same_telescope=False, return_df=False):

        original_fm = FitsManager(self._original_files_df.fillna(""))
        obs = self._observations.loc[i]

        days_limit = (1 if future is not None else -1)
        if future is not None:
            days_limit = -future
        elif past is not None:
            days_limit = past

        telescope = obs.telescope if same_telescope else "."
        dimensions = obs.dimensions

        dates_before = (
                    pd.to_datetime(obs.date) - pd.to_datetime(original_fm.files_df.date) >= timedelta(days=days_limit))

        flats = original_fm.get(
            telescope=telescope + "*", filter=obs["filter"].replace("+", "\+"),
            type="flat",
            dimensions=dimensions).loc[dates_before]
        darks = original_fm.get(telescope=telescope + "*", type="dark", dimensions=dimensions).loc[dates_before]
        bias = original_fm.get(telescope=telescope + "*", type="bias", dimensions=dimensions).loc[dates_before]
        dfs = []

        # To keep only calibration from a single day (the most recent possible)
        if len(flats) > 0:
            flats = flats.loc[flats.date == flats.date.max()]
            dfs.append(flats)
        if len(darks) > 0:
            darks = darks.loc[darks.date == darks.date.max()]
            dfs.append(darks)
        if len(bias) > 0:
            bias = bias.loc[bias.date == bias.date.max()]
            dfs.append(bias)

        others = original_fm.get(
            telescope=obs.telescope,
            filter=obs["filter"].replace("+", "\+"),
            target=obs.target.replace("+", "\+"),
            date=obs.date)

        for _type in ["dark", "flat", "bias"]:
            others = others.reset_index(drop=True)
            others = others.drop(np.argwhere(FilesDataFrame._get(others, type=_type, return_conditions=True).values).flatten())

        dfs.append(others)

        new_df = pd.concat([pd.concat(dfs)])

        if return_df:
            return new_df
        else:
            return self.__class__(new_df)

    @property
    def unique_obs(self):
        """Return whether the object contains a unique observation (observation is defined as a unique combinaison of date, telescope, target and filter).

        Returns
        -------
        bool
        """
        return len(self._observations) == 1

    @property
    def images(self):
        """fits paths of the observation science images

        Returns
        -------
        list of str
        """
        return self.get(type=self.image_kw).path.values.astype(str)

    @property
    def darks(self):
        """fits paths of the observation dark images

        Returns
        -------
        list of str
        """
        return self.get(type="dark").path.values.astype(str)

    @property
    def bias(self):
        """fits paths of the observation bias images

        Returns
        -------
        list of str
        """
        return self.get(type="bias").path.values.astype(str)

    @property
    def flats(self):
        """fits paths of the observation flats images

        Returns
        -------
        list of str
        """
        return self.get(type="flat").path.values.astype(str)

    @property
    def stack(self):
        """fits paths of the observation stack image if present

        Returns
        -------
        list of str
        """
        return self.get(type="stack").path.values.astype(str)

    @property
    def calibrated(self):
        """fits paths of the observation calibrated images if present

        Returns
        -------
        list of str
        """
        return self.get(type="reduced").path.values.astype(str)

    @property
    def products_denominator(self):
        assert self.unique_obs, "observation should be unique, please use set_observation"
        obs = self._observations.loc[0]
        return f"{obs.telescope}_{obs.date.replace('-', '')}_{obs.target}_{obs['filter']}"

    @property
    def images_dict(self):
        return {
            "flats": self.flats,
            "darks": self.darks,
            "bias": self.bias,
            "images": self.images,
        }

    @property
    def obs_name(self):
        if self.unique_obs:
            return self.products_denominator
        else:
            raise AssertionError("obs_name property is only available for FitsManager containing a unique observation")

    def observation_id(self, target, date, telescope='', error=True):
        obs = self._observations
        there = np.argwhere((
                    clean(obs.date).str.contains(sub(date)) &
                    clean(obs.target).str.contains(sub(target)) &
                    clean(obs.telescope).str.contains(sub(telescope))
            ).values).flatten()

        if len(there) > 0:
            if len(there) == 1 or not error:
                i = there[0]
                return i
            elif error:
                raise AssertionError(f"multiple observations found {there}")
        else:
            return None