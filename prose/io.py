import os
from os import path
import datetime
import pandas as pd
from astropy.io import fits
import numpy as np
from tabulate import tabulate
from collections import OrderedDict
from tqdm import tqdm
from prose import utils
from prose.telescope import Telescope
import glob
from prose import CONFIG
import warnings
from prose.lightcurves import LightCurves
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import shutil
# import fitsio


def phot2dict(filename, format="fits"):
    if format == "fits":
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

    files = [path.abspath(f) for f in files]

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


class FitsManager:
    # TODO: Put all private method as _
    # TODO: Check if light_kw is used or useless
    # TODO: Documentation!

    """
    A class to manage data folder containing FITS files organized in arbitrary ways. This class explore all sub-folders and
    retrieve header information to trace single FITS files.

    Parameters
    ----------
    folder : str
        Path of the folder to be explored
    verbose : bool, optional
        Wether or not to print processing info, by default True
    telescope_kw : str, optional
        FITS header keyword where to find telescope name, by default "TELESCOP"
    depth : int, optional
        Depth of exploration in terms of sub-folders, by default 1
    light_kw : str, optional
        Main image type you want to retrieve, by default "light"
    update : bool, optional
        Wether to update the index if `index=True`, by default False
    index : bool, optional
        Wether to use index file if available, by default False

    Raises
    ------
    FileNotFoundError
        If folder doesn't exists
    """
    def __init__(self, folder, verbose=True, telescope_kw="TELESCOP", depth=1, light_kw="light", update=False, index=None):

        self.depth = depth
        self._temporary_files_headers = None
        self._temporary_files_paths = None
        self.files_df = None
        self.verbose = verbose
        self.telescope_kw = telescope_kw
        self.light_kw = light_kw

        self.folder = folder
        if not path.isdir(self.folder):
            raise FileNotFoundError("folder does not exists")

        self.config = CONFIG

        self._check_telescope_file()
        self.telescope = Telescope()
        self.index_file = None
        
        if index:
            has_index = self._load_index(index=index)
            if update or not has_index:
                self._build_files_df()
        else:
            self._build_files_df()

        self._original_files_df = self.files_df.copy()

        assert len(self._original_files_df) != 0, "No data found"

        # To embed useful files
        self.files = None

    def _load_index(self, force_single=False, index=None):
        """Load index files (default is prose_index.csv within self.folder)

        Parameters
        ----------
        force_single : bool, optional
            If True errors will be raised if index file is missing or if multiple present, by default False

        Returns
        -------
        bool
            Wheater proper index file has been loaded to the files DataFrame self.files_df

        Raises
        ------
        ValueError
            raised if force_single is True and index file is missing or if multiple present
        ValueError
            raised if force_single is True and multiple index files are present
        """
        if index is None:
            index_files = get_files("*index.csv", self.folder, single_list_removal=False, none_for_empty=False)
            if len(index_files) == 1:
                self.index_file = index_files[0]
            else:
                if len(index_files) == 0:
                    if force_single:
                        raise ValueError("No *index.csv found")
                    return False
                elif len(index_files) > 1:
                    if force_single:
                        raise ValueError("Too many *index.csv found, should be unique")
                    return False
        else:
            self.index_file = index
            if not path.exists(self.index_file):
                return False

        self.files_df = pd.read_csv(self.index_file, na_filter=False)
        self.files_df["complete_date"] = pd.to_datetime(self.files_df["complete_date"])
        self.files_df["date"] = pd.to_datetime(self.files_df["date"]).apply(lambda x: x.date())
        return True

    def _build_files_df(self):
        """
        Build the main files Dataframe including paths, dates, telescope, types, targets, filters, combined, 
        complete_date, dimensions, flip and jd
        """
        # TODO: Add file size evaluation in files_df
        self._temporary_files_paths = get_files(".f*ts", self.folder, depth=self.depth, single_list_removal=False)
        paths = []
        dates = []
        telescope = []
        types = []
        targets = []
        filters = []
        combined = []
        complete_date = []
        dimensions = []
        flip = []
        jd = []

        _types = ["flat", "dark", "bias", self.light_kw]

        last_telescope_name = "fake_00000"
        _temporary_telescope = Telescope()

        if self.verbose:
            _tqdm = tqdm
        else:
            _tqdm = lambda x: x

        if self.files_df is not None:
            existing_paths = self.files_df["path"].values
        else:
            existing_paths = []

        for i, file_path in enumerate(_tqdm(self._temporary_files_paths)):
            if file_path not in existing_paths:
                # header = fitsio.read_header(file_path)
                header = fits.getheader(file_path)

                try:
                    telescope_name = header[self.telescope_kw].lower()
                    if telescope_name != last_telescope_name:
                        _temporary_telescope.load(
                            CONFIG.match_telescope_name(telescope_name)
                        )

                    _path = self._temporary_files_paths[i]
                    _complete_date = utils.format_iso_date(header.get(_temporary_telescope.keyword_observation_date), night_date=False)
                    _date = utils.format_iso_date(header.get(_temporary_telescope.keyword_observation_date), night_date=True)
                    _telescope = _temporary_telescope.name
                    _target = header.get(_temporary_telescope.keyword_object, "")
                    _type = header.get(_temporary_telescope.keyword_image_type, "").lower()
                    _flip = header.get(_temporary_telescope.keyword_flip, "")
                    
                    if _temporary_telescope.keyword_flat_images.lower() in _type:
                        _type = "flat"
                    elif _temporary_telescope.keyword_dark_images.lower() in _type:
                        _type = "dark"
                    elif _temporary_telescope.keyword_bias_images.lower() in _type:
                        _type = "bias"
                    elif "stack" in _type:
                        _type = "stack"
                    elif _type == self.light_kw:
                        _type = self.light_kw

                    _filter = header.get(_temporary_telescope.keyword_filter, "")
                    _combined = "{}_{}_{}_{}_{}".format(
                        _date.strftime("%Y%m%d"), _type, _telescope, _target, _filter,
                    )
                    _dimensions = "{}x{}".format(header["NAXIS1"], header["NAXIS2"])
                    _jd = header.get(_temporary_telescope.keyword_julian_date, "")

                    paths.append(_path)
                    dates.append(_date)
                    telescope.append(_telescope)
                    types.append(_type)
                    targets.append(_target)
                    filters.append(_filter)
                    combined.append(_combined)
                    complete_date.append(_complete_date)
                    dimensions.append(_dimensions)
                    flip.append(_flip)
                    jd.append(_jd)
                except:
                    print("cannot read {}, ignored...".format(file_path))

        files_df = pd.DataFrame(
            {
                "date": dates,
                "complete_date": complete_date,
                "path": paths,
                "telescope": telescope,
                "dimensions": dimensions,
                "type": types,
                "target": targets,
                "filter": filters,
                "combined": combined,
                "flip": flip,
                "jd": jd
            }
        )

        if self.files_df is None:
            self.files_df = files_df
        else:
            self.files_df = pd.concat([self.files_df, files_df], ignore_index=True)

        self._sort_by_date()
        self._save_index()
    
    def _save_index(self):
        """
        Save current files Dataframe (self.files_df) into an index files (default is prose_index.csv)
        """
        # current file name, not implemented but TODO later
        #datetime.datetime.now().strftime("pwd_%Y%m%d_%H%M_index.csv")
        if self.index_file is None:
            self.index_file = path.join(self.folder, "prose_index.csv")
        self.files_df.to_csv(self.index_file, index=False)

    def _check_telescope_file(self):
        """
        Check for telescope.id file in folder and copy it to specphot config folder
        """
        id_files = get_files(".id", self.folder, single_list_removal=False)

        if len(id_files) > 0:
            assert (
                len(id_files) == 1
            ), "Multiple .id files in your folder, please clean up"
            self.config.save_telescope_file(id_files[0])

    def reset(self):
        """
        Reset the orginial files DataFrame
        """
        self.files_df = self._original_files_df.copy()

    def get(
        self,
        im_type=None,
        telescope=None,
        date=None,
        filter=None,
        target=None,
        return_conditions=False,
        n_images=None
    ):
        """ Filter files based on header info and get their paths (or a filter on the files Dataframe self.files_df)

        Parameters
        ----------
        im_type : str, optional
            type of image (e.g. "light", "dark", "bias", "flat"), by default None for all
        telescope : str, optional
            telescope name, by default None for all
        date : str, optional
            date as %Ym%%d, by default None for all
        filter : str, optional
            filter name, by default None for all
        target : str, optional
            target name, by default None for all
        return_conditions : bool, optional
            weather to return bool Serie matching filters or paths, by default False i.e. returning path of files martching filters
        n_images: int, optional
            number of images to keep, default is None for all images
        Returns
        -------
        list or pandas.Series
        """
        if not filter:
            filter = None
        if not target:
            target = None
        if not telescope:
            telescope = None

        conditions = pd.Series(np.ones(len(self.files_df)).astype(bool), index=self.files_df.index)
        if im_type is not None:
            conditions = conditions & self.files_df["type"].str.contains(im_type)
        if date is not None:
            if isinstance(date, datetime.date):
                date = date.strftime("%Y%m%d")

            conditions = conditions & (
                self.files_df["date"].apply(lambda _d: _d.strftime("%Y%m%d")) == date
            )
        if telescope is not None:
            conditions = conditions & (
                self.files_df["telescope"]
                .str.lower()
                .str.contains(telescope.lower() + "*")
            )
        if filter is not None:
            conditions = conditions & (
                self.files_df["filter"].str.lower().str.contains(filter.lower() + "*")
            )
        if target is not None:
            conditions = conditions & (
                self.files_df["target"]
                .str.lower()
                .str.contains(target.replace("+", "\+").lower() + "*")
            )

        if n_images is None and im_type not in ["dark", "bias", "flat"]:
            n_images = len(self.files_df.loc[conditions]["path"].values)

        if return_conditions:
            return conditions
        else:
            return self.files_df.loc[conditions]["path"].values[0:n_images]

    def _set_telescope(self, name=None):
        """
        Set telescope object

        Parameters
        ----------
        name : str
            name of the telescope to set
        """
        self.telescope.load(self.config.match_telescope_name(name))

    def keep(
        self,
        telescope=None,
        date=None,
        im_filter=None,
        target=None,
        calibration=True,
        check_telescope=True,
        calibration_date_limit=0,
    ):
        """
        Keep in the files DataFrame only files matching the filters. The kept files should all be from the same (and unknown if check_telescope=False) telescope.

        Parameters
        ----------
        telescope : str, optional
            telescope name, by default None for all
        date : str, optional
            date as %Ym%%d, by default None for all
        im_filter : str, optional
            filter name, by default None for all
        target : str, optional
            target name, by default None for all
        calibration : bool, optional
            Weather to keep closest calibration images (in time), by default True
        check_telescope : bool, optional
            Weather to check if calibration images are from the same telescope, by default True. Calibration images witout telescope specified in FITS header are treated from unknown telescope.
        calibration_date_limit : int, optional
            number of days in the past (in the future negative) up to which to consider calibration images, by default 0

        Raises
        ------
        AssertionError
            No files match the filters
        AssertionError
            Multiple telescopes found in matched files
        """

    # TODO: allow date range
        self.files_df = self.files_df.loc[
            self.get(
                return_conditions=True,
                telescope=telescope,
                filter=im_filter,
                target=target,
                date=date,
            )
        ]

        obs_telescopes = np.unique(self.files_df["telescope"])
        obs_telescopes = obs_telescopes[obs_telescopes != "Unknown"]
        assert len(obs_telescopes) != 0, "No files match the filters"
        assert (
            len(np.unique(obs_telescopes)) == 1
        ), "Multiple telescopes found in matched files, please add constraints"

        obs_telescope = np.unique(self.files_df["telescope"])[0]
        obs_dimensions = np.unique(self.files_df["dimensions"])[0]

        self._set_telescope(obs_telescope)

        if calibration:

            # date of the kept observation
            obs_date = np.unique(self.files_df["date"])[0]

            if not check_telescope:
                obs_telescope = None
            dark = self._find_closest_calibration(
                obs_date,
                "dark",
                obs_dimensions,
                obs_telescope,
                days_limit=calibration_date_limit,
            )
            bias = self._find_closest_calibration(
                obs_date,
                "bias",
                obs_dimensions,
                obs_telescope,
                days_limit=calibration_date_limit,
            )
            flat = self._find_closest_calibration(
                obs_date,
                "flat",
                obs_dimensions,
                obs_telescope,
                days_limit=calibration_date_limit,
            )

            self.files_df = pd.concat([self.files_df, dark, bias, flat])
            self._sort_by_date()

    def _find_closest_calibration(
        self, observation_date, im_type, obs_dimensions, telescope=None, days_limit=0
    ):
        """

        Parameters
        ----------
        observation_date : str
            date as %Y%m%d of the observation from which closest calibration data need to be found
        im_type : str
            calibration type "bias", "dark" or "flat"
        obs_dimensions: str
            {pixels}x{pixels} dimension of the images from observation, example: 2000x2000
        telescope : str, optional
            telescope from which closest calibration data need to be found, default is None for all

        Returns
        -------

        """
        original_df = self._original_files_df.copy()

        # Find all dark
        condition = original_df["type"].str.contains(im_type + "*")

        condition_checker = bool(len(original_df[condition]))

        if not condition_checker:
            raise ValueError("No '{}' calibration could be retrieved".format(im_type))

        # Check telescope
        if telescope is not None:
            condition = condition & original_df["telescope"].str.lower().str.contains(
                telescope.lower() + "*"
            )

        condition_checker = bool(len(original_df.loc[condition]))

        if not condition_checker:
            raise ValueError(
                "No '{}' calibration from {} could be retrieved. Common error when calibration "
                "files do not provide telescope information".format(im_type, telescope)
            )

        # Check dimensions
        condition = condition & (original_df["dimensions"] == obs_dimensions)
        condition_checker = bool(len(original_df.loc[condition]))

        if not condition_checker:
            raise ValueError(
                "Could not find calibration images of {} pixels for {}".format(
                    obs_dimensions, telescope
                )
            )

        calibration = original_df.loc[condition]

        # sorted calibration rows
        sorted_calib = calibration.loc[
            (observation_date - calibration["date"])
            >= datetime.timedelta(days=-days_limit)
        ].dropna(
            subset=["date"]
        )  # We only look for files prior or during the day of observation
        closest_combined = sorted_calib.iloc[
            (observation_date - sorted_calib["date"]).argsort()
        ]["combined"]

        if len(closest_combined) == 0:
            raise AssertionError(
                "Calibration could not be found. Check days-limit or check_telescope"
            )

        closest_combined = closest_combined.iloc[0]

        calibration = original_df.loc[
            original_df["combined"].str.contains(closest_combined + "*") & condition
        ]

        return calibration

    def _sort_by_date(self):
        """
        Sort files Dataframe by dates
        """
        self.files_df = self.files_df.sort_values(["complete_date"]).reset_index(
            drop=True
        )

    def _has_calibration(self):
        """
        Return weather calibration files are present in current files DataFrame

        Returns
        -------
        bool
            weather calibration files are present
        """
        return (
            len(self.get("dark")) > 0
            and len(self.get("flat")) > 0
            and len(self.get("bias")) > 0
        )

    def _has_stack(self):
        """
        Return weather stack is present and unique in current files DataFrame

        Returns
        -------
        bool
            weather calibration files are present
        """
        return len(self.get("stack")) == 1

    def describe(self, table_format="obs", return_string=False, original=False):
        """
        Print (or return str) a table synthetizing files DataFrame content

        Parameters
        ----------
        table_format : str, optional
            "obs": show all observations (defined by unique telescope + date + target + filter)
            "calib": show all observations and calibration files
            "files": show all files
            default is "obs"
        return_string : bool, optional
            weather return the string of the table or print it, by default False, i.e. printing
        original : bool, optional
            weather to show the current files DataFrame or the original one before filtering, by default False, i.e. show the current one

        Returns
        -------
        str if return_string is True else None
            string of table

        Raises
        ------
        ValueError
            table_format should be 'obs', 'calib' or 'files'
        """

        if original:
            files_df = self._original_files_df.copy()
        else:
            files_df = self.files_df.copy()

        files_df = files_df.fillna('')

        if "obs" in table_format:
            headers = ["index", "date", "telescope", "target", "filter", "quantity"]

            observations = self._observations
            rows = OrderedDict(observations[headers].to_dict(orient="list"))

            table_string = tabulate(rows, headers, tablefmt="fancy_grid")

        elif "calib" in table_format:
            headers = [
                "date",
                "telescope",
                "type",
                "target",
                "dimensions",
                "filter",
                "quantity",
            ]

            multi_index_obs = files_df.pivot_table(
                index=["date", "telescope", "type", "target", "dimensions", "filter"],
                aggfunc="size",
            )

            single_index_obs = (
                multi_index_obs.reset_index()
                .rename(columns={0: "quantity"})
                .reset_index(level=0)
            )
            rows = OrderedDict(single_index_obs[headers].to_dict(orient="list"))

            table_string = tabulate(rows, tablefmt="fancy_grid", headers="keys",)

        elif "files" in table_format:
            headers = [
                "index",
                "date",
                "telescope",
                "type",
                "dimensions",
                "target",
                "filter",
            ]
            rows = OrderedDict(files_df.reset_index()[headers].to_dict(orient="list"))
            table_string = tabulate(rows, tablefmt="fancy_grid", headers="keys")

        else:
            raise ValueError(
                "{} is not an accepted format. Accepted format are 'obs', 'calib' and 'files'".format(
                    table_format
                )
            )

        if return_string:
            return table_string
        else:
            print(table_string)

    def _trim(self, image, raw=False, wcs=None):
        # TODO: investigate a flip option, cf calibrate
        if raw:
            if isinstance(image, np.ndarray):
                pass
            elif isinstance(image, str):
                if path.exists(image) and image.lower().endswith((".fts", ".fits")):
                    image = fits.getdata(image)
            else:
                raise ValueError("{} should be a numpy array or a fits file")

            trim_0, trim_1 = image.shape

            if self.telescope.trimming[1] != 0:
                trim_0 = -self.telescope.trimming[1]
            if self.telescope.trimming[0] != 0:
                trim_1 = -self.telescope.trimming[0]

            return image[
                self.telescope.trimming[1] : trim_0,
                self.telescope.trimming[0] : trim_1,
            ]
        else:
            if isinstance(image, np.ndarray):
                shape = np.array(image.shape)
                center = shape[::-1]/2
                dimension = shape - 2*np.array(self.telescope.trimming[::-1])
                if wcs is not None:
                    return Cutout2D(image, center, dimension, wcs=wcs)
                else:
                    return Cutout2D(image, center, dimension)
            elif isinstance(image, str):
                if path.exists(image) and image.lower().endswith((".fts", ".fits")):
                    image_data= fits.getdata(image)
                    shape = np.array(image_data.shape)
                    center = shape[::-1]/2
                    dimension = shape - 2*np.array(self.telescope.trimming[::-1])
                    return Cutout2D(image_data, center, dimension, wcs=WCS(image))
            else:
                raise ValueError("{} should be a numpy array or a fits file")
    
    @property
    def _observations(self):
        light_rows = self.files_df.loc[self.files_df["type"].str.contains(self.light_kw)]
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
    def products_denominator(self):
        """Return a string formatted as :code:`telescope_date_target_filter`

        Returns
        -------
        str
            :code:`telescope_date_target_filter`
        """
        single_obs = self._observations

        assert len(single_obs) == 1, "Multiple or no observations found"

        single_obs = single_obs.iloc[0]

        return "{}_{}_{}_{}".format(
            self.telescope.name,
            single_obs["date"].strftime("%Y%m%d"),
            single_obs["target"],
            single_obs["filter"],
        )

    def set_observation(
            self,
            observation_id,
            check_calib_telescope=True,
            calibration=False,
            calibration_date_limit=0
        ):
        """Set the observation to keep

        Parameters
        ----------
        observation_id : int
            index of the observations (observation indexes are listed with :code:`FitsManager.describe()`)
        check_calib_telescope : bool, optional
            Check weather calibration FITS files come from the same telescope as science FITS, by default True. Useful when telescope name is not specified on caliobration files heaeders.
        calibration : bool, optional
            wether to keep calibration files along science images, by default False
        calibration_date_limit : int, optional
            minimum number of days prior of the observation to look for calibration files, by default 0 (i.e. up to the same day). If negative it conrresponds to days in the future of the observation date.
        """

        observations = self._observations

        assert observation_id in observations["index"], "index {} do not match any observation".format(observation_id)

        obs = observations[observations["index"] == observation_id].iloc[0]

        self.keep(
            telescope=obs["telescope"],
            date=obs["date"],
            im_filter=obs["filter"],
            target=obs["target"],
            check_telescope=check_calib_telescope,
            calibration=calibration,
            calibration_date_limit=calibration_date_limit
        )

    # TODO: We absolutely never use kwargs in here. The natural flow is to 
    # instanciate, keep and then copy. This reset is a killer,
    def copy_files(
            self, 
            destination, 
            calibration=True, 
            check_calib_telescope=False, 
            overwrite=False, 
            **kwargs
        ):
        """
        Locally copy files from the files DataFrame into the following structure:
        
        - Target
            - date0
            - date1
            - ...


        Parameters
        ----------
        destination : str
            path where to copy files
        calibration : bool, optional
            wether to keep calibration files, by default True
        check_calib_telescope : bool, optional
            Check weather calibration FITS files come from the same telescope as science FITS, by default True. Useful when telescope name is not specified on caliobration files heaeders.
        overwrite : bool, optional
            wether to obverwrite a file if exists already, by default False
        """
        # TODO: check available space if size is in files_df, else suggest to use kwargs force
        self.reset()
        self.keep(calibration=False, check_telescope=False, **kwargs)
        n_obs = len(self._observations)

        for i, observation in self._observations.iterrows():
            target = observation["target"]
            date_str = observation["date"].strftime("%Y%m%d")

            target_folder = path.join(destination, target)
            if not path.exists(target_folder):
                os.mkdir(target_folder)

            date_folder = path.join(destination, target, date_str)
            if not path.exists(date_folder):
                os.mkdir(date_folder)

            self.reset()
            self.keep(check_telescope=check_calib_telescope, **kwargs)
            self.set_observation(
                i,
                check_calib_telescope=check_calib_telescope, 
                calibration=calibration)
            files = self.get()

            print("{}/{} : copying {} - {} in {}".format(i+1, n_obs, target, date_str, date_folder))
            for file in tqdm(files):
                new_file = path.join(date_folder, path.split(file)[-1])
                if not path.exists(new_file) or overwrite:
                    shutil.copyfile(file, new_file)


def fits_keyword_values(fits_files, keywords, default_value=None, verbose=False):
    """

    Get the values of specific keywords in a list of fits files

    Parameters
    ----------
    fits_files: list(str)
        List of fits files (string path)
    default_value : any
        Value to be returned if keyword is not found (default is None). If None return a keyError instead
    keywords: list(str)
        List of keywords

    Returns
    -------
    List of keyword values for any files with shape (number_of_keywords, number_of_files)

    """
    if not verbose:
        _tqdm = lambda l, **kwargs: l
    else:
        _tqdm = tqdm

    if isinstance(keywords, str):
        _keywords = [keywords]
    else:
        _keywords = keywords

    header_values = []

    for f in _tqdm(fits_files):
        fits_header = fits.getheader(f)
        if default_value is None:
            header_values.append([fits_header[keyword] for keyword in _keywords])
        else:
            header_values.append([fits_header[keyword] for keyword in _keywords])

    # If only one keyword the list is flattened
    if isinstance(keywords, str):
        header_values = [hv for hvs in header_values for hv in hvs]

    return header_values

def set_hdu(hdu_list, value):
    key = value.name 
    if key in hdu_list:
        hdu_list[key] = value
    else:
        hdu_list.append(value)
