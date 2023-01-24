import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from .io import get_files, fits_to_df
from IPython.display import display
from ..console_utils import progress, info

# Convenience
# -----------
SQL_DAYS_BETWEEN = "date >= date('{date}', '-{past:.0f} days') AND date <= date('{date}', '+{future:.0f} days')"

UNIQUE_FIELDS_LIST = ["date", "telescope", "filter", "target", "type", "width", "height", "exposure"]
UNIQUE_FIELDS = ",".join(UNIQUE_FIELDS_LIST)
QMARKS_UNIQUE = ",".join(['?']*len(UNIQUE_FIELDS.split(",")))

PWD = Path(__file__).parent


def in_value(value):
    return f"'{value}'" if isinstance(value, str) else value

def exposure_constraint(exposure=0, tolerance=1000000):
    return f"exposure between {exposure-tolerance} and {exposure+tolerance}"
# ------------


class FitsManager:
    """Object to parse and retrieve FITS files from folder and sub-folders

    This object scans files in a folder and stores the data in a mysql database for conveniance (but all products can be accessed without knowledge of the SQL language)

    Parameters
    ----------
    folder : str
        path of the folder to parse (or list of folders as of prose 2.0.1)
    depth : int, optional
        maxiumum depth of the sub-folders to explore, by default 0
    hdu : int, optional
        by default 0
    extension : str, optional
        by default ".f*ts*"
    file : _type_, optional
        _description_, by default None
    batch_size : bool or int, optional
        - if False: update database after all FITS files are parsed
        - if int: update database every time ``batch_size`` FITS files are parsed

        by default False
    """
    
    def __init__(self, folders=None, files=None, depth=0, hdu=0, extension=".f*t*", file=None, batch_size=False, scan=None, verbose=True, to_df=None):
        if file is None:
            file = ":memory:"

        self.con = sqlite3.connect(file)
        self.cur = self.con.cursor()

        # check if file Table exists
        tables = list(self.cur.execute("SELECT name FROM sqlite_master WHERE type='table';"))
        if len(tables) == 0:
            db_creation = open(PWD / "create_fm_db.sql", "r").read()
            self.cur.executescript(db_creation)
        
        if folders is not None:
            assert files is None, "Only 'folders' or 'files' must be provided, not both"
            files = self.get_files(folders, extension, depth=depth, scan=scan)
        
        if to_df is None:
            self.fits_to_df = fits_to_df
        else:
            self.fits_to_df = to_df

        if files is not None:
            if len(files) > 0:
                self.scan_files(files, batch_size=batch_size, hdu=hdu, verbose=verbose)

    def _insert(self, path, date, telescope, type, target, filter, dimensions, _, jd, exposure, id=None, update_obs=True):
        """Insert FITS data to object database
        """
        if isinstance(filter, float):
            filter = ""
        else:
            filter = filter or ""
        telescope = telescope or ""
        target = target or ""
        id = id or "NULL"
        width, height = dimensions
        filter = filter.replace("'", "p")

        # update observation
        if update_obs:
            obs = (date, telescope, filter, target, type, width, height, exposure)
            self.con.execute(f"INSERT or IGNORE INTO observations({UNIQUE_FIELDS}, files) VALUES ({QMARKS_UNIQUE}, 0)", obs)
            query = " AND ".join([f"{str(key)} = {in_value(value)}" for key, value in zip(UNIQUE_FIELDS_LIST, obs)])
            id = self.con.execute(f"SELECT id FROM observations where {query}").fetchall()[0][0]
            self.con.execute(f"UPDATE observations SET files = files + 1 WHERE id = {id}")
            
        # or IGNORE to handle the unique constraint
        self.cur.execute(
            f"INSERT or IGNORE INTO files VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (date, path, telescope, filter, type, target, width, height, jd, id, exposure))
            

    def get_files(self, folders, extension, scan=None, depth=0):
        """Return path of files with specific extension in the specified folder(s) 

        Parameters
        ----------
        folders : str or list of str
            path or list of paths of folders to look into
        extension : str
            wildcard pattern for file extension
        scan : function, optional
            function to use, called on each folder path, by default None
        depth : int, optional
            number of sub-folders to look into, by default 0
        """
        def _get_files(folder):
            if scan is None:
                return get_files(extension, folder, depth=depth)
            else:
                return scan(folder)

        if isinstance(folders, (list, tuple)):
            files = [] 
            for folder in folders:
                assert Path(folder).exists(), f"Folder {folder} does not exists"
                files += _get_files(folder)
        else:
            assert Path(folders).exists(), f"Folder {folders} does not exists"
            files = _get_files(folders)

        return files

    def _path_in(self, path):
        return self.con.execute(f"SELECT * FROM files WHERE path='{path}'").fetchone() is not None

    def scan_files(self, files, batch_size=False, verbose=True, hdu=0, telescope=None, verbose_new=False, verbose_os=False):
        """Scan files and add data to database

        Parameters
        ----------
        files : list of str or Path
            paths of files
        batch_size : bool or int, optional
            - if False: update database after all FITS files are parsed
            - if int: update database every time ``batch_size`` FITS files are parse

            by default False
        verbose : bool, optional
            whether to show progress bar during parsing, by default True
        hdu: int
            FITS data unit extension where header will be parsed
        """

        if len(files) > 0:
            files_to_scan = [path for path in files if not self._path_in(path)]
            if verbose_new:
                info(f"{len(files_to_scan)} new files to scan")

            if len(files_to_scan) > 0:
                if batch_size is None:
                    batches = [files_to_scan]
                elif batch_size is False:
                    pass
                elif batch_size is True:
                    batches = [files_to_scan]
                else:
                    assert isinstance(batch_size, int), "batch_size must be an int"

                    if len(files_to_scan) < batch_size:
                        batches = [files_to_scan]
                    else:
                        batches = np.array_split(files_to_scan, len(files_to_scan)//batch_size)
                
                _verbose = verbose and batch_size is not False
                _progress = progress(_verbose, desc="Reading fits", unit="files")

                if batch_size is not False:
                    for batch in _progress(batches):
                        try:
                            df = self.fits_to_df(batch, verbose=False, hdu=hdu, verbose_os=verbose_os)
                            for row in df.values:
                                if telescope is not None:
                                    row[2] = telescope
                                self._insert(*row)
                            self.con.commit()
                        except:
                            "ERROR, batch ignored"
                else:
                    df = self.fits_to_df(files_to_scan, verbose=True, hdu=hdu, verbose_os=verbose_os)
                    for row in df.values:
                        if telescope is not None:
                            row[2] = telescope
                        self._insert(*row)
                    self.con.commit()
            else:
                if not verbose_new:
                    f"{len(files_to_scan)} new files to scan"
        else:
            raise AssertionError(f"No files provided")

    def observations(self, hide_exposure=True, **kwargs):
        """return a pandas DataFrame of observations given some metadata constraints in the form of wildcards

        Parameters
        ----------
        hide_exposure : bool, optional
            wether to include exposure in the pandas.DataFrame header, by default True
        **kwargs:
            wildcards value for telescope, target, filter, type or id, default is '*'

        Returns
        -------
        pd.DataFrame
            pandas DataFrame filtered with kwargs wildcards
        """
        columns = {c[1]: "%" for c in self.con.execute("PRAGMA table_info(observations)").fetchall()[1:-3]}
        inputs = kwargs.copy()
            
        for key, value in inputs.items():
            inputs[key] = "%" if value is None else str(value).replace("*", "%")
            
        columns.update(inputs)
        
        where = " AND ".join([f"{key} LIKE {in_value(value)}" for key, value in columns.items()])
        query = f"select * from observations where {where}"
        
        if hide_exposure:
            query = f"select *, SUM(files) from observations where {where} GROUP BY date, telescope, target, filter, type"
            df = self.to_pandas(query)
            df["files"]
            df = df.drop(columns=["files", "exposure"]).rename(columns={"SUM(files)": "files"})
        else:
            query = f"select * from observations where {where}"
            df = self.to_pandas(query)
            
        return df.set_index(["id"])

    def calibrations(self, **kwargs):
        """return a pandas DataFrame of calibrations observations given some metadata constraints in the form of wildcards

        Parameters
        ----------
        **kwargs:
            wildcards value for telescope, target, filter or id, default is '*'
        """
        darks = self.observations(type="dark", **kwargs)
        flats = self.observations(type="flat", **kwargs)
        bias = self.observations(type="bias", **kwargs)
        
        return pd.concat([darks, flats, bias], axis=0)
        
    def files(self, id=None, path=False, exposure=0, tolerance=1000, **kwargs):
        """Return a pandas DataFrame of files given some metadata constraints in the form of wildcards

        Parameters
        ----------
        id : int, optional
            id of the observation for which files are retrieved, by default None, i.e. all files
        path : bool, optional
            whether to include files paths in the pandas.DataFrame header, by default False
        exposure : int, optional
            exposure constraint on the files to retrieve, by default 0
        tolerance : int, optional
            tolerance on the exposure constraint, by default 1000. For example: if exposure is set to 10 and tolerance to 2, all
            files with exposure = 10 +- 2 will be retrieved
        """
        columns = {c[1]: "%" for c in self.con.execute("PRAGMA table_info(files)").fetchall()}
        if not path:
            del columns["path"]
        columns["id"] = id or "%"
        inputs = kwargs.copy()

        for key, value in inputs.items():
            inputs[key] = "%" if value is None else str(value).replace("*", "%")

        columns.update(inputs)

        where = " AND ".join([f"{key} LIKE {in_value(value)}" for key, value in columns.items()])
        where += f" AND {exposure_constraint(exposure, tolerance)}"

        del columns["id"]
        df = self.to_pandas(f"select {','.join(columns.keys())} from files where {where} order by jd")
        return df
    
    def observation_files(self, i, past=1e3, future=0, tolerance=1e15, same_telescope=True, lights="images", show=True):
        files = {}

        obs_dict = self.observations(id=i, hide_exposure=False).to_dict("records")[0]
        sql_days = SQL_DAYS_BETWEEN.format(date=obs_dict["date"], future=future, past=past)
        sql_exposure = exposure_constraint(exposure=obs_dict["exposure"], tolerance=tolerance)

        files[lights] = self.to_pandas(f"SELECT path from files where id = {i} order by jd").values.flatten()
        dfs = []

        if show:
            dfs.append(self.to_pandas(f"SELECT * from observations where id = {i}"))

        for type in ("dark", "bias", "flat"):
            fields = ["width", "height"]
            if same_telescope:
                fields.append("telescope")
            if type == "flat":
                fields.append("filter")

            query = " AND ".join([f"{key} = {in_value(obs_dict[key])}" for key in fields])
            query += f" AND type = '{type}'"
            query = query.format(**obs_dict)

            obs_ids = self.to_pandas(f"""SELECT id FROM observations WHERE {sql_exposure} AND {query}
                AND date = (SELECT MAX(date) FROM files WHERE {sql_days} AND {query})
            """).values.flatten()
            
            if show:
                dfs.append(self.to_pandas(f"""SELECT * FROM observations WHERE {sql_exposure} 
                    AND {query} AND type = '{type}'
                    AND date = (SELECT MAX(date) FROM files 
                    WHERE {sql_days} AND {query})
                """))

            _files = [self.to_pandas(f"select path from files where id={j} order by jd").values.flatten() for j in obs_ids]
            if len(_files) > 0:
                _files = np.hstack(_files)

            files[type + ("s" if type[-1] != "s" else "")] = _files
            
        if show:
            df = pd.concat(dfs, axis=0).set_index(["id"])
            display(df)
        
        return files

    @property
    def unique_obs(self):
        """Return whether the object contains a unique observation (observation is defined as a unique combinaison of date, telescope, target and filter).

        Returns
        -------
        bool
        """
        return len(self.observations(show=False)) == 1

    @property
    def all_images(self):
        """fits paths of the observation science images

        Returns
        -------
        list of str
        """
        return self.files(type='light', path=True).path.values

    def images(self, i, show=False, **kwargs):
        return self.observation_files(i, show=show, **kwargs)['images']

    @property
    def all_darks(self):
        """fits paths of the observation dark images

        Returns
        -------
        list of str
        """
        return self.files(type='dark', path=True).path.values

    def bias(self, i, show=False, **kwargs):
        return self.observation_files(i, show=show, **kwargs)['bias']


    @property
    def all_bias(self):
        """fits paths of the observation bias images

        Returns
        -------
        list of str
        """
        return self.files(type='bias', path=True).path.values

    def darks(self, i, show=False, **kwargs):
        return self.observation_files(i, show=show, **kwargs)['darks']

    @property
    def all_flats(self):
        """fits paths of the observation flats images

        Returns
        -------
        list of str
        """
        return self.files(type='flat', path=True).path.values

    def flats(self, i, show=False, **kwargs):
        return self.observation_files(i, show=show, **kwargs)['flats']

    @property
    def stack(self):
        """fits paths of the observation stack image if present

        Returns
        -------
        list of str
        """
        return self.files(imtype="stack")

    @property
    def reduced(self):
        """fits paths of the observation calibrated images if present

        Returns
        -------
        list of str
        """
        return self.files(imtype="reduced")

    def label(self, i):
        date, telescope, filter, _, target, *_ = self.observations(id=i).values[0]
        return f"{telescope}_{date.replace('-', '')}_{target}_{filter}"

    @property
    def obs_name(self):
        """Observation name ({telescope}_{date}_{target}_{filter}) if a single observation is present
        """
        if self.unique_obs:
            return self.label()
        else:
            raise AssertionError("obs_name property is only available for FitsManager containing a unique observation")

    def __repr__(self):
        return str(self.observations())

    def _repr_html_(self):
        return self.observations()._repr_html_()

    def to_pandas(self, query):
        return pd.read_sql_query(query, self.con)

    def _update_observations(self, verbose=False):
        """
        Slow! This should never be used (use update_obs=True in _insert)
        """
        observations = self.to_pandas(f"select {UNIQUE_FIELDS} from files WHERE id is NULL GROUP BY {UNIQUE_FIELDS}").values

        _progress = progress(verbose, desc="observations scan", unit=" obs")

        for obs in _progress(observations):
            # insert obs
            self.con.execute(f"INSERT or IGNORE INTO observations({UNIQUE_FIELDS}, files) VALUES ({QMARKS_UNIQUE}, 0)", obs)
            self.con.commit()

            # get its id
            query = " AND ".join([f"{str(key)} = {in_value(value)}" for key, value in zip(UNIQUE_FIELDS_LIST, obs)])
            obs_id = self.con.execute(f"SELECT id FROM observations where {query}").fetchall()[0][0]

            # and fill files id values
            self.con.execute(f"UPDATE files SET id = ? WHERE id is NULL AND {query}", [obs_id])
            files_updated = self.con.execute("select changes()").fetchall()[0][0]
            self.con.execute("UPDATE observations SET files = files + ? WHERE id = ?", [files_updated, obs_id])