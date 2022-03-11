import sqlite3
import numpy as np
import tabulate
from tqdm import tqdm
from pathlib import Path
from .io import get_files, fits_to_df

sql_days_between = "date >= date(:date, '-' || :min_days || ' days') AND date <= date(:date, :max_days || ' days')"

sql_light = f"""
SELECT * FROM files WHERE
type = 'light' AND telescope LIKE :telescope || '%' AND target LIKE :target AND date = :date AND filter LIKE :filter || '%' ORDER BY jd
"""

sql_flat = f"""
SELECT path FROM files WHERE
type = 'flat' AND telescope LIKE :telescope || '%' AND filter = :filter AND width = :w AND height = :h AND
date = (
    SELECT MAX(date) FROM files WHERE 
    type = 'flat' AND telescope LIKE :telescope || '%' AND filter = :filter AND width = :w AND height = :h AND
    {sql_days_between} 
)
"""

def sql_other(kind, exposure=None, tolerance=1000000):
    exposure_constraint = f"exposure between {exposure-tolerance} and {exposure+tolerance} AND" if exposure is not None else ""

    return f"""
    SELECT path FROM files WHERE
    type = '{kind}' AND telescope LIKE :telescope || '%' AND width = :w AND height = :h AND date = (
        SELECT MAX(date) FROM files WHERE 
        type = '{kind}' AND telescope LIKE :telescope || '%' AND width = :w AND height = :h AND
        {exposure_constraint}
        {sql_days_between}
    )
    """

class FitsManager:
    """Object to parse and retrive FITS files from a folder and its sub-folders

        Parameters
        ----------
        folder : str
            path of the folder to parse
        depth : int, optional
            maxiumum depth of the sub-folders to explore, by default 0
        hdu : int, optional
            by default 0
        extension : str, optional
            by default ".f*ts*"
    """
    
    def __init__(self, folder=None, depth=0, hdu=0, extension=".f*t*", file=None, batch_size=False):
        if file is None:
            file = ":memory:"

        self.con = sqlite3.connect(file)
        self.cur = self.con.cursor()

        # check if file Table exists
        tables = list(self.cur.execute("SELECT name FROM sqlite_master WHERE type='table';"))
        if len(tables) == 0:
            self.cur.execute('''CREATE TABLE files (date text, path text UNIQUE, 
            telescope text, filter text, type text, target text, width int, height int, 
            jd real, observation_id int, exposure real)''')
            self.con.commit()
        
        if folder is not None:
            self.scan_files(folder, extension, batch_size=batch_size, depth=depth)
        # else:
        #     raise AssertionError(f"No files with extension '{extension}'found")

        _observations = self.observations(show=False, index=False)
        self._observations = np.array([f"{o[0]}_{o[1]}_{o[2]}_{o[3]}" for o in _observations])

    def insert(self, _path, date, telescope, _type, target, _filter, dimensions, _, jd, exposure):
        if isinstance(_filter, float):
            _filter = ""
        width, height = dimensions
        # or IGNORED to handle the unique constraint
        self.cur.execute(
            f"INSERT or IGNORE INTO files VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (date, _path, telescope, _filter, _type, target, width, height, jd, "NULL", exposure))

    def scan_files(self, folder, extension, batch_size=False, verbose=True, depth=0):
        assert Path(folder).exists(), "Folder does not exists"
        files = get_files(extension, folder, depth=depth)

        if len(files) > 0:
            current_files = [v[0] for v in self.cur.execute("SELECT path from files").fetchall()]
            files_to_scan = np.setdiff1d(files, current_files)

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
                
                if verbose:
                    if batch_size is not False:
                        def progress(x): return tqdm(x, bar_format='Reading fits {l_bar}{bar}{r_bar}')
                else:
                    def progress(x): return x

                if batch_size is not False:
                    for batch in progress(batches):
                        df = fits_to_df(batch, verbose=False)
                        for row in df.values:
                            self.insert(*row)
                        self.con.commit()
                else:
                    df = fits_to_df(files_to_scan, verbose=True)
                    for row in df.values:
                        self.insert(*row)
                    self.con.commit()
            else:
                print(f"No new files to scan")
        else:
            raise AssertionError(f"No files with extension '{extension}' found")
        
    def print(self, calib=True, repr=False):
        txt = []
        fields = ["date", "telescope", "target", "filter", "exposure", "type", "quantity"]

        observations = self.observations(telescope="*", target="*", date="*", afilter="*", show=False, index=False)
        _observations = observations.copy()
        if len(observations):
            for i, obs in enumerate(observations):
                observations[i] = (i, ) + obs
            txt.append("Observations:")
            txt.append(tabulate.tabulate(observations, headers=["index", *fields], tablefmt="fancy_grid"))
            
        calibs = [
            self.observations(telescope="*", target="*", date="*", afilter="*", imtype=imtype, show=False, index=False)
            for imtype in ["bias", "dark", "flat"]
        ]
        calibs = [c for ca in calibs for c in ca] # flattening

        if calib:
            if len(calibs):
                txt.append("Calibrations:")
                txt.append(tabulate.tabulate(calibs, headers=fields, tablefmt="fancy_grid"))
            
        others = set(self.observations(telescope="*", target="*", date="*", afilter="*", imtype="*", show=False, index=False))
        obs_and_calibs = set(_observations)
        obs_and_calibs.update(calibs)
        others = others.difference(obs_and_calibs)

        if len(others):
            txt.append("Others:")
            txt.append(tabulate.tabulate(others, headers=fields, tablefmt="fancy_grid"))

        txt = "\n".join(txt)
        
        if repr:
            return txt
        else:
            print(txt)
        
    def observations(self, telescope="", target="", date="", afilter="", imtype="light", show=True, index=True):
        fields = ["date", "telescope", "target", "filter", "exposure", "type", "quantity"]
        telescope=telescope.replace("*", "%")
        target=target.replace("*", "%")
        date=date.replace("*", "%")
        afilter=afilter.replace("*", "%")
        imtype=imtype.replace("*", "%")

        result = list(
            self.cur.execute(
                f"""
                SELECT {",".join(fields[0:-1])}, COUNT(date) FROM files WHERE
                type LIKE :imtype || '%' AND 
                (telescope LIKE :telescope || '%' OR telescope IS NULL) AND 
                (target LIKE :target || '%' OR target IS NULL) AND 
                date LIKE :date || '%' AND 
                filter LIKE :filter || '%'
                GROUP BY date, telescope, target, filter, exposure type ORDER BY date
                """
                , {"telescope":telescope, "target":target, "date": date, "filter":afilter, "imtype":imtype}))
        
        if index:
            fields.insert(0, "index")
            if len(result) > 0:
                for i, r in enumerate(result):
                    result[i] = list(r)
                    j = np.flatnonzero(self._observations == f"{r[0]}_{r[1]}_{r[2]}_{r[3]}")[0]
                    result[i].insert(0, j)
        
        if show:
            print(tabulate.tabulate(result, headers=fields, tablefmt="fancy_grid"))
        else: 
            return result
    
    def files(
        self,
        telescope="", 
        target="", 
        date="", 
        afilter="", 
        imtype="",
    ):
        telescope=telescope.replace("*", "%")
        target=target.replace("*", "%")
        date=date.replace("*", "%")
        afilter=afilter.replace("*", "%")
        imtype=imtype.replace("*", "%")
        
        files_paths = list(
            self.cur.execute(
                f"""
                SELECT path FROM files WHERE
                type LIKE :imtype || '%' AND 
                (telescope LIKE :telescope || '%' OR telescope IS NULL) AND 
                (target LIKE :target || '%' OR target IS NULL) AND 
                date LIKE :date || '%' AND 
                filter LIKE :filter || '%'
                ORDER BY JD
                """
                , {"telescope":telescope, "target":target, "date": date, "filter":afilter, "imtype":imtype}))
        
        return [f[0] for f in files_paths]

    def observation_files(
        self, 
        i, 
        min_days = 100000, 
        max_days = 0, 
        same_telescope=True, 
        lights="images", 
        darkexp_tolerance=100000000
    ):
        """Return a dictionary of all the files (including calibration ones) corresponding to a given observation.

        The index ``i`` corresponds to the observation index displayed when printing the ``FitsManager`` object

        Parameters
        ----------
        i : int
            index of the observation
        min_days : int, optional
            number of days in the past before which calibration files are considered valid, by default 100000
        max_days : int, optional
            number of days in the future after which calibration files are considered valid, by default 0. Set to more than 0 to use calibration files taken after the observation date
        same_telescope : bool, optional
            Wether the calibration images should come from the same telescope (if its name is written in the headers), by default True 
        lights : str, optional
            keyword used in the written dictionary for science images, by default "images"

        Returns
        -------
        dict
            The keys of the dictionnary are:
                - ``images` for the science images
                - ``flats``, ``darks`` and ``bias`` for the corresponding calibration images
        """
        obs = self.observations(show=False, index=False)
        if len(obs) == 0:
            return None
        elif i>len(obs)-1:
            raise AssertionError(f"observation {i} out of range ({len(obs)} found)")
        else:
            date, telescope, target, afilter, exposure, _, _ = obs[i]

        kwargs = {
            "min_days": min_days,
            "max_days": max_days,
            "target": target,
            "date": date,
            "telescope": telescope if same_telescope else '',
            "filter": afilter
        }

        images = {}

        _lights = np.array(list(self.cur.execute(sql_light, kwargs))).T
        images[lights] = _lights[1]
        w = np.unique(_lights[6])[0]
        h = np.unique(_lights[7])[0]
        kwargs.update({"w": w, "h": h})

        images["bias"] = np.array(list(self.cur.execute(sql_other("bias"), kwargs)))
        images["darks"]  = np.array(list(self.cur.execute(sql_other("dark", exposure, darkexp_tolerance), kwargs)))
        images["flats"]  = np.array(list(self.cur.execute(sql_flat, kwargs)))
        
        if len(images["bias"]):
            images["bias"] = [im for ims in images["bias"] for im in ims]
        if len(images["flats"]):
            images["flats"] = [im for ims in images["flats"] for im in ims]
        if len(images["darks"]):
            images["darks"] = [im for ims in images["darks"] for im in ims]

        return images
    
    @property
    def unique_obs(self):
        """Return whether the object contains a unique observation (observation is defined as a unique combinaison of date, telescope, target and filter).

        Returns
        -------
        bool
        """
        return len(self.observations(show=False)) == 1

    @property
    def images(self):
        """fits paths of the observation science images

        Returns
        -------
        list of str
        """
        return self.files(imtype="light")

    @property
    def darks(self):
        """fits paths of the observation dark images

        Returns
        -------
        list of str
        """
        return self.files(imtype="dark")

    @property
    def bias(self):
        """fits paths of the observation bias images

        Returns
        -------
        list of str
        """
        return self.files(imtype="bias")

    @property
    def flats(self):
        """fits paths of the observation flats images

        Returns
        -------
        list of str
        """
        return self.files(imtype="flat")

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
    
    def products_denominator(self, i=0):
        date, telescope, target, afilter, _, _, _ = self.observations(show=False, index=False)[i]
        return f"{telescope}_{date.replace('-', '')}_{target}_{afilter}"
    
    @property
    def obs_name(self):
        if self.unique_obs:
            return self.products_denominator()
        else:
            raise AssertionError("obs_name property is only available for FitsManager containing a unique observation")

    def __repr__(self) -> str:
        return self.print(repr=True)