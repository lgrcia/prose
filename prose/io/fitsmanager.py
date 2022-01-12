import sqlite3
import numpy as np
import tabulate
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

def sql_other(kind):
    return f"""
    SELECT path FROM files WHERE
    type = '{kind}' AND telescope LIKE :telescope || '%' AND width = :w AND height = :h AND date = (
        SELECT MAX(date) FROM files WHERE 
        type = '{kind}' AND telescope LIKE :telescope || '%' AND width = :w AND height = :h AND
        {sql_days_between}
    )
    """

class FitsManager:
    
    def __init__(self, folder, depth=0, hdu=0, extension=".f*ts*"):
        self.folder = folder
        self.con = sqlite3.connect(':memory:')
        self.cur = self.con.cursor()
        self.cur.execute('''CREATE TABLE files (date text, path text UNIQUE, 
        telescope text, filter text, type text, target text, width int, height int, jd real, observation_id int)''')
        
        files = get_files(extension, folder, depth=depth)
        df = fits_to_df(files, hdu=hdu)
        for row in df.values:
            _path, date, telescope, _type, target, _filter, dimensions, _, jd = row
            if isinstance(_filter, float):
                _filter = ""
            width, height = dimensions
            # or IGNORED to handle the unique constraint
            self.cur.execute(
                f"INSERT or IGNORE INTO files VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (date, _path, telescope, _filter, _type, target, width, height, jd, "NULL"))
            self.con.commit()
        
    
    def print(self, calib=True, repr=False):
        txt = []
        fields = ["date", "telescope", "target", "filter", "type", "quantity"]
        
        observations = self.observations(telescope="*", target="*", date="*", afilter="*", show=False)
        observations = [list(o) for o in observations]
        if len(observations):
            for i, obs in enumerate(observations):
                obs.insert(0, i)
            txt.append("Observations:")
            txt.append(tabulate.tabulate(observations, headers=["index", *fields], tablefmt="fancy_grid"))
            
        calibs = [
            self.observations(telescope="*", target="*", date="*", afilter="*", imtype=imtype, show=False)
            for imtype in ["bias", "dark", "flat"]
        ]
        calibs = [c for ca in calibs for c in ca] # flattening

        if calib:
            if len(calibs):
                txt.append("Calibrations:")
                txt.append(tabulate.tabulate(calibs, headers=fields, tablefmt="fancy_grid"))
            
        reduced = self.observations(telescope="*", target="*", date="*", afilter="*", imtype="reduced", show=False)
        if len(reduced):
            txt.append("Reduced:")
            txt.append(tabulate.tabulate(reduced, headers=fields, tablefmt="fancy_grid"))

        txt = "\n".join(txt)
        
        if repr:
            return txt
        else:
            print(txt)
        
    def observations(self, telescope="", target="", date="", afilter="", imtype="light", show=True):
        fields = ["date", "telescope", "target", "filter", "type", "quantity"]
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
                GROUP BY date, telescope, target, filter, type ORDER BY date
                """
                , {"telescope":telescope, "target":target, "date": date, "filter":afilter, "imtype":imtype}))
        if show:
            print(tabulate.tabulate(result, headers=fields, tablefmt="fancy_grid"))
        else: 
            return result
    
    def files(self, telescope="", target="", date="", afilter="", imtype=""):
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
                ORDER BY date
                """
                , {"telescope":telescope, "target":target, "date": date, "filter":afilter, "imtype":imtype}))
        
        return [f[0] for f in files_paths]
        
    def observation_files(self, i, min_days = 100000, max_days = 0, same_telescope=True):
        obs = self.observations(show=False)
        if len(obs) == 0:
            return None
        elif i>len(obs)-1:
            raise AssertionError(f"observation {i} out of range ({len(obs)} found)")
        else:
            date, telescope, target, afilter, _, _ = obs[i]

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
        images["lights"] = _lights[1]
        w = np.unique(_lights[6])[0]
        h = np.unique(_lights[7])[0]
        kwargs.update({"w": w, "h": h})

        images["bias"] = np.array(list(self.cur.execute(sql_other("bias"), kwargs)))
        images["darks"]  = np.array(list(self.cur.execute(sql_other("dark"), kwargs)))
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
    def calibrated(self):
        """fits paths of the observation calibrated images if present

        Returns
        -------
        list of str
        """
        return self.files(imtype="reduced")
    
    def products_denominator(self, i=0):
        date, telescope, target, afilter, _, _ = self.observations(show=False)[i]
        return f"{telescope}_{date.replace('-', '')}_{target}_{afilter}"
    
    @property
    def obs_name(self):
        if self.unique_obs:
            return self.products_denominator()
        else:
            raise AssertionError("obs_name property is only available for FitsManager containing a unique observation")

    def __repr__(self) -> str:
        return self.print(repr=True)