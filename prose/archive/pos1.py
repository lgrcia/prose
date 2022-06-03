import numpy as np
from astropy.table import Table
import requests
from io import StringIO
from .. import Image, blocks, utils
from prose.console_utils import info
from astropy.io import fits
import astropy.units as u
from astropy.time import Time

ps1_pixel_scale = 0.258

class PS1Image(Image):

    ra = None
    dec = None
    filter = None
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        
    @property
    def pixel_scale(self):
        return ps1_pixel_scale * u.arcsec
    
    @property
    def jd_utc(self):
        return self.header["MJD-OBS"] + 2400000.5
        
    @property
    def date(self):
        """datetime of the observation

        Returns
        -------
        datetime.datetime
        """
        return Time(self.jd_utc, format="jd", scale="utc").to_datetime()
 

def pos1_image(skycoord, fov, filter="z"):
    """
    A function to retrieve an PS1 ``Image`` object

    Parameters
    ----------
    skycoord : list, tuple or SkyCoord
        Coordinate of the image center, either:
            - a list of int (interpreted as deg)
            - a str (interpreted as houranlgle, deg)
            - a SkyCoord object
    fov : list, tuple or Quantity array
        field of view of the image in the two axes. If list or tuple, interpreted as arcmin
    filter : str
        one of ["g", "r", "i", "z", "y"]
    
    helped by https://outerspace.stsci.edu/display/PANSTARRS/PS1+Image+Cutout+Service
    """

    skycoord = utils.check_skycoord(skycoord)

    if isinstance(fov, (tuple, list)):
        fov = np.array(fov)*u.arcmin
    
    w = (fov[0]/(ps1_pixel_scale * u.arcsec)).decompose().value
    h = (fov[1]/(ps1_pixel_scale * u.arcsec)).decompose().value
    ra = skycoord.ra.to(u.deg).value
    dec = skycoord.dec.to(u.deg).value
    
    size = int(np.max([w, h]))
        
    cbuf = StringIO()
    cbuf.write(f"{ra} {dec}")
    cbuf.seek(0)

    info("Querying https://ps1images.stsci.edu/cgi-bin/ps1filenames.py")
    r = requests.post(
        "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py", 
        data=dict(filters=filter, type="stack"),
        files=dict(file=cbuf)
    )
    r.raise_for_status()
    tab = Table.read(r.text, format="ascii")
 
    urlbase = "{}?size={}&format={}".format("https://ps1images.stsci.edu/cgi-bin/fitscut.cgi", size, "fits")
    tab["url"] = ["{}&ra={}&dec={}&red={}".format(urlbase,ra,dec,filename)
            for (filename,ra,dec) in zip(tab["filename"],tab["ra"],tab["dec"])]
    
    url = tab[0]["url"]
    query = requests.get(url)
    hdu = fits.HDUList.fromstring(query.content)
    image = PS1Image(data=hdu[0].data, header=hdu[0].header, verbose=False)
    image = blocks.Trim(trim=[int((size-w)/2), int((size-h)/2)])(image)
    image.ra = skycoord.ra
    image.dec = skycoord.dec
    image.filter = filter
    return image