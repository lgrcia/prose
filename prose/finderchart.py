import numpy as np
from astropy.coordinates import SkyCoord
from prose.console_utils import info
import requests
from astropy.io import fits
from . import Image
import astropy.units as u

class PhotographicPlate(Image):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @property
    def pixel_scale(self):
        return self.header["PLTSCALE"]*(u.arcsec/u.mm) * (self.header["XPIXELSZ"]*u.um).to(u.mm)
    
    @property
    def skycoord(self):
        return SkyCoord(*self.wcs.pixel_to_world_values([np.array(self.shape)/2])[0] * u.deg)
    

def sdss_image(skycoord, fov, filter="poss1_blue", return_hdu=False):
    """A function to retrieve an SDSS ``Image`` object

    Parameters
    ----------
    skycoord : list, tuple or SkyCoord
        coordinate of the image center
    fov : list, tuple or Quantity array
        field of view of the image in the two axes
    filter : str, optional
        type of image to retrieve, by default "poss1_blue". Available are:
            - poss1_blue
            - poss1_red
            - poss2ukstu_blue
            - poss2ukstu_red
            - poss2ukstu_ir
            - quickv

    Returns
    -------
    Image
        ``Image`` object of the SDSS field
    """

    if isinstance(skycoord, (tuple, list)):
        if isinstance(skycoord[0], float):
            skycoord = SkyCoord(*skycoord, unit=(u.deg, u.deg))
        elif isinstance(skycoord[0], str):
            skycoord = SkyCoord(*skycoord, unit=("hourangle", "deg"))
        else:
            if not isinstance(skycoord, SkyCoord):
                assert "'skycoord' must be a list of int (interpreted as deg), str (interpreted as houranlgle, deg) or SkyCoord object"
        
    if isinstance(fov, (tuple, list)):
        fov = np.array(fov)*u.arcmin
    ra, dec = skycoord.to_string().split(' ')
    h, w = fov.to(u.arcmin).value
    url = f"https://archive.stsci.edu/cgi-bin/dss_search?v={filter}&r={ra}&d={dec}&e=J2000&h={h}&w={w}&f=fits&c=none&s=on&fov=NONE&v3="
    info("Querying https://archive.stsci.edu/cgi-bin/dss_form")
    query = requests.get(url)
    hdu = fits.HDUList.fromstring(query.content)
    
    if return_hdu:
        return hdu
    
    else:
        return PhotographicPlate(data=np.array(hdu[0].data).astype(float), header=hdu[0].header)
