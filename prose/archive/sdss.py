import astropy.units as u
import numpy as np
import requests
from astropy.io import fits

from .. import FITSImage, Telescope, utils
from ..console_utils import info


def sdss_image(skycoord, fov, filter="poss1_blue", return_hdu=False):
    """A function to retrieve an SDSS ``Image`` object

    Parameters
    ----------
    skycoord : list, tuple or SkyCoord
        Coordinate of the image center, either:
            - a list of int (interpreted as deg)
            - a str (interpreted as houranlgle, deg)
            - a SkyCoord object
    fov : list, tuple or Quantity array
        field of view of the image in the two axes. If list or tuple, interpreted as arcmin
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

    skycoord = utils.check_skycoord(skycoord)

    if isinstance(fov, (tuple, list)):
        fov = np.array(fov) * u.arcmin

    ra, dec = skycoord.to_string().split(" ")
    h, w = fov.to(u.arcmin).value
    url = f"https://archive.stsci.edu/cgi-bin/dss_search?v={filter}&r={ra}&d={dec}&e=J2000&h={h}&w={w}&f=fits&c=none&s=on&fov=NONE&v3="
    info("Querying https://archive.stsci.edu/cgi-bin/dss_form")
    query = requests.get(url)
    hdu = fits.HDUList.fromstring(query.content)

    if return_hdu:
        return hdu
    else:
        pixel_scale = (
            hdu[0].header["PLTSCALE"]
            * (u.arcsec / u.mm)
            * (hdu[0].header["XPIXELSZ"] * u.um).to(u.mm)
        ).value
        telescope = Telescope(pixel_scale=pixel_scale)

        hdu[0].header["DATE-OBS"] = hdu[0].header["DATE-OBS"][0:10]
        image = FITSImage(hdu[0], telescope=telescope)
        image.metadata["ra"] = ra
        image.metadata["dec"] = dec
        image.metadata["filter"] = filter
        return image
