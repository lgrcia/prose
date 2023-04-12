from numpy import trim_zeros

default = dict(
    # Name(s)
    # -------
    name="Unknown",
    names=(),
    # Keywords
    # --------
    keyword_telescope="TELESCOP",
    keyword_object="OBJECT",
    keyword_image_type="IMAGETYP",
    keyword_light_images="light",
    keyword_dark_images="dark",
    keyword_flat_images="flat",
    keyword_bias_images="bias",
    keyword_observation_date="DATE-OBS",
    keyword_exposure_time="EXPTIME",
    keyword_filter="FILTER",
    keyword_airmass="AIRMASS",
    keyword_fwhm="FWHM",
    keyword_seeing="SEEING",
    keyword_ra="RA",
    keyword_dec="DEC",
    keyword_jd="JD",
    keyword_bjd="BJD",
    keyword_flip="PIERSIDE",
    keyword_observation_time=None,
    # Units, formats and scales
    # -------------------------
    ra_unit="deg",
    dec_unit="deg",
    jd_scale="utc",
    bjd_scale="utc",
    mjd=0,
    # Specs
    # -----
    trimming=(0, 0),  # pixels along y/x
    read_noise=9,  # A
    gain=1,  # e-/ADU
    altitude=2000,  # meters
    diameter=100,  # meters
    pixel_scale=None,  # arcsec/pixel
    latlong=(None, None),
    saturation=55000,  # ADUs
    hdu=0,
    camera_name="Andor ikon-L",
)


speculoos_south = dict(
    default,
    name="SS0",
    keyword_telescope="OBSERVAT",
    trimming=(8, 22),  # in pixel along y/x
    read_noise=10,  # in ADU
    gain=1.02,  # in e-/ADU
    altitude=2000,  # in meters
    diameter=100,  # in meters
    pixel_scale=0.33,  # arcsec/pixel
    latlong=(-24.6275, -70.4044),  # (latitude, longitude) in degree
)

callisto = dict(
    speculoos_south,
    name="Callisto",
    names=("SPECULOOS-CALLISTO", "Callisto", "ACP->CALLISTO"),
)

io = dict(speculoos_south, name="Io", names=("SPECULOOS-IO", "Io", "ACP->IO"))

ganymede = dict(
    speculoos_south,
    name="Ganymede",
    names=("SPECULOOS-GANYMEDE", "Ganymede", "ACP->GANYMEDE"),
)

europa = dict(
    speculoos_south, name="Europa", names=("SPECULOOS-EUROPA", "Europa", "ACP->EUROPA")
)

artemis = dict(
    speculoos_south,
    name="Artemis",
    names=("Artemis", "ACP->Artemis", "SPECULOOS-ARTEMIS"),
    ra_unit="hourangle",
    latlong=(28.4754, -16.3089),
)

trappist = dict(
    speculoos_south,
    name="TRAPPIST-South",
    names=("TRAPPIST-S", "TRAPPIST", "ACP->TRAPPIST"),
    trimming=(40, 40),
    pixel_scale=0.64,
    diameter=60,
    ra_unit="deg",
    latlong=(-29.2563, -70.7380),
    camera_name="FLI ProLine PL3041-BB",
)

trappistN = dict(
    speculoos_south,
    name="TRAPPIST-North",
    names=("Trappist-North", "ntm", "ACP->NTM"),
    pixel_scale=0.60,
    diameter=60,
    ra_unit="hourangle",
    latlong=(31.2027, -7.8586),
    camera_name="Andor IKONL BEX2 DD",
)

saintex = dict(
    speculoos_south,
    name="Saint-Ex",
    names=("SAINT-Ex", "ACP->SAINT-EX"),
    ra_unit="hourangle",
    latlong=(31.0439, -115.4637),
)

liverpool = dict(
    speculoos_south,
    name="Liverpool",
    names=("Liverpool Telescope"),
    keyword_telescope="TELESCOP",
    trimming=(8, 22),  # pixels
    read_noise=12,  # ADU
    gain=2.4,  # ADU/e-
    altitude=2363,  # m
    diameter=200,  # cm
    pixel_scale=0.22,  # arcsec/pixel
    latlong=(28.7624, -17.8792),  # deg
    keyword_object="OBJECT",
    keyword_image_type="OBSTYPE",
    keyword_light_images="expose",
    keyword_observation_date="DATE-OBS",
    keyword_exposure_time="EXPTIME",
    keyword_filter="FILTER1",
    keyword_jd="JD_UTC",
    keyword_ra="RA",
    keyword_dec="DEC",
    ra_unit="hourangle",
    TTF_link=None,
    camera_name=None,
)

spirit = dict(
    speculoos_south,
    name="Spirit",
    keyword_telescope="TELESCOP",
    names=("PIRT1280SciCam2"),
    pixel_scale=0.306,
    ra_unit="hourangle",
    trimming=(0, 0),
    saturation=14500,
    read_noise=17,
    gain=5,
    camera_name="PIRT1280SciCam2",
)

built_in_telescopes = {
    telescope["name"].lower(): telescope
    for telescope in (
        callisto,
        io,
        ganymede,
        europa,
        artemis,
        trappist,
        trappistN,
        saintex,
        liverpool,
        spirit,
    )
}
