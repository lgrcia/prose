speculoos_south = {
    "keyword_object": "OBJECT", # keyword holding object or target name
    "keyword_image_type": "IMAGETYP", # keyword holding image type (calibration, science... etc)
    "keyword_light_images": "light", # value of `keyword_image_type` for science images ...
    "keyword_dark_images": "dark", # for dark images ...
    "keyword_flat_images": "flat", # for flat images ...
    "keyword_bias_images": "bias", # for bias images
    "keyword_observation_date": "DATE-OBS", # keyword holding observation date (ISO)
    "keyword_exposure_time": "EXPTIME", # keyword holding exposure time
    "keyword_filter": "FILTER", # keyword holding filter name
    "keyword_observatory": "OBSERVAT", # keyword holding observatory name
    "keyword_julian_date": "JD", # keyword holding time in JD
    "keyword_ra": "RA", # keyword holding RA
    "keyword_dec": "DEC", # keyword holding DEC
    "keyword_flip": "PIERSIDE", # keyword holding meridan flip (WEST or EAST) 
    "keyword_fwhm": "FWHM",
    "keyword_seeing": "SEEING",
    "keyword_airmass": "AIRMASS",
    "name": "SSO",
    "trimming": [8, 22], # in piwel along y/x
    "read_noise": 10, # in ADU
    "gain": 1.02, # in e-/ADU
    "altitude": 2000, # in meters
    "diameter": 100, # in meters
    "pixel_scale": 0.33, # in arcseconds
    "latlong": [24.6275, 70.4044] # [latitude, longitude] in degree
}

callisto = speculoos_south.copy()
callisto.update({"name": "Callisto"})

io = speculoos_south.copy()
io.update({"name": "Io"})

ganymede = speculoos_south.copy()
ganymede.update({"name": "Ganymede"})

europa = speculoos_south.copy()
europa.update({"name": "Europa"})

saintex = speculoos_south.copy()
saintex.update({"name": "SaintEx"})

artemis = speculoos_south.copy()
artemis.update({
    "name": "Artemis",
    "latlong": [28.4754, 16.3089]})

trappist = speculoos_south.copy()
trappist.update({
    "name": "Trappist",
    "trimming": [40, 40],
    "pixel_scale": 0.66, 
    "latlong": [29.2563, 70.7380]})

trappistN = trappist.copy()
trappistN.update({
    "name": "NTM",  
    "latlong": [31.2027, 7.8586]})

built_in_telescopes = {
    "trappist": trappist,
    "artemis": artemis,
    "europa": europa,
    "saintex": saintex,
    "ganymede": ganymede,
    "io": io,
    "callisto": callisto,
    "ntm": trappistN
}