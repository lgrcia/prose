speculoos_south = {
    "keyword_object": "OBJECT",  # keyword holding object or target name
    "keyword_image_type": "IMAGETYP",  # keyword holding image type (calibration, science... etc)
    "keyword_light_images": "light",  # value of `keyword_image_type` for science images ...
    "keyword_dark_images": "dark",  # for dark images ...
    "keyword_flat_images": "flat",  # for flat images ...
    "keyword_bias_images": "bias",  # for bias images
    "keyword_observation_date": "DATE-OBS",  # keyword holding observation date (ISO)
    "keyword_exposure_time": "EXPTIME",  # keyword holding exposure time
    "keyword_filter": "FILTER",  # keyword holding filter name
    "keyword_observatory": "OBSERVAT",  # keyword holding observatory name
    "keyword_jd": "JD",  # keyword holding time in BJD
    "keyword_bjd": "BJD",  # keyword holding time in BJD
    "keyword_ra": "RA",  # keyword holding RA
    "keyword_dec": "DEC",  # keyword holding DEC
    "keyword_flip": "PIERSIDE",  # keyword holding meridan flip (WEST or EAST)
    "keyword_fwhm": "FWHM",
    "keyword_seeing": "SEEING",
    "keyword_airmass": "AIRMASS",
    "name": "SSO",
    "trimming": [8, 22],  # in piwel along y/x
    "read_noise": 10,  # in ADU
    "gain": 1.02,  # in e-/ADU
    "altitude": 2000,  # in meters
    "diameter": 100,  # in meters
    "pixel_scale": 0.33,  # in arcseconds
    "latlong": [-24.6275, -70.4044], # [latitude, longitude] in degree
    "TTF_link": "https://astro.swarthmore.edu/telescope/tess-secure/print_eclipses.cgi?observatory_string=-24.6272%3B-70.4039%3"
                "BAmerica%2FSantiago%3BSPECULOOS-South+Observatory+%284x1.0-m%29%3BSPECULOOS+South&use_utc=1&observatory_latitude=-29.2563"
                "&observatory_longitude=-70.738&timezone=UTC&start_date={date}&days_to_print=90&days_in_past=0&minimum_start_elevation="
                "23&and_vs_or=or&minimum_end_elevation=23&minimum_ha=-12&maximum_ha=12&baseline_hrs=1.5&show_unc=1&maximum_priority=4&"
                "minimum_depth=0&maximum_V_mag=&target_string={tic_id}&lco_only=0&single_object=0&ra=&dec=&epoch=&period=&duration=&"
                "target=&show_ephemeris=0&print_html=1&twilight=-12&max_airmass=2.4"
}

callisto = speculoos_south.copy()
callisto.update({"name": "Callisto"})

io = speculoos_south.copy()
io.update({"name": "Io"})

ganymede = speculoos_south.copy()
ganymede.update({"name": "Ganymede"})

europa = speculoos_south.copy()
europa.update({"name": "Europa"})

artemis = speculoos_south.copy()
artemis.update({
    "name": "Artemis",
    "ra_unit": "hourangle",
    "latlong": [28.4754, -16.3089],
    "TTF_link": "https://astro.swarthmore.edu/telescope/tess-secure/print_eclipses.cgi?observatory_string=28.2999%3B-16.5058%3BAtlantic%2FCanary%3"
                "BSPECULOOS-Northern+Observatory+1.0m%3BSPECULOOS-North+1.0m&use_utc=1&observatory_latitude=-29.2563&observatory_longitude=-70.738&"
                "timezone=UTC&start_date={date}&days_to_print=90&days_in_past=0&minimum_start_elevation=23&and_vs_or=or&minimum_end_elevation=23&"
                "minimum_ha=-12&maximum_ha=12&baseline_hrs=1.5&show_unc=1&maximum_priority=4&minimum_depth=0&maximum_V_mag=&target_string={tic_id}"
                "&lco_only=0&single_object=0&ra=&dec=&epoch=&period=&duration=&target=&show_ephemeris=0&print_html=1&twilight=-12&max_airmass=2.4"})

trappist = speculoos_south.copy()
trappist.update({
    "name": "Trappist-South",
    "trimming": [40, 40],
    "pixel_scale": 0.64,
    "ra_unit": "deg",
    "latlong": [-29.2563, -70.7380],
    "TTF_link": "https://astro.swarthmore.edu/telescope/tess-secure/print_eclipses.cgi?observatory_string=-29.2563%3B-70.738%3BAmerica%2FSantiago%3" \
                "BTRAPPIST-South+%280.6-m%29%3BTRAPPIST+South&use_utc=1&observatory_latitude=-29.2563&observatory_longitude=-70.738&timezone=UTC&" \
                "start_date={date}&days_to_print=90&days_in_past=0&minimum_start_elevation=23&and_vs_or=or&minimum_end_elevation=23&minimum_ha=-12" \
                "&maximum_ha=12&baseline_hrs=1.5&show_unc=1&maximum_priority=4&minimum_depth=0&maximum_V_mag=&target_string={tic_id}&lco_only=0&" \
                "single_object=0&ra=&dec=&epoch=&period=&duration=&target=&show_ephemeris=0&print_html=1&twilight=-12&max_airmass=2.4"})

trappistN = trappist.copy()
trappistN.update({
    "name": "Trappist-North",
    "names": ["Trappist-North", "ntm"],
    "pixel_scale": 0.60,
    "ra_unit": "hourangle",
    "latlong": [31.2027, -7.8586],
    "TTF_link": "https://astro.swarthmore.edu/telescope/tess-secure/print_eclipses.cgi?observatory_string=31.2061%3B-7.8664%3BAfrica%2FCasablanca%3"
                "BTRAPPIST-North+%280.6-m%29%3BTRAPPIST+North&use_utc=1&observatory_latitude=-29.2563&observatory_longitude=-70.738&timezone=UTC&"
                "start_date={date}&days_to_print=90&days_in_past=0&minimum_start_elevation=23&and_vs_or=or&minimum_end_elevation=23&minimum_ha=-12"
                "&maximum_ha=12&baseline_hrs=1.5&show_unc=1&maximum_priority=4&minimum_depth=0&maximum_V_mag=&target_string={tic_id}&lco_only=0&"
                "single_object=0&ra=&dec=&epoch=&period=&duration=&target=&show_ephemeris=0&print_html=1&twilight=-12&max_airmass=2.4"})

saintex = artemis.copy()
saintex.update({
    "name": "Saint-Ex",  
    "ra_unit": "hourangle",
    "latlong": [31.0439, -115.4637],
    "TTF_link": "https://astro.swarthmore.edu/telescope/tess-secure/print_eclipses.cgi?observatory_string=31.04342%3B-115.45476%3BPST8PDT%3"
                "BSAINT-EX+1.0m%3BSAINT-EX+1.0m&use_utc=1&observatory_latitude=-29.2563&observatory_longitude=-70.738&timezone=UTC&"
                "start_date={date}&days_to_print=90&days_in_past=0&minimum_start_elevation=23&and_vs_or=or&minimum_end_elevation=23&"
                "minimum_ha=-12&maximum_ha=12&baseline_hrs=1.5&show_unc=1&maximum_priority=4&minimum_depth=0&maximum_V_mag=&"
                "target_string={tic_id}&lco_only=0&single_object=0&ra=&dec=&epoch=&period=&duration=&target=&show_ephemeris=0&"
                "print_html=1&twilight=-12&max_airmass=2.4"})

liverpool = speculoos_south.copy()
liverpool.update ({
    "name": "Liverpool Telescope",
    "trimming": [8, 22], # pixels
    "read_noise": 12, # ADU
    "gain": 2.4, # ADU/e-
    "altitude": 2363, # m
    "diameter": 200, # cm
    "pixel_scale": 1.0752, # arcsec
    "latlong": [28.7624, -17.8792], # deg
    "keyword_object": "OBJECT",
    "keyword_image_type": "OBSTYPE",
    "keyword_light_images": "expose",
    "keyword_observation_date": "DATE-OBS",
    "keyword_exposure_time": "EXPTIME",
    "keyword_filter": "FILTER1",
    "keyword_observatory": "TELESCOP",
    "keyword_jd": "JD_UTC",
    "keyword_ra": "RA",
    "keyword_dec": "DEC",
    "TTF_link": None
})

built_in_telescopes = {
    "trappist": trappist,
    "artemis": artemis,
    "europa": europa,
    "saintex": saintex,
    "ganymede": ganymede,
    "io": io,
    "callisto": callisto,
    "ntm": trappistN,
    "trappist-north": trappistN,
    "saint-ex": saintex,
    "liverpool": liverpool
}

TESS_pixel = 2.5 # arcmin