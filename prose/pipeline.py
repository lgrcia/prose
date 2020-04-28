from prose import io
from prose import utils
from prose.pipeline_methods import alignment, \
    calibration, psf, detection

from prose.pipeline_methods import photometry as phot

import os
import imageio
import warnings
import astroalign
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os import path
from tqdm import tqdm
from astropy.io import fits
from scipy.spatial import KDTree
from skimage.transform import resize, AffineTransform, warp
from astropy.time import Time
from prose.console_utils import TQDM_BAR_FORMAT, INFO_LABEL
from prose import visualisation as viz
from astropy.wcs import WCS, FITSFixedWarning
from astropy.nddata import Cutout2D


def return_method(name):
    """
    Return the method identified byt the given name
    
    Parameters
    ----------
    name : str or callable
        If str, identifier (name) of the method. See code for details. If callable will return this callable
    
    Returns
    -------
    function
        callable method
    
    Raises
    ------
    ValueError
        If name is not in methods
    ValueError
        If name is not str or callable
    """
    if isinstance(name, str):
        # Alignment
        if name == "xyshift":
            return alignment.xyshift
        elif name == "astroalign":
            return alignment.astroalign_optimized_find_transform
        # PSF fitting
        elif name == "gaussian2d_linear":
            return psf.fit_gaussian2d_linear
        elif name == "gaussian2d":
             return psf.fit_gaussian2d
        elif name == "projected_psf_fit":
            return psf.photutil_epsf
        # Stars finder
        elif name == "daofind":
            return detection.daofindstars
        elif name == "segmentation":
            return detection.segmented_peaks
        # Photometry
        elif name == "aperture":
            return phot.aperture_photometry_annulus
        elif name == "psf":
            return phot.psf_photometry_basic
        # Caolibration
        elif name == "calibration":
            return calibration.calibration
        elif name == "no_calibration":
            return calibration.no_calibration
        # Exception
        else:
            raise ValueError(
                "{} does not correspond to any built in method".format(name)
            )
    elif callable(name):
        return name
    else:
        raise ValueError(
            "{} must be the name of a built-in function or a function".format(name)
        )


class Calibration:
    def __init__(
        self,
        folder=None,
        verbose=True,
        telescope_kw="TELESCOP",
        fits_manager=None,
        _calibration="calibration",
        deepness=1
    ):
        if fits_manager is not None:
            assert isinstance(
                fits_manager, io.FitsManager
            ), "fits_manager must be a FitsManager object"
            self.fits_explorer = fits_manager
        else:
            self.fits_explorer = io.FitsManager(
                folder, verbose=verbose, telescope_kw=telescope_kw, deepness=deepness
            )

        self.telescope = self.fits_explorer.telescope

        self.calibration = _calibration

        self.master_dark = None
        self.master_flat = None
        self.master_bias = None

    @property
    def calibration(self):
        return self._calibration

    @calibration.setter
    def calibration(self, name):
        self._calibration = return_method(name)
        self.calibration_kwargs = {}

    def _produce_master(self, image_type):
        _master = []
        kw_exp_time = self.telescope.keyword_exposure_time
        images = self.fits_explorer.get(image_type)
        assert len(images) > 0, "No {} images found".format(image_type)
        for fits_path in images:
            hdu = fits.open(fits_path)
            primary_hdu = hdu[0]
            image, header = primary_hdu.data, primary_hdu.header
            hdu.close()
            image = self.fits_explorer.trim(image, raw=True)
            if image_type == "dark":
                _dark = (image - self.master_bias) / header[kw_exp_time]
                _master.append(_dark)
            elif image_type == "bias":
                _master.append(image)
            elif image_type == "flat":
                _flat = image - (self.master_bias + self.master_dark)*header[kw_exp_time]
                _flat /= np.mean(_flat)
                _master.append(_flat)
        
        if image_type == "dark":
            self.master_dark = np.mean(_master, axis=0)
        elif image_type == "bias":
            self.master_bias = np.mean(_master, axis=0)
        elif image_type == "flat":
            self.master_flat = np.median(_master, axis=0)


    def produce_masters(self):
        self._produce_master("bias")
        self._produce_master("dark")
        self._produce_master("flat")

    def plot_masters(self):
        plt.figure(figsize=(40, 10))
        plt.subplot(131)
        plt.title("Master bias")
        im = plt.imshow(utils.z_scale(self.master_bias), cmap="Greys_r")
        viz.add_colorbar(im)
        plt.subplot(132)
        plt.title("Master dark")
        im = plt.imshow(utils.z_scale(self.master_dark), cmap="Greys_r")
        viz.add_colorbar(im)
        plt.subplot(133)
        plt.title("Master flat")
        im = plt.imshow(utils.z_scale(self.master_flat), cmap="Greys_r")
        viz.add_colorbar(im)

    def calibrate(self, im_path, flip=False):
        hdu = fits.open(im_path)
        primary_hdu = hdu[0]
        image, header = self.fits_explorer.trim(primary_hdu.data, raw=True), primary_hdu.header
        hdu.close()
        exp_time = header[self.telescope.keyword_exposure_time]
        calibrated_image = self.calibration(image, exp_time, self.master_bias, self.master_dark, self.master_flat)

        if flip:
            calibrated_image = calibrated_image[::-1, ::-1]

        return calibrated_image


class Reduction:
    def __init__(
        self,
        folder=None,
        verbose=False,
        alignment="xyshift",
        fwhm="gaussian2d",
        stars_detection="segmentation",
        calibration="calibration",
        fits_manager=None,
        deepness=1
    ):
        self.calibration = Calibration(
            folder, verbose=verbose, fits_manager=fits_manager, deepness=deepness
        )
        self.fits_explorer = self.calibration.fits_explorer

        self.telescope = self.fits_explorer.telescope

        self.light_files = None

        self.alignment = alignment
        self.stars_detection = stars_detection
        self.fwhm = fwhm

        self.alignment_kwargs = {}
        self.stars_detection_kwargs = {"n_stars": 50}
        self.fwhm_kwargs = {}

        self.calibration.calibration = calibration

        self.data = pd.DataFrame()

    @property
    def alignment(self):
        return self._alignment

    @alignment.setter
    def alignment(self, name):
        self._alignment = return_method(name)
        self.alignment_kwargs = {}

    @property
    def fwhm(self):
        return self._fwhm

    @fwhm.setter
    def fwhm(self, name):
        self._fwhm = return_method(name)
        self.fwhm_kwargs = {}

    @property
    def stars_detection(self):
        return self._stars_detection

    @stars_detection.setter
    def stars_detection(self, name):
        self._stars_detection = return_method(name)
        self.stars_detection_kwargs = {}

    def set_observation(
        self,
        observation_id,
        check_calib_telescope=True,
        keep_closest_calibration=True,
        calibration_date_limit=0,
        produce_masters=True
    ):

        self.fits_explorer.set_observation(
            observation_id,
            check_calib_telescope,
            keep_closest_calibration,
            calibration_date_limit)

        self.light_files = self.fits_explorer.get("light")

        if produce_masters:
            self.calibration.produce_masters()

    def describe(self):
        self.fits_explorer.describe("calib")

    def describe_observations(self):
        n_observations = len(self.fits_explorer.observations())
        print("{} observation{} found :\n{}".format(
                n_observations, "s" if n_observations > 1 else "",
                self.fits_explorer.describe(return_string=True),
            ))

    def run(
        self,
        destination,
        reference_frame=1/2,
        save_gif=True,
        save_stack=True,
        gif_factor=0.25,
        overwrite=False,
        n_images=None,
    ):

        if not path.exists(destination):
            os.mkdir(destination)

        if n_images is None:
            n_images = len(self.light_files)

        self.fwhm_kwargs = {}

        gif_path = "{}{}".format(
            path.join(destination, self.fits_explorer.products_denominator),
            "_movie.gif",
        )

        stack_path = "{}{}".format(
            path.join(destination, self.fits_explorer.products_denominator),
            "_stack.fits",
        )

        reference_frame = int(reference_frame*len(self.light_files))
        reference_image_path = self.light_files[reference_frame]
        reference_image = self.fits_explorer.trim(reference_image_path)
        
        reference_flip = self.fits_explorer.files_df[
            self.fits_explorer.get(im_type="light", return_conditions=True)].iloc[reference_frame]["flip"]

        if self.alignment == alignment.astroalign_optimized_find_transform:
            reference_stars = astroalign._find_sources(reference_image)[
                : astroalign.MAX_CONTROL_POINTS
            ]
            reference_invariants, reference_asterisms = astroalign._generate_invariants(
                reference_stars
            )
        else:
            reference_stars = self.stars_detection(
                reference_image.data,
                **self.stars_detection_kwargs,
            )

        with imageio.get_writer(gif_path, mode="I") as writer:
            for i, image in enumerate(tqdm(
                self.light_files[0:n_images],
                desc="Reduction",
                unit="files",
                ncols=80,
                bar_format=TQDM_BAR_FORMAT,
            )):

                _flip = self.fits_explorer.files_df[
                        self.fits_explorer.get(im_type="light", return_conditions=True)].iloc[i]["flip"]
                flip = not reference_flip == _flip

                # Calibration
                calibrated_frame = self.calibration.calibrate(image, flip=flip)

                if self.alignment == alignment.astroalign_optimized_find_transform:
                    # Translation estimation + stars detection
                    transform, _detected_stars = self.alignment(
                        calibrated_frame,
                        reference_stars,
                        KDTree(reference_invariants),
                        reference_asterisms,
                    )
                    detected_stars = _detected_stars[0]
                    shift = transform.translation

                else:
                    # Stars detection
                    detected_stars = self.stars_detection(
                        calibrated_frame, **self.stars_detection_kwargs
                    )
                    # Image translation estimation
                    shift = self.alignment(
                        detected_stars, reference_stars, **self.alignment_kwargs
                    )

                # Image alignment
                translation = AffineTransform(translation=-shift.astype("int"))
                aligned_frame = warp(
                    calibrated_frame,
                    translation,
                    mode="constant",
                    cval=np.mean(calibrated_frame),
                )

                # Seeing estimation
                try:
                    _fwhm = self.fwhm(calibrated_frame, detected_stars)
                except RuntimeError:
                    _fwhm = -1, -1, -1

                # Stack image production
                if save_stack:
                    if i==0:
                        stacked_image = aligned_frame
                    else:
                        stacked_image += aligned_frame
                else:
                    save_stack = None

                # Gif production
                if save_gif:
                    gif_im = utils.z_scale(
                        resize(
                            aligned_frame,
                            np.array(np.shape(aligned_frame)).astype(int) * gif_factor,
                            anti_aliasing=True,
                        )
                    )
                    writer.append_data((gif_im * 255).astype("uint8"))

                # Reduced image HDU construction
                new_hdu = fits.PrimaryHDU(aligned_frame)
                new_hdu.header = fits.getheader(image)

                h = {
                    "TRIMMING": self.calibration.telescope.trimming[0],
                    "FWHM": np.mean(_fwhm),
                    "FWHMX": _fwhm[0],
                    "FWHMY": _fwhm[1],
                    "DX": shift[0],
                    "DY": shift[1],
                    "ORIGFWHM": new_hdu.header.get(self.telescope.keyword_fwhm, ""),
                    "BZERO": 0,
                    "ALIGNALG": self.alignment.__name__,
                    "FWHMALG": self.fwhm.__name__,
                    "REDDATE": Time.now().to_value("fits"),
                    self.telescope.keyword_image_type: "reduced"
                }

                new_hdu.header.update(h)

                # Astrometry (wcs)
                new_hdu.header.update(reference_image.wcs.to_header(relax=True))

                fits_new_path = os.path.join(
                    destination,
                    path.splitext(path.basename(image))[0] + "_reduced.fits",
                )

                new_hdu.writeto(fits_new_path, overwrite=overwrite)

                if save_stack and image == reference_image_path:
                    stacked_image_header = new_hdu.header

        if save_stack:
            stacked_image /= len(self.light_files[0:n_images])
            stack_hdu = fits.PrimaryHDU(stacked_image)
            stacked_image_header[self.telescope.keyword_image_type] = "Stack image"
            stacked_image_header["REDDATE"] = Time.now().to_value("fits")
            stacked_image_header["NIMAGES"] = len(self.light_files[0:n_images])

            changing_flip_idxs = np.array([
                idx for idx, (i, j) in enumerate(zip(self.fits_explorer.files_df["flip"],
                                                     self.fits_explorer.files_df["flip"][1:]), 1) if i != j])

            if len(changing_flip_idxs) > 0:
                stacked_image_header["FLIPTIME"] = self.fits_explorer.files_df["jd"].iloc[changing_flip_idxs].values[0]

            stack_hdu.header = stacked_image_header
            stack_hdu.writeto(stack_path, overwrite=overwrite)


class Photometry:
    def __init__(
        self,
        folder,
        verbose=False,
        photometry="aperture",
        fwhm="gaussian2d",
        stars_detection="daofind",
    ):

        self.folder = folder

        self.verbose = verbose
        self.fits_explorer = io.FitsManager(folder, verbose=False, light_kw="reduced")
        self.fits_explorer.set_observation(0, check_calib_telescope=False)
        self.telescope = self.fits_explorer.telescope

        self.stars_detection = stars_detection
        self.photometry = photometry
        self.fwhm = fwhm

        self.stars_detection_kwargs = {
            "sigma_clip": 2.5,
            "lower_snr": 50,
            "n_stars": 500,
        }
        self.fwhm_kwargs = {}
        if self.photometry == phot.aperture_photometry_annulus:
            self.apertures = np.arange(0.1, 10, 0.25)
            self.photometry_kwargs = {"apertures": self.apertures}
        else:
            self.apertures = None
            self.photometry_kwargs = {}

        self.stack_path = io.get_files("stack.fits", folder)

        self.photometric_data = None
        self.data = pd.DataFrame()

        self.hdu = None

        self.stars = None

    @property
    def light_files(self):
        return self.fits_explorer.get("reduced")

    @property
    def fwhm(self):
        return self._fwhm

    @fwhm.setter
    def fwhm(self, name):
        self._fwhm = return_method(name)
        self.fwhm_kwargs = {}

    @property
    def stars_detection(self):
        return self._stars_detection

    @stars_detection.setter
    def stars_detection(self, name):
        self._stars_detection = return_method(name)
        self.stars_detection_kwargs = {}

    @property
    def photometry(self):
        return self._photometry

    @photometry.setter
    def photometry(self, name):
        self._photometry = return_method(name)
        self.photometry_kwargs = {}

    @property
    def default_destination(self):
        return path.join(
            self.folder,
            "{}.phots".format(self.fits_explorer.products_denominator
            ))

    def run(self, n_images=None, save=True, overwrite=False):
        if save and not overwrite:
            if path.exists(self.default_destination):
                raise FileExistsError("file already exists, use 'overwrite' kwarg")

        if n_images is None:
            n_images = len(self.light_files)

        stack_data = fits.getdata(self.stack_path)

        self.stars = self.stars_detection(stack_data, **self.stars_detection_kwargs)

        print("{} {} stars detected".format(INFO_LABEL, len(self.stars)))

        if "fixed_fwhm" in self.photometry_kwargs:
            fixed_fwhm = self.photometry_kwargs["fixed_fwhm"]
        else:
            fixed_fwhm = True

        if not fixed_fwhm:
            fwhm = self.fwhm[0:n_images]
        else:
            fwhm = np.mean(self.fwhm(stack_data, self.stars, **self.fwhm_kwargs)[0:2])

        self.photometric_data, data = self.photometry(
            self.light_files[0:n_images],
            self.stars,
            fwhm,
            **self.photometry_kwargs,
        )

        for keyword in ["sky"]:
            if keyword in data:
                self.load_data(keyword, data=data[keyword], n_images=n_images)

        for keyword in [
            "fwhm",
            "dx",
            "dy",
            "airmass",
            self.telescope.keyword_exposure_time,
            self.telescope.keyword_julian_date,
        ]:
            try:
                self.load_data(keyword, n_images=n_images)
            except KeyError:
                pass
        
        if save:
            self.save(overwrite=overwrite)

    def save(self, destination=None, overwrite=False):
        if destination is None:
            destination = self.default_destination

        if self.stack_path is not None:
            header = fits.PrimaryHDU(header=fits.getheader(self.stack_path))
        else:
            # Cambridge fits output issue - primary hdu is not complete ans stack not provided
            header = fits.PrimaryHDU()

        header.header["REDDATE"] = Time.now().to_value("fits")

        hdu_list = [
            header,
            fits.ImageHDU(self.photometric_data, name="photometry"),
            fits.ImageHDU(self.stars, name="stars"),
        ]

        for keyword in [
            "fwhm",
            "sky",
            "dx",
            "dy",
            "airmass",
            (self.telescope.keyword_exposure_time.lower(), "exptime",),
            (self.telescope.keyword_julian_date.lower(), "jd",),
        ]:
            if isinstance(keyword, str):
                if keyword in self.data:
                    hdu_list.append(fits.ImageHDU(self.data[keyword], name=keyword))
            elif isinstance(keyword, tuple):
                if keyword[0] in self.data:
                    hdu_list.append(
                        fits.ImageHDU(self.data[keyword[0]], name=keyword[1])
                    )

        if self.photometry == phot.aperture_photometry_annulus:
            hdu_list.append(fits.ImageHDU(self.apertures, name="apertures"))
        else:
            hdu_list.append(fits.ImageHDU([0], name="apertures"))

        self.hdu = fits.HDUList(hdu_list)
        self.hdu.writeto(destination, overwrite=overwrite)

    def load_data(self, keyword, data=None, n_images=None):
        """
        Load data to the self.data pandas DataFrame. If data is not provided (usually the case) data wiil be loaded from
        self.light_files fits headers using keyword (always lowercase). Data is loaded in the DataFrame with keyword as
        header

        Parameters
        ----------
        keyword : string
            name of the data to load (always lowercase, even if targeting an uppercase fits keyword)
        data : (np.array, list) (optional)
            data to be loaded. If not provided, data will be load from self.light_files headers using keyword

        Returns
        -------

        """
        if n_images is None:
            n_images = len(self.light_files)

        if data is None:
            data = np.array(
                io.fits_keyword_values(
                    self.light_files[0:n_images], [keyword]
                )
            ).flatten()

        self.data[keyword.lower()] = data
