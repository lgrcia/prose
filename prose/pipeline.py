from prose import io
from prose import utils

from prose.pipeline_methods import photometry as phot
from prose.pipeline_methods.detection import StarsDetection, DAOFindStars, SegmentedPeaks
from prose.pipeline_methods.psf import NonLinearGaussian2D
from prose.characterization import Characterize
from prose.pipeline_methods.alignment import Alignment, XYShift
from prose.pipeline_methods.photometry import BasePhotometry, AperturePhotometry

import os
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os import path
from tqdm import tqdm
from astropy.io import fits
from astropy.time import Time
from prose.console_utils import TQDM_BAR_FORMAT, INFO_LABEL
from prose import visualisation as viz
from astropy.nddata import Cutout2D
from astropy.table import Table


class Calibration:
    """
    calibration task, included in :py:class:`~prose.pipeline.Reduction`
    """
    def __init__(
        self,
        folder=None,
        verbose=True,
        telescope_kw="TELESCOP",
        depth=1
    ):
        if isinstance(folder, io.FitsManager):
            self.fits_explorer = folder            
        else:
            self.fits_explorer = io.FitsManager(
                folder, verbose=verbose, telescope_kw=telescope_kw, depth=depth
            )

        self.telescope = self.fits_explorer.telescope

        self.master_dark = None
        self.master_flat = None
        self.master_bias = None

    def calibration(self, image, exp_time):
        return (image - (self.master_dark * exp_time + self.master_bias)) / self.master_flat

    def _produce_master(self, image_type):
        _master = []
        kw_exp_time = self.telescope.keyword_exposure_time
        images = self.fits_explorer.get(image_type)
        assert len(images) > 0, "No {} images found".format(image_type)
        for i, fits_path in enumerate(images):
            hdu = fits.open(fits_path)
            primary_hdu = hdu[0]
            image, header = primary_hdu.data, primary_hdu.header
            hdu.close()
            image = self.fits_explorer.trim(image, raw=True)
            if image_type == "dark":
                _dark = (image - self.master_bias) / header[kw_exp_time]
                if i == 0:
                    _master = _dark
                else:
                    _master += _dark
            elif image_type == "bias":
                if i == 0:
                    _master = image
                else:
                    _master += image
            elif image_type == "flat":
                _flat = image - (self.master_bias + self.master_dark)*header[kw_exp_time]
                _flat /= np.mean(_flat)
                _master.append(_flat)
                del image
        
        if image_type == "dark":
            self.master_dark = _master/len(images)
        elif image_type == "bias":
            self.master_bias = _master/len(images)
        elif image_type == "flat":
            # To avoid memory errors, we split the median computation in 50
            _master = np.array(_master)
            shape_divisors = utils.divisors(_master.shape[1])
            n = shape_divisors[np.argmin(np.abs(50 - shape_divisors))]
            self.master_flat = np.concatenate([np.median(im, axis=0) for im in np.split(_master, n, axis=1)])
            del _master

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

    def calibrate(self, im_path, flip=False, return_wcs=False, only_trim=False):
        # TODO: Investigate flip
        hdu = fits.open(im_path)
        primary_hdu = hdu[0]
        image, header = self.fits_explorer.trim(im_path), primary_hdu.header
        hdu.close()
        exp_time = header[self.telescope.keyword_exposure_time]
        if not only_trim:
            calibrated_image = self.calibration(image.data, exp_time)
        else:
            calibrated_image = image

        if flip:
            calibrated_image = calibrated_image[::-1, ::-1]

        if return_wcs:
            return calibrated_image, image.wcs
        else:
            return calibrated_image


class Reduction:
    """
    calibration, alignment, stacking and other reduction tasks
    """
    def __init__(
        self,
        folder=None,
        verbose=False,
        alignment=None,
        fwhm=None,
        stars_detection=None,
        depth=1
    ):
        self.calibration = Calibration(
            folder, verbose=verbose, depth=depth
        )

        if isinstance(folder, io.FitsManager):
            self.light_files = folder.get("light")
        else:
            # should be set later in set_observation
            self.light_files = None

        self.fits_explorer = self.calibration.fits_explorer
        self.telescope = self.fits_explorer.telescope
        self.data = pd.DataFrame()

        self.fwhm = utils.check_class(fwhm, Characterize, NonLinearGaussian2D(cutout_size=15))
        self.stars_detection = utils.check_class(stars_detection, StarsDetection, SegmentedPeaks(n_stars=50))
        self.alignment = utils.check_class(alignment, Alignment, XYShift(detection=self.stars_detection))

        if len(self.fits_explorer.observations) == 1:
            self.set_observation(0)

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
        n_observations = len(self.fits_explorer.observations)
        print("{} observation{} found :\n{}".format(
                n_observations, "s" if n_observations > 1 else "",
                self.fits_explorer.describe(return_string=True),
            ))

    def run(
        self,
        destination=None,
        reference_frame=1/2,
        save_stack=True,
        overwrite=False,
        n_images=None,
        raise_exists=True,
        only_trim=False
    ):
        """Run reduction task

        Parameters
        ----------
        destination : str, optional
            Destination of the created reduced folder, by default None, i.e. {self.folder}/{self.products_denominator}
        reference_frame : float, optional
            TODO: allow int
            reference frame as a float between 0 (first image) and 1 (last image), by default 1/2
        save_stack : bool, optional
            weather to save stack image, by default True
        overwrite : bool, optional
            weather to overwrite existing files, by default False
        n_images : int, optional
            number of images to process starting from first image, by default None
        raise_exists : bool, optional
            raise exists error if folder already exists, by default True
        calibrate : bool, optional
            weather to calibrate images, by default True
        only_trim: bool, optional,
            weather to skip calibration and only do trimming on images, default False

        Returns
        -------
        str
            destination of created folder (usefull if destination kwargs is None)

        Raises
        ------
        ValueError
            fits explorer should contain a single observation
        AssertionError
            Folder already exists and stack is present, meaning reduction has been done for current folder
        """

        if len(self.fits_explorer.observations) != 1:
            raise ValueError("multiple observations founb, please set an observation")

        if destination is None:
            destination = path.join(
                path.dirname(self.fits_explorer.folder), 
                self.fits_explorer.products_denominator
            )
                
        if not path.exists(destination):
            os.mkdir(destination)
        
        if n_images is None:
            n_images = len(self.light_files)

        stack_path = "{}{}".format(
            path.join(destination, self.fits_explorer.products_denominator),
            "_stack.fits",
        )

        if path.exists(stack_path) and not overwrite:
            if raise_exists:
                raise AssertionError("stack {} already exists".format(stack_path))
            else:
                return destination

        reference_frame = int(reference_frame*len(self.light_files))
        reference_image_path = self.light_files[reference_frame]
        reference_image = self.fits_explorer.trim(reference_image_path)
        
        reference_flip = self.fits_explorer.files_df[
            self.fits_explorer.get(im_type="light", return_conditions=True)].iloc[reference_frame]["flip"]

        self.alignment.set_reference(reference_image.data)

        ref_shape = np.array(reference_image.shape)
        ref_center = ref_shape[::-1]/2

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
            calibrated_frame, calibrated_wcs = self.calibration.calibrate(
                image, flip=flip,
                return_wcs=True,
                only_trim=only_trim
            )

            # Stars detection and shift computation
            detected_stars, shift = self.alignment.run(calibrated_frame)

            # Image alignment
            aligned_frame = Cutout2D(
                calibrated_frame, 
                ref_center-shift.astype("int"), 
                ref_shape, 
                mode="partial", 
                fill_value=np.mean(calibrated_frame),
                wcs=calibrated_wcs
                )

            # Seeing/psf estimation
            try:
                _fwhm = self.fwhm.run(calibrated_frame, detected_stars)
            except RuntimeError:
                _fwhm = -1, -1, -1

            # Stack image production
            if save_stack:
                if i == 0:
                    stacked_image = aligned_frame.data
                else:
                    stacked_image += aligned_frame.data
            else:
                save_stack = None

            # Reduced image HDU construction
            new_hdu = fits.PrimaryHDU(aligned_frame.data)
            new_hdu.header = fits.getheader(image)

            h = {
                "TRIMMING": self.calibration.telescope.trimming[0],
                "FWHM": np.mean([_fwhm[0], _fwhm[1]]),
                "FWHMX": _fwhm[0],
                "FWHMY": _fwhm[1],
                "PSFANGLE": _fwhm[2],
                "FWHMALG": self.fwhm.__class__.__name__,
                "DX": shift[0],
                "DY": shift[1],
                "ALIGNALG": self.alignment.__class__.__name__,
                "SEEING": new_hdu.header.get(self.telescope.keyword_seeing, ""),
                "BZERO": 0,
                "REDDATE": Time.now().to_value("fits"),
                self.telescope.keyword_image_type: "reduced"
            }

            new_hdu.header.update(h)

            # Astrometry (wcs)
            new_hdu.header.update(aligned_frame.wcs.to_header(relax=True))

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

        return destination


class Photometry:
    """
    photometric extraction task
    """
    def __init__(
        self,
        folder,
        verbose=False,
        photometry=None,
        stars_detection=None,
    ):

        self.folder = folder

        self.verbose = verbose
        self.fits_explorer = io.FitsManager(folder, verbose=False, light_kw="reduced")
        self.fits_explorer.set_observation(0, check_calib_telescope=False)
        self.telescope = self.fits_explorer.telescope

        self.stars_detection = utils.check_class(
            stars_detection,
            StarsDetection,
            DAOFindStars(sigma_clip=2.5, lower_snr=20, n_stars=500)
        )
        self.photometry = utils.check_class(
            photometry,
            BasePhotometry,
            AperturePhotometry()
        )
        self.photometry.set_fits_explorer(self.fits_explorer)

        self.stack_path = io.get_files("stack.fits", folder)

        self.fluxes = None
        self.fluxes_errors = None
        self.other_data = {}
        self.data = pd.DataFrame()

        self.hdu = None
        self.stars = None

    @property
    def light_files(self):
        return self.fits_explorer.get("reduced")

    @property
    def default_destination(self):
        return path.join(
            self.folder,
            "{}_{}.phots".format(
                self.fits_explorer.products_denominator,
                self.photometry.__class__.__name__.lower()
            ))

    def run(self, n_images=None, save=True, overwrite=False, remove_reduced=False, raise_exists=True):
        if save and not overwrite:
            if path.exists(self.default_destination):
                if raise_exists:
                    raise FileExistsError("file already exists, use 'overwrite' kwarg")
                else:
                    return None

        if n_images is None:
            n_images = len(self.light_files)

        stack_data = fits.getdata(self.stack_path)
        self.stars = self.stars_detection.run(stack_data)

        print("{} {} stars detected".format(INFO_LABEL, len(self.stars)))

        for keyword in [ 
            "sky",
            "fwhm",
            "fwhmx",
            "fwhmy",
            "psf_angle",
            "dx",
            "dy",
            "airmass",
            self.telescope.keyword_exposure_time,
            self.telescope.keyword_julian_date,
            self.telescope.keyword_seeing,
            self.telescope.keyword_ra,
            self.telescope.keyword_dec,
        ]:
            # TODO: is jd shifted with exposure/2?
            try:
                self.load_data(keyword.replace("_", ""), n_images=n_images)
            except KeyError:
                pass

        self.fluxes, self.fluxes_errors, self.other_data = self.photometry.run(self.stars)
        
        if save:
            self.save(overwrite=overwrite)
        
        if remove_reduced:
            for file_path in self.fits_explorer.get("reduced"):
                os.remove(file_path)

    def save(self, destination=None, overwrite=False):
        if destination is None:
            destination = self.default_destination

        if self.stack_path is not None:
            header = fits.PrimaryHDU(header=fits.getheader(self.stack_path))
        else:
            # Cambridge fits output issue - primary hdu is not complete ans stack not provided
            header = fits.PrimaryHDU()

        header.header["REDDATE"] = Time.now().to_value("fits")

        if len(self.fluxes.shape) == 2:
            self.fluxes = np.array([self.fluxes])
            self.fluxes_errors = np.array([self.fluxes_errors])

        hdu_list = [
            header,
            fits.ImageHDU(self.fluxes, name="photometry"),
            fits.ImageHDU(self.fluxes_errors, name="photometry errors"),
            fits.ImageHDU(self.stars, name="stars"),
        ]
        
        # temporary, TO DELETE, TODO
        if isinstance(self.photometry, AperturePhotometry):
            sky = np.mean(self.other_data["annulus_sky"], axis=0)
            self.data["sky"] = sky

        data_table = Table.from_pandas(self.data)
        hdu_list.append(fits.BinTableHDU(data_table, name="time series"))
        
        # These are other data produced by the photometry task wished to be saved in the .phot
        for other_data_key in self.other_data:
            data = self.other_data[other_data_key]
            hdu_list.append(fits.ImageHDU(data, name=other_data_key.replace("_", " ")))
        
        self.hdu = fits.HDUList(hdu_list)
        self.hdu.writeto(destination, overwrite=overwrite)

    def load_data(self, keyword, data=None, n_images=None):
        """
        Load data to the self.data pandas DataFrame. If data is not provided (usually the case) data will be loaded from
        self.light_files fits headers using keyword (always lowercase). Data is loaded in the DataFrame with keyword as
        header

        Parameters
        ----------
        n_images : int, optional
            number of images to include, default is None
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
                    self.light_files[0:n_images], [keyword.replace("_", "").upper()]
                )
            ).flatten()

        self.data[keyword.lower()] = data


def produce_gif(reduced_folder, light_kw="reduced"):
    
    fits_explorer = io.FitsManager(reduced_folder, verbose=False, light_kw=light_kw)
    fits_explorer.set_observation(0, check_calib_telescope=False)

    gif_path = "{}{}".format(
        path.join(reduced_folder, fits_explorer.products_denominator),
        "_movie.gif",
    )

    with imageio.get_writer(gif_path, mode="I") as writer:
        for image in tqdm(
            fits_explorer.get("reduced"),
            desc="Gif",
            unit="files",
            ncols=80,
            bar_format=TQDM_BAR_FORMAT,
        ):
            writer.append_data(viz.gif_image_array(fits.getdata(image)))
