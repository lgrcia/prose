from prose import load
from pathlib import Path
import matplotlib.pyplot as plt
import os

def make_lightcurve(phot_path, offline=True):
	obs = load(phot_path)
	#obs.keep_good_stars()
	obs.import_stack(os.path.join(Path(phot_path).parent, 'solved.fits')) # This will load the plate solved stack image in the .phot file.
	obs.save()

	obs.set_tic_target(offline=offline)
	obs.save()

	#Differential photometry + lightcurve plot
	obs.broeg2005()
	obs.save()
	plt.figure()
	obs.plot()
	plt.ylim(0.98, 1.02)
	plt.savefig(os.path.join(Path(phot_path).parent, 'lightcurve.png'))

	#Systematics and comparison stars plots
	plt.figure(figsize=(8, 7))
	plt.subplot(121)
	obs.plot_comps_lcs()
	plt.subplot(122)
	obs.plot_systematics()
	plt.savefig(os.path.join(Path(phot_path).parent, 'systematics.png'))

	#Target PSF plot
	plt.figure()
	obs.plot_star_psf()
	plt.savefig(os.path.join(Path(phot_path).parent, 'psf.png'))

	#Save target data and systematics to csv file
	obs.to_csv(os.path.join(Path(phot_path).parent, 'measurements.csv'), sep=',')

	return