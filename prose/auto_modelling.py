import math as mh
import numpy as np
from prose import load
import matplotlib.pyplot as plt
from astropy import units as u
import theano
import pymc3 as pm
import exoplanet as xo
import pymc3_ext as pmx
from prose.utils import earth2sun
from prose import models
from matplotlib.ticker import AutoMinorLocator
from prose import viz
import arviz as az
from astropy.constants import G,M_sun,R_sun, R_earth
from prose.reports import Report, Summary, TESSSummary, TransitModel
import corner
from prose.reports.core import copy_figures


def fill_table(transitmodel,obs, az_sum, opt):
	b = opt['b']
	period = (opt['P']*u.d).to("s")
	R_p = opt['r'] * R_earth
	pi = (np.pi)*u.rad
	a = (((period**2) *(G*obs.host_data['M']*M_sun)) / (4*(np.pi**2)))**(1/3)
	cos_i = ((b * obs.host_data['R']*R_sun) / a)
	i = (np.arccos(cos_i).to('deg'))
	depth = (R_p / (obs.host_data['R']*R_sun))**2 

	transitmodel.obstable[1][1]=[round(obs.quad_limb[0][0][0],3),round(obs.quad_limb[0][0][1],3)]
	transitmodel.obstable[2][2]=f"{obs.host_data['R']}"+" R$_{\\odot}$" if obs.host_data['R']!=1 else None
	transitmodel.obstable[3][2]=f"{obs.host_data['M']}"+" M$_{\\odot}$" if obs.host_data['M']!=1 else None
	transitmodel.obstable[4][1]=f'{az_sum["mean"]["P"]} +/- {az_sum["sd"]["P"]} d'
	transitmodel.obstable[4][2]=f"{obs.period} d"
	transitmodel.obstable[5][1]=f'{az_sum["mean"]["r"]} +/- {az_sum["sd"]["r"]}'+' R$_{\\oplus}$'
	transitmodel.obstable[6][1]=f'{az_sum["mean"]["t0"]} +/- {az_sum["sd"]["t0"]}'
	transitmodel.obstable[6][2]=obs.t0
	transitmodel.obstable[7][1]=f'{az_sum["mean"]["b"]} +/- {az_sum["sd"]["b"]}'
	transitmodel.obstable[8][2]=f"{round(obs.duration*24*60,2)} min"
	transitmodel.obstable[9][1]=format(depth.value, '3.2E')
	transitmodel.obstable[9][2]=format(obs.depth, '3.2E')
	transitmodel.obstable[11][1]=round((a/(obs.host_data['R']*R_sun)).value, 3) if obs.host_data['R']!=1 else None
	transitmodel.obstable[12][1]=round(i.value,3)

def auto_model(phot_path, offline=True):
	obs = load(phot_path)
	if obs.period == None:
		return
	print("Searching the TIC to set host data. This can take several minutes...")
	obs.set_host_data(offline)
	print("Done.")
	print("Estimating limb darkening coefficients...")
	obs.estimate_LDC(offline)
	print("Done.")
	print(f"Limb darkening coefficients estimated as u1={obs.quad_limb[0][0][0]} and u2={obs.quad_limb[0][0][1]}")
	print("Checking best polynomial coeffients to detrent the data...")
	best = obs.best_polynomial(airmass=2, sky=2, dx=5,dy=5,fwhm=5, add=obs.transit(obs.t0,obs.duration))
	print("Done.")
	print(f"Best coefficients are {best}")
	dm = np.hstack([obs.polynomial(**best),obs.transit(obs.t0,obs.duration)])
	trend,transit = obs.lstsq(dm,split=-1)
	rp_guess=np.sqrt(obs.depth)*obs.host_data['R']/earth2sun
	rp_e=rp_guess/2


	theano.config.compute_test_value = "warn"
	systematics = models.LinearModel(obs.polynomial(**best).T)
	print("Setting up the model...")
	with pm.Model() as model:
	    # Transit parameters
	    # -----------------
	    #period = pm.Uniform("period",4,6)
	    r = pm.Normal("r", rp_guess, rp_e)*earth2sun
	    t0_= pm.Normal('t0',obs.t0,0.5)
	    b = pm.Uniform("b",lower=0, upper=1, testval=0.5)
	    p = pm.Normal('P',obs.period,0.5)
	    #Rs = pm.Uniform('Rs', lower=0.9, upper=1.8, testval=1.2)

	    # Keplerian orbit
	    # ---------------
	    orbit = xo.orbits.KeplerianOrbit(period=p, t0=t0_, m_star=obs.host_data['M'], 
	                                     r_star=obs.host_data['R'], b=b)
	    # starry light-curve
	    light_curves = xo.LimbDarkLightCurve(
	        [obs.quad_limb[0][0][0], obs.quad_limb[0][0][1]]
	    ).get_light_curve(orbit=orbit, r=r, t=obs.time) 
	    transit = pm.Deterministic("transit", pm.math.sum(light_curves, axis=-1))

	    # Systematics and final model
	    # ---------------------------
	    residuals = pm.Deterministic("residuals", obs.diff_flux - transit)
	    mu = pm.Deterministic("mu", transit + systematics(residuals))

	    # Likelihood function
	    # -----------------------------
	    pm.Normal("obs", mu=mu, sd=obs.diff_error, observed=obs.diff_flux)

	    # Maximum a posteriori
	    # --------------------
	    opt = pmx.optimize(start=model.test_point)

	#NEED AN IF STATEMENT HERE TO CATCH NON DETECTIONS!
	print("Done.")
	np.random.seed(42)

	print("Running the fit...")
	with model:
	    trace = pm.sample(
		        tune=2000,
		        draws=2000,
		        start=opt,
		        cores=2,
		        chains=2,
		        init="adapt_full",
		        target_accept=0.9,
		        return_inferencedata=False
		   )

	print("Done.")
	with model:
	    az_summary = az.summary(
	        trace, var_names=["P", "t0", "b", "r"]
	    )

	b = opt['b']
	period = (opt['P']*u.d).to("s")
	R_p = opt['r'] * R_earth
	pi = (np.pi)*u.rad
	a = (((period**2) *(G*obs.host_data['M']*M_sun)) / (4*(np.pi**2)))**(1/3)
	cos_i = ((b * obs.host_data['R']*R_sun) / a)
	i = (np.arccos(cos_i).to('deg'))
	depth = (R_p / (obs.host_data['R']*R_sun))**2 

	print("Building transit model and report summary...")
	summary = TESSSummary(obs,expected=(obs.t0,obs.duration))
	transitmodel = TransitModel(obs,expected =(obs.t0,obs.duration),transit=opt['transit'], trend= opt['mu']-opt['transit'],rms_bin=5/24/60)
	fill_table(transitmodel, obs, az_summary, opt)

	report = Report([summary,transitmodel])
	report.make(f"{obs.denominator}_report")
	transitmodel.to_csv_report()

	plt.figure()
	transitmodel.plot_lc_model() 
	viz.plot_expected_transit(obs.time, obs.t0, obs.period, obs.duration, obs.depth, color='#FFA533')
	plt.ylim(0.95,1.045)
	plt.savefig(f"{obs.denominator}_report/transitmodel/figures/model.png")

	samples = pm.trace_to_dataframe(trace, varnames=["P", "r",'t0','b'])
	fig = corner.corner(samples,truths=[opt['P'],opt['r'],opt['t0'],opt['b']])
	fig.patch.set_facecolor('xkcd:white')
	plt.savefig(f"{obs.denominator}_report/transitmodel/figures/Corner_plot.png")

	report.compile()
	print("Done")

	if "TOI" in obs.name:
		copy_figures(folder=f'{obs.denominator}_report',prefix=obs.tfop_prefix,destination=f'{obs.denominator}_report/figures')

	print(f"Modelling and reporting complete. All files can be found in {obs.denominator}_report")
