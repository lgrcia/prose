#imports

import requests as req
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prose import load
from prose import Image
from prose import viz
from prose import Telescope
import os
from astropy.io import fits
from prose.tess import TFOPObservation
import theano
import pymc3 as pm
import exoplanet as xo
import pymc3_ext as pmx
from prose.utils import earth2sun
from prose import models
import aesara_theano_fallback.tensor as tt
theano.config.compute_test_value = "warn"
from prose.reports import Report, Summary
from prose.tess import TESSSummary,TransitModel, TESSNotes
from prose import blocks
import arviz as az
from matplotlib import gridspec
from prose.reports.core import copy_figures
import corner
import ipywidgets as widgets
from IPython.display import clear_output

loc = os.getcwd()
style = {'description_width': 'initial'}

#build obs

USR=widgets.Text(
    placeholder='type here',
    description='Username',
    style=style
)

PSSWD=widgets.Password(
    description='Password',
    style=style
)

OBS_SCOPE=widgets.Dropdown(
    description='Telescope',
    options=['Europa', 'Ganymede', 'Callisto', 'Io', 'Artemis'],
    style=style
)

OBS_DATE=widgets.Text(
    description='Date',
    placeholder='yyyymmdd',
    style=style
)

OBS_FILTER=widgets.Dropdown(
    options=['I+z', 'r', 'g', 'B', 'I'],
    description='Filter',
    style=style
)

OBS_GAIA=widgets.Text(
    description='Gaia ID',
    style=style
)
OBS_ID=widgets.Text(
    description='Target ID',
    style=style
)

BUILD_XAR_B=widgets.Button(description='Build Obs')


def scrape_portal():
    global obs_scope, obs_date, obs_filter, obs_gaia, obs_id, out,usr,psswd
    usr=USR.value
    psswd=PSSWD.value
    
    obs_scope=OBS_SCOPE.value
    obs_date=OBS_DATE.value
    obs_filter=OBS_FILTER.value
    obs_gaia=OBS_GAIA.value
    obs_id=OBS_ID.value

    url2check=f'http://www.mrao.cam.ac.uk/SPECULOOS/portal-night-data/{obs_scope}_{obs_date}_{obs_filter}_{obs_gaia}_{obs_id}.json'

    out=req.get(url2check, auth = (usr, psswd)).json()
    return

def get_stack():
    global directory, path2stack
    stack_url=f'http://www.mrao.cam.ac.uk/SPECULOOS/{obs_scope}/output/v2/StackImages_nightly/{obs_date}/{obs_id}_stack.fits'
    directory = os.path.join(loc, obs_id)
    if not os.path.exists(directory):
        os.makedirs(directory)
    path2stack=os.path.join(directory, stack_url.split('/')[-1])
    get_fits = req.get(stack_url, auth = (usr, psswd), stream=True).content
    with open(path2stack, 'wb') as f:
        f.write(get_fits)
    return

def build_xar():
    scrape_portal()
    get_stack()
    global APERTURE,SCOPE, EXPTIME,TIME,obs
    ap_list=[2,2.8,4,5.7,8,11.3,16,20,24,28,32,40,48]
    ap_dict={f'{i+1}':a for i,a in zip(range(13), ap_list)}

    EXPTIME=out['environment']['EXPOSURE']
    APERTURE=out['best_ap']
    SCOPE=f"SPECULOOS-{out['telescope']}"

    TIME=out['environment']['BJD-OBS']
    diff_fluxes=[[i['DIFF_FLUX'] for i in out['stars'][f'{APERTURE}']]]

    stars=[[i['CCDX'],i['CCDY']] for i in out['stars'][APERTURE]]

    dat=xr.Dataset({
        'airmass': (['time'], out['environment']['AIRMASS']),
        'fwhm': (['time'], out['environment']['FWHM']),
        'dx': (['time'], out['environment']['RA_MOVE']),
        'dy': (['time'], out['environment']['DEC_MOVE']),
        'sky': (['time'], out['environment']['SKYLEVEL']),
        'bjd_tdb': (['time'], TIME),
        'diff_fluxes':(['apertures', 'star','time'], np.array(diff_fluxes)),
        'diff_errors':(['apertures', 'star','time'], [[i['ERROR'] for i in out['stars'][f'{APERTURE}']]]),
        'comps':(['apertures', 'ncomps'], [[i for i in range(10)]]),
        'alc':(['apertures', 'time'], [[i for i in out['ALC'][f'{APERTURE}']]]),
        'apertures_radii':(['apertures','time'],[[ap_dict[APERTURE] for i in range(len(TIME))]])
    }, coords={'stars':(['star', 'n'], stars),
              'time':(['time'], TIME)})

    dat.attrs={'EXPTIME':EXPTIME,
              'aperture':APERTURE,
              'telescope':SCOPE,
              'target':0,
              'filter':out['filter'],
               'NIMAGES':len(TIME),
               'time_format':'BJD',
               'exptime':EXPTIME,
               'name':out['sp_id'],
               'gaia_id':out['gaia_id']
              }

    obs=TFOPObservation(dat)

    obs.aperture=0
    obs.target=0
    obs.stack=Image(path2stack)

    obs.stack.stars_coords=obs.stars
    obs.stack.target=0
    obs.stack=blocks.Cutouts(size=21)(obs.stack)
    obs.stack.psf=obs.stack.cutouts[0].data
    obs.stack=blocks.psf.Gaussian2D()(obs.stack)

    obs.xarray.to_netcdf(f'{directory}/{obs_id}.phot')
    obs.xarray.attrs.update(obs.stack.header)
    if "'" in obs.stack.filter:
        obs.stack.FILTER=obs.stack.filter.replace("'",'')
    obs.plot()
    print(f"https://exofop.ipac.caltech.edu/tess/target.php?id={obs.tic_id}")

OUT=widgets.Output(layout={'border': '1px solid black'})
    
def build_xar_b(b):
    with OUT:
        clear_output()
        build_xar()

BUILD_XAR_B.on_click(build_xar_b)

BOX_ONE=widgets.VBox([widgets.HBox([USR, PSSWD], layout=widgets.Layout(height='45px')),
                     widgets.HBox([OBS_SCOPE, OBS_DATE,OBS_FILTER], layout=widgets.Layout(height='45px')),
                    widgets.HBox([OBS_GAIA,OBS_ID], layout=widgets.Layout(height='45px')),
                     BUILD_XAR_B,
                     OUT])


#modelling

OBS_DEPTH=widgets.FloatText(
    description='Depth (ppt)',
    style=style
)
OBS_T0=widgets.FloatText(
    description='T0 (bjd)',
    style=style
)
OBS_PER=widgets.FloatText(
    description='Period (days)',
    style=style
)
OBS_DUR=widgets.FloatText(
    description='Duration (hrs)',
    style=style
)

MODEL_IT_B=widgets.Button(description='Model it!')

OUT2=widgets.Output(layout={'border': '1px solid black'})

def model_it():
    global obs_depth,obs_t0,obs_per,obs_ms,obs_rs,obs_dur
    obs_depth=OBS_DEPTH.value
    obs_t0=OBS_T0.value
    obs_per=OBS_PER.value
    obs_ms=obs.priors_dataframe['Stellar Mass (M_Sun)'][0]
    obs_rs=obs.priors_dataframe['Stellar Radius (R_Sun)'][0]
    obs_dur=OBS_DUR.value/24
    
    global opt,model
    X= obs.polynomial(airmass=2,fwhm=2,sky=2).T
    c=np.linalg.lstsq(X.T,obs.diff_flux,rcond=None)[0]

    with pm.Model() as model:
        # Systematics
        # -----------------
        w = pm.Flat('w',shape=len(X),testval=np.array(c))
        systematics = pm.Deterministic('systematics',w@X)

        # Stellar parameters
        # -----------------
        u = xo.distributions.QuadLimbDark("u", testval=np.array([0.1878, 0.3844]))
        star = xo.LimbDarkLightCurve(u[0],u[1])
        r_s = pm.Normal('r_s',obs_rs ,0.01)
        m_s = pm.Normal('m_s',obs_ms ,0.01)

        #r = pm.Normal("r", 2.113546581762,1.3841863)*earth2sun

        # Orbital parameters
        # -----------------
        t0= pm.Normal('t0',obs_t0,0.005)
        p = pm.Normal('P',obs_per,0.1228680778)# If log_p, use Deterministic here : pm.Deterministic("period", tt.exp(log_period))
        b = pm.Uniform("b",0.,1)
        depth = pm.Uniform("depth", 0, obs_depth*2/1000,testval=obs_depth/1000)#Prior normal sur log(param) = prior uniform sur param
        ror = pm.Deterministic("ror",star.get_ror_from_approx_transit_depth(depth, b))
        r_p = pm.Deterministic("r_p", ror * r_s) # In solar radius 
        r = pm.Deterministic('r',r_p*1/earth2sun)
        #duration= pm.Uniform('duration',0.045,0.055)

        # Dilution 
        # --------
        #alpha = pm.Normal('alpha',2.5,0.1)
        #y_p = (obs.diff_flux*(1+alpha)) - alpha

        # Keplerian orbit
        # ---------------
        orbit = xo.orbits.KeplerianOrbit(period=p, t0=t0, r_star=r_s, b=b, m_star=m_s)

        # starry light-curve
        # ---------------
        light_curves = star.get_light_curve(orbit=orbit, r=r_p, t=obs.time) 
        transit = pm.Deterministic("transit", pm.math.sum(light_curves, axis=-1))

        #Let's track some parameters : 
        pm.Deterministic("a",orbit.a)
        pm.Deterministic ('i',orbit.incl * 180/np.pi)
        pm.Deterministic('a/r_s', orbit.a / orbit.r_star)

        # Systematics and final model
        # ---------------------------
        #residuals = pm.Deterministic("residuals", obs.diff_flux - transit)
        mu = pm.Deterministic("mu", transit + systematics)

        # Likelihood function
        # -----------------------------
        pm.Normal("obs", mu=mu, sd=obs.diff_error, observed=obs.diff_flux)

        # Maximum a posteriori
        # --------------------
        opt = pmx.optimize(start=model.test_point)
        
    np.random.seed(42)
    global trace
    with model:
        trace = pm.sample(
            tune=1000,
            draws=1000,
            start=opt,
            cores=3,
            chains=2,
            init="adapt_full",
            target_accept=0.9,
            return_inferencedata=False
        )
        

def model_it_b(b):
    with OUT2:
        clear_output()
        model_it()
        
MODEL_IT_B.on_click(model_it_b)

BOX_TWO=widgets.VBox([widgets.Label('Priors and Modelling'),
                      widgets.Label('Guess the transit midtime from the lightcurve above - grab the rest from TFOP'),
    widgets.HBox([OBS_T0,OBS_PER,OBS_DUR,OBS_DEPTH], layout=widgets.Layout(height='45px')),
                      MODEL_IT_B, OUT2
], style=style)

#reporting

variables = ["P", "r",'t0','b','u','r_s','m_s','ror','depth','a','a/r_s','i']

def make_report():
    global tm,summary,report

    with model:
        summary = az.summary(
            trace, var_names=variables, round_to=4
        )
    #This should go in the results table of transitmodel

    posteriors={}

    for i in summary.index:
        posteriors[i]=summary['mean'][i]
        posteriors[i+'_e']=summary['sd'][i]

    tm = TransitModel(obs,name=obs.name,expected =(obs_t0,obs_dur),transit=opt['transit'], trend= opt['systematics'],rms_bin=5/24/6, posteriors=posteriors)

    optimal_aperture=np.mean(obs.apertures_radii[obs.aperture,:])
    mean_fwhm=np.mean(obs.x.fwhm.values)
    mean_target_fwhm=np.mean([obs.stack.fwhmx, obs.stack.fwhmy])

    summary = TESSSummary(obs, mean_target_fwhm,optimal_aperture,mean_fwhm,expected=(obs_t0, obs_dur))
    notes = TESSNotes()

    # Make the full report
    report = Report([summary,tm,notes])
    report.make(f"{obs.label}_report")


def plot_syst(self, size=(6, 8)):
    fig = plt.figure(figsize=size)
    fig.patch.set_facecolor('xkcd:white')
    ax = fig.add_subplot(111)

    self.plot_systematics()
    self.plot_meridian_flip()
    _ = plt.gcf().axes[0].set_title("", loc="left")
    plt.xlabel(f"BJD")
    plt.ylabel("diff. flux")
    plt.tight_layout()
    
def plot_comps(self, size=(6, 8)):
    fig = plt.figure(figsize=size)
    fig.patch.set_facecolor('xkcd:white')
    ax = fig.add_subplot(111)

    self.plot_comps_lcs()
    _ = plt.gcf().axes[0].set_title("", loc="left")
    self.plot_meridian_flip()
    plt.xlabel(f"BJD")
    plt.ylabel("diff. flux")
    plt.tight_layout()
    
def plot_raw(self, size=(6, 4)):
    fig = plt.figure(figsize=size)
    fig.patch.set_facecolor('xkcd:white')
    plt.plot(self.time, self.alc[self.aperture], c="C0", label="artificial")
    plt.legend()
    plt.tight_layout()
    plt.xlabel(f"BJD")
    plt.ylabel("norm. flux")
    #self.style()


def plot_psf_model(obs,figsize):
    cmap="inferno"
    c="blueviolet"

    psf_data=obs.stack.psf
    psf_model=obs.stack.psf_model

    plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, width_ratios=[9, 2], height_ratios=[2, 9])
    gs.update(wspace=0, hspace=0)
            #axtt = plt.subplot(gs[1, 1])
    ax = plt.subplot(gs[1, 0])
    axr = plt.subplot(gs[1, 1], sharey=ax)
    axt = plt.subplot(gs[0, 0], sharex=ax)

    ax.imshow(obs.stack.psf, alpha=1, cmap=cmap, origin="lower")
    ax.contour(obs.stack.psf_model, colors="w", alpha=0.7)

    x, y = np.indices(psf_data.shape)

    axt.plot(y[0], np.mean(psf_data, axis=0), c=c, label="data")
    axt.plot(y[0], np.mean(psf_model, axis=0), "--", c="k", label="model")
    axt.axis("off")
    #axt.set_title(f"{'Median' if star is None else f'Star {star}'} PSF Model ({obs.stack.psf_model_block})", loc="left")
    axt.legend(fontsize='xx-small')

    axr.plot(np.mean(psf_data, axis=1), y[0], c=c)
    axr.plot(np.mean(psf_model, axis=1), y[0], "--", c="k")
    axr.axis("off")
    ax.text(1, 1, f"FWHM x: {obs.stack.fwhmx:.2f} pix\n"
                f"FWHM y: {obs.stack.fwhmy:.2f} pix\n"
                f"angle: {obs.stack.theta/np.pi*180:.2f}°", c="w", fontsize='xx-small')

def make_plots():
    obs.stack.show(stars=True, figsize=(8,8))
    viz.plot_marks(*obs.stars[0].T, label=0, color='y')
    plt.tight_layout()
    plt.savefig(os.path.join(summary.destination, 'figures/stars.png'),dpi=100)

    plot_psf_model(obs,figsize=(3,2))
    plt.tight_layout()
    plt.savefig(os.path.join(summary.destination,'figures/psf.png'), dpi=100)

    plt.figure(figsize=(6,7))
    viz.plot_systematics_signal(obs.time,obs.diff_flux,opt['systematics'],signal=opt['transit'])
    viz.paper_style()
    plt.legend()
    std = 2 * np.std(obs.diff_flux)
    viz.plot_section(1 + std, "expected transit", obs_t0, obs_dur, c="k")
    plt.xlabel(f"BJD")
    plt.ylabel("diff. flux")
    plt.tight_layout()
    plt.savefig(os.path.join(tm.destination,'figures/model.png'),dpi=100)

    samples = pm.trace_to_dataframe(trace, varnames=variables)
    fig = corner.corner(samples,truths=[opt['P'],opt['r'],opt['t0'],opt['b'],opt['u'][0],opt['u'][1],opt['r_s'],opt['m_s'],opt['ror'],
                                        opt['depth'],opt['a'],opt['a/r_s'],opt['i']])
    fig.patch.set_facecolor('xkcd:white')
    plt.tight_layout()
    plt.savefig(os.path.join(tm.destination,"figures/Corner_plot.png"), dpi=100)

    plt.figure(figsize=(6, 4))
    obs.plot()
    plt.tight_layout()
    plt.savefig(os.path.join(summary.destination,'figures/lightcurve.png'), dpi=100)


    plot_syst(obs)
    plt.savefig(os.path.join(summary.destination,'figures/systematics.png'),dpi=100)


    plot_comps(obs)
    plt.savefig(os.path.join(summary.destination,'figures/comparison.png'),dpi=100)


    plot_raw(obs)
    plt.savefig(os.path.join(summary.destination,'figures/raw.png'),dpi=100)

OUT3=widgets.Output(layout={'border': '1px solid black'})

MAKE_REPORT_B=widgets.Button(description='Build Report')

def make_report_b(b):
    with OUT3:
        clear_output()
        make_report()
        
MAKE_REPORT_B.on_click(make_report_b)
    
MAKE_PLOTS_B=widgets.Button(description='Make Plots!')

def make_plots_b(b):
    with OUT3:
        clear_output()
        make_plots()
        
MAKE_PLOTS_B.on_click(make_plots_b)

BOX_THREE=widgets.VBox([widgets.Label('Try making the report first. If that goes ok, make the plots!'),
                       MAKE_REPORT_B,MAKE_PLOTS_B,OUT3])


#finishing

def wrap():
    tm.to_csv_report()
    report.compile()
    copy_figures(folder=tm.destination.parent,
                 prefix=obs.tfop_prefix,
                 destination=os.path.join(tm.destination.parent,"figures"))

    obs.stack.show_cutout(star=0)
    plt.savefig(os.path.join(summary.destination.parent, 'figures/field_zoom.png'),dpi=100)
    
WRAP_B=widgets.Button(description='Compile it!')

def wrap_b(b):
    with OUT4:
        clear_output()
        wrap()
    
WRAP_B.on_click(wrap_b)
    
OUT4=widgets.Output(layout={'border': '1px solid black'})

BOX_FOUR=widgets.VBox([widgets.Label("Finally, compile the report and check it's ok"),
                      WRAP_B,OUT4])
