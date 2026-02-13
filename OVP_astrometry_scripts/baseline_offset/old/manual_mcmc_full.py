from scipy.optimize import least_squares,curve_fit
import numpy as np
import emcee
import importlib
import astropy.units as un
from scipy.interpolate import RegularGridInterpolator
from glob import glob
from outriggers_vlbi_pipeline.vlbi_pipeline_config import gbo# as og_hco

from functools import lru_cache

#@lru_cache(maxsize=None)
#def cached_interpolant(index, theta):
#    return tau_interpolants[index](theta)[0]

from numba import jit
@jit(nopython=True)
def fast_log_likelihood(tau_meas, taus, snrs):
    lls = -((tau_meas - taus) ** 2) * snrs ** 2
    return np.sum(lls)



def log_likelihood(theta, tau_meas, tau_interpolants, snrs,bounds):
    """Evaluate the log-likelihood at given parameters using the interpolant."""

    # Check bounds using NumPy (much faster than looping)
    theta = np.asarray(theta)  # Ensure array
    bounds = np.asarray(bounds)

    if np.any(np.clip(theta, bounds[:, 0], bounds[:, 1]) != theta):
        return -np.inf

    # Vectorized interpolation: evaluate all interpolants at once
    #taus = np.array([cached_interpolant(i, tuple(theta)) for i in range(len(tau_interpolants))])
    taus = np.array([tau_interpolant(theta)[0] for tau_interpolant in tau_interpolants])

    # Vectorized log-likelihood calculation
    lls = -((tau_meas - taus) ** 2 * snrs ** 2)
    return fast_log_likelihood(tau_meas, taus, snrs)



import time

from glob import glob
import importlib
import time
import astropy.units as un
from scipy.interpolate import RegularGridInterpolator
import pandas

all_snrs=[]

tau_interpolants=[]
all_tau_meas=[]
all_tau_meas_no_iono=[]
delta_ys=[]
delta_xs=[]
files_used=[] 

used=[]
delta_ys=[]
delta_xs=[]
tecs=[]
out_files=[]


import time

tau_interpolants=[]
all_tau_meas=[]
all_tau_meas_no_iono=[]
all_snrs=[]
files=glob('/arc/projects/chime_frb/vlbi/manual_triggers/M12_OVP_astrometry*/grid/*J1153+8058_calibrated_to_J1135+4258*.npy')
files=glob('/arc/projects/chime_frb/vlbi/manual_triggers/M12_OVP_astrometry*/grid/**.npy')

for f in files:
    print(f)
    if 'single' not in f:#s and 'gbo-hco' in f:
        grid=np.load(f)

        gbo_xs=np.unique(grid['x_gbo'])
        gbo_ys=np.unique(grid['y_gbo'])
        gbo_zs=np.unique(grid['z_gbo'])

        hco_xs=np.unique(grid['x_hco'])
        hco_ys=np.unique(grid['y_hco'])
        hco_zs=np.unique(grid['z_hco'])

        out_taus=np.zeros(
            shape=(len(gbo_xs),len(gbo_ys),len(gbo_zs),len(hco_xs),len(hco_ys),len(hco_zs)),
            dtype=float
        )
        for i,xgbo in enumerate(gbo_xs):
            for j,ygbo in enumerate(gbo_ys):
                for k,zgbo in enumerate(gbo_zs):
                    for l,xhco in enumerate(hco_xs):
                        for m,yhco in enumerate(hco_ys):
                            for n,zhco in enumerate(hco_zs):
                                val=grid['tau'][np.where(
                                    (
                                        (
                                            (grid['x_gbo']==xgbo)&(grid['y_gbo']==ygbo)
                                        )
                                        &(grid['z_gbo']==zgbo)
                                    )
                                    &
                                    (
                                        (
                                            (grid['x_hco']==xhco)&(grid['y_hco']==yhco)
                                        )
                                        &(grid['z_hco']==zhco)
                                    )
                                )][0]
                                out_taus[i,j,k,l,m,n]=val

        tau_meas=grid['tau_meas'][0]
        interpolant=RegularGridInterpolator((gbo_xs,gbo_ys,gbo_zs,hco_xs,hco_ys,hco_zs),out_taus)
        tau_interpolants.append(interpolant)
        all_tau_meas.append(tau_meas)
        all_tau_meas_no_iono.append(grid['tau_meas_no_iono'][0])
        all_snrs.append(grid['incoh_snr_xx'][0])#grid['snr_xx'][0])
print(len(all_tau_meas))
all_tau_meas=np.array(all_tau_meas)
tau_interpolants=np.array(tau_interpolants)
all_snrs=np.array(all_snrs)

bounds = [(min(gbo_xs),max(gbo_xs)),(min(gbo_ys),max(gbo_ys)),(min(gbo_zs),max(gbo_zs)),
         (min(hco_xs),max(hco_xs)),(min(hco_ys),max(hco_ys)),(min(hco_zs),max(hco_zs))]
initial_guess =[np.median(gbo_xs),np.median(gbo_ys),np.median(gbo_zs),
               np.median(hco_xs),np.median(hco_ys),np.median(hco_zs)]

import emcee
import numpy as np

# Number of dimensions (parameters to estimate)
ndim = len(bounds)

# Number of walkers (should be at least 2-3 times ndim)
nwalkers = 10 * ndim

# Initialize walkers around the initial guess with small random noise
initial_pos = np.array(initial_guess) + 1e-3 * np.random.randn(nwalkers, ndim)

# Define the MCMC sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=(all_tau_meas, tau_interpolants, all_snrs,bounds))

# Run the MCMC for a number of steps
nsteps = 6000  # Choose based on convergence
sampler.run_mcmc(initial_pos, nsteps, progress=True)

# Get the samples after discarding burn-in steps
burn_in = int(nsteps*0.1)
samples = sampler.get_chain(discard=burn_in, flat=True)
out=f'joint_mcmc_fit_{len(all_snrs)}.npy'
print(out)
np.save(out, samples)

# Get the best-fit parameters (median and confidence intervals)
best_fit = np.median(samples, axis=0)
lower_bound = np.percentile(samples, 16, axis=0)
upper_bound = np.percentile(samples, 84, axis=0)

print("Best-fit parameters:", best_fit)
print("Uncertainty (16th-84th percentile):", upper_bound - lower_bound)
