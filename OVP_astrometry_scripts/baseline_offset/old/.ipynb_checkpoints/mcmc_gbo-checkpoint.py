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



def log_likelihood(theta, tau_meas, tau_interpolants, snrs, bounds):
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

tel2='hco'
tag='A22_manual_fit_all'
df=pandas.read_csv(f'/arc/home/shiona/scripts/hco_comissioning2_{tag}_{tel2}.csv') #/arc/home/shiona/scripts/hco_test2_baseline_offset_fit_data2.csv')
outdir=f'/arc/projects/chime_frb/vlbi/hco_comissioning2/{tag}_{tel2}/grid/'

tag=f'A22_manual_fit_all_{tel2}_refined'
outdir=f'/arc/projects/chime_frb/vlbi/hco_comissioning2/{tag}/grid/'
import time

#df=pandas.read_csv(f'/arc/home/shiona/scripts/hco_comissioning2_M4_true_pos_fit_all_gbo.csv') 
for i in range(len(df)):
    tar=df['name'][i]
    cal=df['calibrator_name'][i]
    eid=df['event_id'][i]
    dy=df['delta_y'][i]
    dx=df['delta_x'][i]
    tec=df['tec_xx'][i]
    f=glob(f'{outdir}{eid}_grid_{tag}_{tar}_calibrated_to_{cal}.npy')
    #f=glob(f'/arc/projects/chime_frb/vlbi/hco_comissioning2/M4_true_pos_gbo/grid/*{eid}*{tar}*{cal}*')
    if len(f)>0 and f not in files_used:
        assert len(f)==1, print(f)
        f=f[0]
        grid=np.load(f)
        snr=grid['incoh_snr_xx'][0]
        cohsnr=grid['snr_xx'][0]
        if True:#np.abs(grid['tau_meas'][0]*1e3)<30 and cohsnr>15:# and snr>10:
            xs=np.unique(grid['x'])
            ys=np.unique(grid['y'])
            zs=np.unique(grid['z'])
            out_taus=np.zeros(shape=(len(xs),len(ys),len(zs)),dtype=float)
            for i,x in enumerate(xs):
                for j,y in enumerate(ys):
                    for k,z in enumerate(zs):

                        val=grid['tau'][np.where(
                            (
                                (grid['x']==x)&(grid['y']==y)
                            )
                            &(grid['z']==z)
                        )][0]
                        out_taus[i,j,k]=val

            tau_meas=grid['tau_meas'][0]
            all_snrs.append(snr)#grid['snr_xx'][0])
            delta_ys.append(dy)
            delta_xs.append(dx)
            out_files.append(f)
            interpolant=RegularGridInterpolator((xs,ys,zs),out_taus)
            tau_interpolants.append(interpolant)
            all_tau_meas.append(tau_meas)
            all_tau_meas_no_iono.append(grid['tau_meas_no_iono'][0])
            used.append(True)
            tecs.append(tec)
            files_used.append(f)
        else:
            print(eid)
            print(tar)
            print(cohsnr)
            print(np.abs(grid['tau_meas'][0]*1e3))
    else:
        print(eid)
    
all_snrs=np.array(all_snrs)        
    
all_tau_meas=np.array(all_tau_meas)

print(len(all_tau_meas))

bounds = [(min(xs),max(xs)),(min(ys),max(ys)),(min(zs),max(zs))]
initial_guess =[np.median(xs),np.median(ys),np.median(zs)]


# Run MCMC
#nsteps = int(1e5)
#sampler.run_mcmc(initial_pos, nsteps, progress=True)


import emcee
import numpy as np

# Number of dimensions (parameters to estimate)
ndim = len(bounds)

# Number of walkers (should be at least 2-3 times ndim)
nwalkers = 10 * ndim

# Initialize walkers around the initial guess with small random noise
initial_pos = np.array(initial_guess) + 1e-3 * np.random.randn(nwalkers, ndim)

# Define the MCMC sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=(all_tau_meas, tau_interpolants, all_snrs, bounds))

# Run the MCMC for a number of steps
nsteps = 20000  # Choose based on convergence
sampler.run_mcmc(initial_pos, nsteps, progress=True)

# Get the samples after discarding burn-in steps
burn_in = int(nsteps*0.1)
samples = sampler.get_chain(discard=burn_in, flat=True)
print(f'hco_comissioning2_{tel2}.npy')
np.save(f'hco_comissioning2_{tel2}.npy', samples)

# Get the best-fit parameters (median and confidence intervals)
best_fit = np.median(samples, axis=0)
lower_bound = np.percentile(samples, 16, axis=0)
upper_bound = np.percentile(samples, 84, axis=0)

print("Best-fit parameters:", best_fit)
print("Uncertainty (16th-84th percentile):", upper_bound - lower_bound)
