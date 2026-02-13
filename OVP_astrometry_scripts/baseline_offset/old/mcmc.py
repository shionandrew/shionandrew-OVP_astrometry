from scipy.optimize import least_squares,curve_fit
import numpy as np
import emcee
import importlib
import astropy.units as un
from scipy.interpolate import RegularGridInterpolator
from glob import glob
from outriggers_vlbi_pipeline.vlbi_pipeline_config import hco as og_hco# as og_hco

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

if __name__=='__main__': 
    import argparse
    parser = argparse.ArgumentParser("Correlator Search Executable")
    parser.add_argument("--out_tag",help='out',type=str,default='synthetic_test')
    parser.add_argument("--tel2",help='tel2',type=str,default='hco')
    parser.add_argument("--in_dir",help='in_dir',type=str,default='/arc/projects/chime_frb/vlbi/hco_comissioning2/synthetic_test_chime-hco_perturbed_single/grid/1_grid_synthetic_test_hco_perturbed_refined_')
    
    cmdargs = parser.parse_args()
    out_tag=cmdargs.out_tag
    in_dir=cmdargs.in_dir+'*.npy'
    tel2=cmdargs.tel2
    out=f'{out_tag}_{tel2}_mcmc.npy'
    print(out)
    
    all_snrs=[]
    tau_interpolants=[]
    all_tau_meas=[]
    all_tau_meas_no_iono=[]

    out_files=[]
    used=[]
    delta_ys=[]
    delta_xs=[]
    tecs=[]
    files_used=[]
    delta_ys=[]
    delta_xs=[]


    #df=pandas.read_csv(f'/arc/home/shiona/scripts/hco_comissioning2_{tag}.csv') #/arc/home/shiona/scripts/hco_test2_baseline_offset_fit_data2.csv')
    #tag=f'A22_manual_fit_all_{tel2}_refined'
    #in_dir=f'/arc/projects/chime_frb/vlbi/hco_comissioning2/{tag}/grid/'
    print(in_dir)
    files=glob(in_dir)
    #print(files)
    import time
    for f in files:#range(len(df)):
        #tar=df['name'][i]
        #cal=df['calibrator_name'][i]
        #eid=df['event_id'][i]
        #dy=df['delta_y'][i]
        #dx=df['delta_x'][i]
        #tec=df['tec_xx'][i]
        #/arc/projects/chime_frb/vlbi/manual_triggers/M12_OVP_astrometry_chime-hco_single/grid/20250418054922_grid_M12_OVP_astrometry_hco_refined_J1145+6958_calibrated_to_J1135+4258.npy
        #print(f'{outdir}{eid}_grid_{tag}_{tar}_calibrated_to_{cal}.npy')
        #if tag==f'A22_manual_fit_all_{tel2}_refined' or tag==f'A22_manual_fit_all_{tel2}':
        #    f=glob(f'{outdir}{eid}_grid_{tag}_{tar}_calibrated_to_{cal}.npy')
        #else:
        #    f=glob(f'{outdir}*_{tar}_calibrated_to_{cal}*.npy')
        #f=glob(f'/arc/projects/chime_frb/vlbi/hco_comissioning2/zoomed_M5_manual_fit_all_/grid/*{eid}*{tar}*{cal}*')
        print(f)
        grid=np.load(f)
        snr=grid['incoh_snr_xx'][0]
        cohsnr=grid['snr_xx'][0]

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
            interpolant=RegularGridInterpolator((xs,ys,zs),out_taus)
            tau_interpolants.append(interpolant)
            all_tau_meas.append(tau_meas)
            all_tau_meas_no_iono.append(grid['tau_meas_no_iono'][0])


    print(len(all_tau_meas))    

    bounds = [(min(xs),max(xs)),(min(ys),max(ys)),(min(zs),max(zs))]
    all_snrs=np.array(all_snrs)
    all_tau_meas=np.array(all_tau_meas)



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
    nsteps = 2000  # Choose based on convergence
    sampler.run_mcmc(initial_pos, nsteps, progress=True)

    # Get the samples after discarding burn-in steps
    burn_in = int(nsteps*0.1)
    samples = sampler.get_chain(discard=burn_in, flat=True)

    #np.save('hco_comissioning2_M5_fit_out_all.npy', samples)
    #np.save(f'hco_comissioning2_{tel2}.npy', samples)
    print(out)
    np.save(out, samples)

    # Get the best-fit parameters (median and confidence intervals)
    best_fit = np.median(samples, axis=0)
    lower_bound = np.percentile(samples, 16, axis=0)
    upper_bound = np.percentile(samples, 84, axis=0)

    print("Best-fit parameters:", best_fit)
    print("Uncertainty (16th-84th percentile):", upper_bound - lower_bound)
