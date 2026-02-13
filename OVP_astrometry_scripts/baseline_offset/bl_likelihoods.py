from scipy.optimize import least_squares,curve_fit
import numpy as np
import emcee
import importlib
import astropy.units as un
from scipy.interpolate import RegularGridInterpolator
from glob import glob
from outriggers_vlbi_pipeline.vlbi_pipeline_config import gbo# as og_hco
from functools import lru_cache
from numba import jit
@jit(nopython=True)
def fast_log_likelihood(tau_meas, taus, snrs):
    lls = -((tau_meas - taus) ** 2) * snrs ** 2
    return np.sum(lls)

def simple_log_likelihood(theta, tau_meas, tau_interpolants, snrs, bounds):
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


def full_log_likelihood(theta, tau_meas, tau_interpolants, snrs, bounds):
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


def uniform_prior(params):
    
    x,y,z, alpha = params
    
    prior_std = 50e-3 # us
    prior_var = prior_std**2
        
    if float(alpha) <=0. : #variance must be strictly positive
        return 1
    else: 
        if float(alpha) >= prior_var: 
            return 1
        else: 
            return 0
    
    
def get_correlation(x1, x2, y1, y2, corr_tol=0.): 
    
    if np.abs(x1 - x2) <= corr_tol: # deg 
        if np.abs(y1 - y2) <= corr_tol: # deg
            return 1
    
    return 0
    
def log_likelihood(
    params, 
    tau_meas, 
    yerr, 
    Nij_sys
):
    
    # Assert prior
    if uniform_prior(params): 
        # If True, guess parameter exceeded prior <--> 0% likelihood
        return -np.inf
    
    else: 
        dx,dy,dz,alpha = params
        nevents = tau_meas.shape[0]
        try:    
            model_delays = interpolate_grid_vec(dx, dy, dz)
        except IndexError: 
            return -np.inf

        
        # Build stat noise matrix 
        Nij_stat = np.eye(int(model_delays.size))
        np.fill_diagonal(Nij_stat, yerr.flatten()**2)
        

        # Combine to get total noise matrix
        Nij = Nij_stat + alpha*Nij_sys
        
        # Get determinant & inverse. If it fails, return -np.inf
            
        det_N = 2*np.pi*np.linalg.det(Nij)
        if np.isnan(det_N): 
            return -np.inf
        if det_N == 0:
            return -np.inf
        
        Ninv = np.linalg.pinv(Nij)
        if np.any(np.isnan(Ninv)): 
            return -np.inf
        
        logllhd = -0.5*(tau_meas.flatten() - model_delays.flatten()).T@Ninv@(tau_meas.flatten() - model_delays.flatten()) - 0.5*np.log(det_N)
        
        if np.isnan(logllhd): 
            return -np.inf
        
        return logllhd 