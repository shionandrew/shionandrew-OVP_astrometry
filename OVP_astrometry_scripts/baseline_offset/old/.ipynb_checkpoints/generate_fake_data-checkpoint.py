## for testing baseline offset fit code

import logging
from glob import glob
import os
import pandas
seconds_to_microsecond = 1e6
from pycalc11 import Calc
import astropy.units as un
import astropy.units as u
import astropy.coordinates as ac
from astropy.time import Time
import numpy as np
from outriggers_vlbi_pipeline.vlbi_pipeline_config import chime,kko,gbo
import astropy.coordinates as ac
import outriggers_vlbi_pipeline
import copy
import coda
from beam_model.utils import get_equatorial_from_position,get_position_from_equatorial
from baseline_offset_fixed import get_tau_applied


new_best_fit_params=[-2523644.20739404, -4123700.3658239 ,  4147773.46403909]
hco = ac.EarthLocation.from_geocentric(
    x = (new_best_fit_params[0]) * un.m,  
    y = (new_best_fit_params[1]) * un.m,  
    z = (new_best_fit_params[2]) * un.m  
)
hco.info.name = 'hco'

hco_perturbed =  ac.EarthLocation.from_geocentric(
    x = (new_best_fit_params[0]-3) * un.m,  
    y = (new_best_fit_params[1]+2) * un.m,  
    z = (new_best_fit_params[2]-4) * un.m  
)
hco_perturbed.info.name = 'hco_perturbed'


new_best_fit_params=[ 883728.02446502, -4924463.3225994 ,  3943957.56097847]
gbo = ac.EarthLocation.from_geocentric(
    x = (new_best_fit_params[0]) * un.m,  
    y = (new_best_fit_params[1]) * un.m,  
    z = (new_best_fit_params[2]) * un.m  
)
gbo.info.name = 'gbo'

gbo_perturbed = ac.EarthLocation.from_geocentric(
    x = (new_best_fit_params[0]-1) * un.m,  
    y = (new_best_fit_params[1]+2) * un.m,  
    z = (new_best_fit_params[2]-4) * un.m  
)
gbo_perturbed.info.name = 'gbo_perturbed'


## for simplicity, just choose one ctime
ctime=1714601774.589819

N=200

x_vals=np.random.uniform(-2,2,N)
y_vals=np.random.uniform(-60,60,N)

x_vals_cal=np.random.uniform(-2,2,N)
y_vals_cal=np.random.uniform(-60,60,N)

ra,dec=get_equatorial_from_position(x_vals,y_vals,ctime)
cal_ra,cal_dec=get_equatorial_from_position(x_vals_cal,y_vals_cal,ctime)

telescopes=[gbo_perturbed,hco_perturbed]
telescopes_true=[gbo,hco]
for j in range(len(telescopes)):
    df=pandas.DataFrame(cal_ra,columns=['calibrator_ra'])
    df['calibrator_dec']=cal_dec
    df['dec']=dec
    df['ra']=ra
    df['ctime']=ctime
    df['calibrator_name']=cal_ra
    df['name']=ra
    
    tau_cal_geo=np.zeros(len(ra),dtype=float)
    tau_tar_geo=np.zeros(len(ra),dtype=float)
    for i in range(len(ra)):
        cal_tau_app=get_tau_applied(ra=cal_ra[i],dec=cal_dec[i],ctime=ctime,tel2s=[telescopes[j]])[0,:] #true delay
        cal_tau_true=get_tau_applied(ra=cal_ra[i],dec=cal_dec[i],ctime=ctime,tel2s=[telescopes_true[j]])[0,:] #true delay

        cal_tau_meas=cal_tau_true-cal_tau_app

        tar_tau_app=get_tau_applied(ra=ra[i],dec=dec[i],ctime=ctime,tel2s=[telescopes[j]])[0,:]
        tar_tau_true=get_tau_applied(ra=ra[i],dec=dec[i],ctime=ctime,tel2s=[telescopes_true[j]])[0,:]

        tar_tau_meas=tar_tau_true-tar_tau_app

        tau_cal_geo[i]=cal_tau_meas
        tau_tar_geo[i]=tar_tau_meas
    
    df['tau_xx']=-(tau_tar_geo-tau_cal_geo) #to match data conjugation
    tel=telescopes[j]
    df[ f'{tel.info.name}_x']=tel.x.value
    df[ f'{tel.info.name}_y']=tel.y.value
    df[ f'{tel.info.name}_z']=tel.z.value
    df['snr_xx']=100
    df['incoh_snr_xx']=100
    df['event_id']=1
    
    df['tau_no_iono_xx']=df['tau_xx']
    df.to_csv(f'/arc/home/shiona/scripts/hco_comissioning2_synthetic_data_chime_{tel.info.name}_test.csv',index=False)
    
for i in range(len(telescopes)):
    for j in range(i+1,len(telescopes)):
        tel1=telescopes[i].info.name
        tel2=telescopes[j].info.name
        df1=pandas.read_csv(f'/arc/home/shiona/scripts/hco_comissioning2_synthetic_data_chime_{tel1}_test.csv')
        df2=pandas.read_csv(f'/arc/home/shiona/scripts/hco_comissioning2_synthetic_data_chime_{tel2}_test.csv')
        df3=copy.deepcopy(df1)
        df3[f'{tel2}_x']=df2[f'{tel2}_x']
        df3[f'{tel2}_y']=df2[f'{tel2}_y']
        df3[f'{tel2}_z']=df2[f'{tel2}_z']
        df3['tau_xx']=df1['tau_xx']-df2['tau_xx']
        df3.to_csv(f'/arc/home/shiona/scripts/hco_comissioning2_synthetic_data_{tel1}_{tel2}_test.csv',index=False)
