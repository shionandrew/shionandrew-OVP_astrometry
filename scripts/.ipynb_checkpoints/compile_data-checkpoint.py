import outriggers_vlbi_pipeline.vlbi_pipeline_config as config
from outriggers_vlbi_pipeline.query_database import update_event_status,get_event_data, find_files,fetch_data_from_sheet,check_correlation_completion,get_target_vis_files,get_cal_vis_files
import coda
from outriggers_vlbi_pipeline.diagnostic_plots import plot_visibility_diagnostics
from outriggers_vlbi_pipeline.localization.localization_model import apply_baseband_prior

from outriggers_vlbi_pipeline.diagnostic_plots import *

from IPython.display import Image
from IPython.core.display import Image, display
from outriggers_vlbi_pipeline.calibration import load_vis_target,load_all_calibrator_vis,create_calibrated_visibilities
import outriggers_vlbi_pipeline
close=False

from beam_model.utils import get_position_from_equatorial,get_equatorial_from_position
import pandas

def get_seps(vis_target,baseline='chime-hco'):
    vis_bl=vis_target[baseline]
    ctime=np.nanmedian(vis_target['chime']['time0']['ctime'][:])
    x_deg, y_deg = get_position_from_equatorial(vis_target.ra[0], vis_target.dec[0],ctime)
    x, y = get_position_from_equatorial(vis_bl['calibrator_ra'][0], vis_bl['calibrator_dec'][0],ctime)
    x_sep=x_deg-x
    y_sep=y_deg-y
    return x_sep,y_sep

def extract_info(vis,tel2,tec_grid=np.array([0]),fmin=800,fmax=800):
    visbl=vis[f'chime-{tel2}']
    mask=np.where((vis.freqs>fmin)&(vis.freqs<fmax))
    visbl['vis'][mask]=0.0
    visbl['vis_uncalibrated'][mask]=0.0

    out_dict={}
    lag,incoh_snrs=np.array(coda.analysis.delay.extract_frame_delay(vis[f'chime-{tel2}']))
    out_dict['incoh_snr_xx']=incoh_snrs[0,0]
    out_dict['incoh_snr_yy']=incoh_snrs[1,1]
    
    ctime=np.nanmedian(vis['chime']['time0']['ctime'][:])
    out_dict['ctime']=ctime
    out_dict['event_id']=vis.event_id
    visbl=vis[f'chime-{tel2}']
    out_dict['calibrator_name']=visbl['calibrator_source_name'][0].astype(str)
    out_dict['calibrator_ra']=visbl['calibrator_ra'][0]
    out_dict['calibrator_dec']=visbl['calibrator_dec'][0]
    out_dict['ra']=vis.ra[0]
    out_dict['name']=vis[f'chime-{tel2}'].index_map['pointing_center']['source_name'][0].astype(str)
    out_dict['dec']=vis.dec[0]
    tel=coda.core.baseline.tels_to_astropy(vis[tel2]['auto'].attrs['station'])
    out_dict[ f'{tel2}_x']=tel.x.value
    out_dict[ f'{tel2}_y']=tel.y.value
    out_dict[ f'{tel2}_z']=tel.z.value

    tau_xx,tau_yy=vis[f'chime-{tel2}']['vis'].attrs['taus']
    tau_no_ionoxx,tau_no_ionoyy=vis[f'chime-{tel2}']['vis'].attrs['taus_no_iono']
    tecs0,tecs1=vis[f'chime-{tel2}']['vis'].attrs['tecs']

    out_dict['tau_xx']=tau_xx
    out_dict['tau_yy']=tau_yy
    out_dict['tau_no_iono_xx']=tau_no_ionoxx
    out_dict['tau_no_iono_yy']=tau_no_ionoyy
    out_dict['tec_xx']=tecs0
    out_dict['tec_yy']=tecs1
    
        
    ### uncalibrated 
    taus,snrs=np.array(coda.analysis.delay._extract_subframe_delay(visbl['vis_uncalibrated'][:,0,:,:,0,0]))
    out_dict['uncalibrated_snr_xx']=snrs[0,0]
    out_dict['uncaibrated_snr_yy']=snrs[1,1]
    
    ### calibrated but no iono fit 
    taus,snrs=np.array(coda.analysis.delay._extract_subframe_delay(visbl['vis'][:,0,:,:,0,0]))
    out_dict['calibrated_snr_xx']=snrs[0,0]
    out_dict['caibrated_snr_yy']=snrs[1,1]
    
    coda.analysis.iono.remove_tec(visbl,delta_tec=tecs0,write=True)
    coh_tau,coh_snrs=np.array(coda.analysis.delay.extract_subframe_delay(visbl))
    out_dict['snr_xx']=coh_snrs[0,0]
    coda.analysis.iono.undo_tec(visbl,verbose=True)
    coda.analysis.iono.remove_tec(visbl,delta_tec=tecs1,write=True)
    coh_tau,coh_snrs=np.array(coda.analysis.delay.extract_subframe_delay(visbl))
    out_dict['snr_yy']=coh_snrs[1,1]
    coda.analysis.iono.undo_tec(visbl,verbose=True)

    
    ### calibrator 
    cal_taus,cal_snrs=np.array(coda.analysis.delay._extract_subframe_delay(visbl['calibrator_vis'][:,0,:,:,0,0]))
    out_dict['cal_snr_xx']=cal_snrs[0,0]
    out_dict['cal_snr_yy']=cal_snrs[1,1]
    
    cal_incohtaus,cal_incohsnrs=np.array(coda.analysis.delay._extract_frame_delay(visbl['calibrator_vis'][:,0,:,:,:,0]))
    out_dict['cal_incoh_snr_xx']=cal_incohsnrs[0,0]
    out_dict['cal_incoh_snr_yy']=cal_incohsnrs[1,1]
    
    # additionally mask out calibrator
    mask=np.where(visbl['vis'][:]==0.0)
    visbl['calibrator_vis'][mask]=0.0

    nfreq=len(np.where(np.abs(visbl['vis'][:,0,0,0,0,0])>0.0)[0])
    out_dict['bw_00']=nfreq
    nfreq=len(np.where(np.abs(visbl['vis'][:,0,1,1,0,0])>0.0)[0])
    out_dict['bw_11']=nfreq
    
    cal_taus,cal_snrs=np.array(coda.analysis.delay._extract_subframe_delay(visbl['calibrator_vis'][:,0,:,:,0,0]))
    out_dict['cal_snr_xx_masked']=cal_snrs[0,0]
    out_dict['cal_snr_yy_masked']=cal_snrs[1,1]
    
    cal_incohtaus,cal_incohsnrs=np.array(coda.analysis.delay._extract_frame_delay(visbl['calibrator_vis'][:,0,:,:,:,0]))
    out_dict['cal_incoh_snr_xx_masked']=cal_incohsnrs[0,0]
    out_dict['cal_incoh_snr_yy_masked']=cal_incohsnrs[1,1]
    
    return out_dict


if __name__=='__main__': 
    import argparse
    parser = argparse.ArgumentParser("Correlator Search Executable")
    parser.add_argument("--tel2", help=" )", type=str,default='gbo')
    parser.add_argument("--fmin", help=" )", type=int,default=400)
    parser.add_argument("--fmax", help=" )", type=int,default=400)
    
    cmdargs = parser.parse_args()
    tel2=cmdargs.tel2
    fmin=cmdargs.fmin
    fmax=cmdargs.fmax
    inputtag=f'M22_true_pos_fit_{tel2}'
            
    from coda.core import VLBIVis
    from glob import glob
    all_vis=[]
    mode='calibrated' 
    tag=f'M22_true_pos_fit_{tel2}'
    calibrated_files=glob(f'/arc/projects/chime_frb/vlbi/OVP_astrometry_{tel2}/{inputtag}/*/{mode}/*')
    print(len(calibrated_files))
    for i,file in enumerate(calibrated_files):
        if i%10==0:
            print(i)
        vis=VLBIVis.from_file(file)
        vis.attrs.filename=file
        all_vis.append(vis)

    import pandas as pd
    import importlib
    out_dicts=[]
    for i,vis in enumerate(all_vis):
        try:
            out_dict=extract_info(vis,tel2=tel2,fmin=fmin,fmax=fmax)
            out_dict['filename']=vis.attrs.filename
            out_dicts.append(out_dict)
        except:
            print(i)
    df = pd.DataFrame(out_dicts)

    import beam_model
    xs=[]
    ys=[]
    for i in range(len(df)):
        ctime=df['ctime'][i]
        ra=df['calibrator_ra'][i]#.ra.deg[0]
        dec=df['calibrator_dec'][i]#.dec.deg[0]
        x,y=beam_model.utils.get_position_from_equatorial(ra, dec,ctime)
        xs.append(x)
        ys.append(y)
    df['x_cal']=xs
    df['y_cal']=ys



    xs=[]
    ys=[]
    for i in range(len(df)):
        ctime=df['ctime'][i]
        ra=df['ra'][i]#.ra.deg[0]
        dec=df['dec'][i]#.dec.deg[0]
        x,y=beam_model.utils.get_position_from_equatorial(ra, dec,ctime)
        xs.append(x)
        ys.append(y)
    df['x_tar']=xs
    df['y_tar']=ys

    df['delta_x']=df['x_tar']-df['x_cal']
    df['delta_y']=df['y_tar']-df['y_cal']



    good=['B0136+57',
    'B0329+54',
    'B0531+21',
    'B1541+09',
    'B2154+40',
    'B2021+51',
    'B2016+28',
    'B0355+54',
    'B0919+06',
    'B2310+42',
    'B1642-03',
    'B1917+00',
    'B1322+83',
    'B0834+06',######
    'B1508+55',
    'B1919+21',
    'B2217+47',
    'B2255+58',
    'B1905+39']
    fs=15
    keeps=[]
    for i in range(len(df)):
        keep=True
        tar_name=df['name'][i]
        cal_name=df['calibrator_name'][i]
        if 'B' in tar_name and tar_name not in good:
            keep=False
        if 'B' in cal_name in cal_name not in good:
            keep=False
        keeps.append(keep)
    df['pulsar_good_pos']=keeps

    df['tau_xx_ns']=df['tau_xx']*1e3
    df['tau_yy_ns']=df['tau_yy']*1e3

    out_file=f'/arc/home/shiona/OVP_astrometry/notebooks/plotting_data/latest_hco_comissioning2_{inputtag}_all_{tel2}_{mode}.csv'
    print(out_file)
    df.to_csv(out_file,index=False)

