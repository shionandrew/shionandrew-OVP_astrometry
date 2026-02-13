
from outriggers_vlbi_pipeline.query_database import get_outrigger_pulsar_disk_subset,get_calibrator_dataframe,find_files
from datetime import datetime
import astropy.units as un
from caput.time import Observer
import astropy.coordinates as ac
import outriggers_vlbi_pipeline.vlbi_pipeline_config as config
from outriggers_vlbi_pipeline.vlbi_pipeline_config import chime,hco
import numpy as np
from outriggers_vlbi_pipeline.cross_correlate_data import recorrelate_data
from outriggers_vlbi_pipeline.calibrator_search.find_fringes import get_all_event_ids
from coda.core import VLBIVis
import copy
import os
import logging
from glob import glob
from outriggers_vlbi_pipeline.calibration import create_calibrated_visibilities,fringefit
import coda

def extract_info(vis,baseline,fmin=800,fmax=800):
    visbl=vis[baseline]
    mask=np.where((vis.freqs>fmin)&(vis.freqs<fmax))
    visbl['vis'][mask]=0.0
    visbl['vis_uncalibrated'][mask]=0.0

    out_dict={}
    lag,incoh_snrs=np.array(coda.analysis.delay.extract_frame_delay(vis[f'{baseline}']))
    out_dict['incoh_snr_xx']=incoh_snrs[0,0]
    out_dict['incoh_snr_yy']=incoh_snrs[1,1]
    import re
    tel1=re.split('-',baseline)[0]
    tel2=re.split('-',baseline)[-1]
    ctime=np.nanmedian(vis[tel1]['time0']['ctime'][:])
    out_dict['ctime']=ctime
    out_dict['event_id']=vis.event_id
    visbl=vis[f'{baseline}']
    out_dict['calibrator_name']=visbl['calibrator_source_name'][0].astype(str)
    out_dict['calibrator_ra']=visbl['calibrator_ra'][0]
    out_dict['calibrator_dec']=visbl['calibrator_dec'][0]
    out_dict['ra']=vis.ra[0]
    out_dict['name']=vis[baseline].index_map['pointing_center']['source_name'][0].astype(str)
    out_dict['dec']=vis.dec[0]

    tel=coda.core.baseline.tels_to_astropy(vis[tel2]['auto'].attrs['station'])
    out_dict[ f'{tel2}_x']=tel.x.value
    out_dict[ f'{tel2}_y']=tel.y.value
    out_dict[ f'{tel2}_z']=tel.z.value
    
    tel=coda.core.baseline.tels_to_astropy(vis[tel1]['auto'].attrs['station'])
    out_dict[ f'{tel1}_x']=tel.x.value
    out_dict[ f'{tel1}_y']=tel.y.value
    out_dict[ f'{tel1}_z']=tel.z.value

    tau_xx,tau_yy=vis[baseline]['vis'].attrs['taus']
    tau_no_ionoxx,tau_no_ionoyy=vis[baseline]['vis'].attrs['taus_no_iono']
    tecs0,tecs1=vis[baseline]['vis'].attrs['tecs']

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



from outriggers_vlbi_pipeline.query_database import get_outrigger_pulsar_disk_subset,get_calibrator_dataframe,find_files
from datetime import datetime
import astropy.units as un
from caput.time import Observer
import astropy.coordinates as ac
import outriggers_vlbi_pipeline.vlbi_pipeline_config as config
from outriggers_vlbi_pipeline.vlbi_pipeline_config import chime,hco
import numpy as np
from outriggers_vlbi_pipeline.cross_correlate_data import recorrelate_data
from outriggers_vlbi_pipeline.calibrator_search.find_fringes import get_all_event_ids
from coda.core import VLBIVis
import copy
import os
import logging
from glob import glob
from outriggers_vlbi_pipeline.calibration import create_calibrated_visibilities,fringefit
import coda

cal_df=get_calibrator_dataframe()
## phase reference everything to brightest cal in dump
tec_grid=np.arange(-5,5,.1)

tag='M12_OVP_astrometry'
if True:
    out_dir=f'/arc/projects/chime_frb/vlbi/manual_triggers/M12_OVP_astrometry/calibrated/20250417054519/'#20250418054922/'#20250417054519#f'/arc/projects/chime_frb/vlbi/manual_triggers/{tag}/calibrated/'
    calibrated_files=glob(f'{out_dir}*.h5')
    print(calibrated_files)
    for i,file in enumerate(calibrated_files):
        print(i)
        vis=VLBIVis.from_file(file)
        baselines=['chime-gbo','chime-hco']#,'gbo-hco']
        for baseline in baselines:
            if 'tecs' not in vis[baseline]['vis'].attrs.keys():
                fringefit(vis[baseline],tec_grid=tec_grid)
        vis.save(file)

