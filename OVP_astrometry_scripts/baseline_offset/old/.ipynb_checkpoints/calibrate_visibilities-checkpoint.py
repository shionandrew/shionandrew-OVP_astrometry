
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
detection_df_hco=get_all_event_ids(cal_df,tel2='hco')



## phase reference everything to brightest cal in dump
tel='hco'
baseline_name=f'chime-{tel}'
valid_keys=['chime',baseline_name,'index_map',tel]
tag='M5_fit'
amplitude_scaling='neutral'
eids=glob(f"/arc/projects/chime_frb/vlbi/hco_comissioning2/{tag}/*")
import re
eids=[int(re.split('/',x)[-1]) for x in eids]
print(eids)

pulsar_locs=True 

if pulsar_locs:
    print("PHASE REFERENCING PULSAR TO ALL CALS")
    for event_id in eids:
        out_dir=f'/arc/projects/chime_frb/vlbi/hco_comissioning2/{tag}/{event_id}/pulsar_calibrated/'
        input_dir=f'/arc/projects/chime_frb/vlbi/hco_comissioning2/{tag}/{event_id}/*'
        if True:#len(glob(f'{out_dir}*{amplitude_scaling}*'))==0:#True:#try:
            dfx=detection_df_hco[detection_df_hco['event_id']==event_id].reset_index(drop=True)
            dfx=dfx[dfx['snr']>15]
            if len(dfx)>0:
                dfx=dfx.sort_values(by='snr',ascending=False).reset_index(drop=True)
                sources=(dfx['name'])
                bright_cal=sources[0]
                print(bright_cal)
                print(f"{input_dir}*{bright_cal}*")
                calibrator_file=glob(f"{input_dir}*{bright_cal}*")[0]

                cal_vis=VLBIVis.from_file(calibrator_file)
                coda.analysis.flagging.get_optimal_rfi_mask(cal_vis[baseline_name],write=True,sub_integration_scan=0)

                target_files=glob(f"{input_dir}*B*")
                vis_target=VLBIVis.from_file(target_files[0])
                coda.analysis.flagging.get_optimal_rfi_mask(vis_target[baseline_name],write=True,sub_integration_scan=0)

                cal_files=[]
                for cal in sources[:]:
                    if 'B' not in cal:
                        print(cal)
                        cal_files.append(glob(f"{input_dir}*{cal}*")[0])
                all_cal_vis=[]
                for cal_file in cal_files:
                    cal_vis=VLBIVis.from_file(cal_file)
                    coda.analysis.flagging.get_optimal_rfi_mask(cal_vis[baseline_name],write=True,sub_integration_scan=0)
                    all_cal_vis.append(cal_vis)

                create_calibrated_visibilities(all_cal_vis,vis_target,out_dir=out_dir,amplitude_scaling=amplitude_scaling,tag=amplitude_scaling+'_')

                calibrated_files=glob(f'{out_dir}*')
                for file in calibrated_files:
                    vis=VLBIVis.from_file(file)
                    mask=np.where((vis.freqs<550)&(vis.freqs>500))
                    vis['chime-hco']['vis'][mask]=0.0
                    if 'tecs' not in vis['chime-hco']['vis'].attrs.keys():
                        fringefit(vis['chime-hco'],tec_grid=np.arange(-20,20,.1))
                        vis.save(file)
        else:#except Exception as e:
            print(f"{event_id} already done")
            
else:
    for event_id in eids:
        out_dir=f'/arc/projects/chime_frb/vlbi/hco_comissioning2/{tag}/{event_id}/calibrated/'
        input_dir=f'/arc/projects/chime_frb/vlbi/hco_comissioning2/{tag}/{event_id}/*'
        if len(glob(f'{out_dir}*{amplitude_scaling}*'))==0:#True:#try:
            dfx=detection_df_hco[detection_df_hco['event_id']==event_id].reset_index(drop=True)
            dfx=dfx[dfx['snr']>15]
            if len(dfx)>0:
                dfx=dfx.sort_values(by='snr',ascending=False).reset_index(drop=True)
                sources=(dfx['name'])
                bright_cal=sources[0]
                print(bright_cal)
                print(f"{input_dir}*{bright_cal}*")
                calibrator_file=glob(f"{input_dir}*{bright_cal}*")[0]

                cal_vis=VLBIVis.from_file(calibrator_file)
                coda.analysis.flagging.get_optimal_rfi_mask(cal_vis[baseline_name],write=True,sub_integration_scan=0)

                target_files=glob(f"{input_dir}*B*")
                for cal in sources[1:]:
                    if 'B' not in cal:
                        print(cal)
                        target_files.append(glob(f"{input_dir}*{cal}*")[0])
                print(len(target_files))
                for tar_file in target_files:
                    vis_target=VLBIVis.from_file(tar_file)
                    coda.analysis.flagging.get_optimal_rfi_mask(vis_target[baseline_name],write=True,sub_integration_scan=0)
                    create_calibrated_visibilities([cal_vis],vis_target,out_dir=out_dir,amplitude_scaling=amplitude_scaling,tag=amplitude_scaling+'_')

                calibrated_files=glob(f'{out_dir}*')
                for file in calibrated_files:
                    vis=VLBIVis.from_file(file)
                    mask=np.where((vis.freqs<550)&(vis.freqs>500))
                    vis['chime-hco']['vis'][mask]=0.0
                    if 'tecs' not in vis['chime-hco']['vis'].attrs.keys():
                        fringefit(vis['chime-hco'],tec_grid=np.arange(-40,40,.1))
                        vis.save(file)
        else:#except Exception as e:
            print(f"{event_id} already done")