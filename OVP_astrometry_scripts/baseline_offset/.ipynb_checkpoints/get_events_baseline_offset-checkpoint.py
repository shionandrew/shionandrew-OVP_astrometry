import re
import copy
import os
import logging
import coda
import pandas
from glob import glob
import coda
from outriggers_vlbi_pipeline.calibration import create_calibrated_visibilities,fringefit,fft_fringefit,get_calibrator_dataframe,find_files
from coda.core import VLBIVis
import os

from outriggers_vlbi_pipeline.query_database import (
    fetch_data_from_sheet,
    write_dataframe_to_sheet,
    find_files,
    get_full_filepath,get_calibrator_dataframe,
)
from outriggers_vlbi_pipeline.calibrator_search.find_fringes import get_all_event_ids
import numpy as np

from beam_model.utils import get_position_from_equatorial,get_equatorial_from_position

def get_beam(vis_target,baseline_name):
    ctime = np.nanmedian(vis_target['chime']['gate']['gate_start_unix'])
    x_deg, y_deg = get_position_from_equatorial(vis_target.ra[0], vis_target.dec[0],ctime)
    vis_target[baseline_name]['vis'].attrs['x_deg']=x_deg
    vis_target[baseline_name]['vis'].attrs['y_deg']=y_deg

    x_deg, y_deg = get_position_from_equatorial(vis_target[baseline_name]['calibrator_ra'][0], 
                                                vis_target[baseline_name]['calibrator_dec'],ctime)

    vis_target[baseline_name]['vis'].attrs['cal_x_deg']=x_deg
    vis_target[baseline_name]['vis'].attrs['cal_y_deg']=y_deg
    
    return


SNR_THRESH=30
df=get_calibrator_dataframe()

detection_df_gbo=get_all_event_ids(df,tel2='gbo')
detection_df_hco=get_all_event_ids(df,tel2='hco')
detection_df_hco['snr'] = pandas.to_numeric(detection_df_hco['snr'], errors='coerce')
detection_df_gbo['snr'] = pandas.to_numeric(detection_df_gbo['snr'], errors='coerce')

detection_df_hco=detection_df_hco[detection_df_hco['snr']>SNR_THRESH].reset_index(drop=True)
detection_df_gbo=detection_df_gbo[detection_df_gbo['snr']>SNR_THRESH].reset_index(drop=True)


def calibrate_bright_visibilities(event_id,tel='hco', tag='trial1',snr_thresh=SNR_THRESH):
    if tel=='hco':
        detection_df=detection_df_hco
    if tel=='gbo':
        detection_df=detection_df_gbo
    baseline_name=f'chime-{tel}'

    dfx=detection_df[detection_df['event_id']==event_id].reset_index(drop=True)
    dfx=dfx[dfx['snr']>snr_thresh]
    dfx=dfx.sort_values(by='snr',ascending=False).reset_index(drop=True)

    input_dir=f'/arc/projects/chime_frb/vlbi/hco*rfi/*/*/*/{event_id}/calibrator_visibilities/'
    out_dir=f'/arc/projects/chime_frb/vlbi/OVP_astrometry_{tel}_F11/{tag}/{event_id}/calibrated/'
    os.makedirs(out_dir,exist_ok=True)

    sources=(dfx['name'])

    tec_grid = np.arange(-40,40,.1)

    for i in range(len(sources)-1):
        go=True
        bright_cal=sources[i]
        try:
            calibrator_file=glob(f"{input_dir}*{bright_cal}*")[0]
        except:
            input_dir=f'/arc/projects/chime_frb/vlbi/hco_comissioning2/*/*/*/{event_id}/calibrator_visibilities/'
            try:
                calibrator_file=glob(f"{input_dir}*{bright_cal}*")[0]
            except:
                print("COULD NOT FIND FILE")
                print(f"{input_dir}*{bright_cal}*")
                go=False
        if go:
            cal_vis=VLBIVis.from_file(calibrator_file)
            coda.analysis.flagging.get_optimal_rfi_mask(cal_vis[baseline_name],write=True,sub_integration_scan=0)
            target_files=[]
            for source in sources[i+1:]:
                try:
                    target_files.append(glob(f"{input_dir}*{source}*")[0])
                except:
                    print(f'could not find {input_dir}*{source}')
                for tar_file in target_files:
                    vis_target=VLBIVis.from_file(tar_file)
                    coda.analysis.flagging.get_optimal_rfi_mask(vis_target[baseline_name],write=True,sub_integration_scan=0)
                    create_calibrated_visibilities([cal_vis],vis_target,out_dir=out_dir)
            print('done calibrating')
            calibrated_files=glob(f'{out_dir}*')
            print(calibrated_files)
            for file in calibrated_files:
                vis=VLBIVis.from_file(file)
                get_beam(vis,f'chime-{tel}')
                if 'tecs' not in vis[f'chime-{tel}']['vis'].attrs.keys():
                    fringefit(vis[f'chime-{tel}'],tec_grid=tec_grid)
                    vis.save(file)
                if 'fft_taus' not in vis[f'chime-{tel}'].attrs.keys():
                        fft_fringefit(vis[f'chime-{tel}'],tec_grid)
                        vis.save(file)

#calibrate_bright_visibilities(438535323)
calibrate_bright_visibilities(440021683)
calibrate_bright_visibilities(441021463)
calibrate_bright_visibilities(1123860224)
calibrate_bright_visibilities(1122984830)
calibrate_bright_visibilities(1138026090)
calibrate_bright_visibilities(1128103969)
calibrate_bright_visibilities(1137928820)
              


'''SNR_THRESH = 30
tag=f'F11_bl_{tel}'
    
cal_df=get_calibrator_dataframe()
tel='hco'
eids=pandas.read_csv(f'{tel}_bl_offset_eids.csv')

if tel=='kko':
    tec_grid=np.array([0])
else:
    tec_grid=np.arange(-90,90,.01)

detection_df=get_all_event_ids(cal_df,tel2=tel)
event_ids=np.unique(detection_df['event_id'])

event_ids_to_use=[]

for event_id in event_ids:
    dfx=detection_df[detection_df['event_id']==event_id].reset_index(drop=True)
    if len(dfx)>1:
        event_ids_to_use.append(event_id)
print(len(event_ids_to_use))

final_events_to_use=[]
configs=[]
events_to_redo=[]
for event_id in event_ids_to_use:
    ver='hco_comissioning2'
    files=glob(f'/arc/projects/chime_frb/vlbi/hco_comissioning2/*/*/*/{event_id}/calibrator_visibilities/*')
    if KL_filter:
        ver='hco_comissioning_rfi'
        files=glob(f'/arc/projects/chime_frb/vlbi/hco_comissioning_rfi/*/*/*/{event_id}/calibrator_visibilities/*')
    if True:
        if len(files)>0:
            vis=VLBIVis.from_file(files[0])
            if f'chime-{tel}' in vis.keys():
                hco_position=coda.core.baseline.tels_to_astropy(vis['hco']['auto'].attrs['station'])
                if hco_position!=hco:
                    events_to_redo.append(event_id)
                else:
                    final_events_to_use.append(event_id)
                    configs.append(ver)
            else:
                events_to_redo.append(event_id)
        else:
            events_to_redo.append(event_id)
df_redo=pandas.DataFrame(events_to_redo,columns=['event_id'])
df_redo.to_csv(f'{tel}_redo_events.csv',index=False)

## phase reference everything to brightest cal in dump
baseline_name=f'chime-{tel}'
valid_keys=['chime',baseline_name,'index_map',tel]


import re

failed_events=[]
for i,event_id in enumerate(final_events_to_use):
    config=configs[i]
    out_dir=f'/arc/projects/chime_frb/vlbi/OVP_astrometry_{tel}_F11/{tag}/{event_id}/calibrated/'
    input_dir=f'/arc/projects/chime_frb/vlbi/{config}/*/*/*/{event_id}/calibrator_visibilities/'
    tar_input_dir=f'/arc/projects/chime_frb/vlbi/{config}/*/*/*/{event_id}/target_visibilities/'
    print(f"input dir: {input_dir}")
    print(f"out_dir dir: {out_dir}")
    if True:#len(glob(f'{out_dir}*'))==0:
        dfx=detection_df[detection_df['event_id']==event_id].reset_index(drop=True)
        dfx=dfx[dfx['snr']>SNR_THRESH]
        dfx=dfx.sort_values(by='snr',ascending=False).reset_index(drop=True)
        sources=(dfx['name'])
        bright_cal=sources[0]
        print(bright_cal)
        print(f"{input_dir}*{bright_cal}*")
        calibrator_file=glob(f"{input_dir}*{bright_cal}*")[0]
        
        target_files=[]
        for source in sources[1:]
            target_files.append(glob(f"{input_dir}*{source}*")[0])

        cal_vis=VLBIVis.from_file(calibrator_file)
        coda.analysis.flagging.get_optimal_rfi_mask(cal_vis[baseline_name],write=True,sub_integration_scan=0)

        #target_files=glob(f"{tar_input_dir}*B*")
        #for cal in sources[1:]:
        #    print(cal)
        #    target_files.append(glob(f"{input_dir}*{cal}*")[0])
        #print(len(target_files))
        for tar_file in target_files:
            vis_target=VLBIVis.from_file(tar_file)
            coda.analysis.flagging.get_optimal_rfi_mask(vis_target[baseline_name],write=True,sub_integration_scan=0)
            create_calibrated_visibilities([cal_vis],vis_target,out_dir=out_dir)

        calibrated_files=glob(f'{out_dir}*')
        print(calibrated_files)
        for file in calibrated_files:
            vis=VLBIVis.from_file(file)
            if 'tecs' not in vis[f'chime-{tel}']['vis'].attrs.keys():
                fringefit(vis[f'chime-{tel}'],tec_grid=tec_grid)
                vis.save(file)

    else:#except Exception as e:
        print(f"{event_id} already done")

print('failed events:')
print(failed_events)'''