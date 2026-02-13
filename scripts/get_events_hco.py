from coda.core import VLBIVis
from glob import glob
import re
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
from coda.core import VLBIVis
from outriggers_vlbi_pipeline.query_database import get_event_data, find_files
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
import pandas



best_fit_params=[-2523649.92736954,-4123697.10433308,4147773.43142168]#[-2523643.44047669,-4123699.84440233 ,4147774.23913098]
#### NEW POSITION, as of June ###### 
new_hco = ac.EarthLocation.from_geocentric(
    x = (best_fit_params[0]) * un.m,  
    y = (best_fit_params[1]) * un.m,  
    z = (best_fit_params[2]) * un.m  
)
new_hco.info.name = 'hco'

hco=new_hco

####################################
#### NEW POSITION, as of June ###### 
####################################
new_best_fit_params=[883729.31850621,-4924463.81125919,3943956.82880664]#[ 883728.02446502, -4924463.3225994 ,  3943957.56097847]
new_gbo = ac.EarthLocation.from_geocentric(
    x = (new_best_fit_params[0]) * un.m,  
    y = (new_best_fit_params[1]) * un.m,  
    z = (new_best_fit_params[2]) * un.m  
)
new_gbo.info.name = 'gbo'
gbo=new_gbo

cal_df=get_calibrator_dataframe()

MASK_RFI=False
tel='hco'
if tel=='hco':
    MASK_RFI=False

if tel=='kko':
    tec_grid=np.array([0])
else:
    tec_grid=np.arange(-40,40,.1)

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
    files=glob(f'/arc/projects/chime_frb/vlbi/hco_comissioning_bl/*/*/*/{event_id}/calibrator_visibilities/*')
    if True:
        if len(files)>0:
            vis=VLBIVis.from_file(files[0])
            if f'chime-{tel}' in vis.keys():
                hco_position=coda.core.baseline.tels_to_astropy(vis['hco']['auto'].attrs['station'])
                if hco_position!=hco:
                    print(event_id)
                    events_to_redo.append(event_id)
                else:
                    final_events_to_use.append(event_id)
                    configs.append('hco_comissioning_bl')
            else:
                events_to_redo.append(event_id)
        else:
            events_to_redo.append(event_id)
df_redo=pandas.DataFrame(events_to_redo,columns=['event_id'])
df_redo.to_csv(f'{tel}_redo_events.csv',index=False)

## phase reference everything to brightest cal in dump
baseline_name=f'chime-{tel}'
valid_keys=['chime',baseline_name,'index_map',tel]
tag=f'M22_true_pos_fit_{tel}'
if MASK_RFI:
    tag+='_MASK_RFI'
import re

failed_events=[]
for i,event_id in enumerate(final_events_to_use):
    config=configs[i]

    out_dir=f'/arc/projects/chime_frb/vlbi/OVP_astrometry_{tel}/{tag}/{event_id}/calibrated/'
    input_dir=f'/arc/projects/chime_frb/vlbi/{config}/*/*/*/{event_id}/calibrator_visibilities/'
    tar_input_dir=f'/arc/projects/chime_frb/vlbi/{config}/*/*/*/{event_id}/target_visibilities/'
    print(f"input dir: {input_dir}")
    print(f"out_dir dir: {out_dir}")
    if len(glob(f'{out_dir}*'))==0:
        try:
            dfx=detection_df[detection_df['event_id']==event_id].reset_index(drop=True)
            dfx=dfx[dfx['snr']>10]
            dfx=dfx.sort_values(by='snr',ascending=False).reset_index(drop=True)
            sources=(dfx['name'])
            bright_cal=sources[0]
            print(bright_cal)
            print(f"{input_dir}*{bright_cal}*")
            calibrator_file=glob(f"{input_dir}*{bright_cal}*")[0]

            cal_vis=VLBIVis.from_file(calibrator_file)
            coda.analysis.flagging.get_optimal_rfi_mask(cal_vis[baseline_name],write=True,sub_integration_scan=0)

            target_files=glob(f"{tar_input_dir}*B*")
            for cal in sources[1:]:
                print(cal)
                target_files.append(glob(f"{input_dir}*{cal}*")[0])
            print(len(target_files))
            for tar_file in target_files:
                vis_target=VLBIVis.from_file(tar_file)
                coda.analysis.flagging.get_optimal_rfi_mask(vis_target[baseline_name],write=True,sub_integration_scan=0)
                create_calibrated_visibilities([cal_vis],vis_target,out_dir=out_dir)

            calibrated_files=glob(f'{out_dir}*')
            print(calibrated_files)
            for file in calibrated_files:
                vis=VLBIVis.from_file(file)
                if 'tecs' not in vis[f'chime-{tel}']['vis'].attrs.keys():
                    if tel=='hco' and MASK_RFI:
                        mask=np.where((vis.freqs<550)&(vis.freqs>490))
                        vis[f'chime-hco']['vis'][mask]=0.0
                    fringefit(vis[f'chime-{tel}'],tec_grid=tec_grid)
                    vis.save(file)
        except Exception as e:
            failed_events.append(event_id)
    else:#except Exception as e:
        print(f"{event_id} already done")

print('failed events:')
print(failed_events)