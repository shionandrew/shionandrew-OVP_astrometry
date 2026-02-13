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
import argparse

if __name__=='__main__': 
    parser = argparse.ArgumentParser("rfi Executable")
    parser.add_argument("--tel", help="telescope (chime gbo hco kko )", type=str)
    parser.add_argument("--config_to_search", help="config_to_search", type=str,default='NONE')
    parser.add_argument("--tag", help="tag", type=str,default='M22_true_pos_fit')
    cmdargs = parser.parse_args()
    tel=cmdargs.tel
    tag=cmdargs.tag
    config_to_search=cmdargs.config_to_search
    if config_to_search=="NONE":
        config_to_search = ['hco_comissioning_bl']#'ovp_upgrade*','hco_comissioning*','manual_triggers']
    else:
        config_to_search = [config_to_search]
    cal_df=get_calibrator_dataframe()

    if tel=='kko':
        tec_grid=np.arange(-10,10,.1)
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
    for event_id in event_ids_to_use:
        done=False
        for config in config_to_search:
            if not done:
                files=glob(f'/arc/projects/chime_frb/vlbi/{config}/*/*/*/{event_id}/calibrator_visibilities/*')
                if len(files)>0:
                    vis=VLBIVis.from_file(files[0])
                    if f'chime-{tel}' in vis.keys():
                        final_events_to_use.append(event_id)
                        c=re.split('vlbi/',files[0])[-1]
                        c=re.split('/',c)[0]
                        configs.append(c)
                        done=True
        if not done:
                print(event_id)


    ## phase reference everything to brightest cal in dump
        
    baseline_name=f'chime-{tel}'
    valid_keys=['chime',baseline_name,'index_map',tel]
    tag=f'{tag}_{tel}'
    import re

    failed_events=[]
    for i,event_id in enumerate(final_events_to_use):
        config=configs[i]
        out_dir=f'/arc/projects/chime_frb/vlbi/OVP_astrometry_{tel}/{tag}/{event_id}/calibrated/'
        input_dir=f'/arc/projects/chime_frb/vlbi/{config}/*/*/*/{event_id}/calibrator_visibilities/'
        tar_input_dir=f'/arc/projects/chime_frb/vlbi/{config}/*/*/*/{event_id}/target_visibilities/'
        print(f"input dir: {input_dir}")
        print(f"out_dir dir: {out_dir}")

        dfx=detection_df[detection_df['event_id']==event_id].reset_index(drop=True)
        dfx=dfx[dfx['snr']>10]
        dfx=dfx.sort_values(by='snr',ascending=False).reset_index(drop=True)
        if len(glob(f'{out_dir}*'))<(len(dfx)+1):
            try:
                sources=(dfx['name'])
                bright_cal=sources[0]
                print(bright_cal)
                print(f"{input_dir}*{bright_cal}*")
                calibrator_file=glob(f"{input_dir}*{bright_cal}*")[0]

                cal_vis=VLBIVis.from_file(calibrator_file)
                coda.analysis.flagging.get_optimal_rfi_mask(cal_vis[baseline_name],write=True,sub_integration_scan=0)

                target_files=glob(f"{tar_input_dir}*B*")
                for cal in sources[1:]:
                    if 'B' not in cal:
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
                        fringefit(vis[f'chime-{tel}'],tec_grid=tec_grid)
                        vis.save(file)
            except Exception as e:
                print('failed')
                print(event_id)
                print(e)
                failed_events.append(event_id)
        else:#except Exception as e:
            print(f"{event_id} already done")


    print('failed events:')
    print(failed_events)