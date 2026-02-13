import sys
from coda.core import VLBIVis
from glob import glob
import re
from outriggers_vlbi_pipeline.query_database import get_outrigger_pulsar_disk_subset,get_calibrator_dataframe,find_files,update_calibrator_dataframe
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
import coda
import pandas
from glob import glob
from outriggers_vlbi_pipeline.vlbi_pipeline_config import chime,new_hco,new_gbo
from coda.analysis.flagging import unmask_channels_single_baseline, get_optimal_rfi_mask
from outriggers_vlbi_pipeline.query_database import get_calibrator_dataframe,update_event_status,get_event_data, get_full_filepath, find_files,fetch_data_from_sheet,check_correlation_completion,get_target_vis_files,get_cal_vis_files
from outriggers_vlbi_pipeline.calibrator_search.find_fringes import existing_cal_detections

config.VERSION='manual_triggers'
event_id=20250509042648#20250418054922#20250417054519#20250418054922#20250417054519#20250417054519
eid=event_id
if eid==20250417054519:
    local=True
else:
    local=False

year=str(eid)[:4]
month=str(eid)[4:6]
day=str(eid)[6:8]
singlebeam_outdir=f'/arc/projects/chime_frb/vlbi/manual_triggers/{year}/{month}/{day}/{eid}/calibrator_visibilities/'
print(singlebeam_outdir)
if local:
    df=pandas.read_csv(f'/arc/home/shiona/{eid}_output.csv')
else:
    df=existing_cal_detections(eid)
    
print(df)
for i in range(len(df)):
    source_name=df['name'][i]
    vis_out_dir =f'/arc/projects/chime_frb/vlbi/manual_triggers/{year}/{month}/{day}/{event_id}/calibrator_visibilities/'#get_full_filepath(event_id=event_id,data_type="visibilities", source_type='calibrator')
    vis_out_file = f"{vis_out_dir}{event_id}_{source_name}_vis.h5"
    if len(glob(vis_out_file))==0:
        try:
            print(source_name)
            from outriggers_vlbi_pipeline.burst_search.run_frb_gating import load_bbdata
            file_dir=f'/arc/projects/chime_frb/vlbi/manual_triggers/*/*/*/{event_id}/calibrator_singlebeams/*{source_name}*'
            print(file_dir)
            new_hco.info.name='hco'
            new_gbo.info.name='gbo'
            job,chime_bbdata,wfall=load_bbdata(event_id,telescopes_all=[chime,new_hco,new_gbo],tel_beamformed_dir=file_dir,DM0=0)    
            assert chime_bbdata['tiedbeam_locations']['source_name'][0].astype(str)==source_name
            gate_spec = job.define_scan_params_continuum(pad=0)
            vis = job.run_correlator_job(
                event_id=event_id,
                gate_spec=gate_spec,
                out_h5_file=False,max_lag=100,
                auto_corr=True,interbase=True,
                clear_bbdata=False,assign_pointing = 'nearest',
            )

            valid_keys=[]
            outriggers=['gbo','hco','kko']
            if 'chime-gbo' in vis.keys():
                valid_keys.append('chime-gbo')

            if 'chime-hco' in vis.keys():
                valid_keys.append('chime-hco')

            if 'chime-kko' in vis.keys():
                valid_keys.append('chime-kko')

            import copy
            for key in valid_keys:
                unmask_channels_single_baseline(vis[key])
                from coda.analysis.error_prop import get_vis_std_lag
                get_vis_std_lag(vis[key],write=True)
                get_optimal_rfi_mask(vis[key], write=True, sub_integration_scan=0)


            vis_out_dir =f'/arc/projects/chime_frb/vlbi/manual_triggers/{year}/{month}/{day}/{event_id}/calibrator_visibilities/'#get_full_filepath(event_id=event_id,data_type="visibilities", source_type='calibrator')
            vis_out_file = f"{vis_out_dir}{event_id}_{source_name}_vis.h5"
            os.makedirs(os.path.dirname(vis_out_file), exist_ok=True, mode=0o777)
            logging.info(f"Saving visibilities to {vis_out_file}")
            vis.save(vis_out_file)
        except Exception as e:
            print(e)
    else:
        vis=VLBIVis.from_file(glob(vis_out_file)[0])
        assert vis.source_name[0].astype(str)==source_name

