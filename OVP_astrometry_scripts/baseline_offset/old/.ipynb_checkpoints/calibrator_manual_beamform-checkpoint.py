""" A simple script that (1) pulls data from minoc (2) forms multibeamformed data towards the target and calibrators (3) correlates the data and saves visibilities.
Note that this does not yet perform the actual localization step which is done in run_localization.py"""
import outriggers_vlbi_pipeline.vlbi_pipeline_config as config
from outriggers_vlbi_pipeline.multibeamform import beamform_multipointings, beamform_calibrators,rebeamform_singlebeam
from outriggers_vlbi_pipeline.vlbi_pipeline_config import kko_backend, chime_backend,frb_events_database,kko_events_database
from outriggers_vlbi_pipeline.query_database import get_event_data,find_files
from outriggers_vlbi_pipeline.cross_correlate_data import correlate_multibeam_data
from outriggers_vlbi_pipeline.query_database import check_correlation_completion, update_event_status,check_baseband_localization_completion
from dtcli.src import functions
from dtcli.utilities import cadcclient
from ch_util import tools
import numpy as np
from outriggers_vlbi_pipeline.query_database import get_calibrator_dataframe,update_event_status,get_event_data, get_full_filepath, find_files,fetch_data_from_sheet,check_correlation_completion,get_target_vis_files,get_cal_vis_files
from astropy.coordinates import SkyCoord
import astropy.units as un
import os 
from glob import glob
import copy
import datetime
import pandas as pd
import gspread
import time
import subprocess
import logging
import parser
import argparse
import shutil
import traceback
import sys
import re
from pathlib import Path
import time
from outriggers_vlbi_pipeline.known_calibrators import add_cal_status_to_catalogue
import pandas
from coda.core import VLBIVis


from outriggers_vlbi_pipeline.diagnostic_plots import plot_visibility_diagnostics,plot_cross_correlation_lag
from multiprocessing import Pool

from outriggers_vlbi_pipeline.query_database import get_full_filepath, get_cal_vis_files,fetch_known_sources
from outriggers_vlbi_pipeline.known_calibrators import get_known_source_pos
from outriggers_vlbi_pipeline.arc_commands import datatrail_pull_or_clear, datatrail_pull_cmd,datatrail_clear_cmd,baseband_exists,delete_baseband,vchmod,delete_multibeam,data_exists_at_minoc,datatrail_pull,datatrail_clear
import logging
from outriggers_vlbi_pipeline.vlbi_pipeline_config import (
    chime,
    kko,
    hco,
    gbo,
    kko_events_database,known_sources,
)
from outriggers_vlbi_pipeline.query_database import generate_logs
from dtcli.src import functions
from dtcli.utilities import cadcclient
from pyfx.bbdata_io import get_multibeam_pointing
from outriggers_vlbi_pipeline.calibrator_search.find_fringes import search_calibrator_visibilities,get_all_event_ids,update_main_db
from outriggers_vlbi_pipeline.vlbi_pipeline_config import current_calibrators
from outriggers_vlbi_pipeline.query_database import get_baseband_localization_info
from outriggers_vlbi_pipeline.query_database import get_frb_superset_events,record_outrigger_frb_disk_subset,get_outrigger_frb_disk_subset,get_outrigger_pulsar_disk_subset,record_outrigger_pulsar_disk_subset
database=kko_events_database


import os
import time

def file_modified_in_last_hour(directory):
    current_time = time.time()
    one_hour_ago = current_time - 3600  # 3600 seconds in an hour

    for filename in glob(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            mod_time = os.path.getmtime(filepath)
            if mod_time > one_hour_ago:
                print('file last modified < 1hr ago')
                return True  # Found at least one recently modified file
    return False  # No recent modifications found

def process_event(event_id,df_events_to_process,telescopes,nstart=0,min_cal=1,n_sub_containers=1,n_parallel=1,parallel_process=False,input_ra=None,input_dec=None,
EW_MAX=3,pulsar=True,input_pointings=None,target_name=None,include_target=False,overwrite=False,max_cal=1000,save_beamformed=False,KLT_filter=False):
    vlbi_head_dir=f'/arc/projects/chime_frb/vlbi/{config.VERSION}/'
    generate_logs(event_id,out_file='cal_search.log')
    tel_names=[tel.info.name for tel in telescopes]
    
    to_beamform_tels=[]
    mult_dirs=[]
    dfx=df_events_to_process[df_events_to_process['event_id']==event_id].reset_index(drop=True)                

    for telescope in tel_names:
        mult_dir=f'{vlbi_head_dir}/*/*/*/{event_id}/multibeams/*{telescope}*'
        logging.info(f"searching in {mult_dir}")
        mult_cap=100#1000
        if telescope=='chime':
            mult_cap=100#700
        if len(glob(mult_dir))>mult_cap and not overwrite:
            logging.info(f'{telescope} multibeam data for event {event_id} already found, will not beamform')
            mult_dirs.append(mult_dir)
            file=glob(mult_dirs[0])[0]

            pointings=get_multibeam_pointing(file, method="single")

            input_pointings={}
            input_pointings['ra_j2000']=pointings['ra'][::2]
            input_pointings['dec_j2000']=pointings['dec'][::2]
            input_pointings['name']=pointings['source_name'][::2]
            to_beamform_tels.append(telescope)

        else:
            to_beamform_tels.append(telescope)
    logging.info(f'beamforming for {to_beamform_tels}')
    if len(to_beamform_tels)>0:
        at_chime=True

        if at_chime:
            logging.info("tels to beamform:")
            logging.info(to_beamform_tels)
            for telescope in to_beamform_tels:
                from outriggers_vlbi_pipeline.query_database import get_calibrator_dataframe
                cal_df=get_calibrator_dataframe()


                zdir='temp/'
                raw_data_dir=f'/arc/projects/chime_frb/{zdir}data/{telescope}/baseband/raw/*/*/*/astro_{event_id}/*.h5'
                
                logging.info(f"searching in {raw_data_dir}")
                pulled=False
                if len(glob(raw_data_dir))==0:
                    pulled=True
                    logging.info("pulling data now")
                    datatrail_pull_or_clear(event_id, telescope=telescope, cmd_str='PULLED')

                logging.info("beamforming now")
                logging.info("KLT_filter:")
                logging.info(KLT_filter)
                logging.info("input_pointings:")
                logging.info(input_pointings)
                if False:#telescope=='chime' and event_id==20250516040716:
                    nstart=634
                else:
                    nstart=0
                n_pointings=beamform_calibrators(
                    event_id,telescope=telescope,ra_target=None,dec_target=None,
                    include_target=include_target,cal_df=cal_df,input_pointings=input_pointings, tel2='kko',
                    overwrite=overwrite,minimum_calibrators=min_cal,max_cal=max_cal,target_name=target_name,KLT_filter=KLT_filter,nstart=nstart,
                    raw_data_dir=raw_data_dir,events_database=database,EW_MAX=EW_MAX,nchunk=1)

                mult_dir=f'{vlbi_head_dir}/*/*/*/{event_id}/multibeams/*{telescope}*'
                mult_dirs.append(mult_dir)
                file=glob(mult_dirs[0])[0]
                pointings=get_multibeam_pointing(file, method="single")
                input_pointings={}
                input_pointings['ra_j2000']=pointings['ra'][::2]
                input_pointings['dec_j2000']=pointings['dec'][::2]
                input_pointings['name']=pointings['source_name'][::2]

                import os
                if pulled and event_id!=439373176:
                    for file in glob(raw_data_dir):
                        os.remove(file)

            logging.info("NOW CORRELATING")
            file=glob(mult_dirs[0])[0]
            pointings=get_multibeam_pointing(file, method="single")
            input_pointings={}
            input_pointings['ra_j2000']=pointings['ra'][::2]
            input_pointings['dec_j2000']=pointings['dec'][::2]
            input_pointings['name']=pointings['source_name'][::2]
            npointings=len(input_pointings['ra_j2000'])
            DMS=[0]*npointings
            source_types=['calibrator']*npointings
            print(f"N POINTINGS: {len(DMS)}")
            print(f"N POINTINGS: {len(source_types)}")
            
            success = correlate_multibeam_data(
                event_id = event_id, save_beamformed=True,
                delete_multibeam_data=False,
                source_types=source_types,
                telescopes=telescopes,
                DMS = DMS,
                target_included=include_target,interbase=True,
                nstart=nstart)

    else:
        logging.info("NOW CORRELATING")
        file=glob(mult_dirs[0])[0]
        pointings=get_multibeam_pointing(file, method="single")
        input_pointings={}
        input_pointings['ra_j2000']=pointings['ra'][::2]
        input_pointings['dec_j2000']=pointings['dec'][::2]
        input_pointings['name']=pointings['source_name'][::2]
        npointings=len(input_pointings['ra_j2000'])
        DMS=[0]*npointings
        source_types=['calibrator']*npointings

        print(f"N POINTINGS: {len(DMS)}")
        print(f"N POINTINGS: {len(source_types)}")
        success = correlate_multibeam_data(
            event_id = event_id, save_beamformed=True,
            delete_multibeam_data=False,
            source_types=source_types,
            telescopes=telescopes,
            DMS = DMS,
            target_included=include_target,
            interbase=True,
            nstart=nstart)

    for telescope in tel_names:
        if telescope!='chime':
            search_calibrator_visibilities(event_id,tel2=telescope)
    event_df=get_event_data(event_id)
    month=int(event_df['month'][0])
    day=int(event_df['day'][0])

if __name__=='__main__': 
    database=kko_events_database
    parser = argparse.ArgumentParser("Correlator Search Executable")
    parser.add_argument("--event_id", help="event_id", type=int,default=1) #20250516040716
    parser.add_argument("--nstart", help="nstart", type=int,default=0)
    parser.add_argument("--KLT_filter",help='KLT_filter',type=int,default=0)
    parser.add_argument("--overwrite",help='overwrite',type=int,default=0)

    
    event_id_to_match=20250418054922

    from outriggers_vlbi_pipeline.vlbi_pipeline_config import (
        chime,
        kko,
        new_hco,
        new_gbo,
        kko_events_database,known_sources,
    )
    new_gbo.info.name='gbo'
    new_hco.info.name='hco'
    telescopes=[chime,new_hco,new_gbo]
    cmdargs = parser.parse_args()
    nstart=cmdargs.nstart
    event_id=cmdargs.event_id
    year=int(str(event_id)[:4])
    month=int(str(event_id)[4:6])
    day=int(str(event_id)[6:8])
    df_events_to_process=get_outrigger_frb_disk_subset(year=year,month=month)
    df_events_to_process=df_events_to_process[df_events_to_process['event_id']==event_id].reset_index(drop=True)
    save_beamformed=True
    config_version='manual_triggers'
    KLT_filter=cmdargs.KLT_filter
    overwrite=cmdargs.overwrite
    if overwrite==1:
        overwrite=True
    else:
        overwrite=False

    if KLT_filter==0:
        KLT_filter=False
    else:
        KLT_filter=True

    target_name=None
    log_folder=f'/arc/home/shiona/logs/'
    log_file_event = os.path.join(log_folder, f'{event_id}_manual_{year}_{month}_correlate.log')
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
        handlers=[logging.FileHandler(log_file_event),  # Output log messages to a log file specific to this event
        logging.StreamHandler()],  # Output log messages to the console
        force=True
    )
    logging.info(f"writing to {log_file_event}")
    
    config.VERSION=config_version
    logging.info(config_version)
    vlbi_head_dir=f'/arc/projects/chime_frb/vlbi/{config_version}'
    done_dir=f'{vlbi_head_dir}/*/*/*/{event_id}/calibrator_visibilities/*'
    nfiles=len(glob(done_dir))
    mult_done_dir=f'{vlbi_head_dir}/*/*/*/{event_id}/multibeams/*'
    n_multfiles=len(glob(mult_done_dir))
    paused=False
    if (n_multfiles)>0:
        mult_ongoing=file_modified_in_last_hour(done_dir)
        if not mult_ongoing:
            paused=True
    print(f"{nfiles} files found in {done_dir}")
    print(f"{n_multfiles} files found in {mult_done_dir}")
    if ((overwrite) or ((nfiles==0) or (paused))) or nstart!=0:
        cal_df=get_calibrator_dataframe()
        detection_df_1=get_all_event_ids(cal_df,tel2='gbo')
        detection_df_2=get_all_event_ids(cal_df,tel2='hco')
        detection_df=pandas.concat([detection_df_1,detection_df_2])
        detection_df=detection_df[detection_df['event_id']==event_id_to_match].reset_index(drop=True)
        print(len(detection_df))
        print(detection_df)
        dfx=detection_df.drop_duplicates(subset='name').reset_index(drop=True)
        print(dfx)
        import time
        #time.sleep(60*120) #wait two hours
        if len(dfx)>0:
            process_event(
                input_pointings=dfx,event_id=event_id,pulsar=True,nstart=nstart,include_target=False,df_events_to_process=df_events_to_process,
                min_cal=1,telescopes=telescopes,KLT_filter=KLT_filter,overwrite=overwrite,save_beamformed=save_beamformed)
        else:
            raise AssertionError
