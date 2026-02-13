""" A simple script that (1) pulls data from minoc (2) forms multibeamformed data towards the target and calibrators (3) correlates the data and saves visibilities.
Note that this does not yet perform the actual localization step which is done in run_localization.py"""
import outriggers_vlbi_pipeline.vlbi_pipeline_config as config
from outriggers_vlbi_pipeline.multibeamform import beamform_multipointings, beamform_calibrators,rebeamform_singlebeam
from outriggers_vlbi_pipeline.vlbi_pipeline_config import kko_backend, chime_backend,frb_events_database,kko_events_database
from outriggers_vlbi_pipeline.query_database import get_event_data,find_files
from outriggers_vlbi_pipeline.cross_correlate_data import correlate_multibeam_data
from outriggers_vlbi_pipeline.query_database import get_outrigger_disk_subset,check_correlation_completion, update_event_status,check_baseband_localization_completion
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
EW_MAX=3,pulsar=True,input_pointings=None,target_name=None,include_target=True,overwrite=False,max_cal=1000,save_beamformed=True,KLT_filter=False):
    vlbi_head_dir=f'/arc/projects/chime_frb/vlbi/{config.VERSION}/'
    generate_logs(event_id,out_file='cal_search.log')
    tel_names=[tel.info.name for tel in telescopes]
    
    to_beamform_tels=[]
    mult_dirs=[]
    dfx=df_events_to_process[df_events_to_process['event_id']==event_id].reset_index(drop=True)                
    if pulsar:
        DM=dfx['dm'][0]
        if target_name is None:
            target_name=dfx['name'][0]
    else:
        target_name="EXT"
        event_info=get_event_data(event_id)
        DM=event_info['DM'][0]

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
            if include_target:
                logging.info("INCLUDING THE TARGET")
                input_pointings={}
                input_pointings['ra_j2000']=pointings['ra'][2::2]
                input_pointings['dec_j2000']=pointings['dec'][2::2]
                input_pointings['name']=pointings['source_name'][2::2]
            else:
                input_pointings={}
                input_pointings['ra_j2000']=pointings['ra'][::2]
                input_pointings['dec_j2000']=pointings['dec'][::2]
                input_pointings['name']=pointings['source_name'][::2]

        else:
            to_beamform_tels.append(telescope)
    logging.info(f'beamforming for {to_beamform_tels}')
    if len(to_beamform_tels)>0:
        at_chime=True
        if pulsar: #only do this extra check for pulsars 
            at_chime,n_files=data_exists_at_minoc(event_id, site='chime',return_len=True)

        if at_chime:
            logging.info("tels to beamform:")
            logging.info(to_beamform_tels)
            for telescope in to_beamform_tels:
                from outriggers_vlbi_pipeline.query_database import get_calibrator_dataframe
                cal_df=get_calibrator_dataframe()
                if pulsar:
                    ctime=max([dfx[f'kko_ctime'][0],dfx[f'gbo_ctime'][0],dfx[f'hco_ctime'][0]])
                    if ctime<1641060417:
                        ctime=1737754387.5351834
                        print("WARNING WARNING: CTIME IS OFF")
                    ra_target,dec_target=get_known_source_pos(name=target_name,ctime=ctime)
                else:
                    if input_ra is None and include_target:
                        bbdata_loc=get_baseband_localization_info(event_id)
                        ra_target=bbdata_loc['ra']
                        dec_target=bbdata_loc['dec']
                        logging.info(bbdata_loc)
                    else:
                        ra_target = input_ra
                        dec_target = input_dec

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
                n_pointings=beamform_calibrators(
                    event_id,telescope=telescope,ra_target=ra_target,dec_target=dec_target,
                    include_target=include_target,cal_df=cal_df,input_pointings=input_pointings, tel2='kko',
                    overwrite=overwrite,minimum_calibrators=min_cal,max_cal=max_cal,target_name=target_name,KLT_filter=KLT_filter,
                    raw_data_dir=raw_data_dir,events_database=database,EW_MAX=EW_MAX,nchunk=1)

                mult_dir=f'{vlbi_head_dir}/*/*/*/{event_id}/multibeams/*{telescope}*'
                mult_dirs.append(mult_dir)
                file=glob(mult_dirs[0])[0]
                pointings=get_multibeam_pointing(file, method="single")
                input_pointings={}
                if include_target:
                    if config.VERSION=='ovp_upgrade':
                        bbdata_loc=get_baseband_localization_info(event_id)
                        assert np.abs(pointings['dec'][0]-bbdata_loc['dec'])<10/3600
                    input_pointings['ra_j2000']=pointings['ra'][2::2]
                    input_pointings['dec_j2000']=pointings['dec'][2::2]
                    input_pointings['name']=pointings['source_name'][2::2]
                else:
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
            if include_target:
                DMS.insert(0,DM)
                DMS=DMS[:-1]
                source_types.insert(0,'target')
                source_types=source_types[:-1]
            print(f"N POINTINGS: {len(DMS)}")
            print(f"N POINTINGS: {len(source_types)}")
            
            success = correlate_multibeam_data(
                event_id = event_id, save_beamformed=True,
                delete_multibeam_data=True,
                source_types=source_types,
                telescopes=telescopes,
                DMS = DMS,
                target_included=include_target,
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
        if include_target:
            DMS.insert(0,DM)
            DMS=DMS[:-1]
            source_types.insert(0,'target')
            source_types=source_types[:-1]
        print(f"N POINTINGS: {len(DMS)}")
        print(f"N POINTINGS: {len(source_types)}")
        success = correlate_multibeam_data(
            event_id = event_id, save_beamformed=True,
            delete_multibeam_data=True,
            source_types=source_types,
            telescopes=telescopes,
            DMS = DMS,
            target_included=include_target,
            nstart=nstart)
        if include_target:
            chime_singlebeam_file=find_files(event_id,data_type='singlebeams',source_type='target',telescope='chime')[0]
            from baseband_analysis.core import BBData
            chime_bbdata = BBData.from_file(chime_singlebeam_file)
            if config.VERSION=='ovp_upgrade':
                dfx=get_baseband_localization_info(event_id)
                assert np.abs(chime_bbdata['tiedbeam_locations']['dec'][0]-dfx['dec'])<10/3600

    if include_target and pulsar==True:
        from outriggers_vlbi_pipeline.burst_search.run_pulsar_gating import gate_pulsar
        gate_pulsar(event_id,all_telescopes=telescopes,sim_bb_loc=False)


if __name__=='__main__': 
    database=kko_events_database
    parser = argparse.ArgumentParser("Correlator Search Executable")
    parser.add_argument("--year", help=" )", type=int,default=2024)
    parser.add_argument("--month", help=" ", type=int,default=1)
    parser.add_argument("--event_id", help="event_id", type=int,default=1)
    parser.add_argument("--nstart", help="nstart", type=int,default=0)
    parser.add_argument("--frb",help='frb, 1 for true',type=int,default=0)
    parser.add_argument("--mode",help='quick, slow, or full_search',type=str,default="slow")
    parser.add_argument("--target_name",help='target_name',type=str,default="")
    parser.add_argument("--pulsar_to_target",help='pulsar_to_target',type=str,default="NONE")
    parser.add_argument("--input_ra",help='ra',type=float,default=None)
    parser.add_argument("--input_dec",help='dec',type=float,default=None)
    parser.add_argument("--tel",help='tel',type=str,default='hco')
    parser.add_argument("--config_version",help='config_version',type=str,default='')
    parser.add_argument("--KLT_filter",help='KLT_filter',type=int,default=0)
    parser.add_argument("--overwrite",help='overwrite',type=int,default=0)
    parser.add_argument("--save_beamformed",help='save_beamformed',type=int,default=1)
    parser.add_argument("--cal_search",type=int,default=0)
    cmdargs = parser.parse_args()
    nstart=cmdargs.nstart
    year=cmdargs.year
    month=cmdargs.month
    frb=cmdargs.frb
    event_id=cmdargs.event_id
    input_ra=cmdargs.input_ra
    input_dec=cmdargs.input_dec
    target_name=cmdargs.target_name
    tel2=cmdargs.tel
    config_version=cmdargs.config_version
    KLT_filter=cmdargs.KLT_filter
    overwrite=cmdargs.overwrite
    mode=cmdargs.mode
    save_beamformed=cmdargs.save_beamformed
    cal_search=cmdargs.cal_search
    import astropy.coordinates as ac

    best_fit_params=[-2523649.92736954,-4123697.10433308,4147773.43142168]#[-2523643.44047669,-4123699.84440233 ,4147774.23913098]
    #### NEW POSITION, as of Mar 5 ###### 
    new_hco = ac.EarthLocation.from_geocentric(
        x = (best_fit_params[0]) * un.m,  
        y = (best_fit_params[1]) * un.m,  
        z = (best_fit_params[2]) * un.m  
    )
    new_hco.info.name = 'hco'


    ####################################
    #### NEW POSITION, as of May 3 ###### 
    ####################################
    new_best_fit_params=[883729.31850621,-4924463.81125919,3943956.82880664]#[ 883728.02446502, -4924463.3225994 ,  3943957.56097847]
    new_gbo = ac.EarthLocation.from_geocentric(
        x = (new_best_fit_params[0]) * un.m,  
        y = (new_best_fit_params[1]) * un.m,  
        z = (new_best_fit_params[2]) * un.m  
    )
    new_gbo.info.name = 'gbo'

    
    if save_beamformed==1:
        save_beamformed=True
        print('will save singlebeams')
    else:
        save_beamformed=False
        print('will not save singlebeams')

    if overwrite==0:
        overwrite=False
    else:
        overwrite=True
    if KLT_filter==0:
        KLT_filter=False
    else:
        KLT_filter=True

    include_target=True

    if target_name=="":
        target_name=None
    log_folder=f'/arc/home/shiona/logs/'
    log_file_event = os.path.join(log_folder, f'{event_id}_{frb}_{year}_{month}_correlate.log')
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
        handlers=[logging.FileHandler(log_file_event),  # Output log messages to a log file specific to this event
        logging.StreamHandler()],  # Output log messages to the console
        force=True
    )
    logging.info(f"writing to {log_file_event}")
    
    if config_version=='':
        if frb:
            config_version='ovp_upgrade'
        else:
            config_version='ovp_upgrade_test_locs'
    config.VERSION=config_version
    logging.info(config_version)
    logging.info(f"event_id {event_id}")
    if frb:
        print("FRB")
        if year!=2025:
            frb_events_database="1MMt0S67cXTvcTy3cux-cA7p7LJKPc5ud3bmyMJ-cYBc"#"18p3DswWs6OD2OJuIXiL2stgVlkU9_3w3fyGleo6UGeI"
            df_events_to_process=get_outrigger_disk_subset(database=frb_events_database,year=year,month=month)
        else:
            df_events_to_process=get_outrigger_frb_disk_subset(year=year,month=month)
        pulsar=False
    else:
        pulsar=True
        df_events_to_process=get_outrigger_pulsar_disk_subset(year=year,month=month)
    if event_id==1:
        if frb and input_ra is None:
            df_events_to_process=df_events_to_process[df_events_to_process['bb_loc']=="True"].reset_index(drop=True)
        df_events_to_process=df_events_to_process[df_events_to_process['chime_datatrails_nfiles']>500].reset_index(drop=True)
        df_events_to_process=df_events_to_process[np.abs(df_events_to_process['hco_datatrails_nfiles']-df_events_to_process['files_at_hco'])<10].reset_index(drop=True)
        df_events_to_process=df_events_to_process[np.abs(df_events_to_process['gbo_datatrails_nfiles']-df_events_to_process['files_at_gbo'])<10].reset_index(drop=True)
        df_events_to_process=df_events_to_process[np.abs(df_events_to_process['kko_datatrails_nfiles']-df_events_to_process['files_at_kko'])<10].reset_index(drop=True)
    else:
        df_events_to_process=df_events_to_process[df_events_to_process['event_id']==event_id].reset_index(drop=True) 

   
    import time
    vlbi_head_dir=f'/arc/projects/chime_frb/vlbi/{config.VERSION}/'

    events_to_process=np.array(df_events_to_process['event_id'])
    logging.info(events_to_process)
    logging.info(len(events_to_process))
    for event_id in events_to_process:
        all_telescopes=[chime,new_gbo,new_hco] 

        if event_id==439373176:
            new_gbo.info.name='gbo'
            new_hco.info.name='hco'
            all_telescopes=[chime,new_gbo,new_hco] 
        telescopes=[]
        for tel in all_telescopes:
            dfx=df_events_to_process[df_events_to_process['event_id']==event_id].reset_index(drop=True) 
            print(dfx)
            if dfx[f'{tel.info.name}_datatrails_nfiles'][0]>100:
                logging.info(f"Will incldue telescope {tel} for event_id {event_id}")
                telescopes.append(tel)
        tel_names=[tel.info.name for tel in telescopes]
        logging.info(f"Will incldue {tel_names} for event_id {event_id}")
        if len(telescopes)>1:
            proceed=True
        else:
            proceed=False

        if frb:
            if input_ra is None and include_target:
                logging.info("using bb loc")
                bbdata_loc=get_baseband_localization_info(event_id)
                try:
                    r=bbdata_loc['ra']
                    d=bbdata_loc['dec']
                    logging.info(bbdata_loc)
                except:
                    logging.info("bb loc not found")
                    proceed=False
        if proceed:
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
                    
                    detection_df_gbo=get_all_event_ids(cal_df,tel2='gbo')
                    detection_df_hco=get_all_event_ids(cal_df,tel2='hco')
                    detection_df_kko=get_all_event_ids(cal_df,tel2='kko')
                    detection_df=pandas.concat([detection_df_gbo,detection_df_hco,detection_df_kko])
                    dfx=detection_df[detection_df['event_id']==event_id].reset_index(drop=True)
                    print(len(dfx))
                    dfx=dfx.drop_duplicates(subset='name').reset_index(drop=True)
                    print(dfx)

                    if len(dfx)>0 and mode=='fast':
                        process_event(
                            input_pointings=dfx,event_id=event_id,pulsar=pulsar,nstart=nstart,include_target=include_target,
                            min_cal=1,df_events_to_process=df_events_to_process,input_ra=input_ra,input_dec=input_dec,telescopes=telescopes,KLT_filter=KLT_filter,overwrite=overwrite,
                            target_name=target_name,save_beamformed=save_beamformed)
                    else:
                        if mode=='full_search':
                            process_event(
                                event_id=event_id,pulsar=pulsar,nstart=nstart,min_cal=1000,include_target=include_target,telescopes=telescopes,KLT_filter=KLT_filter,overwrite=overwrite,
                                df_events_to_process=df_events_to_process,
                                target_name=target_name,input_ra=input_ra,input_dec=input_dec,save_beamformed=save_beamformed)
                        else:
                            process_event(
                                event_id=event_id,pulsar=pulsar,nstart=nstart,min_cal=1,include_target=include_target,telescopes=telescopes,KLT_filter=KLT_filter,overwrite=overwrite,
                                df_events_to_process=df_events_to_process,
                                target_name=target_name,input_ra=input_ra,input_dec=input_dec,save_beamformed=save_beamformed)
        else:
            logging.info(f"FAILED {event_id}")
