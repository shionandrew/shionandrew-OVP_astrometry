#frb: 332280830,331641308
#B1905+39 - 331894109, #333653382,333072855,332763763
#B0450+55 - 318618360, 325000711#, 332883030
#B0919+06 - 332595666, 331162390 
#331125326, 334075975 - bright crab
#312913987,319995608,325634353 - R117

import sys
#!{sys.executable} -m pip install -e /arc/home/shiona/outriggers_vlbi_pipeline/
#!{sys.executable} -m pip install -e /arc/home/shiona/coda/
#!{sys.executable} -m pip install -e /arc/home/shiona/pyfx/
#!{sys.executable} -m pip install scikit-learn
#!{sys.executable} -m pip install gspread_formatting

from baseband_analysis.core.sampling import fill_waterfall

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
from outriggers_vlbi_pipeline.calibrator_search.find_fringes import existing_cal_detections
eid=20250516040716#20250509042648#20250418054922#20250417054519#
if eid==20250417054519:
    local=True
else:
    local=False
event_id=eid
year=str(eid)[:4]
month=str(eid)[4:6]
day=str(eid)[6:8]
singlebeam_outdir=f'/arc/projects/chime_frb/vlbi/manual_triggers/{year}/{month}/{day}/{eid}/calibrator_singlebeams/'
if local:
    file=glob(f'/arc/projects/chime_frb/ebarkho/calibrator_survey/hco_multibeams/{eid}/{eid}_hco_*')[0]
    df=pandas.read_csv(f'/arc/home/shiona/{eid}_output.csv')
else:
    config.VERSION='manual_triggers'
    file=glob(f'/arc/projects/chime_frb/vlbi/manual_triggers/*/*/*/{eid}/*multibeams*/*_hco_*')[0]
    df=existing_cal_detections(eid)
    
from pyfx.bbdata_io import get_multibeam_pointing
x=[]
start=0#np.where(np.array(df['name'].astype(str))=='J1135-0021')[0][0]
for i in range(start,len(df)):
    source_name=df['name'][i]

    vis_out_dir =f'/arc/projects/chime_frb/vlbi/manual_triggers/{year}/{month}/{day}/{event_id}/calibrator_visibilities/'#get_full_filepath(event_id=event_id,data_type="visibilities", source_type='calibrator')
    vis_out_file = f"{vis_out_dir}{event_id}_{source_name}_vis.h5"
    if True:#event_id==20250418054922 or len(glob(vis_out_file))==0:
        print(source_name)
        from pyfx.bbdata_io import get_multibeam_pointing,get_bbdatas_from_index,extract_singlebeam
        tels=['chime','hco','gbo']
        for tel in tels:
            if local:
                multibeam_dir=f'/arc/projects/chime_frb/ebarkho/calibrator_survey/{tel}_multibeams/{eid}/{eid}_{tel}_*'
            else:
                multibeam_dir=f'/arc/projects/chime_frb/vlbi/manual_triggers/*/*/*/{eid}/*multibeams*/*_{tel}_*'

            file=glob(multibeam_dir)[0]
            print(file)
            pointings=get_multibeam_pointing(file,method='single')
            print(pointings['source_name'])
            print(len(pointings))
            n=np.where(source_name==pointings['source_name'].astype(str)[::2])[0]
            if len(n)>1:
                print(n)
            n=np.min(n)
                
            n=int(n)
            print(n)
        
            tel_bbdata = extract_singlebeam(multibeam_dir, n=n)
            fill_waterfall(tel_bbdata,write=True)
            print(tel_bbdata['tiedbeam_locations'])
            print(tel_bbdata['tiedbeam_baseband'])
            name=tel_bbdata['tiedbeam_locations']['source_name'][0].astype(str)
            assert name==source_name
            out_file = (f"{singlebeam_outdir}{eid}_{source_name}_{tel}.h5")
            if True:
                print(f"saving singlebeam data to: {out_file}")
                os.makedirs(os.path.dirname(out_file), exist_ok=True, mode=0o777)
                tel_bbdata.save(out_file)
                del tel_bbdata
