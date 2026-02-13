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

new_tag='M7_fit'
best_fit_params=[-2523643.44047669,-4123699.84440233,4147774.23913098]#[-2523644.20461515,-4123700.15073725,4147774.06853484]
#### NEW POSITION, as of Mar 7###### 
new_hco = ac.EarthLocation.from_geocentric(
    x = (best_fit_params[0]) * un.m,  
    y = (best_fit_params[1]) * un.m,  
    z = (best_fit_params[2]) * un.m  
)
new_hco.info.name = 'hco'



# recorrelate
def repoint_data(old_vis,telescopes,source_type,vis_out_dir,tag=''):
    pointing_spec = np.array(copy.deepcopy(old_vis['index_map']['pointing_center']))    
    event_id=old_vis.event_id
    new_vis=recorrelate_data(
        old_vis=old_vis,
        telescopes=telescopes,source_type=source_type,
        pointing_spec=pointing_spec)
    source_name=pointing_spec['source_name'][0].astype(str)
    vis_out_file = f"{vis_out_dir}{event_id}_{source_name}_{tag}_vis.h5"
    os.makedirs(os.path.dirname(vis_out_file), exist_ok=True, mode=0o777)
    logging.info(f"Saving visibilities to {vis_out_file}")
    new_vis.save(vis_out_file)
    return new_vis

year=2025
month=2

cal_df=get_calibrator_dataframe()
detection_df_hco=get_all_event_ids(cal_df,tel2='hco')

df=get_outrigger_pulsar_disk_subset(year=year,month=month)
df=df[df['hco_target_fringes']=='True'].reset_index(drop=True)
#df=df[df['kko_target_fringes']=='True'].reset_index(drop=True)
config.VERSION='hco_comissioning2'
eids=(np.unique(df['event_id']))
print(f"{len(eids)} total events to process")

eids=[436619713,
 436772516,
 436802603,
 436784711,
 436766267,
 436804362,
 437508931,
 437537279,
 437534283,
 436397573,
 436353810,
 436370623,
 436402232,
 436684665,
 436679546,
 436721462,
 437149440,
 437199188,
 437154236,
 437196157,
 436926607,
 436929449,
 437113902,
 437034900,
 438120977,
 437407705,
 437414077,
 437454404,
 437460619,
 437347226,
 437239989,
 437244984,
 437268215,
 436877732,
 436849595,
 436843917,
 436858787,
 436879458,
 436463837,
 436471590,
 436525203]

for event_id in eids[:]:
    print(event_id)
    dfx=detection_df_hco[detection_df_hco['event_id']==event_id].reset_index(drop=True)
    cals=np.unique(dfx['name'])
    files_to_recorrelate=[]#find_files(event_id,data_type='visibilities',source_type='target')
    for cal in cals:
        if 'B' not in cal:
            files_to_recorrelate.append(find_files(event_id,data_type='visibilities',source_type='calibrator',filename_suffix=cal)[0])
    print(f"{len(files_to_recorrelate)} files for {event_id}, searching '/arc/projects/chime_frb/vlbi/hco_comissioning2/{new_tag}/{event_id}/*.h5")
    if len(glob(f'/arc/projects/chime_frb/vlbi/hco_comissioning2/{new_tag}/{event_id}/*.h5'))<2:
        for i,f in enumerate(files_to_recorrelate):
            print(i)
            print(f)
            vis=VLBIVis.from_file(f)
            eid=vis.event_id
            vis_out_dir=f'/arc/projects/chime_frb/vlbi/hco_comissioning2/{new_tag}/{eid}/'
            source_type='calibrator'
            if 'B' in f:
                source_type='target'
            repoint_data(vis,telescopes=[chime,new_hco],tag=new_tag,vis_out_dir=vis_out_dir,source_type=source_type)


