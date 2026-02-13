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


new_tag='M4_fit_gbo'
#### NEW POSITION, as of Mar ###### 

GBOLATITUDE = 38.43587891
GBOLONGITUDE = -79.8262027
GBOALTITUDE = 814.23857806

gbo = ac.EarthLocation.from_geodetic(lon=GBOLONGITUDE,lat=GBOLATITUDE,height=GBOALTITUDE)
gbo.info.name = 'gbo'

new_gbo = ac.EarthLocation.from_geocentric(
    x = (gbo.x.value+70) * un.m, #-2,111,752.244
    y = (gbo.y.value-90) * un.m, #-3,581,453.556
    z = (gbo.z.value+60) * un.m #4821610.081
)
new_gbo.info.name = 'gbo'



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
detection_df_hco=get_all_event_ids(cal_df,tel2='gbo')

df=get_outrigger_pulsar_disk_subset(year=year,month=month)
df=df[df['gbo_target_fringes']=='True'].reset_index(drop=True)

config.VERSION='hco_comissioning2'
eids=(np.unique(df['event_id']))
print(f"{len(eids)} total events to process")
for event_id in eids[:]:
    print(event_id)
    dfx=detection_df_hco[detection_df_hco['event_id']==event_id].reset_index(drop=True)
    cals=np.unique(dfx['name'])
    files_to_recorrelate=find_files(event_id,data_type='visibilities',source_type='target')
    for cal in cals:
        if 'B' not in cal:
            try:
                files_to_recorrelate.append(find_files(event_id,data_type='visibilities',source_type='calibrator',filename_suffix=cal)[0])
            except:
                print(f"{cal} not found for {event_id}")
    print(f"{len(files_to_recorrelate)} files for {event_id}")
    if len(files_to_recorrelate)>1:
        for i,f in enumerate(files_to_recorrelate):
            print(i)
            print(f)
            vis=VLBIVis.from_file(f)
            event_id=vis.event_id
            vis_out_dir=f'/arc/projects/chime_frb/vlbi/hco_comissioning2/{new_tag}/{event_id}/'
            source_type='calibrator'
            if 'B' in f:
                source_type='target'
            #repoint_data(vis,telescopes=[chime,new_hco],tag=new_tag,vis_out_dir=vis_out_dir,source_type=source_type)
            repoint_data(vis,telescopes=[chime,new_gbo],tag=new_tag,vis_out_dir=vis_out_dir,source_type=source_type)


