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



if __name__=='__main__': 
    import argparse
    parser = argparse.ArgumentParser("Correlator Search Executable")
    parser.add_argument("--tel", help=" )", type=str)
    cmdargs = parser.parse_args()
    tel=cmdargs.tel

    print(tel)
    fmin=400
    fmax=600

    if tel=='kko':
        tec_grid=np.arange(-5,5,.1)
    else:
        tec_grid=np.arange(-40,40,.1)

    baseline_name=f'chime-{tel}'
    valid_keys=['chime',baseline_name,'index_map',tel]
    import re
    import time
    inputtag=f'M22_true_pos_fit_{tel}'
    if tel=='hco':
        inputtag=f'M22_true_pos_fit_hco_MASK_RFI'
    calibrated_files=glob(f'/arc/projects/chime_frb/vlbi/OVP_astrometry_{tel}/{inputtag}/*/calibrated/*')
    print(len(calibrated_files))
    calibrated_files=glob(f'/arc/projects/chime_frb/vlbi/OVP_astrometry_{tel}/{inputtag}/*/calibrated/*')
    print(len(calibrated_files))
    for i,file in enumerate(calibrated_files):
        print(i)
        new_file = file.replace("calibrated", f"calibrated_bw_{fmin}_{fmax}_masked")
        print(new_file)
        if len(glob(new_file))==0:
            os.makedirs(os.path.dirname(new_file), exist_ok=True)
            vis=VLBIVis.from_file(file)
            mask=np.where((vis.freqs>fmin)&(vis.freqs<fmax))
            vis[f'chime-{tel}']['vis'][mask]=0.0
            event_id=vis.event_id
            fringefit(vis[f'chime-{tel}'],tec_grid=tec_grid)
            vis.save(new_file)



    fmin=600
    fmax=800

    calibrated_files=glob(f'/arc/projects/chime_frb/vlbi/OVP_astrometry_{tel}/{inputtag}/*/calibrated/*')
    print(len(calibrated_files))
    for i,file in enumerate(calibrated_files):
        print(i)
        new_file = file.replace("calibrated", f"calibrated_bw_{fmin}_{fmax}_masked")
        print(new_file)
        if len(glob(new_file))==0:
            os.makedirs(os.path.dirname(new_file), exist_ok=True)
            vis=VLBIVis.from_file(file)
            mask=np.where((vis.freqs>fmin)&(vis.freqs<fmax))
            vis[f'chime-{tel}']['vis'][mask]=0.0
            event_id=vis.event_id
            fringefit(vis[f'chime-{tel}'],tec_grid=tec_grid)
            vis.save(new_file)





    fmin=500
    fmax=700

    calibrated_files=glob(f'/arc/projects/chime_frb/vlbi/OVP_astrometry_{tel}/{inputtag}/*/calibrated/*')
    print(len(calibrated_files))
    for i,file in enumerate(calibrated_files):
        print(i)
        new_file = file.replace("calibrated", f"calibrated_bw_{fmin}_{fmax}_masked")
        print(new_file)
        if len(glob(new_file))==0:
            os.makedirs(os.path.dirname(new_file), exist_ok=True)
            vis=VLBIVis.from_file(file)
            mask=np.where((vis.freqs>fmin)&(vis.freqs<fmax))
            vis[f'chime-{tel}']['vis'][mask]=0.0
            event_id=vis.event_id
            fringefit(vis[f'chime-{tel}'],tec_grid=tec_grid)
            vis.save(new_file)

