import pandas
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


df_ref=pandas.read_csv('/arc/home/shiona/iri_April13.csv')
from scipy.stats import t

def get_prior(gps_tec,scale=0.9):
    def prior_pdf(tec_vals):
        return t.pdf(tec_vals, df=1, loc=gps_tec, scale=scale)
    return prior_pdf

#nu: 2.214263583065715
#-0.20509773140475696
#s: 0.9019212776533722

#nu: 2.100585293477786
#0.31347833479016707
#s: 0.7934553981783774

def get_dstec(event_id,name,calibrator_name):
    dfx=df_ref[df_ref['event_id']==event_id]
    dftar=dfx[dfx['name']==calibrator_name].reset_index(drop=True)
    dfcal=dfx[dfx['name']==name].reset_index(drop=True)
    dstec=(dftar[f'dstec_iri_{tel}'][0]-dfcal[f'dstec_iri_{tel}'][0])
    return dstec

if __name__=='__main__': 
    import argparse
    parser = argparse.ArgumentParser("Correlator Search Executable")
    parser.add_argument("--tel", help=" )", type=str)
    cmdargs = parser.parse_args()
    tel=cmdargs.tel

    print(tel)

    if tel=='kko':
        tec_grid=np.arange(-2,2,.1)
    else:
        tec_grid=np.arange(-30,30,.1)

    baseline_name=f'chime-{tel}'
    valid_keys=['chime',baseline_name,'index_map',tel]
    import re
    import time
    inputtag=f'M22_true_pos_fit_{tel}'
    #if tel=='hco':
    #    inputtag=f'M22_true_pos_fit_hco_MASK_RFI'

    scales=[0.1]#,0.5]
    for scale in scales:
        tag='_scale_' + str(scale)
        fmins=[400]#,600,400,500]
        fmaxs=[400]#,800,600,800]
        for i in range(len(fmins)):
            fmin=fmins[i]
            fmax=fmaxs[i]
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
                    if False:#tel!='kko':
                        gps_tec=get_dstec(event_id,vis.source_name[0].astype(str),vis[f'chime-{tel}']['calibrator_source_name'][0].astype(str))
                        prior=get_prior(gps_tec=gps_tec,scale=scale)
                        tec_grid=np.arange(-5,5,.1)+gps_tec
                        fringefit(vis[f'chime-{tel}'],tec_grid=tec_grid,prior=prior,tag=tag)
                    fringefit(vis[f'chime-{tel}'],tec_grid=tec_grid,tag=tag)
                    vis.save(new_file)

