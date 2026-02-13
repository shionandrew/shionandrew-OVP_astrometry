import logging
from glob import glob
import os
import pandas
seconds_to_microsecond = 1e6
from pycalc11 import Calc
import astropy.units as un
import astropy.units as u
import astropy.coordinates as ac
from astropy.time import Time
import numpy as np
from outriggers_vlbi_pipeline.vlbi_pipeline_config import chime,kko,gbo
import astropy.coordinates as ac
import outriggers_vlbi_pipeline
import copy
import coda


####################################
#### NEW POSITION, as of May 3 ###### 
####################################
new_best_fit_params=[ 883728.02446502, -4924463.3225994 ,  3943957.56097847]
new_gbo = ac.EarthLocation.from_geocentric(
    x = (new_best_fit_params[0]) * un.m,  
    y = (new_best_fit_params[1]) * un.m,  
    z = (new_best_fit_params[2]) * un.m  
)
new_gbo.info.name = 'gbo'
####################################
####################################
####################################


####################################
#### NEW POSITION, as of May 3 ###### 
####################################
new_best_fit_params=[-2523644.20739404, -4123700.3658239 ,  4147773.46403909]
new_hco = ac.EarthLocation.from_geocentric(
    x = (new_best_fit_params[0]) * un.m,  
    y = (new_best_fit_params[1]) * un.m,  
    z = (new_best_fit_params[2]) * un.m  
)
new_hco.info.name = 'hco'


def make_telescope(x,y,z):
    tel = ac.EarthLocation.from_geocentric(
        x = (x) * un.m,
        y = (y) * un.m, 
        z = (z) * un.m
    )
    return tel

def make_new_telescopes(dxs,dys,dzs,old_tel):
    new_hcos=[]
    for i in range(len(dxs)):
        dx=dxs[i]
        dy=dys[i]
        dz=dzs[i]
        new_hco = ac.EarthLocation.from_geocentric(
            x = (old_tel.x.value+dx) * un.m,
            y = (old_tel.y.value+dy) * un.m, 
            z = (old_tel.z.value+dz) * un.m
        )
        new_hco.info.name = f'{dx}_{dy}_{dz}_{old_tel.info.name}'
        new_hcos.append(new_hco)
    return new_hcos

def get_tau_applied(ra,dec, ctime,tel2s,tel1):
    src = ac.SkyCoord(ra=np.array([ra]) * un.degree, dec=np.array([dec]) * un.degree, frame="icrs")

    ctime = Time(ctime-1,
            format="unix",
        )
    times = np.array([ctime])

    station_coords=[tel1]
    for tel in tel2s:
        station_coords.append(tel)
    station_names=[i.info.name for i in station_coords]
    #print(len(station_coords))
    #print([tel.x.value for tel in station_coords])
    ci = Calc(
        station_names=station_names,
        station_coords=station_coords,
        source_coords=src,
        start_time=ctime,
        duration_min=3,
        base_mode="geocenter",
        dry_atm=True,
        wet_atm=True,
        d_interval=1,
    )
    ci.run_driver()
    delays=np.array([ci.interpolate_delays(time) for time in times]) #ntime,1,ntel,nras
    del ci #prevent initialization issues
    return (delays[:,0,1:,0]-delays[:,0,0,0])*seconds_to_microsecond #,ras,decs  # of shape (nfreq,nbaseline)



def offset_grid_single_model(event_id,tel1_name,tel2_name,
                             ctime,cal_name,tar_name,cal_ra,cal_dec,ra,dec,tf_meas,tf_meas_no_iono,
                             x1_old,y1_old,z1_old,
                             x2_old,y2_old,z2_old,
                             tag,snr_xx,incoh_snr_xx,outdir): 
    file=f'{outdir}{event_id}_grid_{tag}_{tar_name}_calibrated_to_{cal_name}.npy'
    print(file)
    if len(glob(file))==0:
        tel1_used=make_telescope(x1_old,y1_old,z1_old)
        tel1_used.info.name=f'old_{tel1_name}'
        tel2_used=make_telescope(x2_old,y2_old,z2_old)
        tel2_used.info.name=f'old_{tel2_name}'
        
        
        tau_cal_app=get_tau_applied(ra=cal_ra,dec=cal_dec,ctime=ctime,tel1=tel1_used,tel2s=[tel2_used])[0] ### SIGN HERE IS WRONG, TEL USED FLIPPED??
        print(f"tau applied cal: {tau_cal_app}")
        tau_tar_app=get_tau_applied(ra=ra,dec=dec,ctime=ctime,tel1=tel1_used,tel2s=[tel2_used])[0]
        print(f"tau applied tar: {tau_tar_app}")
        tfs=[]

        N_res=3
        if tel1_name=='chime':
            N_res1=3
        else:
            N_res1=N_res
        
        endpoints=10
        dx1s = np.linspace(-endpoints, endpoints, N_res1,endpoint=True)  # GBO
        dy1s = np.linspace(-endpoints, endpoints, N_res1,endpoint=True) 
        dz1s = np.linspace(-endpoints, endpoints, N_res1,endpoint=True) 

        dx2s = np.linspace(-endpoints, endpoints, N_res,endpoint=True)  # HCO
        dy2s = np.linspace(-endpoints, endpoints, N_res,endpoint=True) 
        dz2s = np.linspace(-endpoints, endpoints, N_res,endpoint=True) 

        # Create 6D grid
        dx1_out, dx2_out, dy1_out, dy2_out, dz1_out, dz2_out = [], [], [], [], [], []
        for dx1 in dx1s:
            for dy1 in dy1s:
                for dz1 in dz1s:
                    for dx2 in dx2s:
                        for dy2 in dy2s:
                            for dz2 in dz2s:
                                dx1_out.append(dx1)
                                dy1_out.append(dy1)
                                dz1_out.append(dz1)
                                dx2_out.append(dx2)
                                dy2_out.append(dy2)
                                dz2_out.append(dz2)
        
        dx1_out=np.array(dx1_out)
        dy1_out=np.array(dy1_out)
        dz1_out=np.array(dz1_out)
        dx2_out=np.array(dx2_out)
        dy2_out=np.array(dy2_out)
        dz2_out=np.array(dz2_out)

        chunk_size=1 ### NOTE THIS DOES NOT WORK AND NEEDS TO BE FIXED
        print(f" tau meas: {tf_meas}")
        taus=np.array([])
        liks_out=np.array([])
        
        new_gbos=make_new_telescopes(dx1_out,dy1_out,dz1_out,new_gbo)
        new_hcos=make_new_telescopes(dx2_out,dy2_out,dz2_out,new_hco)
        
        if tel1_name=='chime':
            new_tel1s=[chime]*len(new_gbos)
        if tel1_name=='gbo':
            new_tel1s=new_gbos
        if tel2_name=='gbo':
            new_tel2s=new_gbos
        if tel2_name=='hco':
            new_tel2s=new_hcos
        
        #print(new_tel2s[0].x.value-new_hco.x.value)
        #print(new_tel1s[0].x.value-new_gbo.x.value)
        
        for i in range(len(new_tel2s)):
            new_tel2=[new_tel2s[i]]
            new_tel1=new_tel1s[i]
            new_tel1.info.name='A_'+new_tel1.info.name
            new_tel2[0].info.name='Z_'+new_tel2[0].info.name
            #print(new_tel1.info.name)
            tau_cal_geo=get_tau_applied(ra=cal_ra,dec=cal_dec,ctime=ctime,tel2s=new_tel2,tel1=new_tel1)[0,:]
            tau_tar_geo=get_tau_applied(ra=ra,dec=dec,ctime=ctime,tel2s=new_tel2,tel1=new_tel1)[0,:]
            #print(tau_cal_geo)
            #print(new_tel2[0].x.value-new_tel1.x.value)
            tf=(tau_tar_geo-tau_tar_app) - (tau_cal_geo-tau_cal_app)
            tf=-tf #need to flip because of cursed data conjugation
            taus=np.append(taus,tf)
            liks_out=np.append(liks_out,(-(tf-tf_meas)**2))

        tau_meas=np.array([tf_meas]*len(liks_out))
        tau_meas_no_iono=np.array([tf_meas_no_iono]*len(liks_out))
        snr_xx=np.array([snr_xx]*len(liks_out))
        incoh_snr_xx=np.array([incoh_snr_xx]*len(liks_out))
                       
        x_gbo=[tel.x.value for tel in new_gbos]
        y_gbo=[tel.y.value for tel in new_gbos]
        z_gbo=[tel.z.value for tel in new_gbos]
        x_hco=[tel.x.value for tel in new_hcos]
        y_hco=[tel.y.value for tel in new_hcos]
        z_hco=[tel.z.value for tel in new_hcos]
                       

        new_out_data=[i for i in zip(x_gbo,y_gbo,z_gbo,
                                     x_hco,y_hco,z_hco,
                                     taus,tau_meas,tau_meas_no_iono,snr_xx,incoh_snr_xx)]
                       
        grid_out=np.array(new_out_data,dtype=[('x_gbo', 'float64'), ('y_gbo', 'float64'), ('z_gbo', 'float64'),
                                              ('x_hco', 'float64'), ('y_hco', 'float64'), ('z_hco', 'float64'),
                                              ('tau','float64'),('tau_meas','float64'),('tau_meas_no_iono','float64'),('snr_xx','float64'),('incoh_snr_xx','float64')])
        np.save(file, grid_out) 
        return grid_out
                       
    else:
        print(file)
        print('already found')
        grid_out=np.load(file)
        grid_out['tau_meas']=tf_meas
        grid_out['tau_meas_no_iono']=tf_meas_no_iono
        grid_out['snr_xx']=snr_xx
        grid_out['incoh_snr_xx']=incoh_snr_xx
        np.save(file, grid_out) 
        return grid_out


if __name__=='__main__': 
    import argparse
    parser = argparse.ArgumentParser("Correlator Search Executable")
    parser.add_argument("--N",help='N',type=int,default=0)
    parser.add_argument("--tag",help='tag',type=str,default='synthetic')
    parser.add_argument("--csv_dir",help='tag',type=str,default='/arc/home/shiona/scripts/hco_comissioning2_synthetic_data_')
    cmdargs = parser.parse_args()
    N=cmdargs.N
    tag=cmdargs.tag
    csv_dir=cmdargs.csv_dir    
    
    baselines=['chime-hco','chime-gbo','gbo-hco']
    for baseline in baselines:
        import re
        tel1_name=re.split('-',baseline)[0]
        tel2_name=re.split('-',baseline)[-1]
        #tag=f'M12_OVP_astrometry'
        #file=f'/arc/home/shiona/scripts/manual_triggers_{tag}_{tel1_name}_{tel2_name}.csv'
    
        tagdir=tag+'_'+f'{tel1_name}-'+tel2_name + '_joint'
        outdir=f'/arc/projects/chime_frb/vlbi/hco_comissioning2/{tagdir}/grid/'
        os.makedirs(outdir, exist_ok=True)
    
        file=f'{csv_dir}{tel1_name}_{tel2_name}_test.csv' 
        df=pandas.read_csv(file) #/arc/home/shiona/scripts/hco_test2_baseline_offset_fit_data2.csv')
        chunk=500
        print(len(df))
        print(f"N: {N}")
        for i in range(len(df))[chunk*N:chunk*(N+1)]:
            print(f"{i} out of {len(df)}")
            offset_grid_single_model(event_id=df['event_id'][i],
                                    ctime=df['ctime'][i],
                                    cal_name=df['calibrator_name'][i],
                                    tar_name=df['name'][i],
                                    cal_ra=df['calibrator_ra'][i],
                                    cal_dec=df['calibrator_dec'][i],
                                    ra=df['ra'][i],
                                    dec=df['dec'][i],
                                    tf_meas=df['tau_xx'][i],
                                    tf_meas_no_iono=df['tau_no_iono_xx'][i],
                                    x1_old=df[ f'{tel1_name}_x'][i],
                                    y1_old=df[ f'{tel1_name}_y'][i],
                                    z1_old=df[ f'{tel1_name}_z'][i],
                                    x2_old=df[ f'{tel2_name}_x'][i],
                                    y2_old=df[ f'{tel2_name}_y'][i],
                                    z2_old=df[ f'{tel2_name}_z'][i],
                                    tag=tag+'_'+baseline,tel2_name=tel2_name,tel1_name=tel1_name,
                                     snr_xx=df[ f'snr_xx'][i],
                                     incoh_snr_xx=df[ f'incoh_snr_xx'][i],outdir=outdir)