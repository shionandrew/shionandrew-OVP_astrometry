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


#gps
og_HCOLATITUDE=40.817520  
og_HCOLONGITUDE=-121.46602   
og_HCOALTITUDE=1019 

og_hco = ac.EarthLocation.from_geodetic(lon=og_HCOLONGITUDE,lat=og_HCOLATITUDE,height=og_HCOALTITUDE)
og_hco.info.name = 'hco'

def load_all_vis_for_fit(files,tel2='hco',snr_threshold=15):
    all_vis_for_fit=[]
    for i,file in enumerate(files):
        if i%5==0:
            print(f"{i} of {len(files)}")
        vis_target=coda.core.VLBIVis.from_file(file)
        coh_tau,coh_snrs=np.array(coda.analysis.delay.extract_subframe_delay(vis_target[f'chime-{tel2}']))
        if np.abs(coh_tau[0,0]-coh_tau[1,1])<10e-3: #10ns consistency
            if coh_snrs[0,0]>snr_threshold:# or coh_snrs[1,1]>snr_threshold:
                vis_target.attrs.filename=file
                all_vis_for_fit.append(vis_target)

    return all_vis_for_fit



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
        new_hco.info.name = f'{dx}_{dy}_{dz}'
        new_hcos.append(new_hco)
    return new_hcos

def get_tau_applied(ra,dec, ctime,tel2s):
    src = ac.SkyCoord(ra=np.array([ra]) * un.degree, dec=np.array([dec]) * un.degree, frame="icrs")

    ctime = Time(ctime-1,
            format="unix",
        )
    times = np.array([ctime])

    station_coords=[chime]
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
        duration_min=1,
        base_mode="geocenter",
        dry_atm=True,
        wet_atm=True,
        d_interval=1,
    )
    ci.run_driver()
    delays=np.array([ci.interpolate_delays(time) for time in times]) #ntime,1,ntel,nras
    del ci #prevent initialization issues
    return (delays[:,0,1:,0]-delays[:,0,0,0])*seconds_to_microsecond #,ras,decs  # of shape (nfreq,nbaseline)



def offset_grid_single_model(event_id,ctime,cal_name,tar_name,cal_ra,cal_dec,ra,dec,tf_meas,tf_meas_no_iono,x_old,y_old,z_old,tag,N_res,tel2,snr_xx,incoh_snr_xx,outdir):
    file=f'{outdir}{event_id}_grid_{tag}_{tar_name}_calibrated_to_{cal_name}.npy'
    print(file)
    if len(glob(file))==0:
        tel_used=make_telescope(x_old,y_old,z_old)
        tel_used.info.name=f'old_{tel2}'
        tau_cal_app=get_tau_applied(ra=cal_ra,dec=cal_dec,ctime=ctime,tel2s=[tel_used])[0]
        print(f"tau applied cal: {tau_cal_app}")
        tau_tar_app=get_tau_applied(ra=ra,dec=dec,ctime=ctime,tel2s=[tel_used])[0]
        print(f"tau applied tar: {tau_tar_app}")
        tfs=[]

        dxs=np.linspace(-50,50,N_res)
        dys=np.linspace(-50,50,N_res)
        dzs=np.linspace(-50,50,N_res)

        dx_out=[]
        dy_out=[]
        dz_out=[]
        for i,dx in enumerate(dxs):
            for j,dy in enumerate(dys):
                for k,dz in enumerate(dzs):
                    dx_out.append(dx)
                    dy_out.append(dy)
                    dz_out.append(dz)
        
        dx_out=np.array(dx_out)
        dy_out=np.array(dy_out)
        dz_out=np.array(dz_out)

        chunk_size=1 ### NOTE THIS DOES NOT WORK AND NEEDS TO BE FIXED
        N_chunks=int(np.ceil(len(dx_out)/chunk_size))
        print(f" tau meas: {tf_meas}")
        taus=np.array([])
        liks_out=np.array([])
        if 'kko' in tel2:
            new_hcos=make_new_telescopes(dx_out,dy_out,dz_out,kko)
        if 'gbo' in tel2:
            new_hcos=make_new_telescopes(dx_out,dy_out,dz_out,gbo)
        elif 'hco' in tel2:
            new_hcos=make_new_telescopes(dx_out,dy_out,dz_out,og_hco)
        x_pos=[tel.x.value for tel in new_hcos]
        y_pos=[tel.y.value for tel in new_hcos]
        z_pos=[tel.z.value for tel in new_hcos]
        print(min(x_pos))
        print(max(x_pos))
        print(len(new_hcos))
        for chunk in range(N_chunks):
            print(f"{chunk} out of {N_chunks} chunks")
            new_hco_chunk=new_hcos[chunk*chunk_size:(chunk+1)*chunk_size]

            tau_cal_geo=get_tau_applied(ra=cal_ra,dec=cal_dec,ctime=ctime,tel2s=new_hco_chunk)[0,:]
            tau_tar_geo=get_tau_applied(ra=ra,dec=dec,ctime=ctime,tel2s=new_hco_chunk)[0,:]
            tf=(tau_tar_geo-tau_tar_app) - (tau_cal_geo-tau_cal_app)
            tf=-tf #need to flip because of cursed data conjugation
            taus=np.append(taus,tf)
            liks_out=np.append(liks_out,(-(tf-tf_meas)**2))

        tau_meas=np.array([tf_meas]*len(liks_out))
        tau_meas_no_iono=np.array([tf_meas_no_iono]*len(liks_out))
        snr_xx=np.array([snr_xx]*len(liks_out))
        incoh_snr_xx=np.array([incoh_snr_xx]*len(liks_out))
        x_pos=[tel.x.value for tel in new_hcos]
        y_pos=[tel.y.value for tel in new_hcos]
        z_pos=[tel.z.value for tel in new_hcos]
        print(min(x_pos))
        print(max(x_pos))
        new_out_data=[i for i in zip(x_pos,y_pos,z_pos,taus,tau_meas,tau_meas_no_iono,snr_xx,incoh_snr_xx)]
        grid_out=np.array(new_out_data,dtype=[('x', 'float64'), ('y', 'float64'), ('z', 'float64'),('tau','float64'),('tau_meas','float64'),('tau_meas_no_iono','float64'),('snr_xx','float64'),('incoh_snr_xx','float64')])
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
    parser.add_argument("--tag",help='tag',type=str,default='A22_manual_fit_all')
    parser.add_argument("--tel2_name",help='tag',type=str,default='hco')
    parser.add_argument("--csv",help='tag',type=str,default='/arc/home/shiona/scripts/hco_comissioning2_A22_manual_fit_all_hco.csv')
    cmdargs = parser.parse_args()
    N=cmdargs.N
    tag=cmdargs.tag
    tel1_name='chime'
    tel2_name=cmdargs.tel2_name    
    csv=cmdargs.csv    
    tel2=tel2_name
        
    tagdir=tag+'_'+'chime-'+tel2_name + '_single'
    outdir=f'/arc/projects/chime_frb/vlbi/hco_comissioning2/{tagdir}/grid/'
    os.makedirs(outdir, exist_ok=True)
    print(outdir)
    df=pandas.read_csv(csv) #/arc/home/shiona/scripts/hco_test2_baseline_offset_fit_data2.csv')
    #df=pandas.read_csv('/arc/home/shiona/scripts/kko_test_baseline_offset_fit_data.csv')
    #tel2='kko'
    chunk=500
    print(len(df))
    #df['abs_tau_xx']=np.abs(df['tau_xx'])
    #df=df.sort_values(by='abs_tau_xx',ascending=False).reset_index(drop=True)
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
                                x_old=df[ f'{tel2}_x'][i],
                                y_old=df[ f'{tel2}_y'][i],
                                z_old=df[ f'{tel2}_z'][i],
                                tag=f'{tag}_{tel2}_refined',tel2=tel2,
                                 snr_xx=df[ f'snr_xx'][i],
                                 incoh_snr_xx=df[ f'incoh_snr_xx'][i],outdir=outdir,
                                N_res=5)