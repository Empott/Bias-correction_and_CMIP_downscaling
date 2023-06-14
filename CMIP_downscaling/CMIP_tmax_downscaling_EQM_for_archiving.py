'''This code is written to statistically downscale maximum daily temperature from global climate output (here CMIP5), using a 39 year gridded dataset (here bias-corrected WRF) in place of observations, using the method described in Potter et al., 2022 following Cannon et al., 2014.
The global climate output has been resampled to the WRF grid
This was written to be run distributed on the JASMIN supercomputer
If you wish to use this code, please feel free to get in contact with the author, Emily Potter, at emily.potter@sheffield.ac.uk'''

#create an output file
import sys

job=sys.argv[1]

outputfile="output_files_QM_tmax45_"+str(int(job)+1)+".txt"

with open(outputfile,"w") as f_out:
    f_out.write('opening DS')

with open(outputfile,"a") as f_out:
    f_out.write('testing')

with open(outputfile,"a") as f_out:
    f_out.write('starting script')

#load in python modules
import inspect
import cftime
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy
from scipy.stats.mstats import mquantiles
from scipy.interpolate import interp1d
import time

##################################
# Load in data
##################################

with open(outputfile,"a") as f_out:
    f_out.write ("Number of arguments:"+ str(len(sys.argv))+ "arguments")
with open(outputfile,"a") as f_out:
    f_out.write ("Argument List:"+ str(sys.argv))

def non_parametric_BC_QDM_rolling_parallel(obs_1d_at_point_month,model_hist_1d_at_point_month,model_fut_1d_at_point_month):
    # find quantiles
    nbins=1000
    binmid = np.arange(-(1./nbins)*0.5, 1.+1./nbins, 1./nbins)
    qo = mquantiles(obs_1d_at_point_month, prob=binmid)
    qc = mquantiles(model_hist_1d_at_point_month, prob=binmid)
    qf = mquantiles(model_fut_1d_at_point_month, prob=binmid)
    #interpolation functions
    f2o = interp1d(qf, qo, kind='linear', bounds_error=False)#,fill_value='extrapolate')
    f2c = interp1d(qf, qc, kind='linear', bounds_error=False)#,fill_value='extrapolate')
    #corrected dataset using Cannon 2014 QDM
    gcm_fut_corr = model_fut_1d_at_point_month + f2o(model_fut_1d_at_point_month)- f2c(model_fut_1d_at_point_month)
    return(gcm_fut_corr)

def correct_full_timeseries_parallel(obs_3d,model_hist_3d,model_fut_3d):
    final_corrected_future=model_fut_3d.loc[dict(time2=slice('2019-01-01','2100-12-31'))].copy(deep=True)
#   final_corrected_future=model_fut_3d.loc[dict(time2=slice(cftime.Datetime360Day(2019,01,01, 0, 0, 0, 0, 0, 0),cftime.Datetime360Day(2100,12,30, 12, 0, 0, 0, 0, 364)))].copy(deep=True)
    for year in range(2019,2101):
        print(year)
        start=time.time()
        with open(outputfile,"a") as f_out:
            f_out.write('\n'+ str(year) +'\n')
        start=time.time()
        xr_fut_roll=model_fut_3d.loc[dict(time2=slice(str(year-19)+'-01-01',str(year+19)+'-12-31'))].copy(deep=True)
        for month_of_year in range(1,13):
            obs_3d_at_month=obs_3d[obs_3d['Time.month']==month_of_year]
            gcm_hist_3d_at_month=model_hist_3d[model_hist_3d['time.month']==month_of_year]
            gcm_fut_3d_at_month=xr_fut_roll[xr_fut_roll['time2.month']==month_of_year]
            corrected_for_year=xr.apply_ufunc(non_parametric_BC_QDM_rolling_parallel,obs_3d_at_month,gcm_hist_3d_at_month,gcm_fut_3d_at_month,input_core_dims=[['Time'],['time'],['time2']],output_core_dims=[['time2']],exclude_dims=set(("time2",)),vectorize=True,)
            corrected_for_year['time2']=gcm_fut_3d_at_month.time2
            corrected_for_year=corrected_for_year.transpose('time2','lat','lon')
            final_corrected_future.loc[(final_corrected_future['time2'].dt.year==year) & (final_corrected_future['time2'].dt.month==month_of_year)]=corrected_for_year.loc[(corrected_for_year['time2'].dt.year==year)]
        end=time.time()
        with open(outputfile,"a") as f_out:
            f_out.write('\n'+ str(year)+' took '+str(end-start)+' seconds'+'\n')
    return(final_corrected_future)


#open the gridded dataset to be used as observations
xr_wrf_max=xr.open_dataset('filepath_to_netcdf.nc')

#make sure the gridded dataset is in the required format
xr_wrf_fixed_max=xr.Dataset(data_vars={'T2':(['Time','lat','lon'],xr_wrf_max.T2.data)},coords={'lon':(['lon'],xr_wrf_max.lon[0,:].data),'lat':(['lat'],xr_wrf_max.lat[:,0].data),'Time':xr_wrf_max.Time.data,'month':xr_wrf_max['Time.month'].data}).T2

output_folder='folderpath_to_store_output'

job=sys.argv[1]
with open(outputfile,"a") as f_out:
    f_out.write(str(job))

f=sys.argv[int(job)+1]
with open(outputfile,"a") as f_out:
    f_out.write(str(f))

print('QM downscaling for '+f)

#open the historical CMIP5 model, which has been regridded to the WRF resolution
xr_hist_wrfres=xr.open_dataset(f)
xr_hist_wrfres=xr_hist_wrfres.rename({'south_north':'lat','west_east':'lon'})
with open(outputfile,"a") as f_out:
    f_out.write('historical CMIP opened')

#open the historical CMIP5 model, which has been regridded to the WRF resolution
xr_hist_wrfres['tasmax']=xr_hist_wrfres.tasmax-273.15 #Kelvin to Celsius
xr_hist_wrfres['lat']=xr_hist_wrfres.lat.assign_coords({'lat':xr_hist_wrfres.lat.values})
xr_hist_wrfres['lon']=xr_hist_wrfres.lon.assign_coords({'lon':xr_hist_wrfres.lon.values})

#open future CMIP5 model, which has been regridded to the WRF resolution
xr_fut_wrfres=xr.open_dataset(f.replace('historical','future'))
xr_fut_wrfres=xr_fut_wrfres.rename({'south_north':'lat','west_east':'lon'})
xr_fut_wrfres['lat']=xr_fut_wrfres.lat.assign_coords({'lat':xr_fut_wrfres.lat.values})
xr_fut_wrfres['lon']=xr_fut_wrfres.lon.assign_coords({'lon':xr_fut_wrfres.lon.values})

with open(outputfile,"a") as f_out:
    f_out.write('future CMIP opened')

xr_fut_wrfres['tasmax']=xr_fut_wrfres.tasmax-273.15 #Kelvin to Celsius
xr_fut_wrfres=xr_fut_wrfres.rename({'time':'time2'})
xr_fut_wrfres_to100=xr_fut_wrfres.loc[dict(time2=slice('2000-01-01','2100-12-31'))]


with open(outputfile,"a") as f_out:
    f_out.write('correction starting now')

#########################################
#Perform the statistical downscaling
#########################################
final_corrected_future_to100=correct_full_timeseries_parallel(xr_wrf_fixed_max,xr_hist_wrfres.tasmax,xr_fut_wrfres_to100.tasmax)

with open(outputfile,"a") as f_out:
    f_out.write('saving')
#save file
f_file=f.replace('historical','future').replace('old_folder_path','new_folder_path')
final_corrected_future_to100.to_netcdf(output_folder+f_file.replace('.nc','output_filename.nc'))
with open(outputfile,"a") as f_out:
    f_out.write('Successfully saved')








