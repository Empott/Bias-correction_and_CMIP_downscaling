
'''This code is used to correct the number of wet days and bias-correct the precipitation output from the WRF model using station data. It assumes you already have a bias-correction factor, please contact Emily Potter at emily.potter@sheffield.ac.uk for help creating the bias-correction factor, or with any questions'''


import xarray as xr
import numpy as np
import matplotlib
import netCDF4 as nc
import glob
from datetime import datetime
import shapefile
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from shapely.geometry import Polygon, Point
import numpy.ma as ma
import scipy
import scipy.stats as stats
from matplotlib import pyplot as plt
import pandas as pd
import pytz


#read in original dataset and convert to local time

timeoffset=5 #THIS IS UTC! set this to 5 if you want local time. set this to 12 if you want peru offset + start at 7 am (for SENAMHI precip data?)

def mypreprocess(ds):
#idea for now: each dataset should either be as long as a normal or leap year (9553 or 9577). Just find the index for each and write an if loop.
    if ds.dims['Time']==9553 and str(ds.Times.sel(Time=744).values)[5:16]=='01-01_00:00' and str(ds.Times.sel(Time=9504).values)[5:16]=='01-01_00:00':
        print('run'+str(datetime.strptime(ds.START_DATE,'%Y-%m-%d_%H:%M:%S').year+1)+' is not a leap year')
        dschopped=ds.isel(Time=slice(744+timeoffset,9504+timeoffset)) 
    elif ds.dims['Time']==9577 and str(ds.Times.sel(Time=744).values)[5:16]=='01-01_00:00' and str(ds.Times.sel(Time=9528).values)[5:16]=='01-01_00:00':
        print('run'+str(datetime.strptime(ds.START_DATE,'%Y-%m-%d_%H:%M:%S').year+1)+' is a leap year')
        dschopped=ds.isel(Time=slice(744+timeoffset,9528+timeoffset)) #adjust for time change from utc to local, and for 7 am bits for SENAMHI
    else:
        print('there is a problem with run'+str(datetime.strptime(ds.START_DATE,'%Y-%m-%d_%H:%M:%S').year+1))
    return(dschopped)

#
def mypreprocess_precip(ds):
#idea for now: each dataset should either be as long as a normal or leap year (9553 or 9577). Just find the index for each and write an if loop.
    if ds.dims['Time']==9553 and str(ds.Times.sel(Time=744).values)[5:16]=='01-01_00:00' and str(ds.Times.sel(Time=9504).values)[5:16]=='01-01_00:00':
        print('run'+str(datetime.strptime(ds.START_DATE,'%Y-%m-%d_%H:%M:%S').year+1)+' is not a leap year')
        dschopped=ds.isel(Time=slice(744+timeoffset,9505+timeoffset)) #adjust for time change from utc to local, and for 7 am bits for SENAMHI
    elif ds.dims['Time']==9577 and str(ds.Times.sel(Time=744).values)[5:16]=='01-01_00:00' and str(ds.Times.sel(Time=9528).values)[5:16]=='01-01_00:00':
        print('run'+str(datetime.strptime(ds.START_DATE,'%Y-%m-%d_%H:%M:%S').year+1)+' is a leap year')
        dschopped=ds.isel(Time=slice(744+timeoffset,9529+timeoffset)) #adjust for time change from utc to local, and for 7 am bits for SENAMHI
    else:
        print('there is a problem with run'+str(datetime.strptime(ds.START_DATE,'%Y-%m-%d_%H:%M:%S').year+1))
    return(dschopped)

listofruns=range(1980,2019)

folder_path_overall=''
output_folder=''

mylist=[sorted(glob.iglob(folder_path_overall+'run'+str(year)+'/wrfoutfiles/wrfout_compact_d02.nc')) for year in listofruns]
flatlist=[item for sublist in mylist for item in sublist]

#open dataset for standard variables (e.g. temperature) and cumulative variables (e.g. precipitation)
DS=xr.open_mfdataset(flatlist,preprocess=mypreprocess,decode_times=False) #5 is conversion from UTC to peru time
DS_precip=xr.open_mfdataset(flatlist,preprocess=mypreprocess_precip,decode_times=False) #

DS = DS.rename({'XTIME':'Time'})

DS_precip = DS_precip.rename({'XTIME':'Time'})
# Re-format timestamp (only hour and minutes, no seconds)
DS_precip = DS_precip.assign_coords(Time=pd.to_datetime(DS_precip['Time'].values).round('60min')) 

def add_variable_along_timelatlon(ds, var, name, units, long_name):
    """ This function adds missing variables to the DATA class """
    ds[name] = (('Time','south_north','west_east'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    return ds


def create_precip_local(ds,ds_fortime):
    modelrain_var=ds.RAINNC.values
    #rain before 2pm - rain before 1pm gives rain from 1pm to 2pm
    modelrain_hourly=modelrain_var[1::]-modelrain_var[0:-1]
    dso_UTC=xr.Dataset()
    dso_UTC.coords['Time'] = (('Time'), ds.Time[0:-1])
    dso_UTC.coords['lat'] = (('south_north', 'west_east'), ds.XLAT[0].values)
    dso_UTC.coords['lon'] = (('south_north', 'west_east'), ds.XLONG[0].values)
    dso_UTC = add_variable_along_timelatlon(dso_UTC, modelrain_hourly,'RAINNC', 'mm', 'total hourly rain for the hour 1pm is the rain for 1-2pm')
    dso_UTC=dso_UTC.sel(Time=~dso_UTC.indexes['Time'].duplicated(keep='last'))
    dso_rainhrly=xr.Dataset()
    #call the rain from 1pm to 2pm '1pm rain' so that the rain from 0am to 1am get's counted in that day, not the day before.
    #Create time for new dataset in local time (find the original times, tell it they're in utc, convert to Peru times, then turn back into turn naive timestamp (can't work out how to put a timezone one in xarray)
    local_time=ds_fortime.Time.to_index().tz_localize(pytz.utc).tz_convert('America/Lima').tz_localize(None)
    dso_rainhrly.coords['Time'] = (('Time'), local_time)
    dso_rainhrly.coords['lat'] = (('south_north', 'west_east'), ds.XLAT[0].values)
    dso_rainhrly.coords['lon'] = (('south_north', 'west_east'), ds.XLONG[0].values)
    dso_rainhrly = add_variable_along_timelatlon(dso_rainhrly,dso_UTC.RAINNC.values ,'RAINNC', 'mm', 'total hourly rain for the hour 1pm is the rain for 1-2pm')
    dso_daily_sum=dso_rainhrly.resample(Time='1D').sum()
    return(dso_daily_sum,dso_rainhrly)


#######################################
# correct the number of wet days
######################################


#read in a csv file with the threshold to be defined a wet day
WD_thresh=pd.read_csv('/gws/nopw/j04/pegasus/users/epotter/PeruGROWS_working/code/bias_correction/BC_finalstats/WDC/monthly/fixed_thresh/fullthresh_median_d02.csv')

#correct the number of wet days to match the observations.
def correctwetdays(dso_daily,dso_rainhrly,WD_thresh_csv,savename):
    dso_daily.coords['month']=(dso_daily.indexes['Time'].month)
    WD_thresh_csv['mean']=WD_thresh_csv[WD_thresh_csv.columns[1::]].mean(axis=1)
    final_thresh=pd.DataFrame(index=WD_thresh_csv['month'],data=WD_thresh_csv['mean'].values)
    dso_WDC_temp=dso_daily.RAINNC[dso_daily['month']==1].where(dso_daily.RAINNC[dso_daily['month']==1]>final_thresh.loc[1].values[0],0)
    for month in final_thresh.index[1::]:
        temp_for_month=dso_daily.RAINNC[dso_daily['month']==month].where(dso_daily.RAINNC[dso_daily['month']==month]>final_thresh.loc[month].values[0],0)
        dso_WDC_temp=xr.concat((dso_WDC_temp,temp_for_month),dim='Time')
    dso_WDC=dso_WDC_temp.sortby('Time')
    dso_WDC_dataset=dso_WDC.to_dataset()
    #save
    dso_WDC_dailypad=dso_WDC_dataset.reindex(Time=dso_rainhrly.Time,method='pad')
    dso_rainhrly_WDC=dso_rainhrly.RAINNC.where(dso_WDC_dailypad.RAINNC!=0,0)
    dso_rainhrly_WDC_dataset=dso_rainhrly_WDC.to_dataset()
    dso_WDC_dataset.to_netcdf(output_folder+savename+'daily_d02.nc')
    dso_rainhrly_WDC_dataset.to_netcdf(output_folder+savename+'hourly_d02.nc')
    return(dso_WDC_dataset,dso_rainhrly_WDC_dataset)

#save precipitaiton dataset with wet days corrected
dso_daily_sum,dso_rainhrly=create_precip_local(DS_precip,DS)
WDC_ds,WDC_hourly_ds=correctwetdays(dso_daily_sum,dso_rainhrly,WD_thresh,'precip_WDC_fixed_thresh_monthly_')

###########################################################
# correct precipitation using a fixed precipitation factor
###########################################################

model_folder='/gws/nopw/j04/pegasus/users/epotter/PeruGROWS_working/bias_corrected_files/monthly/fixed_a/'

a_median_WDC=pd.read_csv('/gws/nopw/j04/pegasus/users/epotter/PeruGROWS_working/code/bias_correction/BC_finalstats/final_corrections/monthly/fixed_a/a_median_d02.csv')
a_median_WDC.set_index('month', inplace=True)

WDC_ds=xr.open_dataset(model_folder+'precip_WDC_fixed_thresh_monthly_daily_d02.nc')
WDC_hourly_ds=xr.open_dataset(model_folder+'precip_WDC_fixed_thresh_monthly_hourly_d02.nc')

def correctfinal_fixed_a(dso_WDC,dso_WDC_hourly,a_median_WDC):
    dso_daily_sum_finalcorrected_temporary=dso_WDC.RAINNC[dso_WDC['month']==1]*a_median_WDC['mean'][1]
    dso_hourly_finalcorrected_temporary=dso_WDC_hourly.RAINNC[dso_WDC_hourly['month']==1]*a_median_WDC['mean'][1]
    for month in a_median_WDC.index[1::]:
        temp_for_month=dso_WDC.RAINNC[dso_WDC['month']==month]*a_median_WDC['mean'][month]
        dso_daily_sum_finalcorrected_temporary=xr.concat((dso_daily_sum_finalcorrected_temporary,temp_for_month),dim='Time')
        temp_for_month_hourly=dso_WDC_hourly.RAINNC[dso_WDC_hourly['month']==month]*a_median_WDC['mean'][month]
        dso_hourly_finalcorrected_temporary=xr.concat((dso_hourly_finalcorrected_temporary,temp_for_month_hourly),dim='Time')
    dso_daily_sum_finalcorrected=dso_daily_sum_finalcorrected_temporary.sortby('Time')
    dso_daily_sum_finalcorrected_dataset=dso_daily_sum_finalcorrected.to_dataset()
    dso_daily_sum_finalcorrected_dataset.to_netcdf(model_folder+'precip_finalcorrection_fixed_a_monthly_daily_d02.nc')
    dso_hourly_finalcorrected=dso_hourly_finalcorrected_temporary.sortby('Time')
    dso_hourly_finalcorrected_dataset=dso_hourly_finalcorrected.to_dataset()
    dso_hourly_finalcorrected_dataset.to_netcdf(model_folder+'precip_finalcorrection_fixed_a_monthly_hourly_d02.nc')

    return()

correctfinal_fixed_a(WDC_ds,WDC_hourly_ds,a_median_WDC)

dso_rainhrly.to_netcdf(model_folder+'precip_rawWRF_hourly_d02.nc')
dso_daily_sum.to_netcdf(model_folder+'precip_rawWRF_daily_d02.nc')





