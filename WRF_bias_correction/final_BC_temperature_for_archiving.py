'''This code is used to bias-correct the temperature output from the WRF model using station data. It assumes you already have a bias-correction factor, please contact Emily Potter at emily.potter@sheffield.ac.uk for help creating the bias-correction factor, or with any questions'''


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

domain=2
#Load in WRF data
timeoffset=5 #THIS IS UTC! set this to 5 if you want local time. 

def mypreprocess(ds):
#idea for now: each dataset should either be as long as a normal or leap year (9553 or 9577). Just find the index for each and write an if loop.
    if ds.dims['Time']==9553 and str(ds.Times.sel(Time=744).values)[5:16]=='01-01_00:00' and str(ds.Times.sel(Time=9504).values)[5:16]=='01-01_00:00':
        print('run'+str(datetime.strptime(ds.START_DATE,'%Y-%m-%d_%H:%M:%S').year+1)+' is not a leap year')
        dschopped=ds.isel(Time=slice(744+timeoffset,9504+timeoffset)) #adjust for time change from utc to local, and for 7 am bits for SENAMHI
    elif ds.dims['Time']==9577 and str(ds.Times.sel(Time=744).values)[5:16]=='01-01_00:00' and str(ds.Times.sel(Time=9528).values)[5:16]=='01-01_00:00':
        print('run'+str(datetime.strptime(ds.START_DATE,'%Y-%m-%d_%H:%M:%S').year+1)+' is a leap year')
        dschopped=ds.isel(Time=slice(744+timeoffset,9528+timeoffset)) #adjust for time change from utc to local, and for 7 am bits for SENAMHI
    else:
        print('there is a problem with run'+str(datetime.strptime(ds.START_DATE,'%Y-%m-%d_%H:%M:%S').year+1))
    return(dschopped)

listofruns=range(1980,2019)

folder_path_overall=''
output_folder=''

mylist=[sorted(glob.iglob(folder_path_overall+'run'+str(year)+'/wrfoutfiles/wrfout_compact_d0'+str(domain)+'.nc')) for year in listofruns]
flatlist=[item for sublist in mylist for item in sublist]

DS=xr.open_mfdataset(flatlist,preprocess=mypreprocess,decode_times=False) #5 is conversion from UTC to peru time

DS = DS.rename({'XTIME':'Time'})

# Re-format timestamp (only hour and minutes, no seconds)
DS = DS.assign_coords(Time=pd.to_datetime(DS['Time'].values).round('60min'))

def add_variable_along_timelatlon(ds, var, name, units, long_name):
    """ This function adds missing variables to the DATA class """
    ds[name] = (('Time','south_north','west_east'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    return ds


def create_temp_local(ds):
    model_var=ds.T2.values-273.15
    #rain before 2pm - rain before 1pm gives rain from 1pm to 2pm
    dso=xr.Dataset()
    dso.coords['Time'] = (('Time'), ds.Time.to_index().tz_localize(pytz.utc).tz_convert('America/Lima').tz_localize(None))
    dso.coords['lat'] = (('south_north', 'west_east'), ds.XLAT[0].values)
    dso.coords['lon'] = (('south_north', 'west_east'), ds.XLONG[0].values)
    dso = add_variable_along_timelatlon(dso, model_var,'T2', 'K', 'air temperature at 2 m')
    dso_daily_max=dso.resample(Time='1D').max()
    dso_daily_min=dso.resample(Time='1D').min()
    return(dso,dso_daily_max,dso_daily_min)


def add_variable_along_timelatlon2(ds, var, name, units, long_name):
    """ This function adds missing variables to the DATA class """
    ds[name] = (('Time','lat','lon'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    return ds

def create_local2(ds,var):
    model_var=ds[var].values
    #rain before 2pm - rain before 1pm gives rain from 1pm to 2pm
    dso=xr.Dataset()
    dso.coords['Time'] = (('Time'), ds.Time.to_index().tz_localize(pytz.utc).tz_convert('America/Lima').tz_localize(None))
    dso.coords['lat'] = (('lat'), ds.XLAT[0,:,50].values)
    dso.coords['lon'] = (('lon'), ds.XLONG[0,50,:].values)
    dso = add_variable_along_timelatlon2(dso, model_var,var, ds[var].units, ds[var].long_name)
    dso_daily_mean=dso.resample(Time='1D').mean()
    return(dso,dso_daily_mean)

####################################################################
# For correction with a spatially-fixed correction factor (a)
####################################################################

#read in the correction factors
a_median_Tmax=pd.read_csv('/gws/nopw/j04/pegasus/users/epotter/PeruGROWS_working/code/bias_correction/BC_finalstats/final_corrections/temp/fixed_a/a_median_Tmax_d0'+str(domain)+'.csv')

a_median_Tmin=pd.read_csv('/gws/nopw/j04/pegasus/users/epotter/PeruGROWS_working/code/bias_correction/BC_finalstats/final_corrections/temp/fixed_a/a_median_Tmin_d0'+str(domain)+'.csv')

#in this case, the correction factors were created for 100 runs, so find the mean (you want a dataframe with one permonth, in the end)
final_a_Tmax=pd.DataFrame(index=a_median_Tmax['month'],columns=['mean'],data=a_median_Tmax['mean'].values)
final_a_Tmin=pd.DataFrame(index=a_median_Tmin['month'],columns=['mean'],data=a_median_Tmin['mean'].values)

print('final a Tmax for domain '+str(domain))
print(final_a_Tmax)

print('final a Tmin for domain '+str(domain))
print(final_a_Tmin)


def correctfinal_fixed_a(dso_daily_max,dso_daily_min,dso_hourly,a_Tmin,a_Tmax,model_folder):
    dso_daily_max.coords['month']=(dso_daily_max.indexes['Time'].month)
    dso_daily_min.coords['month']=(dso_daily_min.indexes['Time'].month)
    dso_daily_max_finalcorrected_temporary=dso_daily_max.T2[dso_daily_max['month']==1]+a_Tmax['mean'][1]
    dso_daily_min_finalcorrected_temporary=dso_daily_min.T2[dso_daily_min['month']==1]+a_Tmin['mean'][1]
    for month in a_Tmax.index[1::]:
        temp_for_month_max=dso_daily_max.T2[dso_daily_max['month']==month]+a_Tmax['mean'][month]
        temp_for_month_min=dso_daily_min.T2[dso_daily_min['month']==month]+a_Tmin['mean'][month]
        dso_daily_max_finalcorrected_temporary=xr.concat((dso_daily_max_finalcorrected_temporary,temp_for_month_max),dim='Time')
        dso_daily_min_finalcorrected_temporary=xr.concat((dso_daily_min_finalcorrected_temporary,temp_for_month_min),dim='Time')
    dso_daily_max_finalcorrected=dso_daily_max_finalcorrected_temporary.sortby('Time')
    dso_daily_max_finalcorrected_dataset=dso_daily_max_finalcorrected.to_dataset()
    dso_daily_max_finalcorrected_dataset.to_netcdf(model_folder+'Tmax_finalcorrection_fixed_a_monthly_daily_d0'+str(domain)+'.nc')
    dso_daily_min_finalcorrected=dso_daily_min_finalcorrected_temporary.sortby('Time')
    dso_daily_min_finalcorrected_dataset=dso_daily_min_finalcorrected.to_dataset()
    dso_daily_min_finalcorrected_dataset.to_netcdf(model_folder+'Tmin_finalcorrection_fixed_a_monthly_daily_d0'+str(domain)+'.nc')
    return(dso_daily_max_finalcorrected_dataset,dso_daily_min_finalcorrected_dataset)

#read in or create the hourly WRF data
raw_temp_hourly,raw_Tmax_daily,raw_Tmin_daily=create_temp_local(DS)
#
raw_temp_hourly.to_netcdf(output_folder+'temp_hourly_raw_d0'+str(domain)+'.nc')
raw_Tmax_daily.to_netcdf(output_folder+'Tmax_daily_raw_d0'+str(domain)+'.nc')
raw_Tmin_daily.to_netcdf(output_folder+'Tmin_daily_raw_d0'+str(domain)+'.nc')
#
Tmax_fixed_BC,Tmin_fixed_BC=correctfinal_fixed_a(raw_Tmax_daily,raw_Tmin_daily,raw_temp_hourly,final_a_Tmin,final_a_Tmax,output_folder)

#############################################################################################
# For correction with a correction factor varying based on latitude, longitude and elevation
#############################################################################################

a_int_Tmin=pd.read_csv('/gws/nopw/j04/pegasus/users/epotter/PeruGROWS_working/code/bias_correction/BC_finalstats/final_corrections/temp/variable_a/a_intercept_Tmin_d0'+str(domain)+'.csv',index_col=0,usecols=[0,101])
a_height_coef_Tmin=pd.read_csv('/gws/nopw/j04/pegasus/users/epotter/PeruGROWS_working/code/bias_correction/BC_finalstats/final_corrections/temp/variable_a/a_height_coef_Tmin_d0'+str(domain)+'.csv',index_col=0,usecols=[0,101])
a_lat_coef_Tmin=pd.read_csv('/gws/nopw/j04/pegasus/users/epotter/PeruGROWS_working/code/bias_correction/BC_finalstats/final_corrections/temp/variable_a/a_lat_coef_Tmin_d0'+str(domain)+'.csv',index_col=0,usecols=[0,101])
a_lon_coef_Tmin=pd.read_csv('/gws/nopw/j04/pegasus/users/epotter/PeruGROWS_working/code/bias_correction/BC_finalstats/final_corrections/temp/variable_a/a_lon_coef_Tmin_d0'+str(domain)+'.csv',index_col=0,usecols=[0,101])

a_int_Tmax=pd.read_csv('/gws/nopw/j04/pegasus/users/epotter/PeruGROWS_working/code/bias_correction/BC_finalstats/final_corrections/temp/variable_a/a_intercept_Tmax_d0'+str(domain)+'.csv',index_col=0,usecols=[0,101])
a_height_coef_Tmax=pd.read_csv('/gws/nopw/j04/pegasus/users/epotter/PeruGROWS_working/code/bias_correction/BC_finalstats/final_corrections/temp/variable_a/a_height_coef_Tmax_d0'+str(domain)+'.csv',index_col=0,usecols=[0,101])
a_lat_coef_Tmax=pd.read_csv('/gws/nopw/j04/pegasus/users/epotter/PeruGROWS_working/code/bias_correction/BC_finalstats/final_corrections/temp/variable_a/a_lat_coef_Tmax_d0'+str(domain)+'.csv',index_col=0,usecols=[0,101])
a_lon_coef_Tmax=pd.read_csv('/gws/nopw/j04/pegasus/users/epotter/PeruGROWS_working/code/bias_correction/BC_finalstats/final_corrections/temp/variable_a/a_lon_coef_Tmax_d0'+str(domain)+'.csv',index_col=0,usecols=[0,101])


modheight=DS.HGT.isel(Time=0)
modlat=DS.XLAT.isel(Time=0)
modlon=DS.XLONG.isel(Time=0)

def plot_a(a,month,minormax):
    a.plot(levels=14)
    plt.title('T'+minormax+' $a$ value for d0'+str(domain)+' in month '+str(month)+'\n for $a$ varying with lat, lon and height')
    plt.savefig('T'+minormax+'varlatlonheightmonth_'+str(month)+'d0'+str(domain)+'.pdf',format='pdf',bbox_inches='tight')
#    plt.show()

def correctfinal_variable_a(dso_daily_max,dso_daily_min,dso_hourly,a_int_Tmin,a_height_coef_Tmin,a_lat_coef_Tmin,a_lon_coef_Tmin,a_int_Tmax,a_height_coef_Tmax,a_lat_coef_Tmax,a_lon_coef_Tmax,modheight,modlat,modlon,model_folder):
    dso_daily_max.coords['month']=(dso_daily_max.indexes['Time'].month)
    dso_daily_min.coords['month']=(dso_daily_min.indexes['Time'].month)
    a_Tmax_1=a_int_Tmax.iloc[1].values+a_height_coef_Tmax.iloc[1].values*modheight+a_lat_coef_Tmax.iloc[1].values*modlat+a_lon_coef_Tmax.iloc[1].values*modlon
    plot_a(a_Tmax_1,1,'max')
    a_Tmin_1=a_int_Tmin.iloc[1].values+a_height_coef_Tmin.iloc[1].values*modheight+a_lat_coef_Tmin.iloc[1].values*modlat+a_lon_coef_Tmin.iloc[1].values*modlon
    plot_a(a_Tmin_1,1,'min')
    dso_daily_max_finalcorrected_temporary=dso_daily_max.T2[dso_daily_max['month']==1]+a_Tmax_1
    dso_daily_min_finalcorrected_temporary=dso_daily_min.T2[dso_daily_min['month']==1]+a_Tmin_1
    for month in a_height_coef_Tmin.index[1::]:
        a_Tmax_month=a_int_Tmax.loc[month].values+a_height_coef_Tmax.loc[month].values*modheight+a_lat_coef_Tmax.loc[month].values*modlat+a_lon_coef_Tmax.loc[month].values*modlon
        plot_a(a_Tmax_month,month,'max')
        a_Tmin_month=a_int_Tmin.loc[month].values+a_height_coef_Tmin.loc[month].values*modheight+a_lat_coef_Tmin.loc[month].values*modlat+a_lon_coef_Tmin.loc[month].values*modlon
        plot_a(a_Tmin_month,month,'min')
        temp_for_month_max=dso_daily_max.T2[dso_daily_max['month']==month]+a_Tmax_month
        temp_for_month_min=dso_daily_min.T2[dso_daily_min['month']==month]+a_Tmin_month
        dso_daily_max_finalcorrected_temporary=xr.concat((dso_daily_max_finalcorrected_temporary,temp_for_month_max),dim='Time')
        dso_daily_min_finalcorrected_temporary=xr.concat((dso_daily_min_finalcorrected_temporary,temp_for_month_min),dim='Time')
    dso_daily_max_finalcorrected=dso_daily_max_finalcorrected_temporary.sortby('Time')
    dso_daily_max_finalcorrected_dataset=dso_daily_max_finalcorrected.to_dataset(name='T2')
    dso_daily_max_finalcorrected_dataset.to_netcdf(model_folder+'Tmax_finalcorrection_variable_a_monthly_daily_d0'+str(domain)+'.nc')
    dso_daily_min_finalcorrected=dso_daily_min_finalcorrected_temporary.sortby('Time')
    dso_daily_min_finalcorrected_dataset=dso_daily_min_finalcorrected.to_dataset(name='T2')
    dso_daily_min_finalcorrected_dataset.to_netcdf(model_folder+'Tmin_finalcorrection_variable_a_monthly_daily_d0'+str(domain)+'.nc')
    return(dso_daily_max_finalcorrected_dataset,dso_daily_min_finalcorrected_dataset)

Tmax_variable_BC,Tmin_variable_BC=correctfinal_variable_a(raw_Tmax_daily,raw_Tmin_daily,raw_temp_hourly,a_int_Tmin,a_height_coef_Tmin,a_lat_coef_Tmin,a_lon_coef_Tmin,a_int_Tmax,a_height_coef_Tmax,a_lat_coef_Tmax,a_lon_coef_Tmax,modheight,modlat,modlon,output_folder)


#############################################################################################
# For correction with a correction factor varying based on elevation only
#############################################################################################

a_int_Tmin2=pd.read_csv('/gws/nopw/j04/pegasus/users/epotter/PeruGROWS_working/code/bias_correction/BC_finalstats/final_corrections/temp/heightvar_a/a_intercept_Tmin_d0'+str(domain)+'.csv',index_col=0,usecols=[0,101])
a_height_coef_Tmin2=pd.read_csv('/gws/nopw/j04/pegasus/users/epotter/PeruGROWS_working/code/bias_correction/BC_finalstats/final_corrections/temp/heightvar_a/a_height_coef_Tmin_d0'+str(domain)+'.csv',index_col=0,usecols=[0,101])

a_int_Tmax2=pd.read_csv('/gws/nopw/j04/pegasus/users/epotter/PeruGROWS_working/code/bias_correction/BC_finalstats/final_corrections/temp/heightvar_a/a_intercept_Tmax_d0'+str(domain)+'.csv',index_col=0,usecols=[0,101])
a_height_coef_Tmax2=pd.read_csv('/gws/nopw/j04/pegasus/users/epotter/PeruGROWS_working/code/bias_correction/BC_finalstats/final_corrections/temp/heightvar_a/a_height_coef_Tmax_d0'+str(domain)+'.csv',index_col=0,usecols=[0,101])



def plot_height_a(a,month,minormax):
    a.plot(levels=14)
    plt.title('T'+minormax+' $a$ value for d0'+str(domain)+' in month '+str(month)+'\n for $a$ varying with height')
    plt.savefig('T'+minormax+'varheightmonth_'+str(month)+'d0'+str(domain)+'.pdf',format='pdf',bbox_inches='tight')
    plt.show()
    return()


def correctfinal_varheight_a(dso_daily_max,dso_daily_min,dso_hourly,a_int_Tmin,a_height_coef_Tmin,a_int_Tmax,a_height_coef_Tmax,modheight,model_folder):
    dso_daily_max.coords['month']=(dso_daily_max.indexes['Time'].month)
    dso_daily_min.coords['month']=(dso_daily_min.indexes['Time'].month)
    a_Tmax_1=a_int_Tmax.iloc[1].values+a_height_coef_Tmax.iloc[1].values*modheight
    plot_height_a(a_Tmax_1,1,'max')
    a_Tmin_1=a_int_Tmin.iloc[1].values+a_height_coef_Tmin.iloc[1].values*modheight
    plot_height_a(a_Tmin_1,1,'min')
    dso_daily_max_finalcorrected_temporary=dso_daily_max.T2[dso_daily_max['month']==1]+a_Tmax_1
    dso_daily_min_finalcorrected_temporary=dso_daily_min.T2[dso_daily_min['month']==1]+a_Tmin_1
    for month in a_height_coef_Tmin.index[1::]:
        a_Tmax_month=a_int_Tmax.loc[month].values+a_height_coef_Tmax.loc[month].values*modheight
        plot_height_a(a_Tmax_month,month,'max')
        a_Tmin_month=a_int_Tmin.loc[month].values+a_height_coef_Tmin.loc[month].values*modheight
        plot_height_a(a_Tmin_month,month,'min')
        temp_for_month_max=dso_daily_max.T2[dso_daily_max['month']==month]+a_Tmax_month
        temp_for_month_min=dso_daily_min.T2[dso_daily_min['month']==month]+a_Tmin_month
        dso_daily_max_finalcorrected_temporary=xr.concat((dso_daily_max_finalcorrected_temporary,temp_for_month_max),dim='Time')
        dso_daily_min_finalcorrected_temporary=xr.concat((dso_daily_min_finalcorrected_temporary,temp_for_month_min),dim='Time')
    dso_daily_max_finalcorrected=dso_daily_max_finalcorrected_temporary.sortby('Time')
    dso_daily_max_finalcorrected_dataset=dso_daily_max_finalcorrected.to_dataset(name='T2')
    dso_daily_max_finalcorrected_dataset.to_netcdf(model_folder+'Tmax_finalcorrection_varheight_a_monthly_daily_d0'+str(domain)+'.nc')
    dso_daily_min_finalcorrected=dso_daily_min_finalcorrected_temporary.sortby('Time')
    dso_daily_min_finalcorrected_dataset=dso_daily_min_finalcorrected.to_dataset(name='T2')
    dso_daily_min_finalcorrected_dataset.to_netcdf(model_folder+'Tmin_finalcorrection_varheight_a_monthly_daily_d0'+str(domain)+'.nc')
    return(dso_daily_max_finalcorrected_dataset,dso_daily_min_finalcorrected_dataset)

Tmax_variable_BC,Tmin_variable_BC=correctfinal_varheight_a(raw_Tmax_daily,raw_Tmin_daily,raw_temp_hourly,a_int_Tmin2,a_height_coef_Tmin2,a_int_Tmax2,a_height_coef_Tmax2,modheight,output_folder)


#############################################################
#correct hourly temperature using daily corrected temperature
#############################################################
print('load in hourly')
raw_temp_hourly=xr.open_dataset(output_folder+'temp_hourly_raw_d0'+str(domain)+'.nc')

print('load in daily')
raw_Tmax_daily=xr.open_dataset(output_folder+'Tmax_daily_raw_d0'+str(domain)+'.nc')
raw_Tmin_daily=xr.open_dataset(output_folder+'Tmin_daily_raw_d0'+str(domain)+'.nc')

print('load in BC daily')

BC_fixed_Tmax=xr.open_dataset(output_folder+'Tmax_finalcorrection_fixed_a_monthly_daily_d0'+str(domain)+'.nc')
BC_variable_Tmin=xr.open_dataset(output_folder+'Tmin_finalcorrection_variable_a_monthly_daily_d0'+str(domain)+'.nc')


def daily_to_hourly(raw_hourly,BC_max_daily,BC_min_daily,raw_max_daily,raw_min_daily):
    BC_max_resampled=BC_max_daily.resample(Time='1H').ffill().interp(Time=raw_hourly.Time,method='nearest',kwargs={'fill_value':'extrapolate'})
    BC_min_resampled=BC_min_daily.resample(Time='1H').ffill().interp(Time=raw_hourly.Time,method='nearest',kwargs={'fill_value':'extrapolate'})
    raw_max_resampled=raw_max_daily.resample(Time='1H').ffill().interp(Time=raw_hourly.Time,method='nearest',kwargs={'fill_value':'extrapolate'})
    raw_min_resampled=raw_min_daily.resample(Time='1H').ffill().interp(Time=raw_hourly.Time,method='nearest',kwargs={'fill_value':'extrapolate'})
    scaling=(BC_max_resampled-BC_min_resampled)/(raw_max_resampled-raw_min_resampled)
    hourly_corrected=(raw_hourly-raw_min_resampled)*scaling+BC_min_resampled
    return(hourly_corrected)

hourly_BC_fixedmax_varmin=daily_to_hourly(raw_temp_hourly,BC_fixed_Tmax,BC_variable_Tmin,raw_Tmax_daily,raw_Tmin_daily)
hourly_BC_fixedmax_varmin.to_netcdf(output_folder+'hourly_BC_fixedmax_varmin_d0'+str(domain)+'.nc')





