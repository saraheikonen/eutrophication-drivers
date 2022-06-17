# -*- coding: utf-8 -*-
from fmiopendata.wfs import download_stored_query
import xarray as xr
import os

# import self-defined reprojection function
import reproject_raster


# Specify save path here:
save_path = 'Y:/Dippa/Data/processing/climate_test/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

def main(save_path):
    
 # defining downloaded periods   
 summer_period = ['06', '09', '01', '30']
 winter_period = ['10', '05', '01', '31']
 years = range(2000,2020)
 years_list = list(years)
 
 # query bbox & product id
 bbox = "bbox=18,55,35,75"
 product = "fmi::observations::weather::monthly::grid"

 get_data(years_list, summer_period, bbox, product, 'summer', save_path)
 get_data(years_list, winter_period, bbox, product, 'winter', save_path)
 
     
# download and process summer temperature and precipitation     
def get_data(years_list, period, bbox, product, season, path):
    
    # lists for collecting annual & seasonal data arrays + index for the lists
    annual_data_paths = []
    
    for year in years_list:
        
        # query start and end parameters
        if season == 'summer':
            start_time = 'starttime='+ str(year) +'-'+ period[0] +'-'+ period[2] +'T00:00:00Z'
        else:
            start_time = 'starttime='+ str(year - 1) +'-'+ period[0] +'-'+ period[2] +'T00:00:00Z'
        end_time = 'endtime='+ str(year) +'-'+ period[1] +'-'+ period[3] +'T20:00:00Z'
        
        # get metadata
        model_data = download_stored_query(product, args=[start_time, end_time, bbox])
        # download data
        latest_run = max(model_data.data.keys())  # datetime.datetime(2020, 7, 7, 12, 0)
        data = model_data.data[latest_run]
        # download the data to a temporary file
        save_path = path + season + '_' + str(year) + '.GRIB2'
        annual_data_paths.append(save_path)
        data.download(fname = save_path)
        
    # open all years as an xarray dataset (temperature)
    with xr.open_mfdataset(annual_data_paths, engine='cfgrib', combine='nested', concat_dim='step') as temp_data:
        # add temperature as celcius deg as a new variable
        c_temp_data = temp_data.assign(t2mc = lambda x: x['t2m'] - 273.15)
        # take a mean of all temps (this is okay to do for the whole dataset at
        # once because all 'sample periods' aka seasons are of equal size/length)
        mean_temp = c_temp_data.t2mc.mean(dim='step')
        #set projection
        mean_temp.rio.set_crs('epsg:4326', inplace=True)
        
        # save as geotif
        if not os.path.exists(path + 'temperature/'):
            os.makedirs(path + 'temperature/')
        mean_temp.rio.to_raster(path + 'temperature/temp_proj' + season + '.tif')
        
        # reproject saved raster
        reproject_raster.main(path + 'temperature/temp_proj' + season + '.tif',
                         path + 'temperature/temp_' + season + '_reproject.tif',
                         'EPSG:3067')
            
    annual_precs = []   
    year_number = years_list[0]
    # open all years as an xarray dataset (precipitation)
    for annual_path in annual_data_paths:        
        with xr.open_dataset(annual_path, engine='cfgrib', backend_kwargs={'filter_by_keys': {'shortName': 'prate'}}) as prec_data:
            #print(prec_data.keys)
            prec_cumul = prec_data.prate.sum(dim='step')
            prec_cumul = prec_cumul.rename('prate'+str(year_number))
            annual_precs.append(prec_cumul)
            year_number = year_number + 1
    # combine cumulative precs into one dataset
    all_precs = xr.merge(annual_precs, compat='override')
    # take a mean of the annual precs (that are in the dataset as separate
    # variables) by aggregating the dataset into a data-array and using an
    # additional, new dimension
    mean_cumul_prec = all_precs.to_array(dim='new').mean('new')
    
    # set projection
    mean_cumul_prec.rio.set_crs('epsg:4326', inplace=True)
    # save as geotif
    if not os.path.exists(path + 'precipitation/'):
        os.makedirs(path + 'precipitation/')
    mean_cumul_prec.rio.to_raster(path + 'precipitation/prec_proj' + season + '.tif')
    
    # reproject saved raster
    reproject_raster.main(path + 'precipitation/prec_proj' + season + '.tif',
                     path + 'precipitation/prec_' + season + '_reproject.tif',
                     'EPSG:3067')
    
    # delete grib files
    for path in annual_data_paths:
        os.remove(path)
    
main(save_path)