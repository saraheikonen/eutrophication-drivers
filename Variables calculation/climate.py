# -*- coding: utf-8 -*-
import pandas as pd
import geopandas as gpd
import numpy as np
from rasterstats import zonal_stats
import sys


def main():
    
 run_path = '/scratch/work/heikons1/dippadata/'
 # get input file path 
 input_id = sys.argv[1].strip()

 lake_table = pd.DataFrame(columns=['Area', 'Year'])
 ids = pd.Series(input_id)
 lake_table.iloc[:, 0] = ids.repeat(4).values
 lake_table.iloc[:, 1] = [2016, 2017, 2018, 2019]

 basins_folder = run_path + 'model_input/basins/lake_catchments_with_depth'
 basins = gpd.read_file(basins_folder)
 basin = basins[basins.VPDTunnus == input_id]

 # reproject gdf
 basin_proj = basin.to_crs(epsg=3067)
 
 table_part = climate_variables(lake_table, basin_proj, run_path)
 #print(input_id)
 table_part.to_csv(run_path + 'model_output/climate_new/' + input_id + '.csv', sep=';', index=False)

def climate_variables(table, basin, run_path):
 prec_folder = run_path + 'model_input/climate_new/precipitation/'
 temp_folder = run_path + 'model_input/climate_new/temperature/'

 # create empty columns for results
 table.insert(2,'prec_gs', np.nan)
 table.insert(3,'prec_ws', np.nan)
 table.insert(4,'temp_gs', np.nan)
 table.insert(5,'temp_ws', np.nan)
 
 # get climate rasters
 ws_prec_path = prec_folder + 'prec_winter_reproject.tif'
 gs_prec_path = prec_folder + 'prec_summer_reproject.tif'
 ws_temp_path = temp_folder + 'temp_winter_reproject.tif'
 gs_temp_path = temp_folder + 'temp_summer_reproject.tif'
    
 # calculate mean values of climate variables and attach to table
 ws_prec_mean = zonal_stats(basin, ws_prec_path, stats=['mean'], all_touched=True)
 table.loc[:,'prec_ws'] = ws_prec_mean[0]['mean']
    
 gs_prec_mean = zonal_stats(basin, gs_prec_path, stats=['mean'], all_touched=True)
 table.loc[:,'prec_gs'] = gs_prec_mean[0]['mean']
   
 ws_temp_mean = zonal_stats(basin, ws_temp_path, stats=['mean'], all_touched=True)
 table.loc[:,'temp_ws'] = ws_temp_mean[0]['mean']
   
 gs_temp_mean = zonal_stats(basin, gs_temp_path, stats=['mean'], all_touched=True)
 table.loc[:,'temp_gs'] = gs_temp_mean[0]['mean']
  
 
 return table
  
main()
 