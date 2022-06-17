import numpy as np
from rasterstats import zonal_stats
import pandas as pd
import geopandas as gpd
import sys

def main():
    
 run_path = '/scratch/work/heikons1/dippadata/'
 # get input file path 
 input_id = sys.argv[1].strip()

 lake_table = pd.DataFrame(columns=['Area', 'Year'])
 ids = pd.Series(input_id)
 lake_table.iloc[:, 0] = ids.repeat(4).values
 lake_table.iloc[:, 1] = [2016, 2017, 2018, 2019]
 
 #load basins file/gdf and select one basin
 basins_folder = run_path + 'model_input/basins/lake_catchments_with_depth'
 basins = gpd.read_file(basins_folder)
 basin = basins[basins.VPDTunnus == input_id]

 # reproject gdf
 basin_proj = basin.to_crs(epsg=3035)
 
 # dictionary for landuse calsses
 land_use_dict = {'agric': [17,18,19,20,21,22],
                  'built': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
                  'peatland': [24,26,29,35,41,43,44,45],
                  'mineral_forest': [23,25,27,28,30,31,32,33,34,36,37,38,39,40],
                  'water': [42,46,47,48,49]}
 
 table_part = land_use_variables(lake_table, basin_proj, land_use_dict, run_path)
 
 print(input_id)
 table_part.to_csv(run_path + 'model_output/land_use/' + input_id + '.csv', sep=';', index=False)

def land_use_variables(table, basin, land_use_dict, run_path):
 # open land use data and reproject into an equal area projection
 corine_path = run_path + 'model_input/corine/Clc2018_FI20m_equala.tif'
 
 # create empty columns for results
 table.insert(2,'agric', np.nan)
 table.insert(3, 'built', np.nan)
 table.insert(4,'peatland', np.nan)
 table.insert(5,'mineral_forest', np.nan)
 table.insert(6, 'water', np.nan)
 table.insert(7, 'basin_area', np.nan)
 
 all_pixels = zonal_stats(basin, corine_path, stats='count', all_touched=True)[0]['count']
 
 # calculate percentages of landuse classes and attach to table
 category_counts = zonal_stats(basin, corine_path, categorical = True, all_touched=True)
 
 # iterate throuhg category_counts and count the number of pixels in each 
 # land use category
 
 #initiate counting variables
 agric = 0
 built = 0
 peatland = 0
 mineral_forest = 0
 water = 0
 
 for key in category_counts[0]:
     if key in land_use_dict['agric']:
         agric += category_counts[0][key]
     elif key in land_use_dict['built']:
         built += category_counts[0][key]
     elif key in land_use_dict['peatland']:
         peatland += category_counts[0][key]
     elif key in land_use_dict['mineral_forest']:
         mineral_forest += category_counts[0][key]
     else:
         water += category_counts[0][key]


 # calculate land use % and add to table
 table.loc[:, 'agric'] = agric / all_pixels
 table.loc[:, 'built'] = built / all_pixels
 table.loc[:, 'peatland'] = peatland / all_pixels
 table.loc[:, 'mineral_forest'] = mineral_forest / all_pixels
 table.loc[:, 'water'] = water / all_pixels
 
 return table

main()
 
