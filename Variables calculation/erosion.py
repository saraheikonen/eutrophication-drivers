# -*- coding: utf-8 -*-
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterstats import zonal_stats
import numpy as np
import sys


def main():

 run_path = '/scratch/project_2002694/diplomityo/'
 # get input file path 
 input_id = sys.argv[1].strip()
    
 #start_t_year = time.time

 lake_table = pd.DataFrame(columns=['Area', 'Year', 'erosion'])
 ids = pd.Series(input_id)
 lake_table.iloc[:, 0] = ids.repeat(4).values
 lake_table.iloc[:, 1] = [2016, 2017, 2018, 2019]

 field_path = run_path + 'field_plots'
 fields = gpd.read_file(field_path)

 basins_path = run_path + 'lake_catchments_with_depth'
 basins = gpd.read_file(basins_path)
 basin = basins[basins.VPDTunnus == input_id]

 erosion_base_path = '/appl/data/geo/'

 # reproject fields and catchments to epsg 3067 (which ndvi is in too)
 fields.to_crs(epsg=3067, inplace=True)
 basin_proj = basin.to_crs(epsg=3067)
 
 table_part = erosion_variable(lake_table, fields, basin_proj, erosion_base_path)
 print(input_id)
 table_part.to_csv(run_path + 'erosion/'+ input_id + '.csv', sep=';', index=False)
 
def erosion_variable(table, fields, basin, erosion_base_path):

 erosion_index_folder = erosion_base_path + 'karelia/erosion_risk'
 erosion_index = gpd.read_file(erosion_index_folder)

 # reproject gdf's
 erosion_index.to_crs(epsg=3067, inplace=True)
 
 # get catchment and fields intersection
 fields_within_catch = gpd.overlay(basin, fields, how = 'intersection')
 
 # dissolve all geometries in the intersection, get area
 fields_within_catch['dissolve_field'] = 1
 fields_dissolved = fields_within_catch.dissolve(by='dissolve_field')
 
 if fields_dissolved.shape[0] > 0:
 
  # find erosion raster tiles (from index) that intersect with the catchment
  erosion_tiles = gpd.overlay(erosion_index, basin, how = 'intersection')
  erosion_tiles['dissolve_field'] = 1
  erosion_tiles_dissolved = erosion_tiles.dissolve(by='dissolve_field')
 
  # see if the erosion tiles covers all fields
  if erosion_tiles.size == 0:
   # if no fields are covered, attach nan to final table
   table.loc[:,'erosion'] = np.nan
  elif fields_dissolved.iloc[0].geometry.within(erosion_tiles_dissolved.iloc[0].geometry.buffer(10)):
   # if yes, get erosion rasters
   erosion_tiles_list = erosion_tiles['location']
   erosion_files = []
  
   # get a list of files to open
   for label in erosion_tiles_list:
    erosion_files.append(erosion_base_path + label[:-1] + 'tif')
   
    # open files
    files_to_mosaic = []
    for file in erosion_files:
     raster = rasterio.open(file)
     files_to_mosaic.append(raster)
    
    if len(files_to_mosaic) > 1: 
     # create a mosaic of erosion rasters in catchment
     mosaic, out_trans = merge(files_to_mosaic)
     data = mosaic[0]
    else:
     data = raster.read(1)
     out_trans = raster.transform

  
    # take mean of erosion categories in catchment
    erosion_mean = zonal_stats(fields_dissolved, data, affine=out_trans, stats=['median'])
    table.loc[:,'erosion'] = (erosion_mean[0]['median'])

    
 return table

main()
