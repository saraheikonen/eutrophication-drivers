
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.crs
from rasterio.merge import merge
from rasterio.mask import mask
from pysheds.grid import Grid
import sys
import os
import fiona


def create_point(basin_path, basin, dem_index, dem_path, basin_dem_path):

 pour_point = pd.DataFrame(columns = ['TunnuEdit', 'x', 'y'])
   
 basin_id = basin.iloc[0].TunnuEdit
 out_path = basin_dem_path + basin_id + "_dem"
 
 # filter out basins with too small area, for these pour points will be added manually
 if (basin.iloc[0].Area < 9000):
  pass

 else:

  # find dem raster tiles that intersect with the basin
  dem_tiles = gpd.overlay(basin, dem_index, how = 'intersection')
  dem_tiles_list = dem_tiles['location']
  dem_files = []
  
  # get a list of files to open
  for label in dem_tiles_list:
   dem_files.append(dem_path + label)

  # open files
  files_to_mosaic = []
  for file in dem_files:
   raster = rasterio.open(file)
   files_to_mosaic.append(raster)
   
  if len(files_to_mosaic) > 1:    
   # create a mosaic of dem rasters in basin
   mosaic, out_trans = merge(files_to_mosaic)
  
   out_meta = raster.meta.copy()
   out_meta.update({"driver": "GTiff",
                  "height": mosaic.shape[1],
                  "width": mosaic.shape[2],
                  "transform": out_trans,
                  "crs": "epsg:3067"})
   
   # save temporary dem raster mosaic
   with rasterio.open(out_path + '.tif', "w", **out_meta) as dest:
    dest.write(mosaic)
 
   # open saved dem mosaic for geometric operations
   with rasterio.open(out_path + '.tif') as dataset:
 
    with fiona.open(basin_path) as shapefile:
     shapes = [feature['geometry'] for feature in shapefile]
     # mask dem with basin geometry
     dem_mask, out_trans_mask = mask(dataset, shapes, crop=True)
    
   # make a grid object out of mosaic
   grid = Grid.from_raster(out_path + '.tif', data_name='dem_mosaic')
  
   # add masked data
   grid.add_gridded_data(data=dem_mask[0], data_name='basin_dem', affine=out_trans_mask, crs=grid.crs, nodata=-999)
    
  else: # only one dem raster contains the whole catchment

   with fiona.open(basin_path) as shapefile:
    shapes = [feature['geometry'] for feature in shapefile]
    
    # mask dem with basin geometry
    dem_mask, out_trans_mask = mask(raster, shapes, crop=True)
   
    # make a grid object out of mosaic
    grid = Grid.from_raster(file, data_name='dem_tile')
   
   # add masked data
   grid.add_gridded_data(data=dem_mask[0], data_name='basin_dem', affine=out_trans_mask, crs=grid.crs, nodata=-999)
 
#-------------------------------------------------------working with pysheds

  # dem conditioning
  try:
   grid.fill_depressions(data='basin_dem', out_name='flooded_dem')
   grid.resolve_flats(data='flooded_dem', out_name='inflated_dem')
   grid.flowdir(data='inflated_dem', out_name='fdir')
   grid.accumulation(data='fdir', out_name='acc')

   # save max accumulation point as the pour point
   coords_index = np.where(grid.acc == np.amax(grid.acc))
   y_index = coords_index[0][0]
   x_index = coords_index[1][0]

   x_size, X_rotation, upper_left_x, y_rotation, y_size, upper_left_y = grid.acc.affine[:6]
   x_coords = x_index * x_size + upper_left_x + (x_size / 2) #add half the cell size
   y_coords = y_index * (y_size) + upper_left_y + (y_size / 2) #to centre the point
 
   pour_point = pour_point.append({'TunnuEdit': basin_id, 'x': x_coords, 'y': y_coords}, ignore_index = True)
   print('pour point for ' + basin_id + ' was generated')
   sys.stdout.flush()
   
  # errors that occur if processing memory requirement becomes too large
  except (ValueError, MemoryError):
   print('basin ' + basin_id + ' was not handled')
   sys.stdout.flush()
    
  # delete temporary dem-mosaics
  if os.path.exists(out_path + '.tif'):
   os.remove(out_path + '.tif')
   
 return pour_point



def main(run_path, input_id):
    
 # load data
 input_path = run_path + 'basins/' + input_id
 basin = gpd.read_file(input_path)
 dem_index_path = run_path + 'dem/index'
 dem_index = gpd.read_file(dem_index_path)

 # reproject geodfs
 basin_proj = basin.to_crs(epsg=3067)
 dem_index_proj = dem_index.to_crs(epsg=3067)

 # define paths
 dem_path = run_path + 'dem/'
 basin_dem_path = run_path + 'basin_dems/'
 
 # calculate point
 point = create_point(input_path, basin_proj, dem_index_proj, dem_path, basin_dem_path)
 
 if point.empty == False:
  save_path = run_path + 'pour_points/' + input_id + '.csv'
  point.to_csv(save_path, sep=';')


if __name__ == '__main__':
 
 run_location = {'cluster': '/scratch/work/heikons1/dippadata/', 'local':'Y:/Dippa/'}
 run_path = run_location['cluster']
 
 # get input file path 
 input_id = sys.argv[1].strip() # lake id, i.e. EU WFD id
 main(run_path, input_id)

    
    






   


