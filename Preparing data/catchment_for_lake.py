# -*- coding: utf-8 -*-

import geopandas as gpd
import pandas as pd

#%% Read basin, pour point and lake files

# original basins geometries fixed & exploded into singlepart in qgis
basins_file = r'W:\Dippa\Data\ei_kayteta\uusi_valuma_singlepart'
# pour points created with pysheds + manually added ones
points_file = r'W:\Dippa\Data\final_pour_points'
# original lakes filtered to match chl observations, plus removed some useless columns
lakes_file = r'W:\Dippa\Data\VHS_vesialueet_2016\Jarvi_filtered'

basins = gpd.read_file(basins_file)
pour_points = gpd.read_file(points_file)
lakes = gpd.read_file(lakes_file)

# load national borders
borders_file = r'W:\Dippa\Data\mml\hallintorajat_10k\2020'
borders = gpd.read_file(borders_file)
borders_ext = borders.exterior
      
#%% Processing basin data

# add field for new unique basin id's:
duplicates = basins[basins.OBJECTID.duplicated(keep=False)]
basins_filter = basins.OBJECTID.isin(duplicates.OBJECTID)

duplicates_tunnu = basins[basins.KohdeTunnu.duplicated(keep=False)]
basins_filter_tunnu = basins.KohdeTunnu.isin(duplicates_tunnu.KohdeTunnu)

basins['IdFilter'] = basins_filter
basins['TunnuFilter'] = basins_filter_tunnu

basins['Area'] = basins.area
basins['UNIQUEID'] = basins.apply(lambda x: (float( str(x['OBJECTID']) +  str(x['Area'])[-4:])) if x['IdFilter'] == True else x['OBJECTID'], axis = 1)
basins['TunnuEdit'] = basins.apply(lambda x: ( str(x['KohdeTunnu']) + '_' + str(x['Area'])[-4:]) if x['TunnuFilter'] == True else x['KohdeTunnu'], axis = 1)

# save unique id basins
basins.to_file(r'W:\Dippa\Data\uusi_valuma_unique\unique_basins.shp')

u_basins = basins

#%% Processing basins data
 
#get neighbor basins for each basin
for index, row in u_basins.iterrows():   
 # get 'not disjoint' basins
 neighbors = u_basins[~u_basins.geometry.disjoint(row.geometry)].UNIQUEID.tolist()
 # remove own 'tunnus' from the list
 neighbors = [ tunnus for tunnus in neighbors if row.UNIQUEID != tunnus ]
 neighbors_str = [ str(x) for x in neighbors ]
 if len(neighbors_str) > 50:
  print('varoitus' + str(row.TunnuEdit))
 # add unique id of neighbors as NEIGHBORS value
 u_basins.at[index, "Naapurit"] = ",".join(neighbors_str)
 
u_basins.to_file('basins_neigh.shp')

#%% function for finding out whether basin extends beyond national borders

def basin_within_borders(borders, lake_basin):

 if lake_basin.geometry.iloc[0].is_valid:
  intersection = lake_basin.geometry.iloc[0].intersection(borders.iloc[0])
 else:
  basin_geom_clean = lake_basin.geometry.iloc[0].buffer(0)
  intersection = basin_geom_clean.intersection(borders.iloc[0])
 
 # check intersection
 if intersection.length > 0:
  return False
 else:
  return True


#%% Recursive function for finding a basin for a lake

def basin_for_lake(basin, all_basins, all_points, lake_basin_list):
 # iterate through neighbors to find pour points TO current basin
 #print('entering with ' + basin.TunnuEdit.values[0])
 
 for tunnus in (basin.Naapurit.values[0].split(',')):
  # check neighbour except if it has already been handled
  if tunnus not in lake_basin_list:
   tunnus_flt = float(tunnus)
   # assign a neighboring basin as 'neighbor'
   neighbor = all_basins[all_basins.UNIQUEID == tunnus_flt]
   
   # find the edge between neighboring basins and create 15 meter buffer around it
   basins_edge = basin.iloc[0].geometry.intersection(neighbor.iloc[0].geometry)
   buffer_edge = basins_edge.buffer(10.0)
   
   neighbor_id = neighbor.TunnuEdit.values[0]
   
   # find the pour point of the neighbor basin
   point = all_points[all_points.TunnuEdit == neighbor_id]
   if point.shape[0] == 0:
    print(neighbor_id)
   point_geom = point.geometry.values
   # Test if pour point intersects with buffered basins edge -> 
   # neighbor pours into current basin. 
   if point_geom.intersects(buffer_edge):
    # add neighbor id into lake's basin list and call function for neighbor
    lake_basin_list.append(tunnus)
    #print('appended' + tunnus)
    basin_for_lake(neighbor, all_basins, all_points, lake_basin_list)
 return lake_basin_list   

#%% Function for dissolving basins
def dissolve_basins(basin_list, basins):
 whole_basin = gpd.GeoDataFrame(columns=['OBJECTID', 'KohdeTunnu', 'Nimi', 'UNIQUEID', 'TunnuFilte', 'TunnuEdit', 'Area',
       'Naapurit', 'geometry'])
 for tunnus in basin_list:
  tunnus_flt = float(tunnus)
  basin = basins[basins.UNIQUEID == tunnus_flt]
  whole_basin = whole_basin.append(basin)
 whole_basin.insert(1, 'dissolve_field', 1)
 dissolved_basin = whole_basin.dissolve(by='dissolve_field')
 return dissolved_basin

#%% Create basin for each lake

whole_basins_gdf = gpd.GeoDataFrame(columns=['TunnuEdit', 'VPDTunnus', 'geometry'], crs='epsg:3067')
# list for lakes whose basins extend outside of national borders
exclude_lakes = []

# loop through lakes
for index, row in lakes.iterrows():
 # list for finally collecting id's of all sub basins in whole basin
 whole_basin_list =[]
 # select a lake from dataframe
 lake_id = row.VPDTunnus
 lake = lakes[lakes.VPDTunnus == lake_id]
 # select lake's subbasin from basins gdf
 og_basin = u_basins[u_basins.geometry.apply(lambda x: x.intersects(lake.iloc[0].geometry.centroid))]
 
 # this condition excludes lakes without basins, i.e. lakes in Ahvenanmaa
 if og_basin.shape[0] > 0:
   
  # lists for this specific lake's basins, for the recursion
  lake_basin_list = []
  # add first subbasin to list     
  lake_basin_list.append(str(og_basin.UNIQUEID.values[0]))
 
  # if basin has neighbors start recursion
  if len(og_basin.Naapurit.values[0]) > 0:
   whole_basin_list = basin_for_lake(og_basin, u_basins, pour_points, lake_basin_list)
  else:
   whole_basin_list = lake_basin_list

  if len(whole_basin_list) > 1:
   # dissolve subbasins & add lake_id (VPDTunnus)
   final_basin = dissolve_basins(whole_basin_list, u_basins)
  else:
   final_basin = og_basin

  final_basin.insert(1, 'VPDTunnus', lake_id)
  final_basin = final_basin.drop(columns = ['OBJECTID', 'KohdeTunnu', 'IdFilter', 'TunnuFilter', 'Area', 'Naapurit'])
  # add dissolved basin to all whole basins gdf
  whole_basins_gdf = whole_basins_gdf.append(final_basin)
  print('added basin ' + str(index) + ', ' + str(lake_id))
  # if lake's basin extends outside of national borders, add lake to exclude list
  if not basin_within_borders(borders_ext, final_basin):
   exclude_lakes.append(lake_id)
 else:
  exclude_lakes.append(lake_id)   

#%%
# save whole basins gdf as shapefile
whole_basins_gdf.to_file(r'W:\Dippa\Data\lake_catchments\lake_catchments.shp')

# save excluded lakes list as csv
exclude_lakes_df = pd.DataFrame(exclude_lakes)
exclude_lakes_df.to_csv(r'W:\Dippa\Data\lakes_outside_borders.csv', index=False)

#%% 
whole_basins_gdf = whole_basins_gdf.drop(columns = ['Shape_Leng',
       'Shape_Area', 'TunnuFilte'])
whole_basins_gdf['Area'] = whole_basins_gdf.area



