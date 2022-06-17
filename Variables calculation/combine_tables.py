# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import geopandas as gpd
import os

os.chdir('C:/Users/heikons1/Git local/eutrophication-drivers/Variables calculation')
# own function
import aggregation_area_processing

#%% Combine variable data from individual catchment files

lake_ids = 'Y:/Dippa/Data/model_input/basins/lake_ids.txt'

# Erosion, combine single files

table_base_path = 'Y:/Dippa/Data/model_output/erosion/'
whole_table = pd.DataFrame(columns = ['Area', 'Year', 'erosion'])

with open(lake_ids, 'r') as ids:
 for line in ids:
  line = line.strip()
  table_path = table_base_path + line + '.csv'
  table = pd.read_csv(table_path, sep=';')
  table['erosion'] = table.erosion.apply(lambda x: np.nan if x == 0 else x)
  whole_table = whole_table.append(table)

whole_table.to_csv(table_base_path + 'erosion.csv', sep=',', index=False)


# Climate variables, combine single files

table_base_path = 'Y:/Dippa/Data/model_output/climate_new_20/'
whole_table = pd.DataFrame()

with open(lake_ids, 'r') as ids:
 for line in ids:
  line = line.strip()
  table_path = table_base_path + line + '.csv'
  table = pd.read_csv(table_path, sep=';')
  whole_table = whole_table.append(table)

whole_table.to_csv(table_base_path + 'climate_new_20.csv', sep=',', index=False)

# Land use variables, combine single files

table_base_path = 'Y:/Dippa/Data/model_output/land_use/'
whole_table = pd.DataFrame(columns = ['Area', 'Year', 'agric', 'built', 'peatland', 'mineral_forest', 'water'])

with open(lake_ids, 'r') as ids:
 for line in ids:
  line = line.strip()
  table_path = table_base_path + line + '.csv'
  table = pd.read_csv(table_path, sep=';')
  table.loc[:, ['agric', 'built', 'peatland', 'mineral_forest', 'water']] = table[['agric', 'built', 'peatland', 'mineral_forest', 'water']].multiply(100) # transform fractions into percentages
  whole_table = whole_table.append(table)

whole_table.to_csv(table_base_path + 'land_use.csv', sep=',', index=False)


#%% Combine all variables into rf table

# target variable: base for table
target_table_path = 'Y:/Dippa/Data/model_input/target/jarviklorofylli_seasonal_medians.csv'
target = pd.read_csv(target_table_path, sep=',')
target['Area_Year'] = target.apply(lambda x: str(x.Area) + '_' + str(x.Year), axis=1)

# other variables
table_base_path = 'Y:/Dippa/Data/model_output/'

erosion_path = table_base_path + 'erosion/erosion.csv'
erosion = pd.read_csv(erosion_path, sep=',')
erosion['Area_Year'] = erosion.apply(lambda x: str(x.Area) + '_' + str(x.Year), axis=1)

climate_path = table_base_path + 'climate_new_20/climate_new_20.csv'
climate = pd.read_csv(climate_path, sep=',')
climate['Area_Year'] = climate.apply(lambda x: str(x.Area) + '_' + str(x.Year), axis=1)

land_use_path = table_base_path + 'land_use/land_use.csv'
land_use = pd.read_csv(land_use_path, sep=',')
land_use['Area_Year'] = land_use.apply(lambda x: str(x.Area) + '_' + str(x.Year), axis=1)

final_table = pd.merge(target, erosion, how='right', on='Area_Year')
final_table.drop(columns=['Area_y', 'Year_y'], inplace=True)
final_table = pd.merge(final_table, climate, how='right', on='Area_Year')
final_table.drop(columns=['Area_x', 'Year_x'], inplace=True)
final_table = pd.merge(final_table, land_use, how='right', on='Area_Year')
final_table.drop(columns=['Area_y', 'Year_y', 'Area_Year'], inplace=True)
final_table.rename(columns={'Area_x':'Area', 'Year_x':'Year'}, inplace=True)

#%% Add lake specific data

# add lake data
lakes_path = r'Y:\Dippa\Data\VHS_vesialueet_2016\Jarvet_filter'
lakes = gpd.read_file(lakes_path)

# choose lake columns to add to final table
lakes.insert(1, 'Tyyppi_lyh', np.nan)
lakes['Tyyppi_lyh'] = lakes.apply(lambda x: x.Tyyppi[-5:], axis=1)
lakes.drop(columns=['Tyyppi'], inplace=True)

# add lake depth
depth_path = r'Y:\Dippa\Data\model_input\basins\catchments_with_depth'
depth = gpd.read_file(depth_path)

depth_column = depth.loc[:, ['mean_depth', 'VPDTunnus']]

# add lake columns to rf table / "final table"
final_table = pd.merge(final_table, lakes, how='right', right_on='VPDTunnus', left_on='Area')

# add depth column
final_table = pd.merge(final_table, depth_column, how='inner', right_on='VPDTunnus', left_on='Area')
final_table = final_table.drop(columns=['VPDTunnus_y', 'Area'])

# rename some columns
final_table.rename(columns={'VHATunnusN':'VHA', 'VPDTunnus_x':'VPDTunnus'}, inplace=True)

#drop na-rows
final_table_nona = final_table.dropna(axis=0, how='any')


#%% Add information about catchment hierarchy

rf_table = aggregation_area_processing.main(lakes, final_table_nona)

# drop geometry
rf_table.drop(columns=['geometry'], inplace=True)

# take mean of variable values from the study period
rf_table = rf_table.groupby(['VPDTunnus','Tyyppi_lyh', 'VHA', 'PaajakoTun'], as_index=False ).mean()

# drop year-column
rf_table.drop(columns=['Year'], inplace=True)

#%% Saving rf-table

# save final table 
rf_table.to_csv(table_base_path + 'Final_tables/rf_table.csv', index=False)

