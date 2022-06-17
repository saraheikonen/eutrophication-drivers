# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from scipy import stats
import geopandas as gpd

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

#%% import rf table data (needed for all cells)

table_path = 'Y:/Dippa/Data/model_output/Final_tables/rf_table.csv' # edit accordingly
table = pd.read_csv(table_path, sep=',')

# create folders for tables and figures
out_path = 'C:/Users/heikons1/Git local/eutrophication-drivers/background_tbl_fig/' #edit accordingly
if not os.path.exists(out_path):
    os.mkdir(out_path)

#%% Parameter stats table (Supplementary Table 1)

table_source = table.drop(columns=['VPDTunnus', 'PaajakoTun',
       'VHA', 'Tyyppi_lyh'])

# create dataframe for storing results
variables = table_source.columns
stats_df = pd.DataFrame(index = variables, columns=['mean','min', 'max', 'stdev', 'CV', 'CC', 'p_val'])

# calculate stats

table_vars = stats_df.index.values

for var in table_vars:
 
 var_min = table_source[var].min()
 var_max = table_source[var].max()
 var_median = table_source[var].median()
 var_mean = table_source[var].mean()
 var_stdev = table_source[var].std()
 var_cv = var_stdev / var_mean 
 var_ccs, pval = stats.spearmanr(table_source[var], table_source['Concentration'])
 
 stats_arr = np.array([var_mean, var_min, var_max, var_stdev, var_cv, var_ccs, pval]) 
 stats_df.loc[var,:] = stats_arr
 stats_df.to_csv(out_path + 'parameter_stats.csv')

#%% plot study catchments and river basin disricts 1-4 (Figure 1A)

samples_path = 'Y:/Dippa/Data/model_input/basins/catchments_with_depth' # study catchments
national_borders_path = 'Y:/Dippa/Data/processing/mml/hallintorajat_10k/2020' # administrative borders
vha_path = r'Y:\Dippa\Data\processing\VHA' # river basin district borders

samples = gpd.read_file(samples_path)
study_samples = samples.merge(table, how='inner', on='VPDTunnus')
study_samples_drop = study_samples.drop_duplicates(subset=['VPDTunnus'], keep='first')
study_samples_drop.insert(1, 'dissolve_field', 1)
samples_dissolved = study_samples_drop.dissolve(by='dissolve_field')
samples_dissolved['simple_geo'] = samples_dissolved.geometry.simplify(5000)
samples_dissolved.set_geometry('simple_geo', inplace=True)
vha = gpd.read_file(vha_path)
vha_name_list = ['VHA1', 'VHA2', 'VHA3', 'VHA4']
vha['dissolve_field'] = vha.apply(lambda x: 'other' if x.VHATunnus not in vha_name_list else x.VHATunnus, axis=1)
vha_dissolved = vha.dissolve(by='dissolve_field')
model_vhas = vha.loc[vha.VHATunnus.isin(vha_name_list)] 
national_borders = gpd.read_file(national_borders_path)

model_vhas_latlon = model_vhas.to_crs('EPSG:4326')
vha_dissolved_latlon = vha_dissolved.to_crs('EPSG:4326')
national_borders_latlon = national_borders.to_crs('EPSG:4326')
samples_dissolved_latlon = samples_dissolved.to_crs('EPSG:4326')

fig, ax = plt.subplots()
ax.axis("off")
samples_dissolved_latlon.plot(ax = ax, legend=True, color= 'dimgrey', zorder=3)
vha_dissolved_latlon.plot(ax = ax, legend=True, color='lightgrey', edgecolor='darkgrey', zorder=2)
vha_dissolved_latlon.apply(lambda x: ax.annotate(text=(x['VHATunnus'][3:] if x['VHATunnus'] != 'VHA7' else ''), xy=x.geometry.centroid.coords[0], ha='center', color = 'black'), axis=1)
plt.show()
fig.savefig(out_path + 'catchments_rbds.pdf', bbox_inches = 'tight')
fig.savefig(out_path + 'catchments_rbds.png', bbox_inches = 'tight')
plt.close()



#%% plot chl-a in catchments (Figure 1B)

map_table = table

borders_path = 'Y:/Dippa/Data/processing/mml/hallintorajat_10k/2020' # administrative borders
catchments_path = 'Y:/Dippa/Data/model_input/basins/catchments_with_depth' # study catchments
parent_df_path = 'Y:/Dippa/Data/processing/valuma-aluejako/lake_catchments_parents_manual.csv' # study catchment hierarchies

borders = gpd.read_file(borders_path)
borders_latlon= borders.to_crs('EPSG:4326')
catchments = gpd.read_file(catchments_path)
parent_df = pd.read_csv(parent_df_path)
parent_df.drop(columns=['Unnamed: 0'], inplace=True)
parent_df['parent'] = parent_df.apply(lambda x: x.VPDTunnus if pd.isna(x.parent) else x.parent, axis=1)

# join parent info to data df and calculate mean and std chl-a within not nested catchments
data_parents = pd.merge(map_table, parent_df, how='inner', on='VPDTunnus')
chla_means = data_parents.groupby('parent').Concentration.mean()
chla_std = data_parents.groupby('parent').Concentration.std().fillna(0)

# join mean and std dfs
chla_stats = pd.merge(chla_means, chla_std, how='inner', on='parent')
chla_stats.rename(columns={'Concentration_x':'mean', 'Concentration_y':'std'}, inplace=True)

# calculate cv
cv = lambda x: x['std'] / x['mean']
chla_stats['cv'] = chla_stats.apply(cv, axis=1)

# join geometries for all not nested catchments
not_nested_catch_chla = pd.merge(chla_stats, catchments, how='inner', left_on='parent', right_on='VPDTunnus')

# convert merged df into gdf for plotting
not_nested_catch_chla_gdf = gpd.GeoDataFrame(not_nested_catch_chla)
# simplify geometry for faster plotting
not_nested_catch_chla_gdf.loc[:,'geometry'] = not_nested_catch_chla_gdf.geometry.simplify(5000)
# reproject to latlon:
chla_latlon = not_nested_catch_chla_gdf.to_crs('EPSG:4326')

                         
chla_fig, chla_ax = plt.subplots()
chla_latlon.plot(column = 'mean', ax=chla_ax, cmap='viridis', vmin = 0, vmax = 25, zorder=2, legend=True, legend_kwds={'label':'Mean of aggregated chl-a concentration (µg/l)'})
borders_latlon.plot(ax=chla_ax, lw=0.25, color='lightgray', zorder=1)

chla_fig.savefig(out_path + 'mean_chla_agg_not_nested_mean.pdf', bbox_inches = 'tight')
chla_fig.savefig(out_path + 'mean_chla_agg_not_nested_mean.png', bbox_inches = 'tight')

#%% plot variable distributions (Supplementary Figure 1)

map_table = table
borders_path = 'Y:/Dippa/Data/processing/mml/hallintorajat_10k/2020' # administrative borders
catchments_path = 'Y:/Dippa/Data/model_input/basins/catchments_with_depth/' # study catchments
parent_df_path = 'Y:/Dippa/Data/processing/valuma-aluejako/lake_catchments_parents_manual.csv' # study catchment hierarchies

borders = gpd.read_file(borders_path)
catchments = gpd.read_file(catchments_path)
parent_df = pd.read_csv(parent_df_path)
parent_df.drop(columns=['Unnamed: 0'], inplace=True)
parent_df['parent'] = parent_df.apply(lambda x: x.VPDTunnus if pd.isna(x.parent) else x.parent, axis=1)

data_parents = pd.merge(map_table, parent_df, how='inner', on='VPDTunnus')
data_means = data_parents.groupby('parent').mean()

parent_vars = parent_df.merge(data_means, how='inner', on='parent')
# join geometries for all not nested catchments
plotting_df = pd.merge(parent_vars, catchments, how='inner', left_on='parent', right_on='VPDTunnus')
plotting_df.drop(columns=['VPDTunnus_x', 'mean_depth_x'], inplace=True)
plotting_df.rename(columns={'VPDTunnus_y':'VPDTunnus', 'mean_depth_y':'mean_depth'}, inplace=True)
# convert merged df into gdf for plotting
plotting_gdf = gpd.GeoDataFrame(plotting_df)
# simplify geometry for faster plotting
plotting_gdf.loc[:,'geometry'] = plotting_gdf.geometry.simplify(5000)

plotting_gdf_latlon = plotting_gdf.to_crs('EPSG:4326')
borders_latlon = borders.to_crs('EPSG:4326')


var_list = ['erosion', 'prec_gs','prec_ws',  'temp_ws',
            'agric', 'peatland']

vmin_vars = {'erosion':0, 'prec_gs':240,'prec_ws':250, 'temp_ws':-6,
             'agric':0, 'peatland':0}

vmax_vars = {'erosion':12, 'prec_gs':320,'prec_ws':425, 'temp_ws':2,
             'agric':25, 'peatland':40}

for var in var_list:
 fig2, ax2 = plt.subplots()
 borders_latlon.plot(ax=ax2, lw=0.25, color='lightgray', zorder=1)
 plotting_gdf_latlon.plot(column=var, ax=ax2, cmap='viridis', zorder=2, vmin=vmin_vars[var], vmax=vmax_vars[var], legend=True, legend_kwds={'label':var}, missing_kwds={'color': 'lightgrey'})
 plt.show()
 fig2.savefig(out_path + var + 'aggregated_mean.pdf', bbox_inches = 'tight')
 fig2.savefig(out_path + var + 'aggregated_mean.png', bbox_inches = 'tight')
 plt.close()