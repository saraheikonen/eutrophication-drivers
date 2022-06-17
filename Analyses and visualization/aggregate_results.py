# -*- coding: utf-8 -*-
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import pearsonr
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

def main(results_base_path):
    
    # read results data into a dataframe
    # 1. read catchments geometries and parent info
    catchments_path = 'Y:/Dippa/Data/model_input/basins/catchments_with_depth'
    parent_df_path = 'Y:/Dippa/Data/processing/valuma-aluejako/lake_catchments_parents_manual.csv' 

    catchments = gpd.read_file(catchments_path)
    parent_df = pd.read_csv(parent_df_path)
    parent_df.drop(columns=['Unnamed: 0'], inplace=True)
    
    parent_catchments = catchments.merge(parent_df, on='VPDTunnus', how='inner')
    parent_catchments.loc[:,'parent'] = parent_catchments.apply(lambda x: x.VPDTunnus if pd.isna(x.parent) else x.parent, axis=1)
    
    # 2. read all results files and combine into a df + combine with parent info
    n_files = 100
    result_df_list = []
    for i in range(n_files):
        result_file_path = results_base_path + 'results/split_' + str(i) + '.csv'
        result = pd.read_csv(result_file_path)
        result_df_list.append(result)
    results_combined = pd.concat(result_df_list)
    # drop original parent column because it's an edited field and wrong dtype
    results_combined.drop(columns=['parent'], inplace=True)
    # combine with new parent info
    results_parents = results_combined.merge(parent_df, on='VPDTunnus', how='inner')
    results_parents.loc[:,'parent'] = results_parents.apply(lambda x: x.VPDTunnus if pd.isna(x.parent) else x.parent, axis=1)
    # results parent catchment ids:
    result_ids = results_parents.VPDTunnus.unique().tolist()
        
    # 3. drop unnecessary columns
    results_clean = results_parents.drop(columns=['erosion',
        'prec_gs', 'prec_ws', 'temp_gs', 'temp_ws',
        'built','peatland', 'mineral_forest'
        ])
    # remap lake types based on naturally occuring chl-a
    cats = {' (Rh)':2, ' (Ph)':1, ' (Mh)':2, ' (Vh)':0, '(MRh)':2, '(MVh)':0, ' (Kh)':1,
    ' (Lv)':1, '(SVh)':0, ' (Sh)':1, ' (Rr)':2, ' (Rk)':0}
    lakes_remap = results_clean.loc[:,'Tyyppi_lyh'].map(cats)
    results_clean.drop(columns=['Tyyppi_lyh'], inplace=True)
    results_clean.insert(0,'Tyyppi_lyh',lakes_remap)
    
    # select catchment geometries that are included in results
    catchments_included = parent_catchments.loc[parent_catchments['VPDTunnus'].isin(result_ids)].copy()
    
    # aggregation levels
    agg_levels = [0, 20, 4, 2]
    # agg_levels = [0]
    # numbers above determine how many characters from VPDTunnus
    # are considered in aggregation, the levels of subdivision
    # are encoded to the VPDTunnus. Here, 0 means no aggregation.
    # First 2 characters:
    # main river basin districts, first 4: sub division level 1,
    # first 20: full VPDTunnus + a few extra chars just to be sure.
    agg_results = []    
    
    # 4. take means of results, grouped by aggregation level
    for level in agg_levels:
        
        # take the required characters of parent-variable for results and catchments
        results_copy = results_clean.copy()
        results_copy['parent'] = results_copy.apply(lambda x: x.VPDTunnus[:level] if pd.isna(x.parent) else x.parent[:level], axis=1)
        catchments_copy = catchments_included.copy()
        catchments_copy['parent'] = catchments_copy.apply(lambda x: x.VPDTunnus[:level] if pd.isna(x.parent) else x.parent[:level], axis=1)
       

        if level > 0:        

            # take a mean of results, grouped by the edited parent-variable + count within group
            results_agg_mean = results_copy.groupby(['parent','Tyyppi_lyh'], as_index=False).mean()
            results_agg_counts = results_copy.groupby(['parent','Tyyppi_lyh'], as_index=False).count()
            results_agg_mean.insert(1,'lake_count',results_agg_counts.iloc[:,1])
            
            # dissolve catchment geometries within aggregation level
            catchments_dissolved = catchments_copy.dissolve(by='parent')
    
            # merge with catchment geometries based on edited parent-variable and convert to geodataframe for plotting
            results_agg_mean_geo = gpd.GeoDataFrame(results_agg_mean.merge(catchments_dissolved, on='parent', how='inner'))
            results_agg_mean_geo.loc[:,'geometry'] = results_agg_mean_geo.geometry.simplify(5000)
            
        else:
            results_agg_mean = results_copy
            results_agg_mean.insert(1,'lake_count',1)

            
        if level > 0:    # only plot aggregated results
            # 5. plot absolute prediction error in Paajako ares
            plot_errors(results_agg_mean_geo, 'abs_diff', 'Absolute error μg/l', level, results_base_path)
            # plot aggregation areas for Supplementary Figure 
            plot_aggregation_levels(results_agg_mean_geo, level, results_base_path)
           
        # 6. fit line to errors, measure linear correlation
        r, r_2, p = linear_fit(results_agg_mean, ['pred', 'Concentration'], level, results_base_path)
        
        # 7. store results
        agg_level_results = np.array([level, r, r_2, p])
        agg_results.append(agg_level_results)
        
    # combine all stats from aggregation levels
    agg_results = pd.DataFrame(np.vstack(agg_results), columns=['agg_interval', 'Pearson_r', 'R_2', 'p_value'])
    #print(agg_results)
    agg_results.to_csv(results_base_path + 'agg_results.csv', index=False)
    
    
# function for plotting a variable within a geometry
def plot_errors(error_df, var, label, level, results_base_path):
    # load additional plotting data
    borders_path = 'Y:/Dippa/Data/processing/mml/hallintorajat_10k/2020'
    borders = gpd.read_file(borders_path)
    
    borders_latlon = borders.to_crs('EPSG:4326')
    error_df_latlon = error_df.to_crs('EPSG:4326')
    
    fig, ax = plt.subplots()
    error_df_latlon.plot(column=var, ax=ax, cmap='Reds', vmin = 0, vmax = 15, legend=True, 
                         legend_kwds={'label':label}, missing_kwds={'color': 'lightgrey'}, zorder=2)
    borders_latlon.plot(ax=ax, lw=0.25, color='lightgray', zorder=1)
    plt.show()
    fig.savefig(results_base_path + 'abs_diff_'+str(level)+'_samescale_latlon.pdf', bbox_inches = 'tight')
    fig.savefig(results_base_path + 'abs_diff_'+str(level)+'_samescale_latlon.png', bbox_inches = 'tight')
    plt.close()
    
def plot_aggregation_levels(error_df, level, results_base_path):
    # load additional plotting data
    borders_path = 'Y:/Dippa/Data/processing/mml/hallintorajat_10k/2020'
    borders = gpd.read_file(borders_path)
    
    borders_latlon = borders.to_crs('EPSG:4326')
    error_df_latlon = error_df.to_crs('EPSG:4326')
    
    fig, ax = plt.subplots()
    # background info map
    error_df_latlon.plot(ax=ax, zorder=2,
                         column='parent', cmap='tab20', edgecolor='dimgray', linewidth=0.4)
    borders_latlon.plot(ax=ax, lw=0.25, color='lightgray', zorder=1)
    plt.show()
    fig.savefig(results_base_path + 'area_borders'+str(level)+'_.pdf', bbox_inches = 'tight')
    fig.savefig(results_base_path + 'area_borders_'+str(level)+'_.png', bbox_inches = 'tight')
    plt.close()
    
# function for fitting a line into a dataset + measuring correlation metrics
# and a scatter plot for visualization  
def linear_fit(error_df, fit_columns, level, results_base_path):
    
    agg_level_names = {0: 'No aggregation', 20:'Parent', 4:'1st subdivision', 2:'Main river basins'}
    # correlation stats for aggregated predictions vs ground truths
    r, p = pearsonr(error_df[fit_columns[0]], error_df[fit_columns[1]])
    r_2 = r**2
    #print(r)
    #print(r_2)
    #print(p)
    #print('\n')
    
    # scatterplot of predictions vs ground truth + trendlines
    # trendline of pred vs truth
    z = np.polyfit(error_df[fit_columns[0]], error_df[fit_columns[1]], 1)
    h = np.poly1d(z)
    
    error_df.insert(1, 'trendline', h(error_df[fit_columns[0]]))
    
    # Pearson stats legend for plots
    if p < 0.01:
        axis_text = 'Pearson r: {0:3.2f}'.format(r) + '\n'+'p-value < 0.01'+'\n'+'Level: '+agg_level_names[level]
    elif p < 0.05:
        axis_text = 'Pearson r: {0:3.2f}'.format(r) + '\n'+'p-value < 0.05'+'\n'+'Level: '+agg_level_names[level]
    else:
        axis_text = 'Pearson r: {0:3.2f}'.format(r) + '\n'+'not statistcially significant < 0.01'+'\n'+'Level: '+agg_level_names[level]

    # plots
    
    # categorical cmap 
    cat_cmap = mpl.cm.get_cmap('viridis', 3)
    fig1, ax1 = plt.subplots()
    error_df.plot.scatter(x=fit_columns[0], y=fit_columns[1], ax=ax1, s=7, zorder=1, c='Tyyppi_lyh', colormap=cat_cmap)
    error_df.plot.line(x=fit_columns[0], y='trendline', c='k', lw=0.25, zorder=2, ax=ax1)
    ax1.text(0.02, 0.75, axis_text,
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax1.transAxes,
        color='black', fontsize=10)
    ax1.set_xlim(0,30)
    ax1.set_xlabel('Model estimate of chl-a concentration (μg/l)')
    ax1.set_ylabel('Observed chl-a concentration (μg/l)')
    cax = fig1.get_axes()[1]
    cax.set_ylabel('Natural euthophication level (0,1,2 = Low, Moderate, High)')
    
    plt.show()

    fig1.savefig(results_base_path + 'pred_test_scatter_color1_'+str(level)+'.pdf', bbox_inches = 'tight')
    fig1.savefig(results_base_path + 'pred_test_scatter_color1_'+str(level)+'.png', bbox_inches = 'tight')
    plt.close()

    
    # residual histogram
    fig5, ax5 = plt.subplots()
    error_df['diff'].plot.hist(bins=15, ax=ax5)
    ax5.text(0.02, 0.75, ('Level: '+agg_level_names[level]),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax5.transAxes,
        color='black', fontsize=10)
    ax5.set_xlabel('Absolute error (μg/l)')
    plt.show()
    fig5.savefig(results_base_path + 'residual_hist_'+str(level)+'.pdf', bbox_inches = 'tight')
    fig5.savefig(results_base_path + 'residual_hist_'+str(level)+'.png', bbox_inches = 'tight')
    
    return r, r_2, p
    
    
    
    
    
    
    
    

