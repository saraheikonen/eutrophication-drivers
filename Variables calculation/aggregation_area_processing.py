# -*- coding: utf-8 -*-

import pandas as pd
import geopandas as gpd
import numpy as np

def main(lakes, final_table):
    
    # add centroid column to lakes
    lakes.insert(1, 'centroid', np.nan)
    lakes['centroid'] = lakes.geometry.centroid 
    lakes_points = lakes.set_geometry('centroid')
    
    # find the lakes that are included in the study once
    # lakes with null variable values ave been excluded
    lakes_included_ids = final_table.VPDTunnus.values
    
    # filter lakes df based on basin id's in the final table  
    lakes_included = lakes_points.loc[lakes_points.VPDTunnus.isin(lakes_included_ids)]
    
    # load mid-level catchment area data
    mid_catchments_path = 'Y:/Dippa/Data/processing/valuma-aluejako'
    mid_catchments = gpd.read_file(mid_catchments_path)
    
    # find mid level catchments that have study lakes, save as csv
    mid_catch_lakes = gpd.sjoin(lakes_included, mid_catchments, how ='inner', op='within')
    mid_catch_lakes.drop(columns=['geometry', 'centroid'], inplace=True)
    mid_catch_lakes.to_csv('Y:/Dippa/Data/processing/valuma-aluejako/paajako_with_lakes.csv')
    
    # count the number of lakes within mid level catchments, save as csv
    weighted_catch = mid_catch_lakes.groupby('PaajakoTun').count()
    mid_catch_weights = weighted_catch.loc[:,'VPDTunnus']
    mid_catch_weights.rename('count', inplace=True)
    mid_catch_weights.to_csv('Y:/Dippa/Data/processing/valuma-aluejako/paajako_weighted.csv')

    # add paajakoTun to the final table
    mid_catch_lakes_short = mid_catch_lakes.loc[:,['VPDTunnus', 'PaajakoTun']]
    final_table_paajako = pd.merge(final_table, mid_catch_lakes_short, how='inner', on='VPDTunnus')
    
    return final_table_paajako

