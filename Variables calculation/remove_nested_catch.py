# -*- coding: utf-8 -*-
import geopandas as gpd
import pandas as pd
import numpy as np

def main():
    
    # load catchment data
    catchment_path = 'Y:/Dippa/Data/model_input/basins/catchments_with_depth'
    catchments = gpd.read_file(catchment_path)
    
    # load rf-table to find the lakes and catchments that are included in the study once
    # lakes with null variable values ave been excluded
    final_table_path = 'W:/Dippa/Data/model_output/Final_tables/rf_table.csv'
    final_table = pd.read_csv(final_table_path)
    lakes_included_ids = final_table.VPDTunnus.values
    
    # filter lakes df based on basin id's in the final table  
    catchments_included = catchments.loc[catchments.VPDTunnus.isin(lakes_included_ids), :]
    
    # make a centroid column
    catchments_included['centroid'] = catchments_included.geometry.centroid
    
    # sort df on catchment area -> start from largest cacthments when finding
    # nested cacthments
    catchments_sorted = catchments_included.sort_values(by=['Area'], ascending=False)
    catchments_sorted.reset_index(inplace=True)
    catchments_copy = catchments_sorted.copy()   
    
    # iterate through sorted dataframe, for each catchment check which
    # other catchments it contains
    
    # list for collecting all catchments that are within other catchments, for
    # test purposes
    all_contained_ids = []
    # insert a column for storing the 'parent' cathments for nested catchments
    catchments_sorted.insert(1, 'parent', np.nan)
    # iterate through catchments, starting from the largest ones, and find
    # catchments within each
    for index, row in catchments_copy.iterrows():
        if row.VPDTunnus not in all_contained_ids:
            contained_ids = check_contained(catchments_copy, row.VPDTunnus)
            if row.VPDTunnus in contained_ids: 
                contained_ids.remove(row.VPDTunnus)
            # if there are catchments within a catchment, add them to the list
            # of nested catchments and add the parent catchment id to catchments df
            if len(contained_ids) > 0:
                for item in contained_ids:
                    catchments_sorted.loc[(catchments_sorted['VPDTunnus'] == item), 'parent'] = row.VPDTunnus
                    all_contained_ids.append(item)
    # print nested catchmens for testing purposes
    #print(all_contained_ids)      

    # make a df of all the parent catchments, with the number of nested catcments
    not_nested_catch = catchments_sorted.groupby('parent').count()
    # drop extra columns, save as csv
    not_nested_catch.drop(columns=['TunnuEdit', 'Nimi', 'UNIQUEID', 'Area', 
                                   'geometry', 'centroid', 'VPDTunnus'], inplace=True)
    not_nested_catch.rename(columns={'index':'lakes_within'}, inplace=True)
    not_nested_catch.to_csv('W:/Dippa/Data/processing/valuma-aluejako/not_nested_catchments.csv')
    # save the df of all the catchments with the ids of their parent catchments
    catchments_parents = catchments_sorted.drop(columns=['index','TunnuEdit', 'Nimi', 
                                                         'UNIQUEID', 'Area', 'geometry', 'centroid'])
    catchments_parents.to_csv('W:/Dippa/Data/processing/valuma-aluejako/lake_catchments_parents.csv')
    
def check_contained(catchments_sorted, catch_id):
    # iterate through sorted dataframe, return list of catchments that are within
    # selected catchment
    catchments_within = []
    catchment = catchments_sorted.loc[catchments_sorted.VPDTunnus == catch_id]
    for index, row in catchments_sorted.iterrows():
        if row.centroid.within(catchment.iloc[0].geometry):
            catchments_within.append(row.VPDTunnus)
    return catchments_within
        
main()

