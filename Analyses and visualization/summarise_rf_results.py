# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def main(results_base_path, model_area):
    
    if model_area == 'VHA':
        VHA_list = ['1', '2', '3', '4']
    else:
        VHA_list = ['']

    for VHA in VHA_list:
        # read performance and importance files + drop columns that are not needed
        if VHA != '':
            performance_path = results_base_path + VHA +'/performance/performance.csv'
            importance_path = results_base_path+ VHA + '/importance/importance.csv'
        else:
            performance_path = results_base_path +'/performance/performance.csv'
            importance_path = results_base_path + '/importance/importance.csv'
        performance = pd.read_csv(performance_path)
        performance.drop(columns=['rf_run_n', 'split_n', 'test_areas'], inplace=True)
        importance = pd.read_csv(importance_path)
        n_splits = importance.split_n.max() + 1
        importance.drop(columns=['rf_run_n', 'split_n', 'test_areas', 'actual_test_size'], inplace=True)
        
        # get performance and importance stats
        performance_stats = get_performance_stats(performance)
        importance_stats = get_importance_stats(importance, n_splits)
        
        # save stats as csv in the VHA folder
        performance_stats.to_csv(results_base_path + VHA + '/performance/stats'+VHA+'.csv')
        importance_stats.to_csv(results_base_path + VHA + '/importance/stats'+VHA+'.csv')
        
# function for calculating performance stats, returns a stats df
def get_performance_stats(performance_df):
    # df for storing stats
    variables = performance_df.columns
    stats_df = pd.DataFrame(index = variables, columns=['min', 'max', 'mean', 'stdev'])

    # calculate stats
    table_vars = stats_df.index.values
    
    for var in table_vars:
     
        var_min = performance_df[var].min()
        var_max = performance_df[var].max()
        var_mean = performance_df[var].mean()
        var_stdev = performance_df[var].std()
         
        stats_arr = np.array([var_min, var_max, var_mean, var_stdev]) 
        stats_df.loc[var,:] = stats_arr
     
    return stats_df
    
# function for calculating variable importance stats, returns a stats df
def get_importance_stats(importance_df, n_splits):
    
    # df for storing stats
    variables = importance_df['var'].unique().tolist()
    imp_stats_df = pd.DataFrame(index = variables, columns=['mean', 'stdev', 'included_prec'])
    
    # iterate through variables, for each find importance stats
    for variable in variables:
        var_subset = importance_df.loc[importance_df['var'] == variable]
        mean_imp = var_subset.importance.mean()
        std_imp = var_subset.importance.std()
        included_perc = var_subset.shape[0]/n_splits
        
        imp_stats_arr = np.array([mean_imp, std_imp, included_perc])
        imp_stats_df.loc[variable,:] = imp_stats_arr
        
    return imp_stats_df



#main()
        
    
    

    

