# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #has to be matplotlib=3.4.3
import matplotlib as mpl
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from datetime import date
import shap
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

os.chdir('C:/Users/heikons1/Git local/eutrophication-drivers/Analyses and visualization') # edit accordingly
# import own functiona
import summarise_rf_results
import aggregate_results

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


def main():
        
    # train test split and rf run parameters
    split_times = 100
    test_size = 0.2
    rf_seed_range = 1
    importance_type = 'both' #'both' # or 'shap' or #'perm'
    model_area = 'global' # global or VHA (river basin district)
    
    # rf input data paths
    data_path = 'Y:/Dippa/Data/model_output/Final_tables/rf_table.csv' # variable data
    parent_df_path = 'Y:/Dippa/Data/processing/valuma-aluejako/lake_catchments_parents_manual.csv'
    
    # saving options
    save_base_path = 'C:/Users/heikons1/Git local/eutrophication-drivers/' # edit accordingly
    save = True #True False -> whether results are saved or not
    
    # no need to modify from here on
    if model_area == 'global':
        VHA_list = ['']
        max_tree_depth = 5
    if model_area == 'VHA':
        VHA_list = [1,2,3,4]
        max_tree_depth = 4
    
    for VHA in VHA_list:
        
        VHA_str = str(VHA)
        print(VHA_str)
        # identifiers for saving results
        today = date.today()
        str_date = today.strftime("%d%m%y")
        save_path = save_base_path + 'rf_results' +str_date+model_area+VHA_str
    
        # making folders for saving results & saving
        if save:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            if not os.path.exists(save_path + '/performance'):
                os.mkdir(save_path + '/performance')
            if not os.path.exists(save_path + '/importance'):
                os.mkdir(save_path + '/importance')
            if not os.path.exists(save_path + '/results'):
                os.mkdir(save_path + '/results')
            if not os.path.exists(save_path + '/figures'):
                os.mkdir(save_path + '/figures')
    
        if model_area == 'global':
            # load model input data        
            data_all = pd.read_csv(data_path, sep=',')
            
            # load parent catchment info 
            parent_df = pd.read_csv(parent_df_path)
            parent_df.drop(columns=['Unnamed: 0'], inplace=True)
            
            # join parent info to data df
            data_parents = pd.merge(data_all, parent_df, how='inner', on='VPDTunnus')
            data_parents['parent'] = data_parents.apply(lambda x: x.VPDTunnus[:4] if pd.isna(x.parent) else x.parent[:4], axis=1)
            
            data_averaged = data_parents
        else:
            # load model input data for specified VHA (= river basin district)       
            data_all = pd.read_csv(data_path, sep=',')
            data_VHA = data_all.loc[data_all['VHA'] == VHA]
            
            # load parent catchment info
            parent_df = pd.read_csv(parent_df_path)
            parent_df.drop(columns=['Unnamed: 0'], inplace=True)
            
            # join parent info to data df
            data_parents = pd.merge(data_VHA, parent_df, how='inner', on='VPDTunnus')
            data_parents['parent'] = data_parents.apply(lambda x: x.VPDTunnus[:4] if pd.isna(x.parent) else x.parent[:4], axis=1)
            
            data_averaged = data_parents
        
        # lists for shap plotting
        X_train_list = []
        shap_values_list = []
        shap_values_enc_list =[]
        
        # lists for storing performance and permutation importamce results
        performance_list = []
        importance_list = []
        
        # get train test split and actual test size
        for split_n in range(split_times):
            print(split_n)
            actual_test_size, test_ids = get_split(data_averaged, split_times, test_size, model_area, parent_df_path)
            # preprocess and split dataset
            X_train_enc, X_train, X_test_enc, X_test, y_train, y_test, test_set, tyyppi_num = preprocess_data(test_ids, data_averaged, model_area)
            # store X-train for plotting
            X_train_list.append(X_train)
            # run random forest & variable importance calculations
            performance_results, importance_results, shap_values_enc, shap_values_comb = run_rf(
                split_n, rf_seed_range, actual_test_size, X_train_enc, X_test_enc, X_train, 
                y_train, y_test, test_set, importance_type, tyyppi_num, save, save_path, max_tree_depth)
            # combine performance and importance results
            performance_list.append(performance_results)
            importance_list.append(importance_results)
            shap_values_list.append(shap_values_comb)
            shap_values_enc_list.append(shap_values_enc)
        
        # combine all performance and permutation importance results
        performance_df = pd.concat(performance_list)
        importance_df = pd.concat(importance_list)
        
        # combine results from shap importance + plot
        if (importance_type == 'both') or (importance_type == 'shap'):
            # combining shapley values into one array & X_test's into one df
            shap_values_all_runs = np.concatenate(shap_values_list)
            shap_values_all_runs_enc = np.concatenate(shap_values_enc_list)
            all_samples = pd.concat(X_train_list)
        
            # shap plots
            plot_shap(shap_values_all_runs, all_samples, save, save_path)
            
        # save performance & importance results
        if os.path.isdir(save_path + '/performance'):
            performance_df.to_csv(save_path + '/performance/performance.csv', index=False)
        else:
            print('Performance files not saved')
        if os.path.isdir(save_path + '/importance'):
            importance_df.to_csv(save_path + '/importance/importance.csv', index=False)
        else:
            print('Importance files not saved')
    
    if save:
        
        # summarize model results
        if model_area == 'VHA':
            results_base_path = save_path[:-1]
        else:
            results_base_path = save_path
        summarise_rf_results.main(results_base_path, model_area)
        
        # plot permutation importance after summarization
        if (importance_type == 'both') or (importance_type == 'permutation'):
            
            plot_permutation_importance(results_base_path, model_area)
            
        # produce aggregated plots for global results
        if model_area == 'global': 
            aggregate_results.main(save_path + '/')
                
            
# get train test split indexig
def get_split(data, split_times, test_size, model_area, split_df_path):
    
    # accepted size interval for test set
    test_lower_lim = (test_size - 0.025)*data.shape[0]
    test_upper_lim = (test_size + 0.025)*data.shape[0]
    
    # load splitting dataframe
    split_df = pd.read_csv(split_df_path)
    split_df['parent'] = split_df.apply(lambda x: x.VPDTunnus[:4] if pd.isna(x.parent) else x.parent[:4], axis=1)
    
    # subset parent catchments based on VHA (or globally, get all unique values)
    subset_split_ids = data.parent.unique()
    split_df_subset = split_df.loc[split_df.parent.isin(subset_split_ids)]
    split_df_subset_counts = split_df_subset.groupby('parent').count()
    split_df_subset_counts.drop(columns=['Unnamed: 0'], inplace=True)
    split_df_subset_counts.rename(columns={'VPDTunnus':'count'}, inplace=True)
        
    split_success = False
    test_lakes_no = 0
    test_ids = []

    # iterate over shuffled mid level catchments df and sum the number of 
    # lakes in the catchment until the desired test set size has been reached
    while not split_success:
        split_df_shuffle = shuffle(split_df_subset_counts)
        split_df_shuffle.reset_index(inplace=True)
        test_lakes_no = 0
        test_ids = []
        for index, row in split_df_shuffle.iterrows():
            if (test_lakes_no >= test_lower_lim) and (test_lakes_no <= test_upper_lim):
                split_success = True
                #print(test_ids)
                break
            elif test_lakes_no > test_upper_lim:
                break
            else:
                test_lakes_no += row['count']
                test_ids.append(row.parent) 
                                    
    # get actual test set size:
    actual_test_size = test_lakes_no/len(data.VPDTunnus.unique())
                        
    return actual_test_size, test_ids

# train test split & one-hot encoding categorical variable
def preprocess_data(test_ids, data, model_area):  

    data_copy = data.copy()    
        
    #df for rf, no unnecessary data columns
    data_rf = data_copy.drop(columns=['VHA','mineral_forest'])
    
    cats = {' (Rh)':'High', ' (Ph)':'Moderate', ' (Mh)':'High', ' (Vh)':'Low', '(MRh)':'High', '(MVh)':'Low', ' (Kh)':'Moderate',
    ' (Lv)':'Moderate', '(SVh)':'Low', ' (Sh)':'Moderate', ' (Rr)':'High', ' (Rk)':'Low'}
    lakes_remap = data.loc[:,'Tyyppi_lyh'].map(cats)
    data_rf.drop(columns=['Tyyppi_lyh'], inplace=True)
    data_rf.insert(0,'Tyyppi_lyh',lakes_remap)
    
    tyyppi_num = len(data_rf.Tyyppi_lyh.unique()) - 1
    
    #save the test set from all data for storing results
    test_set = data_copy.loc[data_rf.parent.isin(test_ids)].copy() #parent ->VPDTunnus
    test_set.reset_index(inplace=True)
    test_set.drop(columns=['index'], inplace=True)
    
    # one hot encode lake type column
    enc = OneHotEncoder(handle_unknown='error', drop='first')
    enc_df = pd.DataFrame(enc.fit_transform(data_rf[['Tyyppi_lyh']]).todense())
    enc_df.columns = enc.get_feature_names_out()
    data_enc = pd.concat([enc_df, data_rf], axis=1)
    data_enc.drop(columns=['Tyyppi_lyh'], inplace=True)

    #split rf data into train and test sets based on parent id
    #split data into train and test
    test_set_rf = data_rf.loc[data_rf.parent.isin(test_ids)].copy() #parent ->VPDTunnus
    train_set_rf = data_rf.loc[~data_rf.parent.isin(test_ids)].copy() #parent ->VPDTunnus
    test_set_rf.reset_index(inplace=True)
    train_set_rf.reset_index(inplace=True)
    test_set_rf.drop(columns=['index', 'PaajakoTun', 'parent', 'VPDTunnus'], inplace=True)
    train_set_rf.drop(columns=['index', 'PaajakoTun', 'parent','VPDTunnus',], inplace=True)
        
    # split into predictor and target variables
    X_train = train_set_rf.drop(columns=['Concentration'])
    X_test = test_set_rf.drop(columns=['Concentration'])
    
    # same as above for encoded data
    test_set_rf_enc = data_enc.loc[data_enc.parent.isin(test_ids)].copy() #parent ->VPDTunnus
    train_set_rf_enc = data_enc.loc[~data_enc.parent.isin(test_ids)].copy() #parent ->VPDTunnus
    test_set_rf_enc.reset_index(inplace=True)
    train_set_rf_enc.reset_index(inplace=True)
    test_set_rf_enc.drop(columns=['index', 'PaajakoTun', 'parent', 'VPDTunnus'], inplace=True)
    train_set_rf_enc.drop(columns=['index', 'PaajakoTun', 'parent','VPDTunnus',], inplace=True)
        
    # split into predictor and target variables
    X_train_enc = train_set_rf_enc.drop(columns=['Concentration'])
    X_test_enc = test_set_rf_enc.drop(columns=['Concentration'])
    
    y_train = train_set_rf.loc[:, 'Concentration']
    y_test = test_set_rf.loc[:, 'Concentration']
    
    return X_train_enc, X_train, X_test_enc, X_test, y_train, y_test, test_set, tyyppi_num

# fit rf regressor for a single train test split, calculate performance metrics
def run_rf(split_n, rf_seed_range, actual_test_size, X_train_enc, X_test_enc, X_train, y_train, y_test, test_set, importance_type, tyyppi_num, save, save_path, max_tree_depth):
    
    # initialize dataframes/lists for storing results
    performance = pd.DataFrame(columns=['split_n', 'rf_run_n', 'actual_test_size','test_areas', 'r2_train', 'r2_test', 'rmse_train', 'rmse_test'])
    var_importance = pd.DataFrame(columns=['split_n', 'rf_run_n', 'actual_test_size','test_areas', 'var', 'importance', 'std'])
    shap_values_rf_runs = np.array([])
    shap_values_rf_runs_enc = np.array([])
    
    # list of test ids (paajako tunnus)
    test_ids = test_set.parent.unique().tolist()
    
    # run random forest regrression with a range of seeds
    for rf_run_n in range(rf_seed_range):
        # initialize random forest regressor
        rf = RandomForestRegressor(n_estimators=300, random_state = rf_run_n, max_depth=max_tree_depth) 
        # run preprocessigng steps and fit regressor
        rf.fit(X_train_enc, y_train)
    
        # test predicot on test and train data and collect metrics
        y_pred = rf.predict(X_test_enc)
        y_pred_train = rf.predict(X_train_enc)
     
        r2_test = metrics.r2_score(y_test, y_pred)
        r2_train = metrics.r2_score(y_train, y_pred_train)
     
        rmse_test = metrics.mean_squared_error(y_test, y_pred, squared=False)
        rmse_train = metrics.mean_squared_error(y_train, y_pred_train, squared=False)
    
        #store rf performance stats in a df
        run_performance = pd.DataFrame([[split_n, rf_run_n, actual_test_size, test_ids, r2_train, r2_test, rmse_train, rmse_test]],
                                       columns=['split_n', 'rf_run_n', 'actual_test_size', 'test_areas', 'r2_train', 'r2_test', 'rmse_train', 'rmse_test'])
        # append to collective performance df
        performance = performance.append(run_performance)
        
        # combine predictions, ground truth, their difference and predictor values
        test_set['pred'] = pd.Series(y_pred)
        test_set['diff'] = test_set.apply(lambda x: x.pred - x.Concentration, axis = 1) 
        test_set['abs_diff'] = test_set['diff'].abs()
        # save test set and predictions for it as csv
        if save and os.path.isdir(save_path + '/results'):
            test_set.to_csv(save_path +'/results/split_' + str(split_n) +'.csv', index=False)
        #else:
            #print('RF run results not saved')
   
        # importance calculations                   
        if importance_type == 'both':
            #print('both')
            # calculate permutation importance & append to summarizing importances df
            var_importance = var_importance.append(get_permutation_importance(split_n, rf_run_n, actual_test_size, test_ids, X_train, y_train, max_tree_depth))
            # calculate shap importance & append to shap values array    
            shap_values_enc, shap_values_comb = get_shap_values(X_train_enc,tyyppi_num, rf)
            if (shap_values_rf_runs.size == 0) and (shap_values_rf_runs_enc.size == 0):
                shap_values_rf_runs = shap_values_comb
                shap_values_rf_runs_enc = shap_values_enc
            else:
               shap_values_rf_runs = np.hstack(shap_values_rf_runs, shap_values_comb)
               shap_values_rf_runs_enc = np.hstack(shap_values_rf_runs_enc, shap_values_enc)
            
        elif importance_type == 'shap':
            #print('shap')
            # calculate shap importance & append to shap values array
            shap_values_enc, shap_values_comb = get_shap_values(X_train_enc, tyyppi_num, rf)
            if (shap_values_rf_runs.size == 0) and (shap_values_rf_runs_enc.size == 0):
                shap_values_rf_runs = shap_values_comb
                shap_values_rf_runs_enc = shap_values_enc
            else:
               shap_values_rf_runs = np.hstack(shap_values_rf_runs, shap_values_comb)
               shap_values_rf_runs_enc = np.hstack(shap_values_rf_runs_enc, shap_values_enc)
            
        else:
            #print('perm')
            # calculate permutation importance & append to summarizing importances df
            var_importance = var_importance.append(get_permutation_importance(split_n, rf_run_n, actual_test_size, test_ids, X_train, y_train, max_tree_depth))

    print(performance[['r2_train', 'r2_test', 'rmse_train', 'rmse_test']]) 
    
    return performance, var_importance, shap_values_rf_runs_enc, shap_values_rf_runs

# shap values for a rf regressor fitted with a single train test split  
def get_shap_values(X_train_enc, tyyppi_num, regressor):
    # initialize
    shap.initjs()
    explainer = shap.TreeExplainer(regressor)
    # get shap values
    shap_values = explainer.shap_values(X_train_enc, approximate=False, check_additivity=True)
    # combine shapley values for lake types = columns 0-11
    lake_combined = shap_values[:,:tyyppi_num].sum(axis=1).reshape(-1,1)
    shap_values_nolake = shap_values[:,tyyppi_num:]
    shap_values_comb = np.hstack((lake_combined, shap_values_nolake))
    #shap_values_comb = shap_values
    return shap_values, shap_values_comb

# shap waterfall plot
def make_shap_waterfall_plot(shap_values, features, save, save_path, num_display=11):
    column_list = features.columns
    feature_ratio = (np.abs(shap_values).sum(0) / np.abs(shap_values).sum()) * 100
    column_list = column_list[np.argsort(feature_ratio)[::-1]]
    feature_ratio_order = np.sort(feature_ratio)[::-1]
    cum_sum = np.cumsum(feature_ratio_order)
    column_list = column_list[:num_display]
    feature_ratio_order = feature_ratio_order[:num_display]
    cum_sum = cum_sum[:num_display]
    
    rename_dict = {'prec_ws':'Winter season precipitation',
                   'prec_gs':'Growing season precipitation',
                   'temp_ws':'Winter season temperature',
                   'temp_gs':'Growing season temperature',
                   'agric':'Agricultural area %', 'water':'Lake area %',
                   'built': 'Built area %', 'peatland': 'Peatland area %',
                   'mean_depth': 'Mean depth', 'Tyyppi_lyh': 'Natural eutrophication level',
                   'erosion': 'Field erosion risk'
                   }
    columns_rename = column_list.map(rename_dict)
    
    num_height = 0
    if (num_display >= 20) & (len(columns_rename) >= 20):
        num_height = (len(columns_rename) - 20) * 0.4
        
    fig, ax1 = plt.subplots(figsize=(8, 8 + num_height))
    ax1.plot(cum_sum[::-1], columns_rename[::-1], c='blue', marker='o')
    ax2 = ax1.twiny()
    ax2.barh(columns_rename[::-1], feature_ratio_order[::-1], alpha=0.6)
    
    ax1.grid(True)
    ax2.grid(False)
    ax1.set_xticks(np.arange(0, round(cum_sum.max(), -1)+1, 10))
    ax2.set_xticks(np.arange(0, round(feature_ratio_order.max(), -1)+1, 10))
    ax1.set_xlabel('Cumulative Ratio')
    ax2.set_xlabel('Composition Ratio')
    ax1.tick_params(axis="y", labelsize=13)
    plt.ylim(-1, len(column_list)) 
    
    if save and os.path.isdir(save_path + '/figures'):
        plt.savefig(save_path + '/figures'+ '/shap_waterfall.pdf', bbox_inches = 'tight')
        plt.savefig(save_path + '/figures'+'/shap_waterfall.png', bbox_inches = 'tight')
    else:
        print('SHAP waterfall plot not saved')
    #plt.close()
    

# shap summary and dependence plots for all model runs    
def plot_shap(shap_values_all_runs, all_samples, save, save_path):
    
    # remap lake types (=categorical data) to numeric codes and add to plotting df
    #codes = {' (Rh)':0, ' (Ph)':1, ' (Mh)':2, ' (Vh)':11, '(MRh)':3, '(MVh)':10, ' (Kh)':4,
    #' (Lv)':9, '(SVh)':8, ' (Sh)':5, ' (Rr)':6, ' (Rk)':7}
    codes = {'Low':0, 'Moderate':1, 'High':2}
    lakes_remap = all_samples.loc[:,'Tyyppi_lyh'].map(codes)
    samples_plot = all_samples.drop(columns=['Tyyppi_lyh'])
    samples_plot.insert(0,'Tyyppi_lyh',lakes_remap)
    
    shap.plots.initjs()
    
    #remap varible names for plotting
    var_names = pd.Index(samples_plot.columns.values)
    
    rename_dict = {'prec_ws':'Winter season precipitation',
                   'prec_gs':'Growing season precipitation',
                   'temp_ws':'Winter season temperature',
                   'temp_gs':'Growing season temperature',
                   'agric':'Agricultural area %', 'water':'Lake area %',
                   'built': 'Built area %', 'peatland': 'Peatland area %',
                   'mean_depth': 'Mean depth', 'Tyyppi_lyh': 'Natural eutrophication level',
                   'erosion': 'Field erosion risk'
                   }
    var_rename = var_names.map(rename_dict)
    var_rename_list = var_rename.to_list()
    
    # summary violin plot
    sum_fig, sum_ax = plt.subplots()
    shap.summary_plot(shap_values_all_runs, samples_plot, feature_names = var_rename_list, show=False, cmap=plt.get_cmap("viridis"))
    ax = plt.gca()
    ax.set_xlim(-6,12)

    if save and os.path.isdir(save_path + '/figures'):
        plt.savefig(save_path + '/figures'+'/shap_summary.pdf', bbox_inches = 'tight')
        plt.savefig(save_path + '/figures'+'/shap_summary.png', bbox_inches = 'tight')
    else:
        print('SHAP summary plot not saved')
    
    # interaction dependance plots for the four most important variables (largest mean effect)
    num_top_features = 4
    for i in range(num_top_features):
       # get top feature
       feature_ind = abs(shap_values_all_runs).mean(axis=0).argsort()[-(i+1)]
       # make dependance plots
       shap.dependence_plot(feature_ind, shap_values_all_runs, samples_plot, xmin="percentile(1)", xmax="percentile(99)",show=False, color="dimgrey", x_jitter = 0.5, dot_size=2, feature_names = var_rename_list, interaction_index=None)
       if feature_ind == 0:
           plt.xticks(ticks=[0,1,2], labels=['Low', 'Moderate', 'High'], rotation=45)
       if save and os.path.isdir(save_path + '/figures'):
           plt.savefig(save_path + '/figures'+'/shap_dependence_0199_onevar_'+str(i)+'.pdf', bbox_inches = 'tight')
           plt.savefig(save_path + '/figures'+'/shap_dependence_0199_onevar_'+str(i)+'.png', bbox_inches = 'tight')
       else:
           print('SHAP dependence plots not saved')
        
    make_shap_waterfall_plot(shap_values_all_runs, samples_plot, save, save_path)
    
# calculates permutation importance for a rf regressor fitted with a single train test split    
def get_permutation_importance(split_n, rf_run_n, actual_test_size, test_ids, X_train, y_train, max_tree_depth):

    # make a new preprocessing & random forest regressor
    # pipeline because otherwise permutation importance
    # can't handle the categorical variable as an entity but instead
    # as separate one-hot encoded variables
    
    categorical_columns = ['Tyyppi_lyh']
    numerical_columns = ['erosion','prec_gs', 'prec_ws', 'temp_gs', 'temp_ws',
        'agric','water', 'peatland','built','mean_depth']

    categorical_pipe = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='error', drop='first'))
        ])
    
    preprocessing = ColumnTransformer([
        ('cat', categorical_pipe, categorical_columns),
        ('num', 'passthrough', numerical_columns)
        ])
        
    rf = Pipeline([
        ('preprocess', preprocessing),
        ('classifier', RandomForestRegressor(n_estimators=300, random_state = rf_run_n, max_depth=max_tree_depth))
        ])
    
    rf.fit(X_train, y_train)  

    var_importance = pd.DataFrame(columns=['split_n', 'rf_run_n', 'actual_test_size','test_areas', 'var', 'importance', 'std'])
    
    # calculate permutation importance
    r = permutation_importance(rf, X_train, y_train, n_repeats=10, random_state=rf_run_n)
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            # store run importances in a df
            imp_stats = pd.DataFrame(
                [[split_n, rf_run_n, actual_test_size, test_ids, X_train.columns.values[i], r.importances_mean[i], r.importances_std[i]]],
                columns = ['split_n', 'rf_run_n', 'actual_test_size', 'test_areas', 'var', 'importance', 'std']
                )
            # append to summarizing importances df
            var_importance = var_importance.append(imp_stats)
                
    return var_importance


def plot_permutation_importance(results_base_path, model_area):
    
    if model_area == 'VHA':
        VHA_list = ['1', '2', '3', '4']
    else:
        VHA_list = ['']

    for VHA in VHA_list:
        # read performance and importance files + drop columns that are not needed
        if VHA != '':
            importance_path = results_base_path+ VHA + '/importance/stats'+ VHA +'.csv'
        else:
            importance_path = results_base_path + '/importance/stats.csv'
    
        var_importance = pd.read_csv(importance_path)
        rename_dict = {'prec_ws':'Winter season precipitation',
                       'prec_gs':'Growing season precipitation',
                       'temp_ws':'Winter season temperature',
                       'temp_gs':'Growing season temperature',
                       'agric':'Agricultural area %', 'water':'Lake area %',
                       'built': 'Built area %', 'peatland': 'Peatland area %',
                       'mean_depth': 'Mean depth', 'Tyyppi_lyh': 'Natural eutrophication level',
                       'erosion': 'Field erosion risk'
                       }
        
        var_importance['rename'] = var_importance['Unnamed: 0'].map(rename_dict)
        var_importance['imp_perc'] = var_importance['mean']*100
        var_importance_sorted = var_importance.sort_values(by='imp_perc')
        
        
        imp_fig, imp_ax = plt.subplots()
        var_importance_sorted.plot.barh(x='rename', y='imp_perc', ax=imp_ax, cmap='Paired')
        plt.ylabel('')
        plt.xlabel('Percent reduction in model performance')
        imp_fig.savefig(results_base_path + VHA +'/figures/permutation_importance_' + model_area + VHA +'.pdf', bbox_inches = 'tight')
        imp_fig.savefig(results_base_path + VHA +'/figures/permutation_importance_' + model_area + VHA +'.png', bbox_inches = 'tight')


main()