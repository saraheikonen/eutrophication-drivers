# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 13:11:46 2021

@author: sarah
"""

from selenium import webdriver
from selenium.webdriver.support.ui import Select
import geopandas as gpd
import numpy as np
import os

# give paths to catchment shapefile (basins_path), to chromedriver.exe and where
# to save catchments with depth
basins_path = 'Y:/Dippa/Data/model_input/basins/lake_catchments'
chromedriver_path = chromedriver_path = 'Y:/Dippa/Koodi local/Download data/chromedriver'
save_path = 'Y:/Dippa/Data/basins/lake_catchments_with_depth'

# if save path doesn't exist, create folder
if not os.path.exists(save_path):
    os.makedirs(save_path)

def build_query(short_id, driver):
    
    # open jarvirajapinta querybuilder
    jarviapi_url = 'http://rajapinnat.ymparisto.fi/api/jarvirajapinta/1.0/ODataQueryBuilder/'
    driver.get(jarviapi_url)
    
    # select top 1 result from dropdown
    element_top= Select(driver.find_element_by_id('top'))
    element_top.select_by_value('1')
    driver.implicitly_wait(5)
    
    # select jarvi from dropdown menu
    element_jarvi = Select(driver.find_element_by_id('entities'))
    element_jarvi.select_by_value('0')
    driver.implicitly_wait(5)
    
    # add where-condition: short lake id
    #driver.implicitly_wait(5)
    driver.find_element_by_id('addCondition').click()
    element_condition = Select(driver.find_element_by_class_name('property'))
    element_condition.select_by_value('1')
    driver.implicitly_wait(5)
    # equals
    element_equals = Select(driver.find_element_by_class_name('propertyFilter'))
    element_equals.select_by_value('0')
    driver.implicitly_wait(5)
    # short id
    driver.find_element_by_class_name('propertyFilterInput').send_keys(short_id)
    
    #select columns: mean depth in meters
    driver.find_element_by_id('addSelectCondition').click()
    short_id = driver.find_element_by_id('selectcolumn_24')
    short_id.click()
    driver.implicitly_wait(5)
    
    # search
    search_button = driver.find_element_by_id('submitQuery')
    search_button.click()
    
    # get result
    driver.implicitly_wait(5)
    mean_depth = driver.find_element_by_xpath('/html/body/div/div[2]/table/tbody/tr/td').text
    if len(mean_depth) > 0:
        depth_value = float(mean_depth)
    else:
        depth_value = np.nan
    
    # refresh page to start over
    search_button = driver.find_element_by_id('clearQuery')
    search_button.click()
    
    return depth_value    
    

def main(basins_path, chromedriver_path, save_path):
    
    # prepare basins df
    basins = gpd.read_file(basins_path)

    # make a new field for shortened VPDTunnus
    basins['VPDLyh'] = basins.apply(lambda x: x.VPDTunnus.split('_')[0], axis = 1)
    
    # set up chromedriver
    driver = webdriver.Chrome(executable_path=chromedriver_path) 
    
    # call guery function on actual df
    basins.insert(5, 'mean_depth', np.nan)
    basins.loc[:,'mean_depth'] = basins.apply(lambda x: build_query(x.VPDLyh, driver), axis = 1)

    driver.close()
    
    basins.to_file(save_path + '/basins_with_lake_depth.shp')

main(basins_path, chromedriver_path, save_path)

