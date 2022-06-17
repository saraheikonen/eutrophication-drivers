# -*- coding: utf-8 -*-

import pandas as pd
import geopandas as gpd

#%% Download a-chlorophyll observations and lake data

chl_file = r'Y:\Dippa\Data\processing\Klorofylli\jarviklorofylli_copy.csv'
chl_df = pd.read_csv(chl_file, sep = ';')

lakes_file = r'Y:\Dippa\Data\VHS_vesialueet_2016\Jarvet'
lakes = gpd.read_file(lakes_file)

# lakes outside Finnish borders (or with catchments outside Finnish borders)
exclude_lakes_path = r'Y:\Dippa\Data\VHS_vesialueet_2016\lakes_outside_borders.csv'
exclude_lakes = pd.read_csv(exclude_lakes_path, sep = ',')

#%% Pre-process lakes df

# get a list of unique values in Area column -> number of lakes
lakes_list = chl_df.Area.unique()

# make a subset of lakes df based on lakes with chl observations
lakes_subset = lakes[lakes.VPDTunnus.isin(lakes_list)]

# make subset of lakes whose basins are within national boeders
exclude_list = exclude_lakes['0'].values
lakes_finland = lakes_subset[~lakes_subset.VPDTunnus.isin(exclude_list)]

#%%
# select needed columns from lakes df
lakes_filtered = lakes_finland.filter(['VPDTunnus', 'Tyyppi', 'VHATunnusN', 'geometry'])

# save filtered lake df as shapefile
lakes_filtered.to_file(r'Y:\Dippa\Data\VHS_vesialueet_2016\Jarvet_filter\jarvi_filter.shp')

#also save as csv
lakes_df = pd.DataFrame(lakes_filtered.drop(columns='geometry'))
lakes_df.to_csv(r'Y:\Dippa\Data\VHS_vesialueet_2016\Jarvet_filter\jarvi_filter.csv')

#%% Pre-process chl df

# change date column to datetime format
chl_df['Date'] = pd.to_datetime(chl_df['Date'], format='%d.%m.%Y')

# rename concentration column
chl_df.rename(columns={'Value (mikrogramma/litra)':'Concentration'}, inplace = True)

#%% Chl only medians dataframe + add month and year columns

chl_df['Year'] = chl_df['Date'].dt.year
chl_df['Month'] = chl_df['Date'].dt.month

chl_medians = chl_df[chl_df.Measure  == 'Median']

#%% Chl dataframe where monthy medians are reduced to sesonal (which also means annual) medians

chl_seasonal = chl_medians.groupby(['Area', 'Year'])['Concentration'].median().reset_index()

# drop 2020, because there aren't observations from the whole summer
chl_seasonal.drop(chl_seasonal[chl_seasonal.Year==2020].index, inplace=True)

#%% Save edited chl dataframes as csv

chl_df.to_csv(r'Y:\Dippa\Data\processing\Klorofylli\jarviklorofylli_ym.csv', index=False)
chl_medians.to_csv(r'Y:\Dippa\Data\processing\Klorofylli\jarviklorofylli_medians.csv', index=False)
chl_seasonal.to_csv(r'Y:\Dippa\Data\processing\Klorofylli\jarviklorofylli_seasonal_medians.csv', index=False)
