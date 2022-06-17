

Code to support the aricle 'Modeling the drivers of eutrophication in Finland with a machine learning approach'. By: Sara Heikonen, Maria Yli-Heikkilä and Matias Heino

The analysis is composed of three main parts and the following Python scripts (1-10) and related functions (7.1, 10.1, 10.2):

    Preparing data (download data and filter data)

    1. FMI_download.py, reproject_raster.py: Download temperature and precipitation data from Finnish Meteorological Institute (FMI) API (1), reproject climate rasters
    2. pour_points.py: Find pour points for catchment areas provided by Finnish Environment Insitute (2). For this step, each catchment area was saved as an individual shapefile in a folder
       thats' name is the identifier of the lake or stream that the catchment contains.
    3. catchment_for_lake.py: Delineate catchments for study lakes 
    4. odata_download.py: Download lake mean depth from Finnish Environment Institute's Lake registry (3). Requires ChromeDriver (4)
    5. chla_lakes_processing.py: filter lake data to match chlorophyll-a observations, calculate seasonal medians from chl-a data

    Variables calculation (prepare data for Random Forest model)

    6. climate.py, erosion.py, land_use.py: Calculate study lake/catchment specific values for some explanatory variables (landcover, climate, erosion)
    7. combine_tables.py: Combine variables from all study catchments into a table that will be used as input in the Random Forest (RF) models
	7.1 aggregation_area_processing.py (a function): Add information about catchment hierarchy.
    8. remove_nested_catch.py: Get information about the spatial hierarchy of study catchments

    Analyses and visualization (produce resuls, figures and tables)

    9. backgound_figs_tbls.py: Make background information tables and figures, i.e. Figure 1, Supplementary Table 1 and Supplementary Figure 1
       In addition to the data from this study (5), requires data on administrative borders (6) and borders of river basin districts (7)
    10. random_forest.py: Run RF models & produce Figures 3 & 4, Supplementary figure 5 and model performance stats tables (Table 2 parts). All outputs are produced for either the
        country scale model or individual RBD scale models. In addition to the data from this study (5), requires data on administrative borders (6).
    	10.1 summarize_rf_results.py (a function): Summarize Random Forest model output from 100 runs.
    	10.2 aggregate_results.py (a function): Aggregate model results & produce Figure 2 and Supplementary Figures 2, 3 & 4.
      	
Data for running the scripts under 'Analysis and visualization' provided in Zenodo (link), and by National Land Survey Finland (6).
Data for the earlier steps is openly available from external sources, or can be requested from Finnish Environmental Institute. List of 
the required datasets is provided in the article that this repository supports.

References:

1) Finnish Meteorological Institute, 2021. Monthly Weather Observations. Available at: https://opendata.fmi.fi/wfs?request=GetFeature&storedquery_id=GetDataSetById&datasetid=1000552 (accessed 7.1.21)
2) Finnish Environment Institute, 2014. Catchment areas used in VALUE catchment delineation online tool (draft). Available by request from Finnish environment Institute, 
   metadata available at: https://ckan.ymparisto.fi/dataset/value-valuma-aluejako-ehdotus (accessed 9.14.20).
3) Finnish Environment Institute, 2012. Lake registry. Available at: http://rajapinnat.ymparisto.fi/api/jarvirajapinta/1.0/odataquerybuilder/ (accessed 7.1.21).
4) Chromium, 2021: ChromeDriver. Available at: https://chromedriver.chromium.org/downloads (accessed 6.18.21)
5) Zenodo link
6) National Land Survey of Finland, 2021. Administrative borders. Available at: https://paituli.csc.fi/download.html?data_id=mml_hallinto_10k_2021_shape_euref
7) Finnish Environment Institute, 2005. River Basin Districts. Available at: https://wwwd3.ymparisto.fi/d3/gis_data/spesific/vha.zip (accessed 11.9.2020)
