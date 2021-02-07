capstone-two / NOTEBOOKS
==============================

This directory contains the Jupyter notebooks for my second capstone project.

Notebook Organization
------------

    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
 
 
 
 01-rrp-data-collection
 -----------
This notebook contains the code that extracts the required weather data from the .tar archive. It does this using a list of Station_IDs generated in notebook 02-rrp-data-wrangling. The data collection is very time-intensive (21+ hours on Macbook Pro, 8GB Ram) and is thus kept separate from the data wrangling.
 
 02-rrp-data-wrangling
 ------------		
This notebook contains the code that imports, inspects and cleans the weather and conflict data. The notebook also executes a spatial join to get a list of Station_IDs that are located within 50km of any conflict incidents. This list is used in notebook 01-rrp-data-collection to extract the weather data from those corresponding stations.
 
 03-rrp-exploratory-data-analysis 	
 -------------
This notebook contains the code that explores the climate and conflict data in detail. The notebook investigates the original features, engineers new features to aid EDA and investigates global spatial and temporal patterns in the data. It then selects a single country to focus on as a preliminary case study: India. PCA is conducted on the new climate features and correlations are assessed. Final target feature is defined as 'death_rate': number of deaths per day. Final dataframe has 33 features (32 predictor, 1 target) and 14623 observations.
 
 04-rrp-preprocessing	
 -------------
This notebook contains the code that preprocesses our final dataframe to prepare it for the modelling stage. Numerical data is standardised, missing values are inspected, and the training and test splits are created and exported.

05-rrp-modelling	
 -------------
This notebook contains the code that creates the predictive models using our final dataframe. Six models are created and evaluated: 1) a Dummy Regressor that predicts the mean (as baseline), 2) an out-of-the-box Linear Regression model, 3) a Linear Regression model with SelectKBest features, 4) an out-of-the-box Random Forest Regressor, 5) a RF Regressor with hyperparameter tuning, and 6) a Lasso Regression with hyperparameter tuning. Conclusion: it is not possible to accurately predict conflict intensity using climate data alone.
 
 xx-rrp-profile-reports-weather-data	
 --------------
This notebook generates and exports the profile reports for the 6 climate measures of the GHCN weather data. The profile reports are only generated here, the analysis is presented in notebook 02-rrp-data-wrangling. 
 
 
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
