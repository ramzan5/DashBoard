'''
Implementation of the Dash Board 
'''

import dash
import pandas as pd
import numpy as np
import math
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

# Importing DataPrepProcessing
import DataPreProcessing as dpp

# Importing MachineLearning Model
import MachineLearningModel as ml

# Importing DataVisualization
import DataVisualization as dv


# Specifying the CSV File Path
FilePath = "DataSet/Bike-Sharing-Dataset/day.csv"

# Chceking the File Existance
dpp.CheckFileExistence(FilePath)

# Reading the Data From File
DataFrame = dpp.ReafFile(FilePath)

# Factor value for given DataSet
UpdatedDataFrame = DataFrame.drop(['dteday'], axis=1)
# Checking the Missing Data in given DataFrame
dpp.CheckMissingData(DataFrame)

# Checking the Plausability of given Dataset
dpp.CheckPlauseabilityDataSet(DataFrame)

# Getting the Numbers of Rows
NumOfRows = dpp.DisplayNumbersOfRows(DataFrame)

# Getting the Numbers of Columns
NumOfColumns = dpp.DisplayNumbersOfColumns(DataFrame)

# Getting the Columns Name
ColumnsName = dpp.GetColumnsName(DataFrame)

# Getting the Correlation of give Data Set
DataCorrelation = dpp.CheckCorrelation(DataFrame)

# Getting the Columns Varience of given DataSet
# DataSetVarience = dpp.CheckColumnsVarience(DataFrame)

# Column used For the Prediction for Casual Bikes
ColumnNameUsedForCasualBikes = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit',
                                'temp', 'atemp', 'hum', 'windspeed']

# Column used For the Prediction for Registered Bikes
ColumnNameUsedForRegisterdBikes = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit',
                                   'temp', 'atemp', 'hum', 'windspeed']

# Column used For the Prediction for Cnt (Total Bikes)
ColumnNameUsedForCntBikes = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit',
                             'temp', 'atemp', 'hum', 'windspeed']

# Getting the Regresser for Casual Bikes and Score
ScoresForCasualBikes, RegressorForCasualBikes = ml.RandomForestRegressionForCasual(
    DataFrame)

# Getting the Regresser for Registerd Bikes and Score
ScoresForRegisteredBikes, RegressorForRegisteredBikes = ml.RandomForestRegressionForRegistered(
    DataFrame)

# Getting the Regresser for Cnt (Total) Bikes and Score
ScoresForCntBikes, RegressorForCntBikes = ml.RandomForestRegressionForCnt(
    DataFrame)


# Getting the Regresser for Casual Bikes and Score Using Multivariate Regression Algorithm
ScoresForCasualBikes1, RegressorForCasualBikes1 = ml.MultivariateLinearRegressionForCasual(
    DataFrame)

# Getting the Regresser for Registerd Bikes and Score Using Multivariate Regression Algorithm
ScoresForRegisteredBikes1, RegressorForRegisteredBikes1 = ml.MultivariateLinearRegressionForRegistered(
    DataFrame)

# Getting the Regresser for Cnt (Total) Bikes and Score Using Multivariate Regression Algorithm
ScoresForCntBikes1, RegressorForCntBikes1 = ml.MultivariateLinearRegressionForCnt(
    DataFrame)

# Selecting the Most Significant Features from the Model for the Prediction Casual Bikes
FeatureImportanceForCasual = pd.DataFrame(
    RegressorForCasualBikes.feature_importances_*100, columns=['Importance'], index=ColumnNameUsedForCasualBikes)
FeatureImportanceForCasual = FeatureImportanceForCasual.sort_values(
    "Importance", ascending=False)


# Selecting the Most Significant Features from the Model for the Prediction Registerd Bikes
FeatureImportanceForRegistered = pd.DataFrame(
    RegressorForRegisteredBikes.feature_importances_*100, columns=['Importance'], index=ColumnNameUsedForRegisterdBikes)
FeatureImportanceForRegistered = FeatureImportanceForRegistered.sort_values(
    "Importance", ascending=False)


# Selecting the Most Significant Features for the Model for the Prediction of Cnt (Total Bikes)

FeatureImportanceForCnt = pd.DataFrame(
    RegressorForCntBikes.feature_importances_*100, columns=['Importance'], index=ColumnNameUsedForCntBikes)
FeatureImportanceForCnt = FeatureImportanceForCnt.sort_values(
    "Importance", ascending=False
)

#----------------------Features Importance Figure for Casual Bike--------------------------------
FetureImportanceFigureForCasualBikes = dv.DrawFeatureImportanceGraph(
    FeatureImportanceForCasual)

#----------------------Features Importance Figure for Registered Bike--------------------------------
FetureImportanceFigureForRegisteredBikes = dv.DrawFeatureImportanceGraph(
    FeatureImportanceForRegistered)

#----------------------Features Importance Figure for Cnt (Total Bikes)--------------------------------
FetureImportanceFigureForCntBikes = dv.DrawFeatureImportanceGraph(
    FeatureImportanceForCnt)

# -------------------------------- Figure For HeatMap---------------------------------------------------
CorrelationFigureForHeatMap = dv.DrawCorrelationHeatMap(UpdatedDataFrame)

# 
