import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# Importing csv file date itno DataFarme
DataFrame = pd.read_csv("DataSet/Bike-Sharing-Dataset/day.csv")

# By Using Linear Model (Linear Regression)
# ------------------------Multivariate Linear Regression for total Bike cnt----------------------------


def MultivariateLinearRegressionForCnt(df):
    x = df[['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit',
            'temp', 'atemp', 'hum', 'windspeed']]
    y = df['cnt']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    multivariateRegression = LinearRegression()
    multivariateRegression.fit(x_train, y_train)
    multivariateRegressionScore = multivariateRegression.score(x_test, y_test)

    return multivariateRegressionScore, multivariateRegression

# print(MultivariateLinearRegressionForCnt(DataFrame))
# ----------------------------------------------------------------------------------------------------

# --------------------------Mutivariate Linear Regression For Casual Bike------------------------------


def MultivariateLinearRegressionForCasual(df):
    x = df[['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit',
            'temp', 'atemp', 'hum', 'windspeed']]
    y = df['casual']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    multivariateRegressionForCasual = LinearRegression()
    multivariateRegressionForCasual.fit(x_train, y_train)
    multivariateRegressionScore = multivariateRegressionForCasual.score(
        x_test, y_test)

    return multivariateRegressionScore, multivariateRegressionForCasual

# print(MultivariateLinearRegressionForCasual(DataFrame))
# ----------------------------------------------------------------------------------------------------

# --------------------------Mutivariate Linear Regression For Registered Bike--------------------------


def MultivariateLinearRegressionForRegistered(df):
    x = df[['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit',
            'temp', 'atemp', 'hum', 'windspeed']]
    y = df['registered']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    multivariateRegressionForRegistered = LinearRegression()
    multivariateRegressionForRegistered.fit(x_train, y_train)
    multivariateRegressionScore = multivariateRegressionForRegistered.score(
        x_test, y_test)

    return multivariateRegressionScore, multivariateRegressionForRegistered

# print(MultivariateLinearRegressionForRegistered(DataFrame))
# --------------------------------------------------------------------------------------------------


# By Using RandomForestRegressor

# -------------------------Random Forest Regression For Total Bike Cnt-----------------------------------------

def RandomForestRegressionForCnt(df):
    x = df[['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit',
            'temp', 'atemp', 'hum', 'windspeed']]
    y = df['cnt']
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0)
    regressorForCnt = RandomForestRegressor(
        n_estimators=200, random_state=0, criterion="squared_error")
    regressorForCnt.fit(x_train, y_train)
    y_pred = regressorForCnt.predict(x_test)
    regressorForCntScore = r2_score(y_test, y_pred)

    return regressorForCntScore, regressorForCnt

# print(RandomForestRegressionForCnt(DataFrame))
# -----------------------------------------------------------------------------------------------------

# -------------------------Random Forest Regression For Casual Bike-----------------------------------------


def RandomForestRegressionForCasual(df):
    x = df[['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit',
            'temp', 'atemp', 'hum', 'windspeed']]
    y = df['casual']
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0)
    regressorForCasual = RandomForestRegressor(
        n_estimators=200, random_state=0, criterion="squared_error")
    regressorForCasual.fit(x_train, y_train)
    y_pred = regressorForCasual.predict(x_test)
    regressorForCntScore = r2_score(y_test, y_pred)

    return regressorForCntScore, regressorForCasual

# print(RandomForestRegressionForCasual(DataFrame))
# ----------------------------------------------------------------------------------------------------

# -------------------------Random Forest Regression For Registered Bike-----------------------------------------


def RandomForestRegressionForRegistered(df):
    x = df[['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit',
            'temp', 'atemp', 'hum', 'windspeed']]
    y = df['registered']
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0)
    regressorForRegistered = RandomForestRegressor(
        n_estimators=200, random_state=0, criterion="squared_error")
    regressorForRegistered.fit(x_train, y_train)
    y_pred = regressorForRegistered.predict(x_test)
    regressorForCntScore = r2_score(y_test, y_pred)

    return regressorForCntScore, regressorForRegistered

# print(RandomForestRegressionForRegistered(DataFrame))
# ---------------------------------------------------------------------------------------------------
