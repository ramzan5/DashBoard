from statistics import correlation
import pandas as pd
from scipy.stats import pearsonr
from pathlib import Path
import itertools

# --------------------File Existence Check------------------------------------------------------------
def CheckFileExistence(filePath):
    file = Path(filePath)
    if not file.is_file():
        raise Exception("Oops! File Does Not Exist")

# print(CheckFileExistence("/home/muhammad/Documents/DashBoard/DataSet/Bike-Sharing-Dataset/day.csv"))
# ----------------------------------------------------------------------------------------------------

# ------------------Read Data From File ---------------------------------------------------------------

def ReafFile(filePath):
    file = Path(filePath)
    if not file.is_file():
        raise Exception("Oops! File Does Not Exist")
    DataFrame = pd.read_csv(filePath)
    return DataFrame


# print(ReafFile("/home/muhammad/Documents/DashBoard/DataSet/Bike-Sharing-Dataset/day.csv"))
# ----------------------------------------------------------------------------------------------------

# ------------------------------Check Missing Data in DataSet -------------------------------------------

def CheckMissingData(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! it is not a DataFrame")
    MissingData = False
    NaNRows = DataFrame.isnull()
    NaNRows = NaNRows.any(axis=1)
    DataFrameWithNaN = DataFrame[NaNRows]
    DataFrameWithNaN.shape[0]
    if DataFrameWithNaN.shape[0] != 0:
        raise Exception("Oop! Some Data is missing in DataSet")
    return MissingData

# -------------------------------------------------------------------------------------------------------

# -----------------------------To Get the Numbers of Column from DataSet
def DisplayNumbersOfColumns(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    return DataFrame.shape[1]


DataFrame = ReafFile(
    "/home/muhammad/Documents/DashBoard/DataSet/Bike-Sharing-Dataset/day.csv")
# print(DisplayNumbersOfColumns(DataFrame))
# -------------------------------------------------------------------------------------------------

# --------------------------------Display the Numbers of rows in DataSet---------------------------
def DisplayNumbersOfRows(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    return DataFrame.shape[0]

# print(DisplayNumbersOfRows(DataFrame))
# -------------------------------------------------------------------------------------------------

# ------------------------------Get the Names of Colums from DataSet ------------------------------

def GetColumnsName(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    return DataFrame.columns.tolist()

# print(GetColumnsName(DataFrame))

# ---------------------------------------------------------------------------------------------------

# ------------------------------Check the Varience of Columns---------------------------------------

def CheckColumnsVarience(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    return DataFrame.var()

# print(CheckColumnsVarience(DataFrame))
# ----------------------------------------------------------------------------------------------------

# -------------------------------Correlation Check of Given Data Set----------------------------------

def CheckCorrelation(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    DataFrameIntegerFloatsColumns = DataFrame.select_dtypes(include=['int64','float64'])
    correlation = {}
    columns = DataFrameIntegerFloatsColumns.columns.tolist()
    for col_a, col_b in itertools.combinations(columns, 2):
        correlation[col_a + '__'+ col_b] = pearsonr(DataFrameIntegerFloatsColumns.loc[:,col_a],DataFrameIntegerFloatsColumns.loc[:,col_b])
    FinalResult = pd.DataFrame.from_dict(correlation, orient='index')
    FinalResult.columns = ['PCC', 'p-value']
    return FinalResult.columns

# print(CheckCorrelation(DataFrame))

# ------------------------------------------------------------------------------------------------------


