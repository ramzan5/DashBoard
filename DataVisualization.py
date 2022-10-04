from plotly.subplots import make_subplots
from hashlib import new
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(color_codes=True)


DataFrame = pd.read_csv("DataSet/Bike-Sharing-Dataset/day.csv")
newdf = DataFrame.drop(['dteday'], axis=1)
# --------------------------------------------HeatMap of the Whole Dataset---------------------------


def DrawCorrelationHeatMap(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    corr = DataFrame.corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values,
                                    x=['instant', 'season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit',
                                       'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt'],
                                    y=['instant ', 'season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit',
                                        'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt'],
                                    colorscale='Inferno'
                                    ))
    fig.update_layout(title_text='Correlation heatmap of the dataset',
                      title_x=0.5,
                      height=500)
    return fig

# DrawCorrelationHeatMap(newdf)

# --------------------------------------------------------------------------------------------------------

# -----------------------------------   Display Feature Importance Chart ---------------------------------


def DrawFeatureImportanceGraph(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")

    FigImportanceFeature = go.Figure()
    FigImportanceFeature.add_trace(
        go.Bar(x=DataFrame.index, y=DataFrame["cnt"]))
    FigImportanceFeature.update_layout(
        title_text='Feature Contribution in The Model',
        title_x=0.5,
        height=500,
        xaxis_title="Features",
        yaxis_title="Importance"
    )
    return FigImportanceFeature

# DrawFeatureImportanceGraph(newdf)


# ---------------------------------------Distribution Plot For Seasons---------------------------------

def SeasonDistributionColumn(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    XLabels = ['Winter', 'Spring', 'Summer', 'Fall']
    YLabels = (DataFrame['season'].value_counts()).tolist()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=XLabels, y=YLabels, marker_color='blue'))
    fig.update_layout(title_text='Distribution Plot Of Seasons',
                      title_x=0.5, height=600, width=800)
    return fig

# SeasonDistributionColumn(newdf)

# -----------------------------------------------------------------------------------------------

# ----------------------------------------Distribution Plot For Years------------------------------


def YearDistributionColumn(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    XLabels = ["2011", "2012"]
    YLabels = (DataFrame['yr'].value_counts()).tolist()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=XLabels, y=YLabels, marker_color='green'))
    fig.update_layout(title_text='Distribution Plot Of Years',
                      title_x=0.7, height=500, width=600)
    return fig


# YearDistributionColumn(newdf)
# --------------------------------------------------------------------------------------------------

# ---------------------------------------------Distribution Plot For Months-------------------------
# some missing values
def MonthsDistributionColumn(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    XLabels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
               'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    YLabels = (DataFrame['mnth'].value_counts()).tolist()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=XLabels, y=YLabels, marker_color='purple'))
    fig.update_layout(title_text='Distribution Plot Of Months',
                      title_x=0.7, height=500, width=600)
    return fig

# MonthsDistributionColumn(newdf)
# ----------------------------------------------------------------------------------------------------


# ---------------------------------------------Distribution Plot of WeekDay---------------------------

def WeekDayDistributionColumn(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    XLabels = ['Sat', 'Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    YLabels = (DataFrame['weekday'].value_counts()).tolist()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=XLabels, y=YLabels))
    fig.update_layout(title_text='Distribution Plot Of Week days',
                      title_x=0.7, height=500, width=600)
    return fig

# WeekDayDistributionColumn(newdf)

# -------------------------------------------------Distribution Plot of Year Month and WeekDay-----------------


def YearMonthWeekDayDistributionColumn(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=('Year', 'Months', 'Weekdays'))
    YearXaxis = ["2011", "2012"]
    YearYaxis = (DataFrame['yr'].value_counts()).tolist()
    MonthXaxis = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
                  'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    MonthYaxis = (DataFrame['mnth'].value_counts()).tolist()
    WeekXaxis = ['Sat', 'Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    WeekYaxis = (DataFrame['weekday'].value_counts()).tolist()
    fig.add_trace(go.Bar(x=YearXaxis, y=YearYaxis), row=1, col=1)
    fig.add_trace(go.Bar(x=MonthXaxis, y=MonthYaxis), row=1, col=2)
    fig.add_trace(go.Bar(x=WeekXaxis, y=WeekYaxis), row=1, col=3)
    fig.update_layout(title_text="Distribution Plot for Years Months and Weekdays ",
                      title_x=0.5, height=500, showlegend=False)
    return fig

# YearMonthWeekDayDistributionColumn(newdf)
# -----------------------------------------------------------------------------------------------------

# -----------------------------------------------Distribution Plot of Working Days----------------------


def WorkingDayDistributionColumn(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    XLabels = ['Working Day', 'Off Day']
    YLabels = (DataFrame['workingday'].value_counts()).tolist()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=XLabels, y=YLabels, marker_color='purple'))
    fig.update_layout(title_text='Distribution Plot Of Working Day and Off Day',
                      title_x=0.7, height=500, width=600)
    return fig
# WorkingDayDistributionColumn(newdf)
# ------------------------------------------------------------------------------------------------------

# -----------------------------------------------Distribution Plot of Holidays--------------------------


def HolidayDistributionColumn(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    XLabels = ['Other Days (Including Weekdays)', 'Holidays']
    YLabels = (DataFrame['holiday'].value_counts()).tolist()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=XLabels, y=YLabels, marker_color='purple'))
    fig.update_layout(title_text='Distribution Plot Of Holidays And Other Working Days',
                      title_x=0.7, height=500, width=600)
    return fig

# HolidayDistributionColumn(newdf)
# -------------------------------------------------------------------------------------------------------


# ---------------------------------------------Distribution Plot of Weathersit--------------------------

def WeatherDistributionColumn(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    XLabels = ['Clear, Few clouds, Partly cloudy, Partly cloudy',
               'Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist',
               'Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds',
               'Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog']
    YLabels = (DataFrame['weathersit'].value_counts()).tolist()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=XLabels, y=YLabels, marker_color='purple'))
    fig.update_layout(title_text='Distribution Plot Of Weather Conditions',
                      title_x=0.7, height=500, width=800)
    return fig

# WeatherDistributionColumn(newdf)
# ------------------------------------------------------------------------------------------------------

# --------------------------------------------------Box Plot------------------------------------------
# ------------------------------------------------------------------------------------------------------

# ---------------------------------------------------Box Plot of Year Column Against Numbers of Bikes (Casual, Registerd, Cnt)

def DrawChartYearAgainstCRC(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Casual vs Year", "Registered vs Year", "Cnt vs Year")
    )
    DataFrame['yr'].replace({0: "2011", 1: "2012"}, inplace=True)
    fig.add_trace(go.Box(x=DataFrame['yr'], 
                  y=DataFrame['casual'],marker_color='purple'), row=1, col=1)
    fig.add_trace(go.Box(x=DataFrame['yr'],
                  y=DataFrame['registered']), row=1, col=2)
    fig.add_trace(go.Box(x=DataFrame['yr'],
                  y=DataFrame['cnt']), row=1, col=3)
    fig.update_layout(title_text='Chart of Bikes(Casual, Registered, total) vs Year',
                      title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(title_text="Year", row=1, col=1)
    fig.update_xaxes(title_text="Year", row=1, col=2)
    fig.update_xaxes(title_text="Year", row=1, col=3)
    fig.update_yaxes(title_text="Casual", row=1, col=1)
    fig.update_yaxes(title_text="Registered", row=1, col=2)
    fig.update_yaxes(title_text="cnt", row=1, col=3)
    return fig

# DrawChartYearAgainstCRC(newdf)
 #----------------------------------------------------------------------------------------------------- 

#---------------------Box Plot of Season Column againt Numbers of Bikes (Casual, Registerd, Cnt)--------


def DrawChartSeasonsAgainstCRC(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Casual vs Season", "Registered vs Season", "Cnt vs Season")
    )
    DataFrame['season'].replace({1: "Winter", 2: "Spring", 3:"Summer", 4:"Fall"}, inplace=True)
    fig.add_trace(go.Box(x=DataFrame['season'],
                  y=DataFrame['casual'], marker_color='purple'), row=1, col=1)
    fig.add_trace(go.Box(x=DataFrame['season'],
                  y=DataFrame['registered'], marker_color='blue'), row=1, col=2)
    fig.add_trace(go.Box(x=DataFrame['season'],
                  y=DataFrame['cnt'], marker_color='purple'), row=1, col=3)
    fig.update_layout(title_text='Chart of Bikes(Casual, Registered, total) vs Season',
                      title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(title_text="Season", row=1, col=1)
    fig.update_xaxes(title_text="Season", row=1, col=2)
    fig.update_xaxes(title_text="Season", row=1, col=3)
    fig.update_yaxes(title_text="Casual", row=1, col=1)
    fig.update_yaxes(title_text="Registered", row=1, col=2)
    fig.update_yaxes(title_text="cnt", row=1, col=3)
    return fig

# DrawChartSeasonsAgainstCRC(newdf)
# -------------------------------------------------------------------------------------------------
# ------------------------------Box Plot of Monts against Numbers of Bikes (Casual)----------------

def DrawChartMonthsAgianstCasual(Dataframe):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    
    DataFrame['mnth'].replace({1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul',
                                 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}, inplace=True)
    fig = go.Figure()
    fig.add_trace(
        go.Box(x=DataFrame['mnth'], y=DataFrame['casual'], marker_color='olive'))
    fig.update_layout(title_text='Box Plot Of Months aginst Numbers of Bikes Casual',
                      title_x=0.7, height=700, width=1200)
    return fig

# DrawChartMonthsAgianstCasual(newdf)
# -----------------------------------------------------------------------------------------------------

# ------------------------------Box Plot of Monts against Numbers of Bikes (Registered)----------------


def DrawChartMonthsAgianstRegistered(Dataframe):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")

    DataFrame['mnth'].replace({1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul',
                               8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}, inplace=True)
    fig = go.Figure()
    fig.add_trace(
        go.Box(x=DataFrame['mnth'], y=DataFrame['registered'], marker_color='blue'))
    fig.update_layout(title_text='Box Plot Of Months aginst Numbers of Bikes (Registered)',
                      title_x=0.7, height=700, width=1200)
    return fig


# DrawChartMonthsAgianstRegistered(newdf)
# -----------------------------------------------------------------------------------------------------

# ------------------------------Box Plot of Monts against Numbers of Bikes (Registered+Casual)----------------


def DrawChartMonthsAgianstCnt(Dataframe):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")

    DataFrame['mnth'].replace({1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul',
                               8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}, inplace=True)
    fig = go.Figure()
    fig.add_trace(
        go.Box(x=DataFrame['mnth'], y=DataFrame['cnt'], marker_color='purple'))
    fig.update_layout(title_text='Box Plot Of Months aginst Numbers of Bikes (Total)',
                      title_x=0.7, height=700, width=1200)
    return fig


# DrawChartMonthsAgianstCnt(newdf)
# -----------------------------------------------------------------------------------------------------
