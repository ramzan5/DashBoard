import plotly.express as px
from plotly.subplots import make_subplots
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
    fig.update_layout(title_text='Correlation Heatmap of the dataset',
                      title_x=0.5,
                      height=500)
    return fig

# DrawCorrelationHeatMap(newdf)

# --------------------------------------------------------------------------------------------------------

# -----------------------------------   Display Feature Importance ---------------------------------


def DrawFeatureImportanceGraph(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")

    FigImportanceFeature = go.Figure()
    FigImportanceFeature.add_trace(
        go.Bar(x=DataFrame.index, y=DataFrame["Importance"], marker_color='purple'))
    FigImportanceFeature.update_layout(
        title_text='Feature Contribution in The Model',
        title_x=0.5,
        height=500,
        xaxis_title="Features",
        yaxis_title="Importance"
    )
    return FigImportanceFeature

# DrawFeatureImportanceGraph(newdf)

# Box-Plot for Casual Registered and Total Numbers of bikes


def BoxPlotForCasualRegisteredAndTotalBikes(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")

    fig = go.Figure()
    fig.add_trace(go.Box(y=DataFrame['casual'],
                  name="Casual Bikes", marker_color='purple'))
    fig.add_trace(go.Box(
        y=DataFrame['registered'], name="Registered Bikes", marker_color='black'))
    fig.add_trace(go.Box(y=DataFrame['cnt'],
                  name="Total Bikes", marker_color='brown'))
    fig.update_layout(title_text="Box Plot for Bikes",
                      title_x=0.5, height=500, xaxis_title="Column Variables",
                      yaxis_title="Distribution")
    return fig
# -----------------------------------------------------------------------------------------------------
# BoxPlotForCasualRegisteredAndTotalBikes(newdf)

# ---------------------------------------Distribution Plot For Seasons---------------------------------


def SeasonDistributionColumn(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    XLabels = ['Winter', 'Spring', 'Summer', 'Fall']
    YLabels = (DataFrame['season'].value_counts()).tolist()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=XLabels, y=YLabels, marker_color='purple'))
    fig.update_layout(title_text='Distribution Plot Of Seasons',
                      title_x=0.5, height=500, width=600)
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
                      title_x=0.7, height=500, width=1225)
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
                  y=DataFrame['casual'], marker_color='purple'), row=1, col=1)
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
 # -----------------------------------------------------------------------------------------------------

# ---------------------Box Plot of Season Column againt Numbers of Bikes (Casual, Registerd, Cnt)--------


def DrawChartSeasonsAgainstCRC(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Casual vs Season",
                        "Registered vs Season", "Cnt vs Season")
    )
    DataFrame['season'].replace({1: "Winter",
                                 2: "Spring",
                                 3: "Summer",
                                 4: "Fall"}, inplace=True)
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

    DataFrame['mnth'].replace({1: 'Jan',
                               2: 'Feb',
                               3: 'Mar',
                               4: 'Apr',
                               5: 'May',
                               6: 'Jun',
                               7: 'Jul',
                               8: 'Aug',
                               9: 'Sep',
                               10: 'Oct',
                               11: 'Nov',
                               12: 'Dec'}, inplace=True)
    fig = go.Figure()
    fig.add_trace(
        go.Box(x=DataFrame['mnth'], y=DataFrame['casual'], marker_color='olive'))
    fig.update_layout(title_text='Box Plot Of Months aginst Numbers of Bikes Casual',
                      title_x=0.7, height=500)
    return fig

# DrawChartMonthsAgianstCasual(newdf)
# -----------------------------------------------------------------------------------------------------

# ------------------------------Box Plot of Monts against Numbers of Bikes (Registered)----------------


def DrawChartMonthsAgianstRegistered(Dataframe):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")

    DataFrame['mnth'].replace({1: 'Jan',
                               2: 'Feb',
                               3: 'Mar',
                               4: 'Apr',
                               5: 'May',
                               6: 'Jun',
                               7: 'Jul',
                               8: 'Aug',
                               9: 'Sep',
                               10: 'Oct',
                               11: 'Nov',
                               12: 'Dec'}, inplace=True)
    fig = go.Figure()
    fig.add_trace(
        go.Box(x=DataFrame['mnth'], y=DataFrame['registered'], marker_color='blue'))
    fig.update_layout(title_text='Box Plot Of Months aginst Numbers of Bikes (Registered)',
                      title_x=0.7, height=500)
    return fig


# DrawChartMonthsAgianstRegistered(newdf)
# -----------------------------------------------------------------------------------------------------

# ------------------------------Box Plot of Months against Numbers of Bikes (Registered+Casual)----------------


def DrawChartMonthsAgianstCnt(Dataframe):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")

    DataFrame['mnth'].replace({1: 'Jan',
                               2: 'Feb',
                               3: 'Mar',
                               4: 'Apr',
                               5: 'May',
                               6: 'Jun',
                               7: 'Jul',
                               8: 'Aug',
                               9: 'Sep',
                               10: 'Oct',
                               11: 'Nov',
                               12: 'Dec'}, inplace=True)
    fig = go.Figure()
    fig.add_trace(
        go.Box(x=DataFrame['mnth'], y=DataFrame['cnt'], marker_color='purple'))
    fig.update_layout(title_text='Box Plot Of Months aginst Numbers of Bikes (Total)',
                      title_x=0.7, height=500)
    return fig


# DrawChartMonthsAgianstCnt(newdf)
# -----------------------------------------------------------------------------------------------------


# ----------------------------------------------Box Plot Weekdays againts Numbers of Bikes(Casual)

def DrawChartWeekdayAgianstCasual(Dataframe):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")

    DataFrame['weekday'].replace({0: 'Sunday',
                                  1: 'Monday',
                                  2: 'Tuesday',
                                  3: 'Wednesday',
                                  4: 'Thursday',
                                  5: 'Friday',
                                  6: 'Saturday'
                                  }, inplace=True)
    fig = go.Figure()
    fig.add_trace(
        go.Box(x=DataFrame['weekday'], y=DataFrame['casual'], marker_color='teal'))
    fig.update_layout(title_text='Box Plot Of Days of Week aginst Numbers of Bikes (Casual)',
                      title_x=0.7, height=500)
    return fig

# DrawChartWeekdayAgianstCasual(newdf)
# ---------------------------------------------------------------------------------------------------------

# ----------------------------------------------Box Plot Weekdays againts Numbers of Bikes(Registered)


def DrawChartWeekdayAgianstRegisterd(Dataframe):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")

    DataFrame['weekday'].replace({0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'
                                  }, inplace=True)
    fig = go.Figure()
    fig.add_trace(
        go.Box(x=DataFrame['weekday'], y=DataFrame['registered'], marker_color='salmon'))
    fig.update_layout(title_text='Box Plot Of Days of Week aginst Numbers of Bikes (Registered)',
                      title_x=0.7, height=500)
    return fig

# DrawChartWeekdayAgianstRegisterd(newdf)
# ---------------------------------------------------------------------------------------------------------


# ----------------------------------------------Box Plot Weekdays againts Numbers of Bikes(Cnt -> Total Bikes)


def DrawChartWeekdayAgianstCnt(Dataframe):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")

    DataFrame['weekday'].replace({0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'
                                  }, inplace=True)
    fig = go.Figure()
    fig.add_trace(
        go.Box(x=DataFrame['weekday'], y=DataFrame['cnt'], marker_color='royalblue'))
    fig.update_layout(title_text='Box Plot Of Days of Week aginst Numbers of Bikes (Total)',
                      title_x=0.7, height=500)
    return fig

# DrawChartWeekdayAgianstCnt(newdf)
# ---------------------------------------------------------------------------------------------------------

# ---------------------Box Plot of Holidays Column againt Numbers of Bikes (Casual, Registerd, Cnt)--------


def DrawChartHolidayDayAgainstCRC(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Casual vs Holiday",
                        "Registered vs Holiday", "Cnt vs Holiday")
    )
    DataFrame['holiday'].replace(
        {0: "Working Day",
         1: "Holiday"},
        inplace=True)
    fig.add_trace(go.Box(x=DataFrame['holiday'],
                  y=DataFrame['casual'], marker_color='brown'), row=1, col=1)
    fig.add_trace(go.Box(x=DataFrame['holiday'],
                  y=DataFrame['registered'], marker_color='blue'), row=1, col=2)
    fig.add_trace(go.Box(x=DataFrame['holiday'],
                  y=DataFrame['cnt'], marker_color='purple'), row=1, col=3)
    fig.update_layout(title_text='Chart of Bikes(Casual, Registered, total) vs Holiday',
                      title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(title_text="Holiday", row=1, col=1)
    fig.update_xaxes(title_text="Holiday", row=1, col=2)
    fig.update_xaxes(title_text="Holiday", row=1, col=3)
    fig.update_yaxes(title_text="Casual", row=1, col=1)
    fig.update_yaxes(title_text="Registered", row=1, col=2)
    fig.update_yaxes(title_text="cnt", row=1, col=3)
    return fig

# DrawChartHolidayDayAgainstCRC(newdf)

# ------------------------------------------------------------------------------------------------------------


# ---------------------Box Plot of Workingday Column againt Numbers of Bikes (Casual, Registerd, Cnt)--------

def DrawChartWorkingDayAgainstCRC(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Casual vs WorkingDay",
                        "Registered vs WorkingDay",
                        "Cnt vs WorkingDay")
    )
    DataFrame['workingday'].replace(
        {0: "Off Day(Including Holidays)",
         1: "Working Day"},
        inplace=True)
    fig.add_trace(go.Box(x=DataFrame['workingday'],
                  y=DataFrame['casual'], marker_color='brown'), row=1, col=1)
    fig.add_trace(go.Box(x=DataFrame['workingday'],
                  y=DataFrame['registered'], marker_color='blue'), row=1, col=2)
    fig.add_trace(go.Box(x=DataFrame['workingday'],
                  y=DataFrame['cnt'], marker_color='purple'), row=1, col=3)
    fig.update_layout(title_text='Chart of Bikes(Casual, Registered, total) vs Working Day',
                      title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(title_text="Working Day", row=1, col=1)
    fig.update_xaxes(title_text="Working Day", row=1, col=2)
    fig.update_xaxes(title_text="Working Day", row=1, col=3)
    fig.update_yaxes(title_text="Casual", row=1, col=1)
    fig.update_yaxes(title_text="Registered", row=1, col=2)
    fig.update_yaxes(title_text="cnt", row=1, col=3)
    return fig


# DrawChartWorkingDayAgainstCRC(newdf)

# -------------------------------------------------------------------------------------------------------------------

# ---------------------Box Plot of Weathersit  Column againt Numbers of Bikes (Casual, Registerd, Cnt)--------

def DrawChartWeathersitAgainstCRC(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Casual vs Weathersit",
                        "Registered vs Weathersit", "Cnt vs Weathersit")
    )
    DataFrame['weathersit'].replace(
        {1: "Clear",
         2: "Mist",
         3: "Light Snow",
         4: "Heavy Rain"},
        inplace=True)
    fig.add_trace(go.Box(x=DataFrame['weathersit'],
                  y=DataFrame['casual'], marker_color='brown'), row=1, col=1)
    fig.add_trace(go.Box(x=DataFrame['weathersit'],
                  y=DataFrame['registered'], marker_color='blue'), row=1, col=2)
    fig.add_trace(go.Box(x=DataFrame['weathersit'],
                  y=DataFrame['cnt'], marker_color='purple'), row=1, col=3)
    fig.update_layout(title_text='Chart of Bikes(Casual, Registered, total) vs Weathersit',
                      title_x=0.5, height=500, showlegend=False)
    fig.update_xaxes(title_text="Weathersit", row=1, col=1)
    fig.update_xaxes(title_text="Weathersit", row=1, col=2)
    fig.update_xaxes(title_text="Weathersit", row=1, col=3)
    fig.update_yaxes(title_text="Casual", row=1, col=1)
    fig.update_yaxes(title_text="Registered", row=1, col=2)
    fig.update_yaxes(title_text="cnt", row=1, col=3)
    return fig


# DrawChartWeathersitAgainstCRC(newdf)

# -----------------------------------------------------------------------------------------------------------

# -----------------------------------------Scatter-Plot of Temperature against Numbers of Bikes (Casual, Registerd, Cnt)---------------

def DrawScatterPlotTempAgainstCRC(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Casual vs Temperature",
                        "Registered vs Temperature",
                        "Cnt vs Temperature"))
    fig.add_trace(go.Scatter(
        x=DataFrame['temp'], y=DataFrame['casual'], mode='markers'), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=DataFrame['temp'], y=DataFrame['registered'], mode='markers'), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=DataFrame['temp'], y=DataFrame['cnt'], mode='markers'), row=1, col=3)
    fig.update_layout(title_text='Chart of Bikes(Casual, Registered, total) vs Temperature',
                      title_x=0.5, height=600, showlegend=False)
    fig.update_xaxes(title_text="Temperature", row=1, col=1)
    fig.update_xaxes(title_text="Temperature", row=1, col=2)
    fig.update_xaxes(title_text="Temperature", row=1, col=3)
    fig.update_yaxes(title_text="Casual Bike", row=1, col=1)
    fig.update_yaxes(title_text="Registered Bike", row=1, col=2)
    fig.update_yaxes(title_text="Cnt Total Bike", row=1, col=3)
    return fig


# DrawScatterPlotTempAgainstCRC(newdf)
# -----------------------------------------------------------------------------------------------------------------

# By Using Dofferent Temperature
# -----------------------------------------Scatter-Plot of Temperature against Numbers of Bikes (Casual, Registerd, Cnt)---------------


def DrawScatterPlotATempAgainstCRC(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Casual vs Temperature",
                        "Registered vs Temperature",
                        "Cnt vs Temperature"))
    fig.add_trace(go.Scatter(
        x=DataFrame['atemp'], y=DataFrame['casual'], mode='markers'), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=DataFrame['atemp'], y=DataFrame['registered'], mode='markers'), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=DataFrame['atemp'], y=DataFrame['cnt'], mode='markers'), row=1, col=3)
    fig.update_layout(title_text='Chart of Bikes(Casual, Registered, total) vs A Temperature',
                      title_x=0.5, height=600, showlegend=False)
    fig.update_xaxes(
        title_text="Temperature ((t-t_min)/(t_max-t_min))", row=1, col=1)
    fig.update_xaxes(
        title_text="Temperature ((t-t_min)/(t_max-t_min)", row=1, col=2)
    fig.update_xaxes(
        title_text="Temperature ((t-t_min)/(t_max-t_min)", row=1, col=3)
    fig.update_yaxes(title_text="Casual Bike", row=1, col=1)
    fig.update_yaxes(title_text="Registered Bike", row=1, col=2)
    fig.update_yaxes(title_text="Cnt Total Bike", row=1, col=3)
    return fig


# DrawScatterPlotATempAgainstCRC(newdf)
# -----------------------------------------------------------------------------------------------------------------


# -----------------------------------------Scatter-Plot of Humidity against Numbers of Bikes (Casual, Registerd, Cnt)---------------

def DrawScatterPoltHumidityAgainstCRC(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Casual Bikes vs Humidity",
                        "Registered Bikes vs Humidity",
                        "Cnt Bikes vs Humidity"))

    fig.add_trace(go.Scatter(
        x=DataFrame['hum'], y=DataFrame['casual'], mode='markers'), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=DataFrame['hum'], y=DataFrame['registered'], mode='markers'), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=DataFrame['hum'], y=DataFrame['cnt'], mode='markers'), row=1, col=3)
    fig.update_layout(title_text='Chart of Bikes(Casual, Registered, total) vs Humidity',
                      title_x=0.5, height=600, showlegend=False)
    fig.update_xaxes(
        title_text="Humidity", row=1, col=1)
    fig.update_xaxes(
        title_text="Humidity", row=1, col=2)
    fig.update_xaxes(
        title_text="Humidity", row=1, col=3)
    fig.update_yaxes(title_text="Casual Bike", row=1, col=1)
    fig.update_yaxes(title_text="Registered Bike", row=1, col=2)
    fig.update_yaxes(title_text="Cnt Total Bike", row=1, col=3)
    return fig

# DrawScatterPoltHumidityAgainstCRC(newdf)

# ---------------------------------------------------------------------------------------------------------


# -----------------------------------------Scatter-Plot of WindSpeed against Numbers of Bikes (Casual, Registerd, Cnt)---------------
def DrawScatterPoltWindSpeedAgainstCRC(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Casual Bikes vs WindSpeed",
                        "Registered Bikes vs WindSpeed",
                        "Cnt Bikes vs WindSpeed"))

    fig.add_trace(go.Scatter(
        x=DataFrame['windspeed'], y=DataFrame['casual'], mode='markers'), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=DataFrame['windspeed'], y=DataFrame['registered'], mode='markers'), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=DataFrame['windspeed'], y=DataFrame['cnt'], mode='markers'), row=1, col=3)
    fig.update_layout(title_text='Chart of Bikes(Casual, Registered, total) vs Windspeed',
                      title_x=0.5, height=600, showlegend=False)
    fig.update_xaxes(
        title_text="Windspeed", row=1, col=1)
    fig.update_xaxes(
        title_text="Windspeed", row=1, col=2)
    fig.update_xaxes(
        title_text="Windspeed", row=1, col=3)
    fig.update_yaxes(title_text="Casual Bike", row=1, col=1)
    fig.update_yaxes(title_text="Registered Bike", row=1, col=2)
    fig.update_yaxes(title_text="Cnt Total Bike", row=1, col=3)
    return fig


# DrawScatterPoltWindSpeedAgainstCRC(newdf)
# ---------------------------------------------------------------------------------------------------------------------------


# 3d Graph for given Data set

def Draw3dGraphForBikeSharing(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")

    fig = go.Figure(data=[go.Surface(z=DataFrame)])
    fig.update_layout(title='3D Graphical Representation', autosize=False,
                      height=500,
                      margin=dict(l=65, r=50, b=65, t=90))
    return fig

# Draw3dGraphForBikeSharing(newdf)

# ---------------------------------------------------------------------------------------------------------


# -----------------------Draw 3D Scatter Plot for Bike Sharing Season  Casual Cnt ------------

def Draw3dScatterPlotForSeasonCasulCnt(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    df = DataFrame[['season', 'casual', 'cnt']]
    fig = px.scatter_3d(df, x='cnt', y='casual', z='season',
                        color='cnt')
    fig.update_layout(title_text='3D Scatter Plot for Season Casual and Cnt',
                      title_x=0.5, height=500)
    return fig

# Draw3dScatterPlotForSeasonCasulCnt(newdf)
# -------------------------------------------------------------------------------------------

# ---------------------------Draw 3D Scatter Plot for Bike Sharing Season Registered and Total------


def Draw3dScatterPlotForSeasonRegisteredCnt(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    df = DataFrame[['season', 'registered', 'cnt']]
    fig = px.scatter_3d(df, x='season', y='registered', z='cnt',
                        color='cnt')
    fig.update_layout(title_text='3D Scatter Plot for Season Registered and Cnt',
                      title_x=0.5, height=500)
    return fig


# Draw3dScatterPlotForSeasonRegisteredCnt(newdf)


# ------------------------------------ 3d Scatter Plot for Bike Sharing Month Casual Cnt ---------------

def Draw3dScatterPlotForMonthsCasulCnt(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    df = DataFrame[['mnth', 'casual', 'cnt']]
    fig = px.scatter_3d(df, x='mnth', y='casual', z='cnt',
                        color='cnt')
    fig.update_layout(title_text='3D Scatter Plot for Months Casual and Cnt',
                      title_x=0.5, height=500)
    return fig


# Draw3dScatterPlotForMonthsCasulCnt(newdf)

# --------------------------------------------------------------------------------------------------------

# ------------------------------------ 3d Scatter Plot for Bike Sharing Month Registered Cnt ---------------

def Draw3dScatterPlotForMonthsRegisteredCnt(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    df = DataFrame[['mnth', 'registered', 'cnt']]
    fig = px.scatter_3d(df, x='mnth', y='registered', z='cnt',
                        color='cnt')
    fig.update_layout(title_text='3D Scatter Plot for Months Registered and Cnt',
                      title_x=0.5, height=500)
    return fig


# Draw3dScatterPlotForMonthsRegisteredCnt(newdf)

# --------------------------------------------------------------------------------------------------------

# -------------------------3d Scatter Plot of Temp , Casual , Total Cnt of Bikes--------------------------

def Draw3dScatterPlotForTempCasualCnt(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    df = DataFrame[['temp', 'casual', 'cnt']]
    fig = px.scatter_3d(df, x='temp', y='casual', z='cnt',
                        color='temp')
    fig.update_layout(title_text='3D Scatter Plot for Temp Casual and Cnt',
                      title_x=0.5, height=500)
    return fig

# Draw3dScatterPlotForTempCasualCnt(newdf)

# ------------------------------------------------------------------------------------------------------------

# -------------------------3d Scatter Plot of Temp , Registered , Total Cnt of Bikes--------------------------


def Draw3dScatterPlotForTempRegisteredCnt(DataFrame):
    if type(DataFrame).__name__ != "DataFrame":
        raise Exception("Oops! It is Not a DataFrame")
    df = DataFrame[['temp', 'registered', 'cnt']]
    fig = px.scatter_3d(df, x='temp', y='registered', z='cnt',
                        color='cnt')
    fig.update_layout(title_text='3D Scatter Plot for Temp Registered and Cnt',
                      title_x=0.5, height=500)
    return fig

# Draw3dScatterPlotForTempRegisteredCnt(newdf)

# ------------------------------------------------------------------------------------------------------------
