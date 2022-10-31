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


# ----------------------Features Importance Figure for Casual Bike--------------------------------
FetureImportanceFigureForCasualBikes = dv.DrawFeatureImportanceGraph(
    FeatureImportanceForCasual)

# ----------------------Features Importance Figure for Registered Bike--------------------------------
FetureImportanceFigureForRegisteredBikes = dv.DrawFeatureImportanceGraph(
    FeatureImportanceForRegistered)

# ----------------------Features Importance Figure for Cnt (Total Bikes)--------------------------------
FetureImportanceFigureForCntBikes = dv.DrawFeatureImportanceGraph(
    FeatureImportanceForCnt)

# -------------------------------- Figure For HeatMap---------------------------------------------------
CorrelationFigureForHeatMap = dv.DrawCorrelationHeatMap(UpdatedDataFrame)

# -----------------------------------Box Plot for Casual Registered and Total Bikes

BoxPlotFigureForCasualRegisteredAndCnt = dv.BoxPlotForCasualRegisteredAndTotalBikes(
    UpdatedDataFrame)

# Details for Casual Bikes Model

DetailInformationForCasualBikeModel = 'The above Machine Simulation Tool can be used to Predict the value of Casual. The model that we have implemented on our dataset is the Random Forest Regression Model. We tried different Models and Techniques as well but we achieved the maximum accuracy with this Model. The accuracy of this model is ' + \
    str(round(ScoresForCasualBikes*100, 2)) + '.So, it is a pretty accurate model as we were expecting. After a lot of trials and errors, we came up with this model and the Features to use in it. With the sliders, you can try different values and use it to predict the Cnt for Casual Bikes. The chart in top right shows the most significant Features of our model which are contributing most in the model'


# Details for Registered Bikes Model

DetailInformationForRegisteredBikeModel = 'The above Machine Simulation Tool can be used to Predict the value of Registered Bikes. The model that we have implemented on our dataset is the Random Forest Regression Model. We tried different Models and Techniques as well but we achieved the maximum accuracy with this Model. The accuracy of this model is ' + \
    str(round(ScoresForRegisteredBikes*100, 2)) + '.So, it is a pretty accurate model as we were expecting. After a lot of trials and errors, we came up with this model and the Features to use in it. With the sliders, you can try different values and use it to predict the Numbers for Registered Bikes. The chart in top right shows the most significant Features of our model which are contributing most in the model'

# ---------------------------Distribution Column Figures -----------------------------------------
SeasonDistributionColumnFigure = dv.SeasonDistributionColumn(UpdatedDataFrame)
YearDistributionColumnFigure = dv.YearDistributionColumn(UpdatedDataFrame)
MonthsDistributionColumnFigure = dv.MonthsDistributionColumn(UpdatedDataFrame)
WeekDayDistributionColumnFigure = dv.WeekDayDistributionColumn(
    UpdatedDataFrame)
YearMonthWeekDayDistributionColumnFigure = dv.YearMonthWeekDayDistributionColumn(
    UpdatedDataFrame)
WorkingDayDistributionColumnFigure = dv.WorkingDayDistributionColumn(
    UpdatedDataFrame)
HolidayDistributionColumnFigure = dv.HolidayDistributionColumn(
    UpdatedDataFrame)
WeatherDistributionColumnFigure = dv.WeatherDistributionColumn(
    UpdatedDataFrame)
# -----------------------------------------------------------------------------------------------

# ------------------------------------BoxPlot Column Figures----------------------------------------
FigureForDrawChartYearAgainstCRC = dv.DrawChartYearAgainstCRC(UpdatedDataFrame)
FigureForDrawChartSeasonsAgainstCRC = dv.DrawChartSeasonsAgainstCRC(
    UpdatedDataFrame)
FigureForDrawChartMonthsAgianstCasual = dv.DrawChartMonthsAgianstCasual(
    UpdatedDataFrame)
FigureForDrawChartMonthsAgianstRegistered = dv.DrawChartMonthsAgianstRegistered(
    UpdatedDataFrame)
FigureForDrawChartMonthsAgianstCnt = dv.DrawChartMonthsAgianstCnt(
    UpdatedDataFrame)
FigureForDrawChartWeekdayAgianstCasual = dv.DrawChartWeekdayAgianstCasual(
    UpdatedDataFrame)
FigureForDrawChartWeekdayAgianstRegistered = dv.DrawChartWeekdayAgianstRegisterd(
    UpdatedDataFrame)
FigureForDrawChartWeekdayAgianstCasualCnt = dv.DrawChartWeekdayAgianstCnt(
    UpdatedDataFrame)
FigureForDrawChartHolidayDayAgainstCRC = dv.DrawChartHolidayDayAgainstCRC(
    UpdatedDataFrame)
FigureForDrawChartWorkingDayAgainstCRC = dv.DrawChartWorkingDayAgainstCRC(
    UpdatedDataFrame)
FigureForDrawChartWeathersitAgainstCRC = dv.DrawChartWeathersitAgainstCRC(
    UpdatedDataFrame)
# ----------------------------------------------------------------------------------------------------

# --------------------------------------Scatter Plot Figures-----------------------------------------
FigureForDrawScatterPlotTempAgainstCRC = dv.DrawScatterPlotTempAgainstCRC(
    UpdatedDataFrame)
FigureForDrawScatterPlotATempAgainstCRC = dv.DrawScatterPlotATempAgainstCRC(
    UpdatedDataFrame)
FigureForDrawScatterPoltHumidityAgainstCRC = dv.DrawScatterPoltHumidityAgainstCRC(
    UpdatedDataFrame)
FigureForDrawScatterPoltWindSpeedAgainstCRC = dv.DrawScatterPoltWindSpeedAgainstCRC(
    UpdatedDataFrame)

# -------------------------------------------------------------------------------------------------------

# -----------------------------Making the Slider for each Significant Value so User can manupulate and Predict value (Casual)-----
SliderForWorkingDayLabel = FeatureImportanceForCasual.index[0]
SliderForWorkingDayMinValue = math.floor(
    DataFrame[SliderForWorkingDayLabel].min())
SliderForWorkingDayMeanValue = round(
    DataFrame[SliderForWorkingDayLabel].mean())
SliderForWorkingDayMaxValue = round(DataFrame[SliderForWorkingDayLabel].max())

SliderForTemp = FeatureImportanceForCasual.index[1]
SliderForTempMinValue = math.floor(DataFrame[SliderForTemp].min())
SliderForTempMeanValue = round(DataFrame[SliderForTemp].mean())
SliderForTempMaxValue = round(DataFrame[SliderForTemp].max())

SliderForATemp = FeatureImportanceForCasual.index[2]
SliderForATempMinValue = math.floor(DataFrame[SliderForTemp].min())
SliderForATempMeanValue = round(DataFrame[SliderForTemp].mean())
SliderForATempMaxValue = round(DataFrame[SliderForTemp].max())

SliderForHum = FeatureImportanceForCasual.index[3]
SliderForHumMinValue = math.floor(DataFrame[SliderForHum].min())
SliderForHumMeanValue = round(DataFrame[SliderForHum].mean())
SliderForHumMaxValue = round(DataFrame[SliderForHum].max())

# -----------------------------------------------------------------------------------------------------

# -----------------------------Making the Slider for each Significant Value so User can manupulate and Predict value Registered -----
SliderForYear = FeatureImportanceForRegistered.index[0]
SliderForYearMinValue = math.floor(DataFrame[SliderForYear].min())
SliderForYearMeanValue = round(DataFrame[SliderForYear].mean())
SliderForYearMaxValue = round(DataFrame[SliderForYear].max())


SliderForTemp1 = FeatureImportanceForRegistered.index[1]
SliderForTempMinValue1 = math.floor(DataFrame[SliderForTemp1].min())
SliderForTempMeanValue1 = round(DataFrame[SliderForTemp1].mean())
SliderForTempMaxValue1 = round(DataFrame[SliderForTemp1].max())


SliderForATemp1 = FeatureImportanceForRegistered.index[2]
SliderForATempMinValue1 = math.floor(DataFrame[SliderForATemp1].min())
SliderForATempMeanValue1 = round(DataFrame[SliderForATemp1].mean())
SliderForATempMaxValue1 = round(DataFrame[SliderForATemp1].max())


SliderForMnt = FeatureImportanceForRegistered.index[3]
SliderForMntMinValue = math.floor(DataFrame[SliderForMnt].min())
SliderForMntMeanValue = round(DataFrame[SliderForMnt].mean())
SliderForMntMaxValue = round(DataFrame[SliderForMnt].max())


SliderForSeason = FeatureImportanceForRegistered.index[4]
SliderForSeasonMinValue = math.floor(DataFrame[SliderForSeason].min())
SliderForSeasonMeanValue = round(DataFrame[SliderForSeason].mean())
SliderForSeasonMaxValue = round(DataFrame[SliderForSeason].max())


# -----------------------------------------------------------------------------------------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# Header Component

HeaderComponent = html.Div(
    "DashBoard for Bike Sharing Data Visualization", className='header')


# Component having the required Slider and Informaton Graph for Casual Bikes

CasualBikeDisplayComponent = html.Div([
    dbc.Row([
        dbc.Col([
            html.Div([
                # div for SliderForWorkingDay
                html.Div([
                    html.H3(SliderForWorkingDayLabel, className='heading'),
                    dcc.Slider(
                        id='sliderforWorkingday', className='slider',
                        min=SliderForWorkingDayMinValue, max=SliderForWorkingDayMaxValue, value=0.5,
                        marks={i: '{}'.format(i) for i in range(
                            SliderForWorkingDayMinValue, SliderForWorkingDayMaxValue+1)}
                    ),
                    html.Div(id='sliderforWorkingday-output')
                ]),
                # div for SliderForTemp Slider
                html.Div([
                    html.H3(SliderForTemp, className='heading'),
                    dcc.Slider(
                        id='sliderforTemp', className='slider',
                        min=SliderForTempMinValue, max=SliderForTempMaxValue, value=0.30,
                        marks={i: '{}'.format(i) for i in range(
                            SliderForTempMaxValue, SliderForTempMaxValue+1)}
                    ),
                    html.Div(id='sliderforTemp-output')
                ]),
                # div for SliderForATemp Slider
                html.Div([
                    html.H3(SliderForATemp, className='heading'),
                    dcc.Slider(
                        id='sliderforATemp', className='slider',
                        min=SliderForATempMinValue, max=SliderForATempMaxValue, value=0.75,
                        marks={i: '{}'.format(i) for i in range(
                            SliderForATempMaxValue, SliderForATempMaxValue+1)}
                    ),
                    html.Div(id='sliderforATemp-output')
                ]),
                # div for Slider Humidity
                html.Div([
                    html.H3(SliderForHum, className='heading'),
                    dcc.Slider(
                        id='sliderforhum', className='slider',
                        min=SliderForHumMinValue, max=SliderForHumMaxValue, value=0.49,
                        marks={i: '{}'.format(i) for i in range(
                            SliderForHumMinValue, SliderForHumMaxValue+1)}
                    ),
                    html.Div(id='sliderforhum-output')
                ]),
                # Predicted Cnt Container for Casual Bikes
                html.Div(html.P(id='prediction-result',
                                className='predict-cnt-div')),

            ], className='div1'),
            # --------   Row Colum Information div Container--------
            html.Div([
                html.Div([
                    dbc.Button(
                        "DataSet Detalis",
                        id="collapse-btn-casual",
                        color="primary",
                        className="mb-3"
                    ),
                    dbc.Collapse(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.P(
                                        "Number of the Columns in DataSet: " + str(NumOfColumns), className='text-btn'),
                                    html.P("Number of the Rows in DataSet: " +
                                           str(NumOfRows), className='text-btn')
                                ]
                            ),
                            style={'margin': 8}),
                        id="collapse-row-info",
                    ),
                ]
                ),
            ], className='div1')
        ], width=4),
        dbc.Col([
            html.Div([
                dcc.Graph(figure=FetureImportanceFigureForCasualBikes)
            ], className='div2')
        ], width=8)
    ])
])

# Copmonent Showing the Details For Casual Bike By Using Machine Model

ShowingDetailofCasualBikesModelComponent = html.Div([
    html.Div([
        dbc.Button(
            "Model Details",
            id="collapse-casual-info-btn",
            color="primary",
            className="casualbike-detail-btn",
        ),
        dbc.Collapse(
            dbc.Card(
                dbc.CardBody(DetailInformationForCasualBikeModel),
                style={'margin': 8}
            ),
            id="collapse-casual-info",
            style={'padding': 10}
        )
    ])
], className='model-information-div')

# Componet for empty spacing specially for styling purpose

EmptyBoxComponent = html.Div([
    html.Div(
        [

        ]
    )
], className='empty-box')


# Componet having information for Registered Bikes and giving the Prediction by using slider and Machine Model
RegisteredBikeDisplayComponent = html.Div([
    dbc.Row([
        dbc.Col([
            html.Div([
                # Div for slider of Year
                html.Div([
                     html.H3(SliderForYear, className='heading'),
                     dcc.Slider(
                         id='sliderforyear', className='slider',
                         min=SliderForYearMinValue, max=SliderForYearMaxValue, value=0.75,
                         marks={i: '{}'.format(i) for i in range(
                            SliderForYearMinValue, SliderForYearMaxValue+1)}
                     ),
                     html.Div(id='sliderforyear-output')
                     ]),
                # Div for Slide for Temperature
                html.Div([
                    html.H3(SliderForTemp1, className='heading'),
                    dcc.Slider(
                        id='sliderforTemp1', className='slider',
                        min=SliderForTempMinValue1, max=SliderForTempMaxValue1, value=0.6,
                        marks={i: '{}'.format(i) for i in range(
                            SliderForTempMinValue1, SliderForTempMaxValue1+1)}
                    ),
                    html.Div(id='sliderforTemp1-output')
                ]),
                # Div for Slider ATemperature (Used Different Technique to measure Temperature)
                html.Div([
                    html.H3(SliderForATemp1, className='heading'),
                    dcc.Slider(
                        id='sliderforATemp1', className='slider',
                        min=SliderForATempMinValue1, max=SliderForATempMaxValue1, value=0.5,
                        marks={i: '{}'.format(i) for i in range(
                            SliderForATempMinValue1, SliderForATempMaxValue1+1)}
                    ),
                    html.Div(id='sliderforATemp1-output')
                ]),
                # Div for Slider for  Months
                html.Div([
                    html.H3(SliderForMnt, className='heading'),
                    dcc.Slider(
                        id='sliderforMnt', className='slider',
                        min=SliderForMntMinValue, max=SliderForMntMaxValue, value=SliderForMntMeanValue,
                        marks={i: '{}'.format(i) for i in range(
                            SliderForMntMinValue, SliderForMntMaxValue+1)}
                    ),
                    html.Div(id='sliderformnt-output')
                ]),
                # Div for Slider for Seasons
                html.Div([
                    html.H3(SliderForSeason, className='heading'),
                    dcc.Slider(
                        id='SliderForSeason', className='slider',
                        min=SliderForSeasonMinValue, max=SliderForSeasonMaxValue, value=SliderForSeasonMeanValue,
                        marks={i: '{}'.format(i) for i in range(
                            SliderForSeasonMinValue, SliderForSeasonMaxValue+1)}
                    ),
                    html.Div(id='sliderforseason-output')
                ]),
                # Div for Predicted Cnt for Registered Bikes Container
                html.Div(html.P(id='prediction-result-regis',
                                className='predict-cnt-div')),
            ], className='div1'),
            # --------   Row Colum Information div Container--------
            html.Div([
                html.Div([
                    dbc.Button(
                        "DataSet Detalis",
                        id="collapse-row-info-cnt-btn",
                        color="primary",
                        className="mb-3"
                    ),
                    dbc.Collapse(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.P(
                                        "Number of the Columns in DataSet: " + str(NumOfColumns), className='text-btn'),
                                    html.P("Number of the Rows in DataSet: " +
                                           str(NumOfRows), className='text-btn')
                                ]
                            ),
                            style={'margin': 8}),
                        id="collapse-cnt-info",
                    ),
                ]
                ),
            ], className='div1')
        ], width=4),
        dbc.Col([
            html.Div([
                dcc.Graph(figure=FetureImportanceFigureForRegisteredBikes)
            ], className='div2')
        ], width=8)
    ])
])

# # Copmonent Showing the Details For Registered Bike By Using Machine Model

ShowingDetailofRegistredBikesModelComponent = html.Div([
    html.Div([
        dbc.Button(
            "Model Details",
            id="collapse-registered-info-btn",
            color="primary",
            className="casualbike-detail-btn",
        ),
        dbc.Collapse(
            dbc.Card(
                dbc.CardBody(DetailInformationForRegisteredBikeModel),
                style={'margin': 8}
            ),
            id="collapse-registered-info",
            style={'padding': 10}
        )
    ])
], className='model-information-div')

# Component fot BoxPlot For Casual Registered and Total Bikes
BoxPlotComponentForCasualRegisteredAndCnt = html.Div([
    dcc.Graph(figure=BoxPlotFigureForCasualRegisteredAndCnt,
              style={'margin': 10})
], className='box-plot-casual-regis-cnt')

# Component showing the Correlation HeatMap Figure

CorrelationHeatMapComponent = html.Div([
    dcc.Graph(figure=CorrelationFigureForHeatMap, style={'margin': 10})
], className='heat-map-plot')

# Distribution Column Components

# Season and year distribution Copmponent
SeasonYearDistributionColumnComponent = html.Div([
    html.Div([
    dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Graph(figure=SeasonDistributionColumnFigure,
                          style={'margin': 10})
            ])
        ], width=6),
        dbc.Col([
            html.Div([
                dcc.Graph(figure=YearDistributionColumnFigure,
                          style={'margin': 10})])
        ], width=6)
    ])], className="distribution-col"),
    ], className='distribution-column')
    
# Component for showing the distribution Plot for months

MonthsDistributionColumnComponent = html.Div([
    dcc.Graph(figure=MonthsDistributionColumnFigure, style={'margin': 10})
],className='Months-distribution')

# Working-day and Holiday distribution column component 

WorkingDayAndHolidayDistributionColumnComponent = html.Div([
    html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Graph(figure=WorkingDayDistributionColumnFigure,
                              style={'margin': 10})
                ])
            ], width=6),
            dbc.Col([
                html.Div([
                    dcc.Graph(figure=HolidayDistributionColumnFigure,
                              style={'margin': 10})])
            ], width=6)
        ])], className="distribution-col"),
], className='distribution-column')

# Year Months Weeeksday Component 
YearMonthWeekDayDistributionColumnComponet = html.Div([
    dcc.Graph(figure=YearMonthWeekDayDistributionColumnFigure, style={'margin': 10})
], className='Year-Months-weekday-distribution')

# Weather Distribution Component

WeatherDistributionColumnComponent = html.Div([
    dcc.Graph(figure=WeatherDistributionColumnFigure,
              style={'margin': 10})
], className='weather-distribution')
# Main Layout
app.layout = html.Div(
                    [
                       HeaderComponent,
                       CasualBikeDisplayComponent,
                       ShowingDetailofCasualBikesModelComponent,
                       EmptyBoxComponent,
                       RegisteredBikeDisplayComponent,
                       ShowingDetailofRegistredBikesModelComponent,
                       EmptyBoxComponent,
                       BoxPlotComponentForCasualRegisteredAndCnt,
                       EmptyBoxComponent,
                       CorrelationHeatMapComponent,
                       EmptyBoxComponent,
                       SeasonYearDistributionColumnComponent,
                       EmptyBoxComponent,
                       MonthsDistributionColumnComponent,
                       EmptyBoxComponent,
                       WorkingDayAndHolidayDistributionColumnComponent,
                       EmptyBoxComponent,
                       YearMonthWeekDayDistributionColumnComponet,
                       EmptyBoxComponent,
                       WeatherDistributionColumnComponent,
                       EmptyBoxComponent,
                    ]
                    , className='Main-background')


@app.callback(
    Output("collapse-row-info", "is_open"),
    [Input("collapse-btn-casual", "n_clicks")],
    [State("collapse-row-info", "is_open")],
)


def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# implementing the CallBack function for Casual Bikes Model Detail showing Button


@app.callback(
    Output("collapse-casual-info", "is_open"),
    [Input("collapse-casual-info-btn", "n_clicks")],
    [State("collapse-casual-info", "is_open")],
)
# checking the toggle Collapse

def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# checking for the registered Collapse


@app.callback(
    Output("collapse-cnt-info", "is_open"),
    [Input("collapse-row-info-cnt-btn", "n_clicks")],
    [State("collapse-cnt-info", "is_open")])


def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# checking for the registered model info Collapse


@app.callback(
    Output("collapse-registered-info", "is_open"),
    [Input("collapse-registered-info-btn", "n_clicks")],
    [State("collapse-registered-info", "is_open")])


def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


# The Callback function Returning the Number of Casual bikes by changing different values using slider

@app.callback(Output(component_id='prediction-result', component_property="children"),
              # the values corresponding to the 4 sliders by using their ids and value property
              [Input('sliderforWorkingday', 'value'), Input(
                  'sliderforTemp', 'value'), Input('sliderforATemp', 'value'), Input('sliderforhum', 'value')]
              )


def UpadtePredictionForCasual(X1, X2, X3, X4):
    inputX = np.array(
        [DataFrame['season'].mean(),
         DataFrame['yr'].mean(),
         DataFrame['mnth'].mean(),
         DataFrame['holiday'].mean(),
         DataFrame['weekday'].mean(),
         X4,
         DataFrame['weathersit'].mean(),
         X3,
         X2,
         X1,
         DataFrame['windspeed'].mean(),

         ]).reshape(1, -1)
    PredictedNumber = RegressorForCasualBikes.predict(inputX)[0]
    return "Predicted Numbers of Casual Bikes: {}".format(round(PredictedNumber, 1))


# The Callback function Returning the Number of Registerd bikes by changing different values using slider

@app.callback(Output(component_id='prediction-result-regis', component_property="children"),
              # the values corresponding to the 4 sliders by using their ids and value property
              [Input('sliderforyear', 'value'), Input(
                  'sliderforTemp1', 'value'), Input('sliderforATemp1', 'value'), Input('sliderforMnt', 'value'), Input('SliderForSeason', 'value')]
              )


def UpdateThePredictionForRegistered(X1, X2, X3, X4, X5):
    inputX = np.array(
        [
            X5,
            X4,
            X3,
            DataFrame['holiday'].mean(),
            DataFrame['weekday'].mean(),
            DataFrame['workingday'].mean(),
            DataFrame['weathersit'].mean(),
            X2,
            X1,
            DataFrame['hum'].mean(),
            DataFrame['windspeed'].mean(),
        ]).reshape(1, -1)
    PredictedNumber = RegressorForRegisteredBikes.predict(inputX)[0]
    return "Predicted Numbers of Registered Bikes: {}".format(round(PredictedNumber, 1))



if __name__ == '__main__':
    # app.run_server(debug=True)
    app.run_server(port=8005, host='0.0.0.0')
