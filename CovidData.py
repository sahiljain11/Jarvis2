import pandas as pd
#import pycountry

#MAPS

#read data global confirmed
cols_to_use = [1,-1]
URL_DATASET1 = r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
df_confirm = pd.read_csv(URL_DATASET1)
df_confirm = df_confirm[df_confirm.columns[cols_to_use]]
df_confirm = df_confirm.groupby('Country/Region').agg('sum')
df_confirm = df_confirm.reset_index() #this line is to fix the header that is moved to an entry after the groupby method

#read data global deaths
cols_to_use = [1,-1]
URL_DATASET2 = r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
df_deaths = pd.read_csv(URL_DATASET2)
df_deaths = df_deaths[df_deaths.columns[cols_to_use]]
df_deaths = df_deaths.groupby('Country/Region').agg('sum')
df_deaths = df_deaths.reset_index()
#potential problem: congo data is split into Brazzaville and Kinshasa. does QML Maps adjust for this?
#print(df_confirm.iloc[:50])


#-------------------->final global confirmed and deaths dictionaries {'Afghanistan': 31517, 'Albania': 2535, 'Algeria': 13907...}
confirm_dict = dict(zip(df_confirm['Country/Region'], df_confirm.iloc[:, -1]))
#print(confirm_dict)

deaths_dict = dict(zip(df_deaths['Country/Region'], df_deaths.iloc[:, -1]))
#print(deaths_dict)


#read data US confirmed
cols_to_use = [6,-1]
URL_DATASET3 = r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
df_usconfirm = pd.read_csv(URL_DATASET3)
df_usconfirm = df_usconfirm[df_usconfirm.columns[cols_to_use]]
df_usconfirm = df_usconfirm.groupby('Province_State').agg('sum')
df_usconfirm = df_usconfirm.reset_index()
#print(df_usconfirm.iloc[:50])

#read data US deaths
cols_to_use = [6,-1]
URL_DATASET4 = r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'
df_usdeaths = pd.read_csv(URL_DATASET4)
df_usdeaths = df_usdeaths[df_usdeaths.columns[cols_to_use]]
df_usdeaths = df_usdeaths.groupby('Province_State').agg('sum')
df_usdeaths = df_usdeaths.reset_index()
#print(df_usdeaths.iloc[:50])

#-------------------->final domestic (US) confirmed and deaths dictionaries {'Alabama': 950, 'Alaska': 14, 'American Samoa': 0...}
usconfirm_dict = dict(zip(df_usconfirm['Province_State'], df_usconfirm.iloc[:, -1]))
#print(usconfirm_dict)
usdeaths_dict = dict(zip(df_usdeaths['Province_State'], df_usdeaths.iloc[:, -1]))
#print(usdeaths_dict)









# #CHARTS: this code takes only a select handful of countries
# #make the console display more, using way too many lines
# desired_width=320
# pd.set_option('display.width', desired_width)
# np.set_printoptions(linewidth=desired_width)
# pd.set_option('display.max_columns',10)
#
# #read data  /// where data is number of days since 1/22/20
# URL_DATASET = r'https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv'
# df1 = pd.read_csv(URL_DATASET, parse_dates=['Date'])
# first_date = df1.loc[0, 'Date']
# df1['Date'] = (df1['Date'] - first_date).dt.days
#
# #obtain data for countries  /// last twenty data points
#
# df_canada = df1[df1['Country'] == 'Canada']
# df_canada = df_canada.tail(20)
# df_us = df1[df1['Country'] == 'US']
# df_us = df_us.tail(20)
# df_mexico = df1[df1['Country'] == 'Mexico']
# df_mexico = df_mexico.tail(20)
# df_guat = df1[df1['Country'] == 'Guatemala']
# df_guat = df_guat.tail(20)
# df_cuba = df1[df1['Country'] == 'Cuba']
# df_cuba = df_cuba.tail(20)
# df_haiti = df1[df1['Country'] == 'Haiti']
# df_haiti = df_haiti.tail(20)
# df_domrep = df1[df1['Country'] == 'Dominican Republic']
# df_domrep = df_domrep.tail(20)
# df_brazil = df1[df1['Country'] == 'Brazil']
# df_brazil = df_brazil.tail(200)
# df_russia = df1[df1['Country'] == 'Russia']
# df_russia = df_russia.tail(200)
# df_uk = df1[df1['Country'] == 'United Kingdom']
# df_uk = df_uk.tail(200)
# df_china = df1[df1['Country'] == 'China']
# df_china = df_china.tail(200)
#
# #graphing data
# x1 = df_canada['Confirmed'].tolist()
# x2 = df_us['Confirmed'].tolist()
# x3 = df_mexico['Confirmed'].tolist()
# x4 = df_guat['Confirmed'].tolist()
# x5 = df_cuba['Confirmed'].tolist()
# x6 = df_haiti['Confirmed'].tolist()
# x7 = df_domrep['Confirmed'].tolist()
#
# x8 = df_us['Confirmed'].tolist()
# x9 = df_brazil['Confirmed'].tolist()
# x10 = df_russia['Confirmed'].tolist()
# x11 = df_uk['Confirmed'].tolist()
# x12 = df_china['Confirmed'].tolist()
#
# x1d = df_canada['Deaths'].tolist()
# x2d = df_us['Deaths'].tolist()
# x3d = df_mexico['Deaths'].tolist()
# x4d = df_guat['Deaths'].tolist()
# x5d = df_cuba['Deaths'].tolist()
# x6d = df_haiti['Deaths'].tolist()
# x7d = df_domrep['Deaths'].tolist()
#
# x8d = df_us['Deaths'].tolist()
# x9d = df_brazil['Deaths'].tolist()
# x10d = df_russia['Deaths'].tolist()
# x11d = df_uk['Deaths'].tolist()
# x12d = df_china['Deaths'].tolist()
#
# y1 = df_canada['Date'].tolist()
# y2 = df_us['Date'].tolist()
# y3 = df_mexico['Date'].tolist()
# y4 = df_guat['Date'].tolist()
# y5 = df_cuba['Date'].tolist()
# y6 = df_haiti['Date'].tolist()
# y7 = df_domrep['Date'].tolist()
#
# y8 = df_us['Date'].tolist()
# y9 = df_brazil['Date'].tolist()
# y10 = df_russia['Date'].tolist()
# y11 = df_uk['Date'].tolist()
# y12 = df_china['Date'].tolist()
#
#
#
# #North America confirmed dictionary
# #dictionary values are [x, y] with x and y both containing the last 20 recorded values
# #y values are days since january 22
# keys = ["Canada", "USA", "Mexico", "Guatemala", "Cuba", "Haiti", "Dominican Republic"]
# values = [[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5], [x6, y6], [x7, y7]]
#
# na_dict = {key: value for key, value in zip(keys, values)}
#
# #World confirmed dictionary
# keys2 = ["USA", "Brazil", "Russia", "UK", "China"]
# values2 = [[x8, y8], [x9, y9], [x10, y10], [x11, y11], [x12, y12]]
#
# w_dict = {key: value for key, value in zip(keys2, values2)}
#
# #North America deaths dictionary
# keys3 = ["Canada", "USA", "Mexico", "Guatemala", "Cuba", "Haiti", "Dominican Republic"]
# values3 = [[x1d, y1], [x2d, y2], [x3d, y3], [x4d, y4], [x5d, y5], [x6d, y6], [x7d, y7]]
#
# nad_dict = {key: value for key, value in zip(keys3, values3)}
#
# #World confirmed dictionary
# keys4 = ["USA", "Brazil", "Russia", "UK", "China"]
# values4 = [[x8d, y8], [x9d, y9], [x10d, y10], [x11d, y11], [x12d, y12]]
#
# wd_dict = {key: value for key, value in zip(keys4, values4)}
