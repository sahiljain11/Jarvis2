import pandas as pd
import sys
import time
import numpy
import datetime
from PySide2 import QtWidgets as qtw
from PySide2 import QtGui as qtg
from PySide2 import QtCore as qtc
from PySide2 import QtQuick as qtq
from PySide2 import QtQml as qtm
from spellchecker import SpellChecker
#import collections #METHOD BETA


class Stats(qtc.QObject):

    #Signals
    countryAllConfirmedChanged = qtc.Signal()

    def __init__(self):
        super(Stats, self).__init__()
        cols_to_use = [1, -1]
        URL_DATASET1 = r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
        df_confirm = pd.read_csv(URL_DATASET1)
        df_confirm = df_confirm[df_confirm.columns[cols_to_use]]
        df_confirm = df_confirm.groupby('Country/Region').agg('sum')
        df_confirm = df_confirm.reset_index()
        self.confirm_dict = dict(zip(df_confirm['Country/Region'], df_confirm.iloc[:, -1]))

        URL_DATASET2 = r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
        df_deaths = pd.read_csv(URL_DATASET2)
        df_deaths = df_deaths[df_deaths.columns[cols_to_use]]
        df_deaths = df_deaths.groupby('Country/Region').agg('sum')
        df_deaths = df_deaths.reset_index()
        self.deaths_dict = dict(zip(df_deaths['Country/Region'], df_deaths.iloc[:, -1]))

        cols_to_use = [6, -1]
        URL_DATASET3 = r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
        df_usconfirm = pd.read_csv(URL_DATASET3)
        df_usconfirm = df_usconfirm[df_usconfirm.columns[cols_to_use]]
        df_usconfirm = df_usconfirm.groupby('Province_State').agg('sum')
        df_usconfirm = df_usconfirm.reset_index()
        self.usconfirm_dict = dict(zip(df_usconfirm['Province_State'], df_usconfirm.iloc[:, -1]))

        URL_DATASET4 = r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'
        df_usdeaths = pd.read_csv(URL_DATASET4)
        df_usdeaths = df_usdeaths[df_usdeaths.columns[cols_to_use]]
        df_usdeaths = df_usdeaths.groupby('Province_State').agg('sum')
        df_usdeaths = df_usdeaths.reset_index()
        self.usdeaths_dict = dict(zip(df_usdeaths['Province_State'], df_usdeaths.iloc[:, -1]))

        #Aggregate ecountry data
        URL_DATASET5 = r'https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv'
        # dates days since day 1
        df1 = pd.read_csv(URL_DATASET5, parse_dates=['Date'])
        first_date = df1.loc[0, 'Date']
        df1['Date'] = (df1['Date'] - first_date).dt.days
        df1.sort_values(by=["Country", "Date"], inplace=True)
        df1.set_index(keys=["Country"], inplace=True)
        self._data = df1

        # Aggregate state data
        URL_DATASET6 = r'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'
        # dates days since day 1
        self.covid_states = pd.read_csv(URL_DATASET6, parse_dates=['date'])
        first_date = self.covid_states.loc[0, 'date']
        self.covid_states['date'] = (self.covid_states['date'] - first_date).dt.days
        self.covid_states.sort_values(by=["state", "date"], inplace=True)
        self.covid_states.set_index(keys=["state"], inplace=True)

        # Aggregate county data
        URL_DATASET7 = r'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'
        self.covid_counties = pd.read_csv(URL_DATASET7, parse_dates=['date'])
        first_date = self.covid_counties.loc[0, 'date']
        self.covid_counties['date'] =  (self.covid_counties['date'] - first_date).dt.days
        self.covid_counties.sort_values(by=["county", "state", "date"], inplace=True)
        self.covid_counties.set_index(keys=["county"], inplace=True)


        yesterday = (datetime.datetime.now() -  datetime.timedelta(days=1)).strftime('%m-%d-%Y')
        url = r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/' + yesterday + ".csv"
        URL_DATASET8 = url
        self.covid_states = pd.read_csv(URL_DATASET8)
        self.covid_states.set_index(keys=["Province_State"], inplace=True)
        self.states = self.covid_states.index.tolist()

        
        # Geographical data for each country and state
        REFERENCE = r"https://raw.githubusercontent.com/datasets/covid-19/master/data/reference.csv"
        self.reference = pd.read_csv(REFERENCE)
        self.counties = self.reference[(self.reference.Country_Region == "US") & (self.reference.Province_State.notnull()) & (self.reference.Admin2.notnull()) & (self.reference.Lat.notnull()) & (self.reference.Long_.notnull()) ][["Admin2", "Province_State", "Lat", "Long_"]].values.tolist()
        self.countries = self.reference[self.reference.Province_State.isnull()][["Country_Region", "Lat", "Long_"]].values.tolist()


        self.counties_names = self.name_list_maker(self.counties)
        self.countries_names = self.name_list_maker(self.countries)
        self.states_names = self.name_list_maker(self.states)
        self.spell = self.initspeller()

    def show(self, *args):
        for i in args:
            if i == 'death':
                print (self.deaths_dict)
            elif i == 'confirm':
                print (self.confirm_dict)
            elif i == 'usdeath':
                print(self.usdeaths_dict)
            elif i == 'usconfirm':
                print(self.usconfirm_dict)
            else:
                print ("that's not even a dictionary bruh")

    # latest date confirmed cases for given state
    @qtc.Slot(str, result=int)
    def confirmus(self, state):
        query = self.usconfirm_dict.get(state)
        if query is None:
            return 0
        return query

    # latest date confirmed cases for given state
    @qtc.Slot(str, result=int)
    def deathus(self, state):
        query = self.usdeaths_dict.get(state)
        if query is None:
            return 0
        return query

    #Get the time data for a county
    @qtc.Slot(str, str, result='QVariant')       
    def get_data_for_county(self, county, state):
        result = self.covid_counties[(self.covid_counties.index == county) & (self.covid_counties.state == state)]
        if result.empty:
            return 
        return result
    
    #Get the time data for a state
    @qtc.Slot(str, result='QVariant')
    def get_data_for_state(self, state):
        result = self.covid_states[self.covid_states.index == state]
        if result.empty:
            return 
        return result.to_dict()

    # all time, all data for a given country
    @qtc.Slot(str, result='QVariant')
    def get_data_for_country(self, country):
        result = self._data[self._data.index == country]
        if result.empty:
            return 
        return result
    
    @qtc.Slot(result='QVariant')
    def get_countries(self):
        return self.countries
    
    @qtc.Slot(result=int)
    def get_num_countries(self):
        return len(self.countries)
    
    @qtc.Slot(result='QVariant')
    def get_states(self):
        return self.states

    @qtc.Slot(result=int)
    def get_num_counties(self):
        return len(self.counties)

    # latest date confirmed cases for given country
    @qtc.Slot(str, result=int)
    def confirmglobal(self, country):
        query = self.confirm_dict.get(country)
        if query is None:
            return 0
        return query

    # latest date deaths for given country
    @qtc.Slot(str, result=int)
    def deathglobal(self, country):
        query = self.deaths_dict.get(country)
        if query is None:
            return 0
        return query
    
    @qtc.Slot(str, result=int)
    def recoverglobal(self, country):
        query = self.get_data_for_country(country)
        # Return nothing if the country is not found
        if (query is None):
            return
        # Retrieve the time series data for deathes
        aye = query[["Date", "Recovered"]]
        return aye.tail(1).at[country,"Recovered"]
    
    @qtc.Slot(str, str, result='QVariant')
    def countyallconfirmed(self, county, state):
        # Attempt to query for the given country
        query = self.get_data_for_county(county, state)
        # Return nothing if the country is not found
        if (query is None):
            return
        # Retrieve the time series data for deathes
        aye = query[["date", "cases"]]
        # Zip the data into a dict and convert the time data to dates
        return dict(zip(aye.date.apply(lambda x: self.epoch_to_date(x, datetime.datetime(2020, 1, 22, 0, 0))), aye.cases))
    

    @qtc.Slot(str, str, result='QVariant')
    def countyalldeath(self, county, state):
        # Attempt to query for the given country
        query = self.get_data_for_county(county, state)
        # Return nothing if the country is not found
        if (query is None):
            return
        # Retrieve the time series data for deathes
        aye = query[["date", "deaths"]]
        # Zip the data into a dict and convert the time data to dates
        return dict(zip(aye.date.apply(lambda x: self.epoch_to_date(x, datetime.datetime(2020, 1, 22, 0, 0))), aye.deaths))

    # dictionary of alltime confirmed cases for given country; hey is a dataframe
    @qtc.Slot(str, result='QVariant')
    def countryallconfirmed(self, country):
        # Attempt to query for the given country
        query = self.get_data_for_country(country)
        # Return nothing if the country is not found
        if (query is None):
            return
        # Retrieve the time series data for cases
        hey = query[["Date", "Confirmed"]]
        # Zip the data into a dictionary and convert the time data to dates
        return dict(zip(hey.Date.apply(lambda x: self.epoch_to_date(x, datetime.datetime(2020, 1, 22, 0, 0))),hey.Confirmed))

    # dictionary of alltime deaths for given country; aye is a dataframe
    @qtc.Slot(str, result='QVariant')
    def countryalldeath(self, country):
        # Attempt to query for the given country
        query = self.get_data_for_country(country)
        # Return nothing if the country is not found
        if (query is None):
            return
        # Retrieve the time series data for deathes
        aye = query[["Date", "Deaths"]]
        # Zip the data into a dict and convert the time data to dates
        return dict(zip(aye.Date.apply(lambda x: self.epoch_to_date(x, datetime.datetime(2020, 1, 22, 0, 0))), aye.Deaths))
    
    @qtc.Slot(str, result='QVariant')
    def countryallrecovered(self, country):
        query = self.get_data_for_country(country)
        # Return nothing if the country is not found
        if (query is None):
            return
        # Retrieve the time series data for deathes
        aye = query[["Date", "Recovered"]]
        # Zip the data into a dict and convert the time data to dates
        return dict(zip(aye.Date.apply(lambda x: self.epoch_to_date(x, datetime.datetime(2020, 1, 22, 0, 0))), aye.Recovered))

    # Converts the number of days since the given epoch to a date in the form Year-Month-Day
    def epoch_to_date(self, num_days, epoch):
        date = epoch + datetime.timedelta(num_days)
        return date.strftime("%Y-%m-%d")

    #Autocorrects Users input query
    def auto_correct_state_query(self,input_message):
        #
        if input_message.lower() in [state.lower() for state in self.states_names]:
            return input_message
        else:
            #corrects the input message to a state
            return self.spell.correction(input_message)

    @qtc.Slot(str, result=str)
    def auto_correct_country_query(self, input_message):
        #if input message is a country return
        if input_message.lower() in [country.lower() for country in self.countries_names]:
            return input_message
        #corrects the input message to country
        else:
            return self.spell.correction(input_message)

    @qtc.Slot(str, result=str)
    def auto_correct_county_query(self,input_message):
        #if input message is a county return
        if input_message.lower() in [county.lower() for county in self.counties_names]:
            return input_message
        #corrects the input message to a county
        else:
            return self.spell.correction(input_message)

    def initspeller(self):
        spell = SpellChecker(language=None, case_sensitive=False)
        spell.word_frequency.load_words(self.counties_names)
        spell.word_frequency.load_words(self.countries_names)
        spell.word_frequency.load_words(self.states_names)
        return spell

    def name_list_maker(self,list_of_lists):
        output_list = [None] * len(list_of_lists)
        #print(list_of_lists[0][0], 'hi')
        for list in range(0,len(list_of_lists)):
            output_list[list] = list_of_lists[list][0]
        return output_list

if __name__ == '__main__':
   
    print("This is the main")

    app = qtw.QApplication(sys.argv)
    engine = qtm.QQmlApplicationEngine()
    root_context = engine.rootContext()

    my_stats = Stats()
    root_context.setContextProperty('corona', my_stats)
    engine.load(qtc.QUrl.fromLocalFile('Covid.qml'))

    #print(my_stats.confirmus('Texas'))
    #print(my_stats.confirmglobal('Albania'))
    #print(my_stats.deathglobal('Albania'))
    #my_stats.show('death')
    #my_stats.show('usdeath')
    #my_stats.show('confirm', 'death')

    #METHOD BETA
    #print(my_stats.get_data_for_country("Russia"))
    #print(my_stats.countryallconfirmed("Russia"))
    #print(my_stats.countryalldeath("Russia"))

    #METHOD ALPHA
    #print(my_stats.countryallconfirmed("Canada")[:50])
    
    sys.exit(app.exec_())