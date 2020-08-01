import pandas as pd
import sys
import datetime
from PySide2 import QtWidgets as qtw
from PySide2 import QtGui as qtg
from PySide2 import QtCore as qtc
from PySide2 import QtQuick as qtq
from PySide2 import QtQml as qtm
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


        # #METHOD ALPHA, not the most effective bc using rows n dictionaries, getting data over a period of time
        # URL_DATASET5 = r'https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv'
        # # dates days since day 1
        # df1 = pd.read_csv(URL_DATASET5, parse_dates=['Date'])
        # first_date = df1.loc[0, 'Date']
        # df1['Date'] = (df1['Date'] - first_date).dt.days
        # self.stats_by_country = collections.defaultdict(list)   #collections is a dict variant: unlike a normal dict that raises a KeyError when you try to access a key that's not there, a defaultdict instead gives you some default. In this case, an empty list, commonly used for counting
        # for row in df1.itertuples():
        #     self.stats_by_country[row.Country].append(row)

        #METHOD BETA, i believe by using dataframes, this is more efficient
        URL_DATASET5 = r'https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv'
        # dates days since day 1
        df1 = pd.read_csv(URL_DATASET5, parse_dates=['Date'])
        first_date = df1.loc[0, 'Date']
        df1['Date'] = (df1['Date'] - first_date).dt.days

        df1.sort_values(by=["Country", "Date"], inplace=True)
        df1.set_index(keys=["Country"], inplace=True)
        self._data = df1

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

    # latest date confirmed cases for given country
    def confirmglobal(self, country):
        return self.confirm_dict[country]

    # latest date deaths for given country
    def deathglobal(self, country):
        return self.deaths_dict[country]

    # latest date confirmed cases for given state
    def confirmus(self, state):
        return self.usconfirm_dict[state]

    # latest date confirmed cases for given state
    def deathus(self, state):
        return self.usdeaths_dict[state]

    #METHOD ALPHA
    # def countryallconfirmed(self, country):
    #     country_stats = self.stats_by_country[country]
    #     return list(zip((entry.Date for entry in country_stats), (entry.Confirmed for entry in country_stats)))
    #
    # def countryalldeath(self, country):
    #     country_stats = self.stats_by_country[country]
    #     return list(zip((entry.Date for entry in country_stats), (entry.Deaths for entry in country_stats)))

    #METHOD BETA (3)
    # all time, all data for a given country
    def get_data_for_country(self, country):
        result = self._data[self._data.index == country]
        if result.empty:
            return
        return result

    # dictionary of alltime confirmed cases for given country; hey is a dataframe
    @qtc.Slot(str, result='QVariant')
    def countryallconfirmed(self, country):
        query = self.get_data_for_country(country)
        if (query is None):
            return
        hey = query[["Date", "Confirmed"]]
        #print([type(k) for k in {1: 2, 3: 4}.keys()])
        #print([type(k) for k in dict(zip(hey.Date,hey.Confirmed)).keys()])
        return dict(zip(hey.Date.apply(lambda x: self.epoch_to_date(x, datetime.datetime(2020, 1, 22, 0, 0))),hey.Confirmed))

    # dictionary of alltime deaths for given country; aye is a dataframe
    @qtc.Property('QVariant')
    def countryalldeath(self, country):
        aye = self.get_data_for_country(country)[["Date", "Deaths"]]
        return dict(zip(aye.Date, aye.Deaths))

    def epoch_to_date(self, num_days, epoch):
        date = epoch + datetime.timedelta(num_days)
        return date.strftime("%Y-%m-%d")


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