import pandas as pd

class Stats:

    def __init__(self):
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


        #getting data over a period of time
        URL_DATASET5 = r'https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv'
        #dates days since day 1
        df1 = pd.read_csv(URL_DATASET5, parse_dates=['Date'])
        first_date = df1.loc[0, 'Date']
        df1['Date'] = (df1['Date'] - first_date).dt.days

        df_alltimecountry = df1[df1['Country'] == 'CountryName']
        xconfirmed = df_alltimecountry['Confirmed'].tolist()
        xdeaths = df_alltimecountry['Deaths'].tolist()
        ydates = df_alltimecountry['Date'].tolist()
        self.countryallconfirmed = dict(zip(ydates, xconfirmed))
        self.countryalldeath = dict(zip(ydates, xdeaths))

    def show(self, *args): #this is called a repr/representation/"predefined method thats true for all classes"? 
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

    def confirmglobal(self, country):
        return self.confirm_dict[country]

    def deathglobal(self, country):
        return self.deaths_dict[country]

    def confirmus(self, state):
        return self.usconfirm_dict[state]

    def deathus(self, state):
        return self.usdeaths_dict[state]

    def countryallconfirmed(self, country):
        return self.countryallconfirmed[country]

    def countryalldeath(self, country):
        return self.countryalldeath[country]

my_stats = Stats()
#print(my_stats.confirmus('Texas'))
#print(my_stats.confirmglobal('Albania'))
#print(my_stats.deathglobal('Albania'))
#my_stats.show('death')
#my_stats.show('usdeath')
#my_stats.show('confirm', 'death')