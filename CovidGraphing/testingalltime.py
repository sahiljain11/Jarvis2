import pandas as pd
import collections

#https://stackoverflow.com/questions/17426292/what-is-the-most-efficient-way-to-create-a-dictionary-of-two-pandas-dataframe-co
#this can be made even more efficient
class Stats:

    def __init__(self):
        cols_to_use = [1, -1]
        URL_DATASET1 = r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
        df_confirm = pd.read_csv(URL_DATASET1) #reads data
        df_confirm = df_confirm[df_confirm.columns[cols_to_use]] #takes columns 1 to 1 before end
        df_confirm = df_confirm.groupby('Country/Region').agg('sum')
        df_confirm = df_confirm.reset_index()
        self.confirm_dict = dict(zip(df_confirm['Country/Region'], df_confirm.iloc[:, -1])) #makes dict using col country/region and the last date

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

        # METHOD BETA, i believe by using dataframes, this is more efficient
        URL_DATASET5 = r'https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv'
        # dates days since day 1
        df1 = pd.read_csv(URL_DATASET5, parse_dates=['Date'])
        first_date = df1.loc[0, 'Date']
        df1['Date'] = (df1['Date'] - first_date).dt.days

        df1.sort_values(by=["Country", "Date"], inplace=True)
        df1.set_index(keys=["Country"], inplace=True)
        self._data = df1

    def show(self, *args):  # this is called a repr/representation/"predefined method thats true for all classes"?
        for i in args:
            if i == 'death':
                print(self.deaths_dict)
            elif i == 'confirm':
                print(self.confirm_dict)
            elif i == 'usdeath':
                print(self.usdeaths_dict)
            elif i == 'usconfirm':
                print(self.usconfirm_dict)
            else:
                print("that's not even a dictionary bruh")

    def confirmglobal(self, country):
        return self.confirm_dict[country]

    def deathglobal(self, country):
        return self.deaths_dict[country]

    def confirmus(self, state):
        return self.usconfirm_dict[state]

    def deathus(self, state):
        return self.usdeaths_dict[state]

    # METHOD ALPHA
    # def countryallconfirmed(self, country):
    #     country_stats = self.stats_by_country[country]
    #     return list(zip((entry.Date for entry in country_stats), (entry.Confirmed for entry in country_stats)))
    #
    # def countryalldeath(self, country):
    #     country_stats = self.stats_by_country[country]
    #     return list(zip((entry.Date for entry in country_stats), (entry.Deaths for entry in country_stats)))

    # METHOD BETA (3)
    def get_data_for_country(self, country):
        return self._data[self._data.index == country]

    def countryallconfirmed(self, country):
        hey = self.get_data_for_country(country)[["Date", "Confirmed"]]
        return dict(zip(hey.Date, hey.Confirmed))

    def countryalldeath(self, country):
        aye = self.get_data_for_country(country)[["Date", "Deaths"]]
        return dict(zip(aye.Date, aye.Deaths))


my_stats = Stats()
# print(my_stats.confirmus('Texas'))
# print(my_stats.confirmglobal('Albania'))
# print(my_stats.deathglobal('Albania'))
# my_stats.show('death')
# my_stats.show('usdeath')
# my_stats.show('confirm', 'death')

# METHOD BETA
# print(my_stats.get_data_for_country("Russia"))
# print(my_stats.countryallconfirmed("Russia"))
# print(my_stats.countryalldeath("Russia"))

# METHOD ALPHA
# print(my_stats.countryallconfirmed("Canada")[:50])


#collections is a dict variant: unlike a normal dict that raises a KeyError when you try to access a key that's not there, a defaultdict instead gives you some default. In this case, an empty list, commonly used for counting
# because without defaultdict, you'd have to do this instead:
#         for row in df1.itertuples():
#           if row.Country not in self.stats_by_country:
#               self.stats_by_country[row.Country] = []
#           self.stats_by_country[row.Country].append(row)
#
# Basically, self.stats_by_country is a dict mapping strings (country names) to lists.
# So each row is just appended to the list for the specific country.
# As for what each row is (what itertuples returns):
# Pandas(Index=0, Date=Timestamp('2020-01-22 00:00:00'), Country='Afghanistan', Confirmed=0, Recovered=0, Deaths=0)
# this is kind of a named tuple you can index like row.Country or row.Confirmed



# method 2
#
# "DataFrames are smart - when you requests elements by comparing with the index, they use binary search"
# 1) sort the DataFrame by the Country.
# 2) Set Country to be the index column
# 3) Now queries like df1[df1['Country'] == country] will be fast.
#
# In [148]: s = Stats()
#
# In [149]: s.countryalldeath("Russia")
# Out[149]:
#          Date  Deaths
# Country
# Russia    122    3388
# Russia     77      63
# Russia    187   13334
# Russia     45       0
# Russia    159    9152
# ...       ...     ...
# Russia    112    2212
# Russia    127    4142
# Russia    133    5208
# Russia    181   12561
# Russia     58       1
#
# [189 rows x 2 columns]
#
#
#
# self._data[self._data.index == country] can be simplified to self._data.loc[country]
# https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html
#
#  get_data_for_country("Russia") to get all entries (with all columns) and select the columns you want.
#
#not by date
#
# import pandas as pd
#
#
# class Stats:
#
#     def __init__(self):
#         URL_DATASET5 = r'https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv'
#         # dates days since day 1
#         df1 = pd.read_csv(URL_DATASET5, parse_dates=['Date'])
#         first_date = df1.loc[0, 'Date']
#         df1['Date'] = (df1['Date'] - first_date).dt.days
#
#         df1.sort_values(by="Country", inplace=True)
#         df1.set_index(keys=["Country"], drop=False, inplace=True)
#         self._data = df1
#
#     def countryallconfirmed(self, country):
#         return self._data[self._data.Country == country][["Date", "Confirmed"]]
#
#     def countryalldeath(self, country):
#         return self._data[self._data.Country == country][["Date", "Deaths"]]