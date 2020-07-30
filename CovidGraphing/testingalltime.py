import pandas as pd
import collections

class Stats:

    def __init__(self):
        # URL_DATASET5 = r'https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv'
        # # dates days since day 1
        # df1 = pd.read_csv(URL_DATASET5, parse_dates=['Date'])
        # first_date = df1.loc[0, 'Date']
        # df1['Date'] = (df1['Date'] - first_date).dt.days
        #
        # self.countryallconfirmed = dict(zip((df1['Date']), (df1['Confirmed'])))
        # self.countryalldeath = dict(zip(iter(df_alltimecountry['Date']), iter(df_alltimecountry['Deaths'])))
        #
        # df_alltimecountry = (df1[df1['Country'] == 'CountryName'])
        # xconfirmed = (df1[df1['Country'] == 'CountryName'])['Confirmed'].tolist()
        # xdeaths = (df1[df1['Country'] == 'CountryName'])['Deaths'].tolist()
        # ydates = df_alltimecountry['Date'].tolist()
        #
        #
        # self.countryallconfirmed = dict(zip(iter(df_alltimecountry['Date']), iter(df_alltimecountry['Confirmed'])))
        # self.countryalldeath = dict(zip(iter(df_alltimecountry['Date']), iter(df_alltimecountry['Deaths'])))
        URL_DATASET5 = r'https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv'
        # dates days since day 1
        df1 = pd.read_csv(URL_DATASET5, parse_dates=['Date'])
        first_date = df1.loc[0, 'Date']
        df1['Date'] = (df1['Date'] - first_date).dt.days
        self.stats_by_country = collections.defaultdict(list)
        for row in df1.itertuples():
            self.stats_by_country[row.Country].append(row)

    def countryallconfirmed(self, country):
        country_stats = self.stats_by_country[country]
        return list(zip((entry.Date for entry in country_stats), (entry.Confirmed for entry in country_stats)))

    def countryalldeath(self, country):
        country_stats = self.stats_by_country[country]
        return list(zip((entry.Date for entry in country_stats), (entry.Deaths for entry in country_stats)))

my_stats = Stats()
print(my_stats.countryallconfirmed("Canada")[:50])