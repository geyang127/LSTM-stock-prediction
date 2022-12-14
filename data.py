import pandas_datareader.data as web
import datetime

start = datetime.datetime(2010,1,1)
end = datetime.datetime(2020,12,31)

def read_data():
    start = datetime.datetime(2010, 1, 1)
    end = datetime.datetime(2020, 12, 31)
    df = web.DataReader('GOOG', 'stooq', start, end)
    df.dropna(inplace=True)
    df.sort_index(inplace=True)
    pre_days = 10
    df['label'] = df['Close'].shift(-pre_days)
    df.to_csv('data.csv')
    return df