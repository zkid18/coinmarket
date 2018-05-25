import pandas as pd
import datetime
from coinmarket import get_all_coins

def transform_btc_to_usdt(btc_dict, usdt_dict, btc):
    df_usdt_coins_historical_data = {}
    df_btc_coins_historical_data = {}

    for usdt_coin, usdt_coin_data in usdt_dict.iteritems():
        df_usdt_coins_historical_data[usdt_coin] = pd.DataFrame(usdt_coin_data)
        df_usdt_coins_historical_data[usdt_coin].date = df_usdt_coins_historical_data[usdt_coin].date.apply(
            lambda x: datetime.datetime.fromtimestamp(x))
        df_usdt_coins_historical_data[usdt_coin].set_index('date', inplace=True)
        df_usdt_coins_historical_data[usdt_coin].drop(["quoteVolume"], axis=1, inplace=True)

    for btc_coin, btc_coin_data in btc_dict.iteritems():
        df_btc_coins_historical_data[btc_coin] = pd.DataFrame(btc_coin_data)
        df_btc_coins_historical_data[btc_coin].date = df_btc_coins_historical_data[btc_coin].date.apply(
            lambda x: datetime.datetime.fromtimestamp(x))
        df_btc_coins_historical_data[btc_coin].set_index('date', inplace=True)
        df_btc_coins_historical_data[btc_coin].drop(["quoteVolume"], axis=1, inplace=True)

        #transform to usdt
        df_usdt_coins_historical_data[btc_coin] = df_btc_coins_historical_data[btc_coin].multiply(btc, axis='columns',
                                                                                                  fill_value=0)
    return df_usdt_coins_historical_data


def get_max_shape(d):
    max_shape = 0
    for k in d:
        if len(d[k]) > max_shape:
            max_shape = len(d[k])
            coin = k
    return max_shape, coin

def get_k_top_traded_coins(history, data,k):
    df_market_cap = get_all_coins()
    traded_coins = []
    for coin in data:
        if len(data[coin][data[coin].close != 0]) > history:
            traded_coins.append(coin)
    return df_market_cap[df_market_cap.Ticker.isin(traded_coins)].head(k).Ticker.values