import requests
from bs4 import BeautifulSoup
import json
import pandas as pd


def get_all_coins():
    url = 'https://api.coinmarketcap.com/v1/ticker/'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    dic = json.loads(soup.prettify())

    # create an empty DataFrame
    df_market_cap = pd.DataFrame(columns=["Ticker", "MarketCap"])

    for i in range(len(dic)):
        df_market_cap.loc[len(df_market_cap)] = [dic[i]['symbol'], dic[i]['market_cap_usd']]

    df_market_cap.MarketCap = df_market_cap.MarketCap.astype(float)
    df_market_cap.sort_values(by=['MarketCap'])

    return df_market_cap