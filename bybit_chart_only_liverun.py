import numpy as np
import pandas as pd
import random
import os
import seaborn as sns
import math
from sklearn.model_selection import train_test_split
import pybit
import ccxt
import bybit
import time
import telegram
from tqdm import tqdm
import pandas_ta as ta
import joblib
from pytorch_tabnet.tab_model import TabNetClassifier , TabNetRegressor

def get_df(bybit_ccxt):
    df = pd.DataFrame(bybit_ccxt.fetch_ohlcv("BTCUSDT", timeframe='4h', limit=200))
    df = df.rename(columns={0:'timestamp',
                            1:'open',
                            2:'high',
                            3:'low',
                            4:'close',
                            5:'volume'})
    return df


def create_timestamps(df):
    bybit = ccxt.bybit()
    dates = df['timestamp'].values
    timestamp = []
    for i in range(len(dates)):
        date_string = bybit.iso8601(int(dates[i]))
        date_string = date_string[:10] + " " + date_string[11:-5]
        timestamp.append(date_string)
    df['datetime'] = timestamp
    df = df.drop(columns={'timestamp'})
    df.set_index(pd.DatetimeIndex(df["datetime"]), inplace=True)
    return df


def preprocess_data(df):
    # add datetime information
    hours = []
    days = []
    months = []
    years = []
    for dt in tqdm(df['datetime']):
        hour = pd.to_datetime(dt).hour
        day = pd.to_datetime(dt).day
        month = pd.to_datetime(dt).month
        year = pd.to_datetime(dt).year
        hours.append(hour)
        days.append(day)
        months.append(month)
        years.append(year)
    df['Hours'] = hours
    df['Days'] = days
    df['Months'] = months
    df['Years'] = years
    print("=== Feature Engineering ===")

    # add rvi
    df['RVI'] = df.ta.rvi(lookahead=False)

    # inertia
    df['inertia'] = df.ta.inertia(lookahead=False)

    # differencing
    for l in range(1,24):
        for col in ['open','high','low','close','volume']:
            val = df[col].values
            val_ret = [None for _ in range(l)]
            for i in range(l, len(val)):
                if val[i-l] == 0:
                    ret = 1
                else:
                    ret = val[i] / val[i-l]
                val_ret.append(ret)
            df['{}_change_{}'.format(col, l)] = val_ret

    # ebsw
    df['ebsw'] = df.ta.ebsw(lookahead=False)

    # rsi
    df['RSI'] = df.ta.rsi(lookahead=False)

    # rsx
    df['RSX'] = df.ta.rsx(lookahead=False)

    # chaikin money flow
    df['cmf'] = df.ta.cmf(lookahead=False)

    # volume weighted average price
    df['vwap'] = df.ta.vwap(lookahead=False)

    # holt-winter moving average
    df['hwma'] = df.ta.hwma(lookahead=False).values

    # fibonacci's weighted moving average
    df['fwma'] = df.ta.fwma(lookahead=False).values

    # add average directional movement index
    df = pd.concat([df, df.ta.adx(lookahead=False)], axis=1)

    # add ultimate oscillator
    df['uo'] = df.ta.uo(lookahead=False)

    # moving average convergence divergence
    df = pd.concat([df, df.ta.macd(lookahead=False)], axis = 1)

    # bollinger bands
    df = pd.concat([df, df.ta.bbands(lookahead=False)], axis=1)

    # exponential moving averages
    df['EMA10'] = df.ta.ema(length=10, lookahead=False)
    df['EMA30'] = df.ta.ema(length=30, lookahead=False)
    df['EMA60'] = df.ta.ema(length=60, lookahead=False)

    df = df.dropna()
    df = df.drop(columns={'datetime','Years'})
    return df

# define session and set leverage

from pybit import HTTP
import pprint

api_key = "<API KEY>"
api_secret = "<SECRET KEY>"


session = HTTP(
    endpoint="https://api.bybit.com",
    api_key=api_key,
    api_secret=api_secret
)

### set leverage ###
leverage = 1
session.set_leverage(symbol='BTCUSDT', buy_leverage=leverage, sell_leverage=leverage)


# load tabnet model
tabnet = TabNetClassifier()
tabnet.load_model('TabNetClassifier_ByBit_with_class_weights_chart_only_74.zip')


# define telegram bot
chat_id = <chat id>
tel_token = "<telegram token>"
bot = telegram.Bot(token=tel_token)

def my_ceil(a, precision=0):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)

def my_floor(a, precision=0):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)

cash_status = [] # store cash amount

action = {0:'long', 1:'short', 2:'hold'}

def execute_trade(leverage):
    iteration = 0
    move = 0 # -1: short, 1: long
    bybit_ccxt = ccxt.bybit()

    high_leverage = False

    while True:
        text = "==== Trade Iteration " + str(iteration) + " ===="
        bot.sendMessage(chat_id=chat_id, text=text)

        t0 = time.time()

        if high_leverage == True:
            text = "resetting leverage..."
            bot.sendMessage(chat_id=chat_id, text=text)
            leverage = 1
            session.set_leverage(symbol='BTCUSDT', buy_leverage=leverage, sell_leverage=leverage)

        high_leverage = False
        ### make prediction ###
        bybit_ccxt = ccxt.bybit()
        df = get_df(bybit_ccxt)
        df = create_timestamps(df)
        df = preprocess_data(df)
        x = df.values[-2].reshape((-1,df.shape[1]))
        pred = float(tabnet.predict(x).item())
        prob = np.max(tabnet.predict_proba(x))

        if prob > 0.9:
            high_leverage = True
            text = "TabNet predicted class {} with probability {}. Resetting leverage to 5x.".format(action[pred], prob)
            bot.sendMessage(chat_id=chat_id, text=text)
            leverage = 5
            session.set_leverage(symbol='BTCUSDT', buy_leverage=leverage, sell_leverage=leverage)
        elif prob <= 0.9:
            text = "TabNet predicted class {} with probability. Using 1x leverage.".format(action[pred], prob)
            bot.sendMessage(chat_id=chat_id, text=text)

        ### open position ###
        if pred == 1.0:
            ### short ###
            text = "Choosing Short Position!"
            bot.sendMessage(chat_id=chat_id, text=text)
            if iteration > 0:
                if session.my_position(symbol="BTCUSDT")['result'][0]['size'] == 0:
                    text = "no positions open... stop loss or take profit was probably triggered."
                    bot.sendMessage(chat_id=chat_id, text=text)
                else:
                    if move == 1:
                        text = "Closing previous long position and opening short position..."
                        bot.sendMessage(chat_id=chat_id, text=text)
                        ### long was chosen, so we take short to close position ###
                        resp = session.place_active_order(
                                symbol="BTCUSDT",
                                side="Sell",
                                order_type="Market",
                                qty=qty,
                                time_in_force="GoodTillCancel",
                                reduce_only=True,
                                close_on_trigger=False
                        )
                    elif move == -1:
                        text = "Closing previous short position and opening short position..."
                        bot.sendMessage(chat_id=chat_id, text=text)
                        ### short was chosen, so we take long to close position ###
                        resp = session.place_active_order(
                                symbol="BTCUSDT",
                                side="Buy",
                                order_type="Market",
                                qty=qty,
                                time_in_force="GoodTillCancel",
                                reduce_only=True,
                                close_on_trigger=False
                        )
                ### after closing the position, we need to recalculate quantity and cash status ###
                cur_price = session.latest_information_for_symbol(symbol='BTCUSDT')['result'][0]['last_price']
                balances = session.get_wallet_balance()
                usdt = balances['result']['USDT']['available_balance']
                text = "current cash status = " + str(usdt)
                bot.sendMessage(chat_id=chat_id, text=text)
                cash_status.append(float(usdt))
                qty = float(usdt) / float(cur_price) * leverage
                qty = my_floor(qty, precision = 5)
                ### set stop loss and take profit ###
                if high_leverage == False:
                    stop_loss = round(float(cur_price) * (1 - 0.75/100))
                    take_profit = round(float(cur_price) * (1 + 3/100))
                elif high_leverage == True:
                    stop_loss = round(float(cur_price) * (1 - 0.75*5/100))
                    take_profit = round(float(cur_price) * (1 + 3*5/100))

                ### take short position as predicted ###
                resp = session.place_active_order(
                    symbol="BTCUSDT",
                    side="Sell",
                    order_type="Market",
                    qty=qty,
                    time_in_force="GoodTillCancel",
                    reduce_only=False,
                    close_on_trigger=False,
                    stop_loss = take_profit, # reversed for short
                    take_profit = stop_loss
                )
            elif iteration == 0:
                ### calculate quantity ###
                cur_price = session.latest_information_for_symbol(symbol='BTCUSDT')['result'][0]['last_price']
                balances = session.get_wallet_balance()
                usdt = balances['result']['USDT']['available_balance']
                text = "current cash status = " + str(usdt)
                bot.sendMessage(chat_id=chat_id, text=text)
                cash_status.append(float(usdt))
                qty = float(usdt) / float(cur_price) * leverage
                qty = my_floor(qty, precision = 5)
                ### set stop loss and take profit ###
                if high_leverage == False:
                    stop_loss = round(float(cur_price) * (1 - 0.75/100))
                    take_profit = round(float(cur_price) * (1 + 3/100))
                elif high_leverage == True:
                    stop_loss = round(float(cur_price) * (1 - 0.75*5/100))
                    take_profit = round(float(cur_price) * (1 + 3*5/100))

                ### open short for the first iteration ###
                resp = session.place_active_order(
                    symbol="BTCUSDT",
                    side="Sell",
                    order_type="Market",
                    qty=qty,
                    time_in_force="GoodTillCancel",
                    reduce_only=False,
                    close_on_trigger=False,
                    stop_loss = take_profit, # reversed for short
                    take_profit = stop_loss
                )
            ### reset move ###
            move = -1
        elif pred == 0.0:
            ### long ###
            text = "Choosing Long Position!"
            bot.sendMessage(chat_id=chat_id, text=text)
            if iteration > 0:
                if session.my_position(symbol="BTCUSDT")['result'][0]['size'] == 0:
                    text = "no positions open... stop loss or take profit was probably triggered."
                    bot.sendMessage(chat_id=chat_id, text=text)
                else:
                    if move == -1:
                        text = "Closing previous short position and opening long position..."
                        bot.sendMessage(chat_id=chat_id, text=text)
                        ### short was chosen, so we take long to close position ###
                        resp = session.place_active_order(
                                symbol="BTCUSDT",
                                side="Buy",
                                order_type="Market",
                                qty=qty,
                                time_in_force="GoodTillCancel",
                                reduce_only=True,
                                close_on_trigger=False
                        )
                    elif move == 1:
                        text = "Closing previous long position and opening long position..."
                        bot.sendMessage(chat_id=chat_id, text=text)
                        ### long was chosen, so we take short to close position ###
                        resp = session.place_active_order(
                                symbol="BTCUSDT",
                                side="Sell",
                                order_type="Market",
                                qty=qty,
                                time_in_force="GoodTillCancel",
                                reduce_only=True,
                                close_on_trigger=False
                        )
                ### after closing the position, we need to recalculate quantity and cash status ###
                cur_price = session.latest_information_for_symbol(symbol='BTCUSDT')['result'][0]['last_price']
                balances = session.get_wallet_balance()
                usdt = balances['result']['USDT']['available_balance']
                text = "current cash status = " + str(usdt)
                bot.sendMessage(chat_id=chat_id, text=text)
                cash_status.append(float(usdt))
                qty = float(usdt) / float(cur_price) * leverage
                qty = my_floor(qty, precision = 5)
                ### set stop loss and take profit ###
                if high_leverage == False:
                    stop_loss = round(float(cur_price) * (1 - 3/100))
                    take_profit = round(float(cur_price) * (1 + 0.75/100))
                elif high_leverage == True:
                    stop_loss = round(float(cur_price) * (1 - 3*5/100))
                    take_profit = round(float(cur_price) * (1 + 0.75*5/100))
                ### take long position as predicted ###
                resp = session.place_active_order(
                    symbol="BTCUSDT",
                    side="Buy",
                    order_type="Market",
                    qty=qty,
                    time_in_force="GoodTillCancel",
                    reduce_only=False,
                    close_on_trigger=False,
                    stop_loss = stop_loss,
                    take_profit = take_profit
                )
            elif iteration == 0:
                ### calculate quantity ###
                cur_price = session.latest_information_for_symbol(symbol='BTCUSDT')['result'][0]['last_price']
                balances = session.get_wallet_balance()
                usdt = balances['result']['USDT']['available_balance']
                text = "current cash status = " + str(usdt)
                bot.sendMessage(chat_id=chat_id, text=text)
                cash_status.append(float(usdt))
                qty = float(usdt) / float(cur_price) * leverage
                qty = my_floor(qty, precision = 5)
                ### set stop loss and take profit ###
                if high_leverage == False:
                    stop_loss = round(float(cur_price) * (1 - 3/100))
                    take_profit = round(float(cur_price) * (1 + 0.75/100))
                elif high_leverage == True:
                    stop_loss = round(float(cur_price) * (1 - 3*5/100))
                    take_profit = round(float(cur_price) * (1 + 0.75*5/100))

                ### open long position for first iteration ###
                resp = session.place_active_order(
                    symbol="BTCUSDT",
                    side="Buy",
                    order_type="Market",
                    qty=qty,
                    time_in_force="GoodTillCancel",
                    reduce_only=False,
                    close_on_trigger=False,
                    stop_loss = stop_loss,
                    take_profit = take_profit
                )
            ### reset move ###
            move = 1
        elif pred == 2.0:
            text = "not much volatility. Skipping this round"
            bot.sendMessage(chat_id=chat_id, text=text)

        iteration += 1
        text = "waiting for the next 4 hours"
        bot.sendMessage(chat_id=chat_id, text=text)
        elapsed = time.time() - t0
        time.sleep(60*60*4 - elapsed)



## run
execute_trade(leverage=1)
