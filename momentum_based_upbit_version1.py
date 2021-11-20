import numpy as np 
import pandas as pd 
import random 
import os 
from tqdm import tqdm
import pyupbit
import ccxt
import time
import datetime
import telegram
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

### telegram tokens ###
token = "<YOUR TOKEN>"
bot = telegram.Bot(token=token) 
chat_id = "<YOUR CHAT ID>" 

### API keys for the test account ###
access_key = "<YOUR ACCESS KEY>"
secret_key = "<YOUR SECRET KEY>"
upbit = pyupbit.Upbit(access_key, secret_key)

### started trading Fri Nov 19 3:11 PM ### 

def geometric_mean(x): 
    x = np.asarray(x) 
    return x.prod() ** (1/len(x))  

def min_max_norm(x):  
    return (x - np.min(x)) / (np.max(x) - np.min(x)) 
    
historical_profits = [] 
big_caps = ["KRW-BTC", "KRW-ETH", "KRW-ADA", "KRW-DOT", "KRW-XRP", "KRW-SOL"] # top 6 crypto market caps 
current_portfolio = [] 
iteration = 0
gma_window = 5
lookback_window = 3
topk = 20 
gamma = 0.05 # commissions 
cash_status = [] 
flag = False # used for stop loss and take profit (if we need these functionalities) 

while True: 
    t0 = time.time() 
    print("getting all available tickers...")
    tickers = pyupbit.get_tickers(fiat="KRW")  
    print("{} cryptos currently listed on the market".format(len(tickers)))
    text = "*** Trade " + str(iteration) + " ***"
    bot.sendMessage(chat_id=chat_id, text=text) 
    ### close positions and fill up historical profits array ### 
    if iteration > 0: 
        portfolio_weights = [1/len(current_portfolio) for i in range(len(current_portfolio))]
        if flag == False: 
            text = "closing positions..." 
            bot.sendMessage(chat_id=chat_id, text=text) 
            for i in range(len(current_portfolio)): 
                while True: 
                    try: 
                        unit = upbit.get_balance(current_portfolio[i])
                        upbit.sell_market_order(current_portfolio[i], unit) 
                        break 
                    except Exception as e: 
                        print(e)
                        continue 
                time.sleep(0.1) 
        flag = False 
        text = "recording historical profits..." 
        bot.sendMessage(chat_id=chat_id, text=text) 
        rets = [] 
        for i in range(len(current_portfolio)): 
            df = pyupbit.get_ohlcv(current_portfolio[i], interval='minute60') 
            while df is None: 
                print("Failed to retrieve data... retrying") 
                df = pyupbit.get_ohlcv(current_portfolio[i], interval='minute60')
            cur = df['close'].values[-1] 
            prev = df['open'].values[-2] 
            ret = cur / prev 
            rets.append(ret)  
            time.sleep(0.1)
        historical_profits.append(np.dot(portfolio_weights, rets))  
        if len(historical_profits) > gma_window: 
            historical_profits.pop(0) 
        
    ### reset current portfolio ### 
    current_portfolio = [] 
    
    ### select portfolio ### 
    infos = [] 
    for ticker in tqdm(tickers, position=0, leave=True): 
        if ticker in big_caps: 
            infos.append(0)
        else:
            df = pyupbit.get_ohlcv(ticker, interval='minute60')  
            while df is None: 
                print("Failed to retrieve data... retrying")
                df = pyupbit.get_ohlcv(ticker, interval='minute60')
            volsum = np.sum(df['volume'].values[-lookback_window:]) 
            price_change = (df['close'].values[-1] - df['close'].values[-lookback_window]) / df['close'].values[-lookback_window]
            infos.append(volsum * price_change) 
        time.sleep(0.1)
    best_idx = np.argsort(infos)[::-1]
    for idx in best_idx[:topk]: 
        current_portfolio.append(tickers[idx]) 
    for ticker in big_caps: 
        current_portfolio.append(ticker) 
    
    text = "investing in " + str(current_portfolio) + "\n" 
    bot.sendMessage(chat_id=chat_id, text=text)
    
    portfolio_weights = [1/len(current_portfolio) for i in range(len(current_portfolio))]
    
    ### open positions ### 
    cash_amount = upbit.get_balance("KRW")  
    text = "current cash amount is " + str(cash_amount) + " won." 
    bot.sendMessage(chat_id=chat_id, text=text) 
    cash_status.append(cash_amount)
    if iteration > 0: 
        gma_profits = geometric_mean(historical_profits)  
        text = "gma profits so far = " + str(gma_profits) + "\n" 
        bot.sendMessage(chat_id=chat_id, text=text)
        if gma_profits <= 1.0: 
            text = "probably a down market... resorting to cash agent..." 
            bot.sendMessage(chat_id=chat_id, text=text) 
        elif gma_profits > 1.0:  
            for i in range(len(current_portfolio)):  
                while True: 
                    try: 
                        upbit.buy_market_order(current_portfolio[i], cash_amount * (1-gamma/100) * portfolio_weights[i])
                        break 
                    except Exception as e: 
                        print(e)
                        continue 
                time.sleep(0.1)
    elif iteration == 0: 
        for i in range(len(current_portfolio)):  
            while True: 
                try: 
                    upbit.buy_market_order(current_portfolio[i], cash_amount * (1-gamma/100) * portfolio_weights[i])
                    break 
                except Exception as e: 
                    print(e)
                    continue 
            time.sleep(0.1)

    iteration += 1 
    
    text = "waiting for the next hour..." 
    bot.sendMessage(chat_id=chat_id, text=text) 
    elapsed = time.time() - t0 
    time.sleep(60*60 - elapsed)
    
