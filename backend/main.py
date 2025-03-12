# backend/main.py
import os
import pandas as pd
import numpy as np
import gym
import yfinance as yf
import datetime
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import spaces
from ib_insync import IB, Stock, MarketOrder
from fastapi import FastAPI

app = FastAPI()

# Connect to Interactive Brokers API
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# AI Trading Environment
class TradingEnv(gym.Env):
    def __init__(self, symbols=['AAPL', 'GOOGL'], start='2022-01-01', end='2023-01-01', initial_balance=10000):
        super(TradingEnv, self).__init__()
        self.symbols = symbols
        self.data = {symbol: yf.download(symbol, start=start, end=end) for symbol in symbols}
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = {symbol: 0 for symbol in symbols}
        
        # Action & Observation Space
        self.action_space = spaces.MultiDiscrete([3] * len(symbols))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(symbols) * 6,), dtype=np.float32)
    
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = {symbol: 0 for symbol in self.symbols}
        return self._next_observation()
    
    def _next_observation(self):
        obs = []
        for symbol in self.symbols:
            data = self.data[symbol].iloc[self.current_step]
            obs.extend([data['Open'], data['High'], data['Low'], data['Close'], data['Volume'], self.balance])
        return np.array(obs, dtype=np.float32)
    
    def step(self, action):
        reward = 0
        for i, symbol in enumerate(self.symbols):
            current_price = self.data[symbol].iloc[self.current_step]['Close']
            if action[i] == 1:  # Buy
                shares_to_buy = self.balance // current_price
                self.shares_held[symbol] += shares_to_buy
                self.balance -= shares_to_buy * current_price
            elif action[i] == 2 and self.shares_held[symbol] > 0:  # Sell
                self.balance += self.shares_held[symbol] * current_price
                self.shares_held[symbol] = 0
                reward += self.balance - self.initial_balance
        
        self.current_step += 1
        done = self.current_step >= len(self.data[self.symbols[0]]) - 1
        return self._next_observation(), reward, done, {}

# Train AI Model
env = DummyVecEnv([lambda: TradingEnv()])
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=50000)
model.save("trading_ai_model")

# Backtesting
def backtest(model, env):
    obs = env.reset()
    rewards = []
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
    plt.plot(rewards)
    plt.xlabel('Time Step')
    plt.ylabel('Profit/Loss')
    plt.title('Backtest Performance')
    plt.show()

backtest(model, env)

# Live Trading
def live_trade(model, symbols=['AAPL', 'GOOGL']):
    for symbol in symbols:
        stock = Stock(symbol, 'SMART', 'USD')
        action, _ = model.predict(env.reset())
        if action[0] == 1:  # Buy
            order = MarketOrder('BUY', 10)
            trade = ib.placeOrder(stock, order)
        elif action[0] == 2:  # Sell
            order = MarketOrder('SELL', 10)
            trade = ib.placeOrder(stock, order)

@app.get("/trade")
def execute_trade():
    live_trade(model)
    return {"status": "Trade executed"}