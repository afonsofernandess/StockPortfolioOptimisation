import pandas as pd
import glob
import random
import numpy as np

# Get all CSV file paths
csv_files = glob.glob("archive/*.csv")


df_list = [pd.read_csv(csv_files[0])]

# Read the rest without headers
df_list += [pd.read_csv(file, header=None) for file in csv_files[1:]]

# Assign column names from the first file to all
for df in df_list[1:]:
    df.columns = df_list[0].columns

# Concatenate all DataFrames
final_df = pd.concat(df_list, ignore_index=True)

final_df['Date'] = pd.to_datetime(final_df['Date'], errors='coerce')

print(final_df)

# ###########################################
#             Auxiliar Functions
# ###########################################

# Evaluates the profit per day
# @param prices: a vector of the asset close prices
# @return: percentual variation of the values
def calculate_daily_returns(prices):
    return prices.pct_change().dropna()

# Evaluated the risk
# @param prices: a vector of the asset close prices
# @return: standard deviation of daily_returns
def calculate_volatility(prices):
    daily_returns = calculate_daily_returns(prices)
    return np.std(daily_returns)

# Evaluates the profit
# @param prices: a vector of the asset close prices
# @return: average of daily_returns
def calculate_average_return(prices):
    daily_returns = calculate_daily_returns(prices)
    return np.mean(daily_returns)

# ###########################################
#             Evaluation Function
# ###########################################


# Computes the portfolio return
# @param weights: a vector of asset weights in the portfolio
# @param returns: a vector of the returns for each asset in the portfolio
# @return: scalar product between weights and returns
def calculate_portfolio_return(weights, returns):
    return np.dot(list(weights.values()), returns)  # Retorno ponderado

# Computes the portfolio volatility
# @param weights: a vector of asset weights in the portfolio
# @param volatilities: a vector of the returns for each asset in the portfolio
# @return: square root of the sum of the weighted_volatilities
def calculate_portfolio_volatility(weights, volatilities):
    weighted_volatilities = [w * v for w, v in zip(weights.values(), volatilities)] # multiplies each asset's weight by its asset's volatility
    portfolio_volatility = np.sqrt(np.sum(weighted_volatilities))
    return portfolio_volatility

# Evaluation Function
# @param weights: a vector of asset weights in the portfolio
# @param returns: a vector of the returns for each asset in the portfolio
# @param volatilities: a vector of the returns for each asset in the portfolio
# @return: ratio of the portfolio's return to its volatility
def objective_function(weights, returns, volatilities):
    portfolio_return = calculate_portfolio_return(weights, returns)
    portfolio_volatility = calculate_portfolio_volatility(weights, volatilities)
    sharpe_ratio = portfolio_return / portfolio_volatility
    return sharpe_ratio

##########################################
#         Simple Implementation
###########################################
# My ideia here was to create a list with a few stocks and set random weights to each stock to start with, 
# then for each stock compute the volatility (risk), the return (profit) and evaluate it by returning the sharpe ratio (objective_function)

# Some stocks names to start with :)
symbols = ['A2M', 'AGL', 'ALL']
prices = {
    'A2M': pd.Series([150, 155, 160, 158, 165]),
    'AGL': pd.Series([2800, 2825, 2850, 2835, 2860]),
    'ALL': pd.Series([3400, 3450, 3480, 3465, 3500])
}

weights = {symbol: random.random() for symbol in symbols}
total_weight = sum(weights.values()) 
weights = {action: weight / total_weight for action, weight in weights.items()} # normalizing weigths so that the sum is 1 -> constraint

profit = [calculate_average_return(prices[symbol]) for symbol in symbols]
risk = [calculate_volatility(prices[symbol]) for symbol in symbols]

sharpe_ratio = objective_function(weights, profit, risk)
print(sharpe_ratio)