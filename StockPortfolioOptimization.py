import pandas as pd
import random
import numpy as np
import os


# Global Variables

stock_prices = {}

# ###########################################
#             Auxiliar Functions
# ###########################################

# Get data range as the user input
# @return: valid data range
def get_date_range():

    dates = pd.date_range('2000-01-02', '2020-04-02')
    print("Choose a start and end date for the analysis (Format: YYYY-MM-DD)")
    while True:
        start_date = input("Start date: ")
        if start_date in dates:
            break
        print("Please enter a valid date")

    while True:
        end_date = input("End date: ")
        if end_date in dates:
            break
        print("Please enter a valid date")

    if start_date > end_date:
        print("Please enter a valid date")
        exit(1)
    return pd.date_range(start_date, end_date)


# Initializate the program
# @return: initial solution of portifolio distribution
def start():

    print("Stock Portfolio Optimization\n")
    valid_dates = get_date_range()

    stocks = [stock.split('.')[0] for stock in sorted(os.listdir('archive'))]

    for stock in stocks:

        date_adjClose_list = []
        pc = pd.read_csv(f'archive/{stock}.csv', usecols=['Date', 'Adj Close'])
        pc['Date'] = pd.to_datetime(pc['Date'])
        pc = pc[pc['Date'].isin(valid_dates)]

        for index, row in pc.iterrows():
            date_adjClose_list.append(row['Adj Close'])
        if len(date_adjClose_list) > 0:
            stock_prices[stock] = pd.Series(date_adjClose_list)

    weights = {symbol: random.random() for symbol in stock_prices.keys()}
    weights = normalize_weight(weights)

    return weights


# Evaluates the profit per day
# @param prices: a vector of the asset close prices
# @return: percentual variation of the values
def calculate_daily_returns(prices):
    return prices.pct_change().dropna()


# Evaluated the risk
# @param prices: a vector of the asset close prices
# @return: standard deviation of daily_returns
def calculate_risk(prices):
    daily_returns = calculate_daily_returns(prices)
    return np.std(daily_returns)


# Evaluates the profit
# @param prices: a vector of the asset close prices
# @return: average of daily_returns
def calculate_average_return(prices):
    daily_returns = calculate_daily_returns(prices)
    return np.mean(daily_returns)


# Normalizes weights so that the sum is 1 -> constraint
# @param weights: a vector of asset weights in the portfolio
# @return: normalized vector of asset weights
def normalize_weight(weight):
    total_weight = sum(weight.values())
    weight = {k: v / total_weight for k, v in weight.items()}
    return weight


# ###########################################
#             Evaluation Function
# ###########################################


# Computes the portfolio return
# @param weights: a vector of asset weights in the portfolio
# @param returns: a vector of the returns for each asset in the portfolio
# @return: scalar product between weights and returns
def calculate_portfolio_return(weights, returns):
    return np.dot(list(weights.values()), returns)  # Retorno ponderado


# Computes the portfolio risk
# @param weights: a vector of asset weights in the portfolio
# @param risks: a vector of the returns for each asset in the portfolio
# @return: square root of the sum of the weighted_risks
def calculate_portfolio_risk(weights, risks):
    weighted_risks = [w * v for w, v in
                      zip(weights.values(), risks)]  # multiplies each asset's weight by its asset's risk
    portfolio_risk = np.sqrt(np.sum(weighted_risks))
    return portfolio_risk


# Evaluation Function
# @param weights: a vector of asset weights in the portfolio
# @return: ratio of the portfolio's return to its risk
def evaluate_solution(weights):

    profit = [calculate_average_return(stock_prices[symbol]) for symbol in stock_prices.keys()]
    risk = [calculate_risk(stock_prices[symbol]) for symbol in stock_prices.keys()]

    portfolio_return = calculate_portfolio_return(weights, profit)
    portfolio_risk = calculate_portfolio_risk(weights, risk)
    sharpe_ratio = portfolio_return / portfolio_risk
    return sharpe_ratio


# ###########################################
#             Neighbour Function
# ###########################################

# Modifies the weight of a randomly selected stock
# @param weights: a vector of asset weights in the portfolio
# @return: a slight modified vector of asset weights in the portfolio.
def generate_neighbor(weights, threshold):

    new_weights = weights.copy()

    idx = random.choice(list(weights.keys()))
    weight_change = random.choice([-threshold, threshold])  # randomly decide whether to add or subtract the threshold
    new_weights[idx] += weight_change

    return normalize_weight(new_weights)


##########################################
#         Algortithm Implementation
###########################################

def hill_climbing(x0):

    threshold = 0.01

    x = x0
    while True:

        neighbors = []
        for i in range (1,5):
            neighbors.insert(0,generate_neighbor(x,threshold))

        # find the neighbor with the highest function value
        best_neighbor = max(neighbors, key=evaluate_solution)

        if evaluate_solution(best_neighbor) <= evaluate_solution(x): # if the best neighbor is not better than x, stop
            return x
        x = best_neighbor # otherwise, continue with the best neighbor

##########################################
#         Simple Implementation
###########################################
# My idea here was to create a list with a few stocks and set random weights to each stock to start with,
# then for each stock compute the risk (risk), the return (profit) and evaluate it by returning the sharpe ratio (objective_function)

if __name__ == '__main__':

    initial_solution = start()

    final_solution = hill_climbing(initial_solution)
    sharpe_ratio = evaluate_solution(final_solution)
    print(sharpe_ratio)