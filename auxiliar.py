import pandas as pd
import random

def get_date_range():
    """Get valid date range from user input"""
    dates = pd.date_range('2000-01-03', '2020-04-01')
    print("Minimum date: 2000-01-03\nMaximum date: 2020-04-01")
    print("Choose a start and end date for the analysis (Format: YYYY-MM-DD)")
    while True:
        start_date = input("Start date (or 'back' to return): ")
        if start_date.lower() == 'back':
            return None
        try:
            if pd.to_datetime(start_date) in dates:
                break
        except:
            pass
        print("Please enter a valid date between 2000-01-02 and 2020-04-02")

    while True:
        end_date = input("End date (or 'back' to return): ")
        if end_date.lower() == 'back':
            return None
        try:
            if pd.to_datetime(end_date) in dates:
                break
        except:
            pass
        print("Please enter a valid date between 2000-01-03 and 2020-04-01")

    if pd.to_datetime(start_date) > pd.to_datetime(end_date):
        print("End date must be after start date")
        exit(1)

    return pd.date_range(start_date, end_date)



def initialize_portfolio(stocks):
    """Generate random initial portfolio weights"""
    weights = {symbol: random.random() for symbol in stocks}
    return normalize_weights(weights)


def normalize_weights(weights):
    """Normalize weights to sum to 1"""
    total_weight = sum(weights.values())
    return {k: v / total_weight for k, v in weights.items()}