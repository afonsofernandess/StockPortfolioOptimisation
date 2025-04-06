import pandas as pd
import random
import numpy as np
import os
from collections import deque
import time
from datetime import datetime

import auxiliar as aux
import visualisation as vis
import instance_management as io

# Global Variables
stock_prices = {}
stock_opening_prices = {}
algorithm_metrics = {
    'hill_climbing': {},
    'simulated_annealing': {},
    'tabu_search': {},
    'genetic_algorithm': {}
}
dates = None

# Create instances directory if it doesn't exist
if not os.path.exists('instances'):
    os.makedirs('instances')

def load_stock_data(valid_date):
    """Load stock opening and closing price for a specific date"""
    global stock_prices
    stock_prices = {}

    stocks = [stock.split('.')[0] for stock in sorted(os.listdir('archive'))]

    for stock in stocks:
        try:
            pc = pd.read_csv(f'archive/{stock}.csv', usecols=['Date', 'Open', 'Adj Close'])
            pc['Date'] = pd.to_datetime(pc['Date'])
            # Filter to only match valid date
            pc = pc[pc['Date'] == valid_date]

            if len(pc) > 0:
                stock_prices[stock] = pc['Adj Close']
                stock_opening_prices[stock] = pc['Open']
        except:
            continue

    return stock_prices

# ###########################################
#             Evaluation Function
# ###########################################

def evaluate_portfolio(weights):
    """Evaluate portfolio multiplying the weight with the relative return of each stock"""
    if not stock_prices:
        raise ValueError("No stock data loaded")

    returns = []
    for symbol in weights.keys():
        open_price = stock_opening_prices[symbol].iloc[0]
        adj_close_price = stock_prices[symbol].iloc[0]
        ret = (adj_close_price - open_price) / open_price
        returns.append(ret)
    portfolio_return = np.dot(list(weights.values()), returns)

    return portfolio_return


# ###########################################
#             Neighbor Generation
# ###########################################

def generate_neighbor(weights, mutation_rate=0.1):
    """Generate a neighboring solution by slightly modifying weights"""
    new_weights = weights.copy()
    stock_to_change = random.choice(list(weights.keys()))

    # Randomly adjust weight
    adjustment = random.uniform(-mutation_rate, mutation_rate)
    new_weights[stock_to_change] += adjustment

    # Ensure weights stay positive
    new_weights = {k: max(v, 0) for k, v in new_weights.items()}
    return aux.normalize_weights(new_weights)


# ###########################################
#             Optimization Algorithms
# ###########################################

def hill_climbing(initial_solution, date, max_iterations=1000, neighbor_count=10):
    """Hill Climbing optimization algorithm"""
    start_time = time.time()
    current_solution = initial_solution
    current_score = evaluate_portfolio(current_solution)
    best_solution = current_solution
    best_score = current_score

    scores = []

    for _ in range(max_iterations):
        neighbors = [generate_neighbor(current_solution) for _ in range(neighbor_count)]
        best_neighbor = max(neighbors, key=evaluate_portfolio)
        neighbor_score = evaluate_portfolio(best_neighbor)

        if neighbor_score > current_score:
            current_solution = best_neighbor
            current_score = neighbor_score
            if current_score > best_score:
                best_solution = current_solution
                best_score = current_score
        scores.append(best_score)

    execution_time = time.time() - start_time
    if date not in algorithm_metrics['hill_climbing']:
        algorithm_metrics['hill_climbing'][date] = {'time': [], 'best_score': []}
    algorithm_metrics['hill_climbing'][date]['time'].append(execution_time)
    algorithm_metrics['hill_climbing'][date]['best_score'].append(best_score)
    return best_solution, best_score, scores, execution_time


def simulated_annealing(initial_solution, date, max_iterations=1000, initial_temp=100, cooling_rate=0.99):
    """Simulated Annealing optimization algorithm"""
    start_time = time.time()
    current_solution = initial_solution
    current_score = evaluate_portfolio(current_solution)
    best_solution = current_solution
    best_score = current_score
    temp = initial_temp

    scores = []

    for _ in range(max_iterations):
        neighbor = generate_neighbor(current_solution)
        neighbor_score = evaluate_portfolio(neighbor)

        if neighbor_score > current_score:
            current_solution = neighbor
            current_score = neighbor_score
            if current_score > best_score:
                best_solution = current_solution
                best_score = current_score
        else:
            # Accept worse solution with some probability
            if random.random() < np.exp((neighbor_score - current_score) / temp):
                current_solution = neighbor
                current_score = neighbor_score

        temp *= cooling_rate
        scores.append(best_score)

    execution_time = time.time() - start_time
    if date not in algorithm_metrics['simulated_annealing']:
        algorithm_metrics['simulated_annealing'][date] = {'time': [], 'best_score': []}
    algorithm_metrics['simulated_annealing'][date]['time'].append(execution_time)
    algorithm_metrics['simulated_annealing'][date]['best_score'].append(best_score)
    return best_solution, best_score, scores, execution_time


def tabu_search(initial_solution, date, max_iterations=1000, tabu_size=10, neighbor_count=10):
    """Tabu Search optimization algorithm"""
    start_time = time.time()
    current_solution = initial_solution
    current_score = evaluate_portfolio(current_solution)
    best_solution = current_solution
    best_score = current_score
    tabu_list = deque(maxlen=tabu_size)

    scores = []

    for _ in range(max_iterations):
        neighbors = [generate_neighbor(current_solution) for _ in range(neighbor_count)]
        # Filter out tabu solutions
        candidates = [n for n in neighbors if str(n) not in tabu_list]

        if not candidates:  # All neighbors are in tabu list
            candidates = neighbors  # Allow tabu solutions if no alternatives

        best_candidate = max(candidates, key=evaluate_portfolio)
        candidate_score = evaluate_portfolio(best_candidate)

        current_solution = best_candidate
        current_score = candidate_score
        tabu_list.append(str(current_solution))

        if current_score > best_score:
            best_solution = current_solution
            best_score = current_score

        scores.append(best_score)

    execution_time = time.time() - start_time
    if date not in algorithm_metrics['tabu_search']:
        algorithm_metrics['tabu_search'][date] = {'time': [], 'best_score': []}
    algorithm_metrics['tabu_search'][date]['time'].append(execution_time)
    algorithm_metrics['tabu_search'][date]['best_score'].append(best_score)
    return best_solution, best_score, scores, execution_time


def genetic_algorithm(date, population_size=20, generations=100, mutation_rate=0.1, crossover_rate=0.8):
    """Genetic Algorithm optimization"""
    start_time = time.time()
    stocks = list(stock_prices.keys())

    # Initialize population
    population = [aux.initialize_portfolio(stocks) for _ in range(population_size)]
    best_solution = max(population, key=evaluate_portfolio)
    best_score = evaluate_portfolio(best_solution)

    scores = []

    for _ in range(generations):
        # Evaluate fitness
        fitness = [evaluate_portfolio(ind) for ind in population]

        # Selection (tournament selection)
        new_population = []
        for _ in range(population_size):
            # Pick 2 random individuals
            a, b = random.sample(range(population_size), 2)
            winner = population[a] if fitness[a] > fitness[b] else population[b]
            new_population.append(winner.copy())

        # Crossover
        for i in range(0, population_size - 1, 2):
            if random.random() < crossover_rate:
                parent1 = new_population[i]
                parent2 = new_population[i + 1]

                # Single-point crossover
                if len(stocks) > 1:
                    crossover_point = random.randint(1, len(stocks) - 1)
                else:
                    crossover_point = 0
                keys = list(parent1.keys())

                child1 = {**parent1}
                child2 = {**parent2}

                # Swap weights after crossover point
                for j in range(crossover_point, len(keys)):
                    child1[keys[j]], child2[keys[j]] = child2[keys[j]], child1[keys[j]]

                new_population[i] = aux.normalize_weights(child1)
                new_population[i + 1] = aux.normalize_weights(child2)

        # Mutation
        for i in range(population_size):
            if random.random() < mutation_rate:
                new_population[i] = generate_neighbor(new_population[i])

        population = new_population

        # Track best solution
        current_best = max(population, key=evaluate_portfolio)
        current_score = evaluate_portfolio(current_best)
        if current_score > best_score:
            best_solution = current_best
            best_score = current_score

        scores.append(best_score)

    execution_time = time.time() - start_time
    if date not in algorithm_metrics['genetic_algorithm']:
        algorithm_metrics['genetic_algorithm'][date] = {'time': [], 'best_score': []}
    algorithm_metrics['genetic_algorithm'][date]['time'].append(execution_time)
    algorithm_metrics['genetic_algorithm'][date]['best_score'].append(best_score)
    return best_solution, best_score, scores, execution_time

# ###########################################
#             Main Interface
# ###########################################

def run_optimization(algorithm, initial_solution, parameters, date, instance_dir=None):
    """Run specified optimization algorithm"""
    if algorithm == 'hill_climbing':
        solution, score, scores, execution_time = hill_climbing(initial_solution, date, **parameters)
    elif algorithm == 'simulated_annealing':
        solution, score, scores, execution_time = simulated_annealing(initial_solution, date, **parameters)
    elif algorithm == 'tabu_search':
        solution, score, scores, execution_time = tabu_search(initial_solution, date, **parameters)
    elif algorithm == 'genetic_algorithm':
        solution, score, scores, execution_time = genetic_algorithm(date, **parameters)
    else:
        raise ValueError("Unknown algorithm")

    timestamp = io.save_results(instance_dir, solution, score, algorithm, parameters, execution_time, date)
    vis.plot_optimization_process(scores, algorithm.replace('_', ' ').title(), date, instance_dir, timestamp)
    vis.plot_solution(solution, algorithm.replace('_', ' ').title(), date, instance_dir, timestamp)

    return solution, score, scores, execution_time


def main_menu():
    """Interactive main menu"""
    print("\nStock Portfolio Optimization")
    print("1. Create new problem instance")
    print("2. Load problem instance from directory")
    print("3. Run optimization algorithms")
    print("4. Compare algorithm performance on a day")
    print("5. Compare algorithm performance on all days")
    print("6. Exit")

    while True:
        choice = input("Select an option (or 'back' to return): ")
        if choice.lower() == 'back':
            return 'back'
        if choice in ['1', '2', '3', '4', '5', '6']:
            return choice
        print("Invalid choice, please try again")

def run_algorithms_interactive(instance_dir):
    """Run optimization algorithms with user parameters"""
    algorithms = {
        '1': ('hill_climbing', {'max_iterations': 1000, 'neighbor_count': 10}),
        '2': ('simulated_annealing', {'max_iterations': 3000, 'initial_temp': 100, 'cooling_rate': 0.97}),
        '3': ('tabu_search', {'max_iterations': 1000, 'tabu_size': 10, 'neighbor_count': 10}),
        '4': ('genetic_algorithm', {'population_size': 25, 'generations': 350, 'mutation_rate': 0.18, 'crossover_rate': 0.6})
    }

    print("\nSelect algorithm to run:")
    print("1. Hill Climbing")
    print("2. Simulated Annealing")
    print("3. Tabu Search")
    print("4. Genetic Algorithm")
    print("5. Run all algorithms")
    print("6. Back to main menu")

    while True:
        choice = input("Enter your choice: ")
        if choice == '6' or choice.lower() == 'back':
            return

        if choice == '5':
            # Run all algorithms
            for date in dates:
                print(f"\nDay: {date.strftime('%Y-%m-%d')}")
                load_stock_data(date)
                if(not stock_prices):
                    print("No stock data available for this date.")
                    continue
                for algo_name, params in algorithms.values():
                    print(f"Running {algo_name.replace('_', ' ').title()}...")

                    initial_solution = aux.initialize_portfolio(stock_prices)
                    solution, score, _, _ = run_optimization(algo_name, initial_solution, params, date, instance_dir)
                    print(f"Best Sharpe Ratio: {score:.4f}")
                    print(f"Obtained solution: {solution}")
            break
        elif choice in algorithms:
            # Run single algorithm
            algo_name, params = algorithms[choice]
            print(f"\nRunning {algo_name.replace('_', ' ').title()}...")

            # Allow parameter customization
            print(f"Current parameters: {params}")
            customize = input("Customize parameters? (y/n/back): ").lower()
            if customize == 'back':
                continue
            if customize == 'y':
                for param in params:
                    new_val = input(f"Enter new value for {param} (current: {params[param]}, or 'back' to cancel): ")
                    if new_val.lower() == 'back':
                        break
                    if new_val:
                        try:
                            params[param] = type(params[param])(new_val)
                        except ValueError:
                            print(f"Invalid value for {param}, keeping default")
                else:  # Only run if we didn't break out of the loop
                    for date in dates:
                        print(f"Day: {date.strftime('%Y-%m-%d')}")
                        load_stock_data(date)
                        if not stock_prices:
                            print("No stock data available for this date.")
                            continue
                        initial_solution = aux.initialize_portfolio(stock_prices)
                        solution, score, _, _ = run_optimization(algo_name, initial_solution, params, date, instance_dir)
                        print(f"Best Score: {score:.4f}")
                        print(f"Obtained solution: {solution}")
            elif customize == 'n':
                for date in dates:
                    print(f"Day: {date.strftime('%Y-%m-%d')}")
                    load_stock_data(date)
                    if not stock_prices:
                        print("No stock data available for this date.")
                        continue
                    initial_solution = aux.initialize_portfolio(stock_prices)
                    solution, score, _, _ = run_optimization(algo_name, initial_solution, params, date, instance_dir)
                    print(f"Best Score: {score:.4f}")
                    print(f"Obtained solution: {solution}")
            break
        else:
            print("Invalid choice, please try again")


def compare_algorithms_interactive(instance_dir):
    """Compare performance of different algorithms for a specific day"""
    if not instance_dir or not os.path.isdir(instance_dir):
        print("\nInvalid or missing instance directory.")
        return

    # Get list of dates that have been run
    available_dates = [d for d in os.listdir(instance_dir) if os.path.isdir(os.path.join(instance_dir, d))]

    if not available_dates:
        print("\nNo results available. Please run some algorithms first.")
        return

    print("\nAvailable dates:")
    for i, date in enumerate(available_dates, 1):
        print(f"{i}. {date}")

    print(f"{len(available_dates) + 1}. Back to main menu")

    while True:
        choice = input("Select date to compare results (number or 'back' to return): ")
        if choice.lower() == 'back':
            return

        try:
            choice_num = int(choice)
            if choice_num == len(available_dates) + 1:
                return
            if 1 <= choice_num <= len(available_dates):
                selected_date = available_dates[choice_num - 1]
                break
        except ValueError:
            print("Invalid input. Please enter a number.")

    day_dir = os.path.join(instance_dir, selected_date)
    dt = pd.to_datetime(selected_date)
    available_algorithms = [algo for algo in algorithm_metrics if algorithm_metrics[algo].get(dt) and algorithm_metrics[algo][dt]['best_score']]
    if not available_algorithms:
        print("\nNo algorithms have been run yet. Please run some algorithms first.")
        return

    print("\nAvailable algorithm results:")
    for i, algo in enumerate(available_algorithms, 1):
        run_count = len(algorithm_metrics[algo][dt]['best_score'])
        print(f"{i}. {algo.replace('_', ' ').title()} ({run_count} run{'s' if run_count > 1 else ''})")

    print(f"{len(available_algorithms) + 1}. All available algorithms")
    print(f"{len(available_algorithms) + 2}. Back to main menu")

    while True:
        choices = input("Select algorithms to compare (comma-separated numbers, or 'back' to return): ")
        if choices.lower() == 'back':
            return

        try:
            selected_indices = [int(choice.strip()) for choice in choices.split(',')]
            algorithms_to_compare = []

            for idx in selected_indices:
                if idx == len(available_algorithms) + 1:
                    algorithms_to_compare = available_algorithms.copy()
                    break
                elif idx == len(available_algorithms) + 2:
                    return
                elif 1 <= idx <= len(available_algorithms):
                    algorithms_to_compare.append(available_algorithms[idx - 1])

            if not algorithms_to_compare:
                print("No valid algorithms selected")
                continue

            vis.plot_algorithm_comparison(selected_date, algorithm_metrics, day_dir, algorithms_to_compare)
            print("\nComparison completed and saved to instance directory")
            break
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas.")

def compare_all_algorithms(instance_dir):
    """Compare performance of all algorithms for each day in the instance"""
    if not instance_dir or not os.path.isdir(instance_dir):
        print("\nInvalid or missing instance directory.")
        return

    available_dates = sorted([d for d in os.listdir(instance_dir) if os.path.isdir(os.path.join(instance_dir, d))], key=lambda x: datetime.strptime(x, '%Y-%m-%d'))

    if not available_dates:
        print("\nNo results available. Please run some algorithms first.")
        return

    all_scores = {algo: [] for algo in algorithm_metrics.keys()}
    dates = []

    for date_str in available_dates:
        date = pd.to_datetime(date_str)
        dates.append(date_str)
        for algo in algorithm_metrics.keys():
            if algorithm_metrics[algo].get(date) and algorithm_metrics[algo][date]['best_score']:
                avg_score = np.mean(algorithm_metrics[algo][date]['best_score'])
                all_scores[algo].append(avg_score)
            else:
                all_scores[algo].append(None)  # No score for this date

    vis.plot_all_algorithm_comparison(dates, all_scores, instance_dir)


# ###########################################
#             Main Execution
# ###########################################

if __name__ == '__main__':
    print("Stock Portfolio Optimization System")
    print("---------------------------------")

    stocks = None
    dates = None
    instance_dir = None

    while True:
        choice = main_menu()

        if choice == 'back':
            continue
        elif choice == '1':
            result = io.create_problem_instance()
            if result[0] is not None and result[1] is not None:  # Only update if not going back
                dates, instance_dir = result
        elif choice == '2':
            result = io.load_problem_instance_interactive()
            if result[0] is not None and result[1] is not None:  # Only update if not going back
                dates, instance_dir = result
        elif choice == '3':
            if dates is None:
                print("\nPlease load or create a problem instance first")
                continue
            run_algorithms_interactive(instance_dir)
        elif choice == '4':
            compare_algorithms_interactive(instance_dir)
        elif choice == '5':
            compare_all_algorithms(instance_dir)
        elif choice == '6':
            print("\nExiting...")
            break
        else:
            print("\nInvalid choice, please try again")