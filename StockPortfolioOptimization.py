import pandas as pd
import random
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import json
from collections import deque
import time
import shutil

# Global Variables
stock_prices = {}
algorithm_metrics = {
    'hill_climbing': {'time': [], 'best_score': []},
    'simulated_annealing': {'time': [], 'best_score': []},
    'tabu_search': {'time': [], 'best_score': []},
    'genetic_algorithm': {'time': [], 'best_score': []}
}

# Create instances directory if it doesn't exist
if not os.path.exists('instances'):
    os.makedirs('instances')


# ###########################################
#             Auxiliar Functions
# ###########################################

def get_date_range():
    """Get valid date range from user input"""
    dates = pd.date_range('2000-01-02', '2020-04-02')
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
        print("Please enter a valid date between 2000-01-02 and 2020-04-02")

    if pd.to_datetime(start_date) > pd.to_datetime(end_date):
        print("End date must be after start date")
        exit(1)

    return pd.date_range(start_date, end_date)


def load_stock_data(valid_dates, stock_limit=None):
    """Load stock data with optional limit on number of stocks"""
    global stock_prices
    stock_prices = {}

    stocks = [stock.split('.')[0] for stock in sorted(os.listdir('archive'))]
    if stock_limit:
        stocks = stocks[:stock_limit]

    for stock in stocks:
        try:
            pc = pd.read_csv(f'archive/{stock}.csv', usecols=['Date', 'Adj Close'])
            pc['Date'] = pd.to_datetime(pc['Date'])
            pc = pc[pc['Date'].isin(valid_dates)]
            if len(pc) > 0:
                stock_prices[stock] = pc['Adj Close']
        except:
            continue

    print(f"Loaded {len(stock_prices)} stocks")
    return stock_prices


def initialize_portfolio(stocks):
    """Generate random initial portfolio weights"""
    weights = {symbol: random.random() for symbol in stocks}
    return normalize_weights(weights)


def normalize_weights(weights):
    """Normalize weights to sum to 1"""
    total_weight = sum(weights.values())
    return {k: v / total_weight for k, v in weights.items()}


def calculate_daily_returns(prices):
    return prices.pct_change(fill_method=None).dropna()  # Explicitly disable filling


def calculate_risk(prices):
    """Calculate risk (std dev) of daily returns"""
    daily_returns = calculate_daily_returns(prices)
    return np.std(daily_returns)


def calculate_average_return(prices):
    """Calculate average daily return"""
    daily_returns = calculate_daily_returns(prices)
    return np.mean(daily_returns)


# ###########################################
#             Evaluation Function
# ###########################################

def evaluate_portfolio(weights):
    """Evaluate portfolio using Sharpe ratio"""
    if not stock_prices:
        raise ValueError("No stock data loaded")

    returns = [calculate_average_return(stock_prices[symbol]) for symbol in weights.keys()]
    risks = [calculate_risk(stock_prices[symbol]) for symbol in weights.keys()]

    portfolio_return = np.dot(list(weights.values()), returns)
    portfolio_risk = np.sqrt(np.sum([w * r for w, r in zip(weights.values(), risks)]))

    # Avoid division by zero
    if portfolio_risk == 0:
        return 0
    return portfolio_return / portfolio_risk


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
    return normalize_weights(new_weights)


# ###########################################
#             Optimization Algorithms
# ###########################################

def hill_climbing(initial_solution, max_iterations=1000, neighbor_count=10):
    """Hill Climbing optimization algorithm"""
    start_time = time.time()
    current_solution = initial_solution
    current_score = evaluate_portfolio(current_solution)
    best_solution = current_solution
    best_score = current_score

    scores = [current_score]

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
    algorithm_metrics['hill_climbing']['time'].append(execution_time)
    algorithm_metrics['hill_climbing']['best_score'].append(best_score)
    return best_solution, best_score, scores, execution_time


def simulated_annealing(initial_solution, max_iterations=1000, initial_temp=100, cooling_rate=0.99):
    """Simulated Annealing optimization algorithm"""
    start_time = time.time()
    current_solution = initial_solution
    current_score = evaluate_portfolio(current_solution)
    best_solution = current_solution
    best_score = current_score
    temp = initial_temp

    scores = [current_score]

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
    algorithm_metrics['simulated_annealing']['time'].append(execution_time)
    algorithm_metrics['simulated_annealing']['best_score'].append(best_score)
    return best_solution, best_score, scores, execution_time


def tabu_search(initial_solution, max_iterations=1000, tabu_size=10, neighbor_count=10):
    """Tabu Search optimization algorithm"""
    start_time = time.time()
    current_solution = initial_solution
    current_score = evaluate_portfolio(current_solution)
    best_solution = current_solution
    best_score = current_score
    tabu_list = deque(maxlen=tabu_size)

    scores = [current_score]

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
    algorithm_metrics['tabu_search']['time'].append(execution_time)
    algorithm_metrics['tabu_search']['best_score'].append(best_score)
    return best_solution, best_score, scores, execution_time


def genetic_algorithm(population_size=20, generations=100, mutation_rate=0.1, crossover_rate=0.8):
    """Genetic Algorithm optimization"""
    start_time = time.time()
    stocks = list(stock_prices.keys())

    # Initialize population
    population = [initialize_portfolio(stocks) for _ in range(population_size)]
    best_solution = max(population, key=evaluate_portfolio)
    best_score = evaluate_portfolio(best_solution)

    scores = [best_score]

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
                crossover_point = random.randint(1, len(stocks) - 1)
                keys = list(parent1.keys())

                child1 = {**parent1}
                child2 = {**parent2}

                # Swap weights after crossover point
                for j in range(crossover_point, len(keys)):
                    child1[keys[j]], child2[keys[j]] = child2[keys[j]], child1[keys[j]]

                new_population[i] = normalize_weights(child1)
                new_population[i + 1] = normalize_weights(child2)

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
    algorithm_metrics['genetic_algorithm']['time'].append(execution_time)
    algorithm_metrics['genetic_algorithm']['best_score'].append(best_score)
    return best_solution, best_score, scores, execution_time


# ###########################################
#             Visualization
# ###########################################

def plot_optimization_process(scores, algorithm_name, instance_dir=None, timestamp=None):
    """Plot the optimization progress over iterations"""
    plt.figure(figsize=(10, 6))
    plt.plot(scores, label='Best Score')
    plt.title(f'{algorithm_name} Optimization Progress')
    plt.xlabel('Iteration')
    plt.ylabel('Sharpe Ratio')
    plt.legend()
    plt.grid(True)

    if instance_dir:
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_name = f"{algorithm_name.replace(' ', '_')}_progress_{timestamp}.png"
        plot_path = os.path.join(instance_dir, plot_name)
        plt.savefig(plot_path)
        plt.close()
    else:
        plt.show()


def plot_algorithm_comparison(instance_dir=None, algorithms_to_compare=None):
    """Compare performance of selected algorithms"""
    if not algorithms_to_compare:
        print("No algorithms selected for comparison")
        return

    plt.figure(figsize=(12, 6))

    # Create display names with run counts below the main name
    display_names = []
    for algo in algorithms_to_compare:
        run_count = len(algorithm_metrics[algo]['best_score'])
        name = algo.replace('_', ' ').title()
        if run_count > 1:
            name += f"\n(avg of {run_count} runs)"
        display_names.append(name)

    # Plot solution quality comparison
    plt.subplot(1, 2, 1)
    scores = [np.mean(algorithm_metrics[algo]['best_score']) for algo in algorithms_to_compare]
    bars = plt.bar(display_names, scores)
    plt.title('Average Solution Quality')
    plt.ylabel('Sharpe Ratio')

    # Adjust the text position for better visibility
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')

    # Plot time comparison
    plt.subplot(1, 2, 2)
    times = [np.mean(algorithm_metrics[algo]['time']) for algo in algorithms_to_compare]
    bars = plt.bar(display_names, times)
    plt.title('Average Execution Time')
    plt.ylabel('Seconds')

    # Adjust the text position for better visibility
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}s',
                 ha='center', va='bottom')

    plt.tight_layout()

    if instance_dir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = os.path.join(instance_dir, f"algorithm_comparison_{timestamp}.png")
        plt.savefig(plot_path)
        plt.close()
    else:
        plt.show()

# ###########################################
#             File I/O
# ###########################################

def save_problem_instance(instance_name, stocks, dates):
    """Save problem instance to file in its directory"""
    instance_dir = os.path.join('instances', instance_name)
    if not os.path.exists(instance_dir):
        os.makedirs(instance_dir)

    data = {
        'stocks': stocks,
        'start_date': str(dates[0]),
        'end_date': str(dates[-1])
    }

    filename = os.path.join(instance_dir, f"{instance_name}_instance.json")
    with open(filename, 'w') as f:
        json.dump(data, f)

    return instance_dir


def load_problem_instance(instance_name):
    """Load problem instance from file in its directory"""
    instance_dir = os.path.join('instances', instance_name)
    filename = os.path.join(instance_dir, f"{instance_name}_instance.json")

    with open(filename, 'r') as f:
        data = json.load(f)
    return data['stocks'], pd.date_range(data['start_date'], data['end_date']), instance_dir


def save_results(instance_dir, solution, score, algorithm_name, parameters, execution_time):
    """Save optimization results to file in instance directory"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {
        'algorithm': algorithm_name,
        'solution': solution,
        'score': score,
        'parameters': parameters,
        'execution_time': execution_time,
        'timestamp': timestamp
    }

    filename = os.path.join(instance_dir, f"results_{algorithm_name}_{timestamp}.json")
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    return timestamp


# ###########################################
#             Main Interface
# ###########################################

def run_optimization(algorithm, initial_solution, parameters, instance_dir=None):
    """Run specified optimization algorithm"""
    if algorithm == 'hill_climbing':
        solution, score, scores, execution_time = hill_climbing(initial_solution, **parameters)
    elif algorithm == 'simulated_annealing':
        solution, score, scores, execution_time = simulated_annealing(initial_solution, **parameters)
    elif algorithm == 'tabu_search':
        solution, score, scores, execution_time = tabu_search(initial_solution, **parameters)
    elif algorithm == 'genetic_algorithm':
        solution, score, scores, execution_time = genetic_algorithm(**parameters)
    else:
        raise ValueError("Unknown algorithm")

    timestamp = save_results(instance_dir, solution, score, algorithm, parameters, execution_time)
    plot_optimization_process(scores, algorithm.replace('_', ' ').title(), instance_dir, timestamp)

    return solution, score, scores, execution_time


def main_menu():
    """Interactive main menu"""
    print("\nStock Portfolio Optimization")
    print("1. Create new problem instance")
    print("2. Load problem instance from directory")
    print("3. Run optimization algorithms")
    print("4. Compare algorithm performance")
    print("5. Exit")

    while True:
        choice = input("Select an option (or 'back' to return): ")
        if choice.lower() == 'back':
            return 'back'
        if choice in ['1', '2', '3', '4', '5']:
            return choice
        print("Invalid choice, please try again")


def create_problem_instance():
    """Create and save a new problem instance"""
    dates = get_date_range()
    if dates is None:  # User chose to go back
        return None, None, None

    stock_limit = input("Enter maximum number of stocks to include (0 for all, or 'back' to return): ")
    if stock_limit.lower() == 'back':
        return None, None, None

    stock_limit = int(stock_limit or "0")
    stocks = load_stock_data(dates, stock_limit if stock_limit > 0 else None)

    instance_name = input("Enter a name for this problem instance (or 'back' to return): ")
    if instance_name.lower() == 'back':
        return None, None, None

    instance_dir = save_problem_instance(instance_name, list(stocks.keys()), dates)
    print(f"Problem instance saved to {instance_dir}")
    return list(stocks.keys()), dates, instance_dir


def load_problem_instance_interactive():
    """Load problem instance from directory"""
    print("\nAvailable instances:")
    instances = [d for d in os.listdir('instances') if os.path.isdir(os.path.join('instances', d))]

    if not instances:
        print("No instances available. Please create one first.")
        return None, None, None

    for i, instance in enumerate(instances, 1):
        print(f"{i}. {instance}")
    print(f"{len(instances) + 1}. Back to main menu")

    while True:
        choice = input("Select instance to load (number or name, or 'back' to return): ")
        if choice.lower() == 'back':
            return None, None, None

        try:
            # Try to interpret as number
            if choice.isdigit():
                choice_num = int(choice)
                if choice_num == len(instances) + 1:
                    return None, None, None
                if 1 <= choice_num <= len(instances):
                    instance_name = instances[choice_num - 1]
                    break
            else:
                # Treat as name
                if choice in instances:
                    instance_name = choice
                    break
        except:
            pass

        print("Invalid choice, please try again")

    stocks, dates, instance_dir = load_problem_instance(instance_name)
    load_stock_data(dates)
    print(f"Loaded problem instance '{instance_name}' with {len(stocks)} stocks")
    return stocks, dates, instance_dir


def run_algorithms_interactive(stocks, instance_dir):
    """Run optimization algorithms with user parameters"""
    initial_solution = initialize_portfolio(stocks)

    algorithms = {
        '1': ('hill_climbing', {'max_iterations': 1000, 'neighbor_count': 10}),
        '2': ('simulated_annealing', {'max_iterations': 1000, 'initial_temp': 100, 'cooling_rate': 0.99}),
        '3': ('tabu_search', {'max_iterations': 1000, 'tabu_size': 10, 'neighbor_count': 10}),
        '4': (
        'genetic_algorithm', {'population_size': 20, 'generations': 100, 'mutation_rate': 0.1, 'crossover_rate': 0.8})
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
            for algo_name, params in algorithms.values():
                print(f"\nRunning {algo_name.replace('_', ' ').title()}...")
                solution, score, _, _ = run_optimization(algo_name, initial_solution, params, instance_dir)
                print(f"Best Sharpe Ratio: {score:.4f}")
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
                    solution, score, _, _ = run_optimization(algo_name, initial_solution, params, instance_dir)
                    print(f"Best Sharpe Ratio: {score:.4f}")
            elif customize == 'n':
                solution, score, _, _ = run_optimization(algo_name, initial_solution, params, instance_dir)
                print(f"Best Sharpe Ratio: {score:.4f}")
            break
        else:
            print("Invalid choice, please try again")


def compare_algorithms_interactive(instance_dir):
    """Interactive algorithm comparison"""
    # Get list of algorithms that have been run
    available_algorithms = [algo for algo in algorithm_metrics if algorithm_metrics[algo]['best_score']]

    if not available_algorithms:
        print("\nNo algorithms have been run yet. Please run some algorithms first.")
        return

    print("\nAvailable algorithm results:")
    for i, algo in enumerate(available_algorithms, 1):
        run_count = len(algorithm_metrics[algo]['best_score'])
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

            plot_algorithm_comparison(instance_dir, algorithms_to_compare)
            print("\nComparison completed and saved to instance directory")
            break
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas.")


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
            result = create_problem_instance()
            if result != (None, None, None):  # Only update if not going back
                stocks, dates, instance_dir = result
        elif choice == '2':
            result = load_problem_instance_interactive()
            if result != (None, None, None):  # Only update if not going back
                stocks, dates, instance_dir = result
        elif choice == '3':
            if not stocks:
                print("\nPlease load or create a problem instance first")
                continue
            run_algorithms_interactive(stocks, instance_dir)
        elif choice == '4':
            compare_algorithms_interactive(instance_dir)
        elif choice == '5':
            print("\nExiting...")
            break
        else:
            print("\nInvalid choice, please try again")