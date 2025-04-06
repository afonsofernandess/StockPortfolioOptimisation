import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

def plot_optimization_process(scores, algorithm_name, date, instance_dir=None, timestamp=None):
    """Plot the optimization progress over iterations"""
    resultdir = os.path.join(instance_dir, date.strftime('%Y-%m-%d'))
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)

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
        plot_path = os.path.join(resultdir, plot_name)
        plt.savefig(plot_path)
        plt.close()
    else:
        plt.show()

def plot_solution(solution, algorithm_name, date, instance_dir=None, timestamp=None):
    """Plot the solution weights for each stock"""
    resultdir = os.path.join(instance_dir, date.strftime('%Y-%m-%d'))
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)

    plt.figure(figsize=(20, 6))
    stocks = list(solution.keys())
    weights = list(solution.values())

    plt.bar(stocks, weights)
    plt.title(f'{algorithm_name} Solution Weights')
    plt.xlabel('Stocks')
    plt.ylabel('Weights')
    plt.xticks(rotation=90)
    plt.grid(True)

    if instance_dir:
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_name = f"{algorithm_name.replace(' ', '_')}_solution_{timestamp}.png"
        plot_path = os.path.join(resultdir, plot_name)
        plt.savefig(plot_path)
        plt.show()
        plt.close()
    else:
        plt.show()


def plot_algorithm_comparison(date, algorithm_metrics, instance_dir=None, algorithms_to_compare=None):
    """Compare performance of selected algorithms"""
    if not algorithms_to_compare:
        print("No algorithms selected for comparison")
        return
    datetime = pd.to_datetime(date)

    plt.figure(figsize=(12, 6))

    # Create display names with run counts below the main name
    display_names = []
    for algo in algorithms_to_compare:
        run_count = len(algorithm_metrics[algo][datetime]['best_score'])
        name = algo.replace('_', ' ').title()
        if run_count > 1:
            name += f"\n(avg of {run_count} runs)"
        display_names.append(name)

    # Plot solution quality comparison
    plt.subplot(1, 2, 1)
    scores = [np.mean(algorithm_metrics[algo][datetime]['best_score']) for algo in algorithms_to_compare]
    bars = plt.bar(display_names, scores)
    plt.title('Average Solution Quality')
    plt.ylabel('Score')

    # Adjust the text position for better visibility
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')

    # Plot time comparison
    plt.subplot(1, 2, 2)
    times = [np.mean(algorithm_metrics[algo][datetime]['time']) for algo in algorithms_to_compare]
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
        plot_path = os.path.join(instance_dir, f"Algorithm_comparison_{timestamp}.png")
        plt.savefig(plot_path)
        plt.close()
    else:
        plt.show()

def plot_all_algorithm_comparison(dates, all_scores, instance_dir):
    """Plot comparison of all algorithms over multiple days"""
    plt.figure(figsize=(12, 6))

    for algo, scores in all_scores.items():
        if any(score is None for score in scores):
            continue
        plt.plot(dates, scores, label=algo.replace('_', ' ').title())

    plt.title('Algorithm Performance Comparison Over Time')
    plt.xlabel('Date')
    plt.ylabel('Average Score')
    plt.legend()
    plt.grid(True)

    if instance_dir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = os.path.join(instance_dir, f"All_algorithm_comparison_{timestamp}.png")
        plt.savefig(plot_path)
        plt.show()
        plt.close()
    else:
        plt.show()