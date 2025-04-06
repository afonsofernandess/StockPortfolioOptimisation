import pandas as pd
import os
from datetime import datetime
import json

import auxiliar as aux

def save_problem_instance(instance_name, dates):
    """Save problem instance to file in its directory"""
    instance_dir = os.path.join('instances', instance_name)
    if not os.path.exists(instance_dir):
        os.makedirs(instance_dir)

    data = {
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
    return pd.date_range(data['start_date'], data['end_date']), instance_dir


def save_results(instance_dir, solution, score, algorithm_name, parameters, execution_time, date):
    """Save optimization results to file in instance directory"""
    newdir = os.path.join(instance_dir, date.strftime('%Y-%m-%d'))
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {
        'algorithm': algorithm_name,
        'solution': solution,
        'score': score,
        'parameters': parameters,
        'execution_time': execution_time,
        'timestamp': timestamp
    }

    filename = os.path.join(newdir, f"Results_{algorithm_name}_{timestamp}.json")
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    return timestamp

def create_problem_instance():
    """Create and save a new problem instance"""
    global dates
    dates = aux.get_date_range()
    if dates is None:  # User chose to go back
        return None, None

    instance_name = input("Enter a name for this problem instance (or 'back' to return): ")
    if instance_name.lower() == 'back':
        return None, None

    instance_dir = save_problem_instance(instance_name, dates)
    print(f"Problem instance saved to {instance_dir}")
    return dates, instance_dir


def load_problem_instance_interactive():
    """Load problem instance from directory"""
    print("\nAvailable instances:")
    instances = [d for d in os.listdir('instances') if os.path.isdir(os.path.join('instances', d))]

    if not instances:
        print("No instances available. Please create one first.")
        return None, None

    for i, instance in enumerate(instances, 1):
        print(f"{i}. {instance}")
    print(f"{len(instances) + 1}. Back to main menu")

    while True:
        choice = input("Select instance to load (number or name, or 'back' to return): ")
        if choice.lower() == 'back':
            return None, None

        try:
            # Try to interpret as number
            if choice.isdigit():
                choice_num = int(choice)
                if choice_num == len(instances) + 1:
                    return None, None
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

    global dates
    dates, instance_dir = load_problem_instance(instance_name)
    return dates, instance_dir

