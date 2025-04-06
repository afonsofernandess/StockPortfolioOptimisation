# Stock Portfolio Optimisation System

## Overview
This project implements a stock portfolio optimization system in Python. 

The system calculates the optimal allocation of weights for a portfolio of stocks on a daily basis, based on the relative returns of each stock.

## Requirements
This project was built with Python 3.12.3 and the packages listed in the [requirements.txt](requirements.txt) file.

## Installation
After cloning the repository and having the required version of python installed, you can install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Usage
To run the system, you can use the following command:

```bash
python main.py
```

This opens the main menu in which you can:
1. Create or load a problem instance;
2. Run the optimisation algorithms;
3. Compare the results of different algorithms;

## Features

**Create a new problem instance**
To create a new instance, select a start and end date within the available range (from January 3rd, 2020 to April 1st, 2020), and give your instance a name. Please note that execution time increases with larger datasets.

Once created, the instance will be saved in the 'instances' directory.

**Load problem instance**
Choosing the option to load an instance will present a list of available instances. You can then select the one you'd like to work with. There is also a default instance named 'two_days', which corresponds to April 4th and 5th, 2015.

**Run optimization algorithms**
This option allows you to run a specific algorithm with customizable parameters, or execute all algorithms at once. After the algorithms are run, the data related to the optimal weights recommended by each algorithm for each day will be saved in a new directory inside the instance directory.

The algorithms include:
  - **Hill Climbing**: A local search algorithm that iteratively makes small changes to the current solution to find a better one.
  - **Simulated Annealing**: A probabilistic algorithm that explores the solution space by allowing worse solutions to be accepted with a certain probability, which decreases over time.
  - **Tabu Search**: A local search algorithm that uses memory structures to avoid revisiting previously explored solutions.
  - **Genetic Algorithm**: An evolutionary algorithm that uses selection, crossover, and mutation to evolve a population of solutions towards better ones.

After running some algorithms you can compare their results.

**Compare Results (Day)**
Compare the results of different algorithms for a specific day.

**Compare Results (All Days)** 
Compare the results of different algorithms for all days in the problem instance.