# Stock Portfolio Optimisation System

## Overview
This project is an implementation of a stock portfolio optimisation system in python.
The system calculates the optimal weights of a portfolio of stocks for each day based on the relative return of each stock.

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
- Create or load a problem instance;
- Run the optimisation algorithms;
- Compare the results of different algorithms;

## Features
- **Problem Instance Creation**: Create a new problem instance with a set of stocks from a chosen date range.
- **Problem Instance Loading**: Load a previously created problem instance.
- **Optimisation Algorithms**: Run different optimisation algorithms (with customizable parameters) to find the optimal weights of the portfolio. The algorithms include:
  - **Hill Climbing**: A local search algorithm that iteratively makes small changes to the current solution to find a better one.
  - **Simulated Annealing**: A probabilistic algorithm that explores the solution space by allowing worse solutions to be accepted with a certain probability, which decreases over time.
  - **Tabu Search**: A local search algorithm that uses memory structures to avoid revisiting previously explored solutions.
  - **Genetic Algorithm**: An evolutionary algorithm that uses selection, crossover, and mutation to evolve a population of solutions towards better ones.
- **Compare Results (Day)**: Compare the results of different algorithms for a specific day.
- **Compare Results (All Days)**: Compare the results of different algorithms for all days in the problem instance.