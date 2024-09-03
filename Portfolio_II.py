# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 16:21:32 2024

@author: ieron
"""

import pandas as pd
import numpy as np
from scipy.stats import norm


# Your transition matrix
transition_matrix = np.array([
    [0.91115, 0.08179, 0.00607, 0.00072, 0.00024, 0.00003, 0.00000, 0.00000],
    [0.00844, 0.89626, 0.08954, 0.00437, 0.00064, 0.00036, 0.00018, 0.00021],
    [0.00055, 0.02595, 0.91138, 0.05509, 0.00499, 0.00107, 0.00045, 0.00052],
    [0.00031, 0.00147, 0.04289, 0.90584, 0.03898, 0.00708, 0.00175, 0.00168],
    [0.00007, 0.00044, 0.00446, 0.06741, 0.83274, 0.07667, 0.00895, 0.00926],
    [0.00008, 0.00031, 0.0015, 0.0049, 0.05373, 0.82531, 0.07894, 0.03523],
    [0.00000, 0.00015, 0.00023, 0.00091, 0.00388, 0.0763, 0.83035, 0.08818],
    [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 1.00000]
])

# Index labels
index_labels = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'D']

# Create a DataFrame
transition_df = pd.DataFrame(transition_matrix, index=index_labels, columns=index_labels)

# Calculate cumulative probability
cumulative_df = transition_df.cumsum(axis=1)

# Calculate NORMSINV(1 - cumulative probability) with shifted values for lower bound
lower_df = pd.DataFrame(norm.ppf(1 - cumulative_df.shift(1, axis=1).fillna(0).values), columns=index_labels)

# Calculate NORMSINV(1 - cumulative probability) and convert to DataFrame for upper bound
upper_df = pd.DataFrame(norm.ppf(1 - cumulative_df.values), columns=index_labels)

# Assign new index labels
cumulative_df.index = [f'c_{label}' for label in index_labels]
lower_df.index = [f'l_{label}' for label in index_labels]
upper_df.index = [f'u_{label}' for label in index_labels]

# Combine values from l_{index} and u_{index} to create new rows
bins_df = pd.DataFrame(index=[f'B_{label}' for label in index_labels], columns=index_labels)
for label in index_labels:
    bins_df.loc[f'B_{label}'] = list(zip(lower_df[label], upper_df[label]))

# Display the bins
#bins_df = bins_df.transpose()
print(bins_df)


###############################################################################

# Set the seed for reproducibility
np.random.seed(1981)

# Number of simulations
num_simulations = 500000

# Correlation value
corr = 1

# Generate 'Y' column with random values from a standard normal distribution
Y_column = norm.ppf(np.random.rand(num_simulations))

# Generate 'e' column with random values from a standard normal distribution
e_column = norm.ppf(np.random.rand(num_simulations))

# Calculate 'X' column based on the given formula
X_column = np.sqrt(corr) * Y_column + np.sqrt(1 - corr) * e_column

# Create a DataFrame to store the results
simulation_df = pd.DataFrame({
    'Y': Y_column,
    'e': e_column,
    'X': X_column
})

# Display the simulation results
print(simulation_df)
###############################################################################
###############################################################################
###############################################################################
# Define the intervals as a dictionary
intervals_BB = {
    'D': (-10, -2.3550542838738737),
    'CCC': (-2.3550542838738737, -2.092207028989833),
    'B': (-2.092207028989833, -1.311289424366028),
    'BB': (-1.311289424366028, 1.4582922699394292),
    'BBB': (1.4582922699394292, 2.577909594360747),
    'A': (2.577909594360747, 3.2849513008401496),
    'AA': (3.2849513008401496, 3.8081682644489683),
    'AAA': (3.8081682644489683, 10),
}

intervals_B = {
    'D': (-10, -1.8089420934059597),
    'CCC': (-1.8089420934059597, -1.2046459757412644),
    'B': (-1.2046459757412644, 1.5504230903381704),
    'BB': (1.5504230903381704, 2.4681853040490207),
    'BBB': (2.4681853040490207, 2.895960480135712),
    'A': (2.895960480135712, 3.3597961587489262),
    'AA': (3.3597961587489262, 3.775011939356454),
    'AAA': (3.775011939356454, 10),
}

intervals_CCC = {
    'D': (-10, -1.3520478766042054),
    'CCC': (-1.3520478766042054, 1.3952515393406255),
    'B': (1.3952515393406255, 2.5642467713605885),
    'BB': (2.5642467713605885, 3.0137974400371323),
    'BBB': (3.0137974400371323, 3.3669662096427815),
    'A': (3.3669662096427815, 3.6153000069246914),
    'AA': (3.6153000069246914, 10),
    'AAA': (10, 11),
}


# Initialize a new column for labels in simulation_df
simulation_df['new_rating_BB','new_rating_B','new_rating_CCC'] = ''

# Compare 'X' values from simulation_df to intervals_BB and assign labels
for label, interval in intervals_BB.items():
    # Check if 'X' values fall within the interval
    mask = (simulation_df['X'] > interval[0]) & (simulation_df['X'] <= interval[1])
    simulation_df.loc[mask, 'new_rating_BB'] = label

# Compare 'X' values from simulation_df to intervals_B and assign labels
for label, interval in intervals_B.items():
    # Check if 'X' values fall within the interval
    mask = (simulation_df['X'] > interval[0]) & (simulation_df['X'] <= interval[1])
    simulation_df.loc[mask, 'new_rating_B'] = label
    
# Compare 'X' values from simulation_df to intervals_CCC and assign labels
for label, interval in intervals_CCC.items():
    # Check if 'X' values fall within the interval
    mask = (simulation_df['X'] > interval[0]) & (simulation_df['X'] <= interval[1])
    simulation_df.loc[mask, 'new_rating_CCC'] = label



# Display the updated simulation_df
print(simulation_df)

###############################################################################
# Base Values
base_values = {
    'AAA': 99.40,
    'AA': 98.39,
    'A': 97.22,
    'BBB': 92.79,
    'BB': 90.11,
    'B': 86.60,
    'CCC': 77.16,
    'D': 0.00
}

# Shocked Values
shocked_values = {
    'AAA': 99.50,
    'AA': 98.51,
    'A': 97.53,
    'BBB': 92.77,
    'BB': 90.48,
    'B': 88.25,
    'CCC': 77.88,
    'D': 60.00
}

# Count how many times it stayed AAA and how many times it didn't
bb_count = (simulation_df['new_rating_BB'] == 'BB').sum() /num_simulations
non_bb_count = (simulation_df['new_rating_BB'] != 'BB').sum() /num_simulations

b_count = (simulation_df['new_rating_B'] == 'B').sum() /num_simulations
non_b_count = (simulation_df['new_rating_B'] != 'B').sum() /num_simulations

ccc_count = (simulation_df['new_rating_CCC'] == 'CCC').sum() /num_simulations
non_ccc_count = (simulation_df['new_rating_CCC'] != 'CCC').sum() /num_simulations

# Display the counts
print(f'Times stayed BB: {bb_count}')
print(f'Times didn\'t stay BB: {non_bb_count}')

print(f'Times stayed B: {b_count}')
print(f'Times didn\'t stay B: {non_b_count}')

print(f'Times stayed CCC: {ccc_count}')
print(f'Times didn\'t stay CCC: {non_ccc_count}')
###############################################################################
# Calculate Expected Value

expected_values_BB = simulation_df['new_rating_BB'].apply(lambda x:  base_values['BB'] 
                                                            if x == 'BB' else  shocked_values[x])
expected_values_B = simulation_df['new_rating_B'].apply(lambda x:  base_values['B'] 
                                                          if x == 'B' else  shocked_values[x])
expected_values_CCC = simulation_df['new_rating_CCC'].apply(lambda x:  base_values['CCC'] 
                                                            if x == 'CCC' else  shocked_values[x])


# Calculate Expected Portfolio Value for each simulation
simulation_df['Exp_Value_BB'] = expected_values_BB
simulation_df['Exp_Value_B'] = expected_values_B
simulation_df['Exp_Value_CCC'] = expected_values_CCC


initial_value = ( 0.6 * 90.11 + 0.35 * 86.6 + 0.05 * 77.16) * 1500

# Calculate the Average Portfolio Value across all simulations
simulation_df['Avg_Exp_Value'] = (simulation_df['Exp_Value_BB'] * 0.6 
                                  + simulation_df['Exp_Value_B'] * 0.35 
                                  + simulation_df['Exp_Value_CCC'] * 0.05) * 1500
#simulation_df['Returns'] = ( simulation_df['Avg_Exp_Value'] - initial_value ) / initial_value

# Calculate the average value over 10 simulations
#avg_BB = expected_values_BB.mean()
#avg_B = expected_values_B.mean()
#avg_CCC = expected_values_CCC.mean()


# Display the expected and average values
#print(expected_values_BB, avg_BB)
#print(expected_values_B, avg_B)
#print(expected_values_CCC, avg_CCC)

Portf_Avg_Value = simulation_df['Avg_Exp_Value'].mean() # millions EUR
print(f'Expected (Average) Value of Porfolio I: {Portf_Avg_Value}')
#########################################################################################################

#%%
# Simulated portfolio values
simulation_values = simulation_df['Avg_Exp_Value']

# Calculate changes in portfolio values
portfolio_changes = simulation_values - initial_value

# Calculate VaR at 90% and 99.5% confidence levels
VaR_90 = np.percentile(portfolio_changes, 10)  # 10th percentile for 90% confidence
VaR_995 = np.percentile(portfolio_changes, 0.5)  # 0.5th percentile for 99.5% confidence

# Calculate ES at 90% and 99.5% confidence levels
# This is the average of losses that are beyond the VaR cutoff
ES_90 = portfolio_changes[portfolio_changes <= VaR_90].mean()
ES_995 = portfolio_changes[portfolio_changes <= VaR_995].mean()

VaR_90, VaR_995, ES_90, ES_995

#########################################################################################################

# Check

#pos_portfolio_changes= np.sum(portfolio_changes > 0)
#neg_portfolio_changes = np.sum(portfolio_changes < 0)

positive_count = np.count_nonzero(portfolio_changes > 0) / num_simulations
negative_count = np.count_nonzero(portfolio_changes < 0) / num_simulations
print(positive_count) 
print(negative_count)


















