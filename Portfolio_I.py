# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 13:24:48 2024

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
# Define the intervals as a dictionary for AAA, AA, BBB rating transitions
intervals_AAA = {
    'D': (-12, -11),
    'CCC': (-11, -10),
    'B': (-10, -4.012810811119328),
    'BB': (-4.012810811119328, -3.460087443038689),
    'BBB': (-3.460087443038689, -3.093215946698764),
    'A': (-3.093215946698764, -2.45419599111397),
    'AA': (-2.45419599111397, -1.3478706173405963),
    'AAA': (-1.3478706173405963, 10),
}

intervals_AA = {
    'D': (-8.209536151601387, -3.5271870106498224),
    'CCC': (-3.5271870106498224, -3.359796158748848),
    'B': (-3.359796158748848, -3.174683527455022),
    'BB': (-3.174683527455022, -2.991071931712996),
    'BBB': (-2.991071931712996, -2.526516429517446),
    'A': (-2.526516429517446, -1.3088062184617912),
    'AA': (-1.3088062184617912, 2.389311141634685),
    'AAA': (2.389311141634685, 10),
}

intervals_BBB = {
    'D': (-10, -2.932726257334291),
    'CCC': (-2.932726257334291, -2.7035651427151923),
    'B': (-2.7035651427151923, -2.3076250501322817),
    'BB': (-2.3076250501322817, -1.6498188118115051),
    'BBB': (-1.6498188118115051, 1.6988895306121583),
    'A': (1.6988895306121583, 2.9147268289727286),
    'AA': (2.9147268289727286, 3.422710813085352),
    'AAA': (3.422710813085352, 10),
}

# Assign new rating based on the intervals for AAA, AA, BBB
simulation_df['new_rating_AAA'] = simulation_df['X'].apply(
    lambda x: next((rating for rating, (low, high) in intervals_AAA.items() if low < x <= high), 'D')
)
simulation_df['new_rating_AA'] = simulation_df['X'].apply(
    lambda x: next((rating for rating, (low, high) in intervals_AA.items() if low < x <= high), 'D')
)
simulation_df['new_rating_BBB'] = simulation_df['X'].apply(
    lambda x: next((rating for rating, (low, high) in intervals_BBB.items() if low < x <= high), 'D')
)

# Display the updated simulation_df with the first few rows
simulation_df.head()


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
    'D': 60.00
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
aaa_count = (simulation_df['new_rating_AAA'] == 'AAA').sum() / num_simulations
non_aaa_count = (simulation_df['new_rating_AAA'] != 'AAA').sum()/ num_simulations

aa_count = (simulation_df['new_rating_AA'] == 'AA').sum()/ num_simulations
non_aa_count = (simulation_df['new_rating_AA'] != 'AA').sum()/ num_simulations

bbb_count = (simulation_df['new_rating_BBB'] == 'BBB').sum()/ num_simulations
non_bbb_count = (simulation_df['new_rating_BBB'] != 'BBB').sum()/ num_simulations

# Display the counts
print(f'Times stayed AAA: {aaa_count}')
print(f'Times didn\'t stay AAA: {non_aaa_count}')

print(f'Times stayed AA: {aa_count}')
print(f'Times didn\'t stay AA: {non_aa_count}')

print(f'Times stayed BBB: {bbb_count}')
print(f'Times didn\'t stay BBB: {non_bbb_count}')

###############################################################################
# Calculate Expected Value

expected_values_AAA = simulation_df['new_rating_AAA'].apply(lambda x:   base_values['AAA'] 
                                                            if x == 'AAA' else  shocked_values[x])
expected_values_AA = simulation_df['new_rating_AA'].apply(lambda x:  base_values['AA'] 
                                                          if x == 'AA' else  shocked_values[x])
expected_values_BBB = simulation_df['new_rating_BBB'].apply(lambda x:  base_values['BBB'] 
                                                            if x == 'BBB' else  shocked_values[x])


# Calculate Expected Portfolio Value for each simulation
simulation_df['Exp_Value_AAA'] = expected_values_AAA
simulation_df['Exp_Value_AA'] = expected_values_AA
simulation_df['Exp_Value_BBB'] = expected_values_BBB


initial_value = ( 0.6 * 99.4 + 0.3 * 98.39 + 0.1 * 92.79) * 1500

# Calculate the Average Portfolio Value across all simulations
simulation_df['Avg_Exp_Value'] = (simulation_df['Exp_Value_AAA'] * 0.6 
                                  + simulation_df['Exp_Value_AA'] * 0.3 
                                  + simulation_df['Exp_Value_BBB'] * 0.1) * 1500
#simulation_df['Returns'] = ( simulation_df['Avg_Exp_Value'] - initial_value ) / initial_value

# Calculate the average value over 10 simulations
#avg_AAA = expected_values_AAA.mean()
#avg_AA = expected_values_AA.mean()
#avg_BBB = expected_values_BBB.mean()


# Display the expected and average values
#print(expected_values_AAA, avg_AAA)
#print(expected_values_AA, avg_AA)
#print(expected_values_BBB, avg_BBB)

Portf_Avg_Value = simulation_df['Avg_Exp_Value'].mean() # millions EUR
print(f'Expected (Average) Value of Porfolio I: {Portf_Avg_Value}.')
#########################################################################################################
#########################################################################################################
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


##################################################################################
# Check

#pos_portfolio_changes= np.sum(portfolio_changes > 0)
#neg_portfolio_changes = np.sum(portfolio_changes < 0)

positive_count = np.count_nonzero(portfolio_changes > 0) / num_simulations
negative_count = np.count_nonzero(portfolio_changes < 0) / num_simulations
print(positive_count) 
print(negative_count)




































