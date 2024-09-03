# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 19:19:08 2024

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

# Set the seed for reproducibility
np.random.seed(42)

# Number of simulations
num_simulations = 500000

# Correlation value
corr = 1

# Create a DataFrame to store the results
simulation_df = pd.DataFrame()

# Number of issuers
num_issuers = 100

# Master DataFrame to store results for all issuers
master_df = pd.DataFrame()

# Generate 'Y' column with random values from a standard normal distribution
Y_column = norm.ppf(np.random.rand(num_simulations))

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

# Loop through each issuer
for i in range(num_issuers):
    # Generate 'e' column with random values from a standard normal distribution
    e_column = norm.ppf(np.random.rand(num_simulations))

    # Calculate 'X' column based on the given formula
    X_column = np.sqrt(corr) * Y_column + np.sqrt(1 - corr) * e_column

    # Add 'X' column to simulation_df
    simulation_df['X'] = X_column

    # Define the intervals as a dictionary
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

    # Initialize new columns for labels
    simulation_df['new_rating_AAA'] = ''
    simulation_df['new_rating_AA'] = ''
    simulation_df['new_rating_BBB'] = ''

    # Compare 'X' values from simulation_df to intervals_AAA and assign labels
    for label, interval in intervals_AAA.items():
        mask = (simulation_df['X'] > interval[0]) & (simulation_df['X'] <= interval[1])
        simulation_df.loc[mask, 'new_rating_AAA'] = label

    # Compare 'X' values from simulation_df to intervals_AA and assign labels
    for label, interval in intervals_AA.items():
        mask = (simulation_df['X'] > interval[0]) & (simulation_df['X'] <= interval[1])
        simulation_df.loc[mask, 'new_rating_AA'] = label

    # Compare 'X' values from simulation_df to intervals_BBB and assign labels
    for label, interval in intervals_BBB.items():
        mask = (simulation_df['X'] > interval[0]) & (simulation_df['X'] <= interval[1])
        simulation_df.loc[mask, 'new_rating_BBB'] = label

    # Append the results for each issuer to a master DataFrame if needed
    master_df = pd.concat([master_df, simulation_df], axis=0, ignore_index=True)

###############################################################################

# Count how many times it stayed AAA and how many times it didn't
aaa_count = (simulation_df['new_rating_AAA'] == 'AAA').sum()
non_aaa_count = (simulation_df['new_rating_AAA'] != 'AAA').sum()

aa_count = (simulation_df['new_rating_AA'] == 'AA').sum()
non_aa_count = (simulation_df['new_rating_AA'] != 'AA').sum()

bbb_count = (simulation_df['new_rating_BBB'] == 'BBB').sum()
non_bbb_count = (simulation_df['new_rating_BBB'] != 'BBB').sum()

# Display the counts
print(f'Times stayed AAA: {aaa_count}')
print(f'Times didn\'t stay AAA: {non_aaa_count}')

print(f'Times stayed AA: {aa_count}')
print(f'Times didn\'t stay AA: {non_aa_count}')

print(f'Times stayed BBB: {bbb_count}')
print(f'Times didn\'t stay BBB: {non_bbb_count}')
###############################################################################
# Calculate Expected Value

expected_values_AAA = simulation_df['new_rating_AAA'].apply(lambda x: (aaa_count/num_simulations) * base_values['AAA'] 
                                                            if x == 'AAA' else (aaa_count/num_simulations) * shocked_values[x])
expected_values_AA = simulation_df['new_rating_AA'].apply(lambda x: (aaa_count/num_simulations) * base_values['AA'] 
                                                          if x == 'AA' else (aaa_count/num_simulations) * shocked_values[x])
expected_values_BBB = simulation_df['new_rating_BBB'].apply(lambda x: (aaa_count/num_simulations) * base_values['BBB'] 
                                                            if x == 'BBB' else (aaa_count/num_simulations) * shocked_values[x])


# Calculate Expected Portfolio Value for each simulation
simulation_df['Exp_Value_AAA'] = expected_values_AAA
simulation_df['Exp_Value_AA'] = expected_values_AA
simulation_df['Exp_Value_BBB'] = expected_values_BBB


initial_value = ( 0.6 * 99.4 + 0.3 * 98.39 + 0.1 * 92.79) * 1500

# Calculate the Average Portfolio Value across all simulations
simulation_df['Avg_Exp_Value'] = (simulation_df['Exp_Value_AAA'] * 0.6 
                                  + simulation_df['Exp_Value_AA'] * 0.3 
                                  + simulation_df['Exp_Value_BBB'] * 0.1) * 1500
simulation_df['Returns'] = ( simulation_df['Avg_Exp_Value'] - initial_value ) / initial_value

# Calculate the average value over 10 simulations
avg_AAA = expected_values_AAA.mean()
avg_AA = expected_values_AA.mean()
avg_BBB = expected_values_BBB.mean()


# Display the expected and average values
print(expected_values_AAA, avg_AAA)
print(expected_values_AA, avg_AA)
print(expected_values_BBB, avg_BBB)

Portf_Avg_Value = simulation_df['Avg_Exp_Value'].mean() # millions EUR
print(f'Expected (Average) Value of Porfolio I: {Portf_Avg_Value}')
#########################################################################################################
#########################################################################################################
#########################################################################################################
#%%

# Sort the simulated portfolio values
sorted_values = simulation_df['Avg_Exp_Value'].sort_values()

# Calculate VaR at 90% and 99.5%
var_90 = np.percentile(sorted_values, 90)
var_995 = np.percentile(sorted_values, 99.5)

print(f'VaR at 90%: {var_90}')
print(f'VaR at 99.5%: {var_995}')

# Calculate Expected Shortfall (ES) at 90% and 99.5%
es_90 = sorted_values[sorted_values >= var_90].mean()
es_995 = sorted_values[sorted_values >= var_995].mean()

print(f'ES at 90%: {es_90}')
print(f'ES at 99.5%: {es_995}')























