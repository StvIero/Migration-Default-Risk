# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 21:42:06 2024

@author: ieron
"""

import matplotlib.pyplot as plt
import numpy as np

# Data
rho_values = [0.0, 0.33, 0.66, 1]

# Portfolio I
portfolio_I_expected_values = [134487.84, 134471.25, 134441.08, 134437.70]
portfolio_I_var_90 = [134607.00, 134591.64, 134562.11, 134557.68]
portfolio_I_var_995 = [135710.54, 135695.05, 135665.28, 135660.81]
portfolio_I_es_90 = [134643.53, 134627.93, 134598.86, 134594.06]
portfolio_I_es_995 = [135744.49, 135729.22, 135698.51, 135693.80]

# Portfolio II
portfolio_II_expected_values = [108989.57, 108929.97, 108911.18, 108839.24]
portfolio_II_var_90 = [109846.47, 109803.53, 109773.67, 109710.44]
portfolio_II_var_995 = [118909.25, 115296.91, 115126.76, 115199.10]
portfolio_II_es_90 = [110.225, 110.179, 110.150, 110.086]
portfolio_II_es_995 = [119.896, 119.659, 118.462, 119.544]

# Plotting
plt.figure(figsize=(12, 16))

# Portfolio I - Expected Values
plt.subplot(4, 1, 1)
plt.plot(rho_values, portfolio_I_expected_values, marker='o', label='Expected Value')
plt.title('Portfolio I - Expected Values vs Rho')
plt.xlabel('Rho (%)')
plt.ylabel('Value (in thousands EUR)')
plt.legend()

# Portfolio I - VaR at 90%
plt.subplot(4, 1, 2)
plt.plot(rho_values, portfolio_I_var_90, marker='o', label='VaR at 90%')
plt.title('Portfolio I - VaR at 90% vs Rho')
plt.xlabel('Rho (%)')
plt.ylabel('Value (in thousands EUR)')
plt.legend()

# Portfolio I - VaR at 99.5%
plt.subplot(4, 1, 3)
plt.plot(rho_values, portfolio_I_var_995, marker='o', label='VaR at 99.5%')
plt.title('Portfolio I - VaR at 99.5% vs Rho')
plt.xlabel('Rho (%)')
plt.ylabel('Value (in thousands EUR)')
plt.legend()

# Portfolio I - ES
plt.subplot(4, 1, 4)
plt.plot(rho_values, portfolio_I_es_90, marker='o', label='ES at 90%')
plt.plot(rho_values, portfolio_I_es_995, marker='o', label='ES at 99.5%')
plt.title('Portfolio I - Expected Shortfall vs Rho')
plt.xlabel('Rho (%)')
plt.ylabel('Value (in thousands EUR)')
plt.legend()

plt.tight_layout()
plt.show()

# Plotting for Portfolio II
plt.figure(figsize=(12, 16))

# Portfolio II - Expected Values
plt.subplot(4, 1, 1)
plt.plot(rho_values, portfolio_II_expected_values, marker='o', label='Expected Value')
plt.title('Portfolio II - Expected Values vs Rho')
plt.xlabel('Rho (%)')
plt.ylabel('Value (in thousands EUR)')
plt.legend()

# Portfolio II - VaR at 90%
plt.subplot(4, 1, 2)
plt.plot(rho_values, portfolio_II_var_90, marker='o', label='VaR at 90%')
plt.title('Portfolio II - VaR at 90% vs Rho')
plt.xlabel('Rho (%)')
plt.ylabel('Value (in thousands EUR)')
plt.legend()

# Portfolio II - VaR at 99.5%
plt.subplot(4, 1, 3)
plt.plot(rho_values, portfolio_II_var_995, marker='o', label='VaR at 99.5%')
plt.title('Portfolio II - VaR at 99.5% vs Rho')
plt.xlabel('Rho (%)')
plt.ylabel('Value (in thousands EUR)')
plt.legend()

# Portfolio II - ES
plt.subplot(4, 1, 4)
plt.plot(rho_values, portfolio_II_es_90, marker='o', label='ES at 90%')
plt.plot(rho_values, portfolio_II_es_995, marker='o', label='ES at 99.5%')
plt.title('Portfolio II - Expected Shortfall vs Rho')
plt.xlabel('Rho (%)')
plt.ylabel('Value (in thousands EUR)')
plt.legend()

plt.tight_layout()
plt.show()

###############################################################################

# Data
rho_values = [0.0, 0.33, 0.66, 1]

# Portfolio I
portfolio_I_expected_values = [134413.77, 134330.03, 134434.88, 134411.20]
portfolio_I_var_90 = [134534.65, 134451.07, 134554.14, 134531.10]
portfolio_I_var_995 = [135637.59, 135553.33, 135657.24, 135634.02]
portfolio_I_es_90 = [134571.16, 1344878.97, 134590.78, 134567.56]
portfolio_I_es_995 = [135670.52, 135584.98, 135690.51, 135668.09]

# Portfolio II
portfolio_II_expected_values = [108944.29, 108934.53, 108998.31, 109061.36]
portfolio_II_var_90 = [109811.61, 109806.51, 109856.31, 109926.69]
portfolio_II_var_995 = [115305.97, 118865.78, 115352.69, 115425.96]
portfolio_II_es_90 = [110.18749, 110.18215, 110.23161, 110.29969]
portfolio_II_es_995 = [119.65864, 119.83141, 119.69006, 119.85198]

# Plotting
plt.figure(figsize=(12, 16))

# Portfolio I - Expected Values
plt.subplot(4, 1, 1)
plt.plot(rho_values, portfolio_I_expected_values, marker='o', label='Expected Value')
plt.title('Portfolio I - Expected Values vs Rho')
plt.xlabel('Rho (%)')
plt.ylabel('Value (in thousands EUR)')
plt.legend()

# Portfolio I - VaR at 90%
plt.subplot(4, 1, 2)
plt.plot(rho_values, portfolio_I_var_90, marker='o', label='VaR at 90%')
plt.title('Portfolio I - VaR at 90% vs Rho')
plt.xlabel('Rho (%)')
plt.ylabel('Value (in thousands EUR)')
plt.legend()

# Portfolio I - VaR at 99.5%
plt.subplot(4, 1, 3)
plt.plot(rho_values, portfolio_I_var_995, marker='o', label='VaR at 99.5%')
plt.title('Portfolio I - VaR at 99.5% vs Rho')
plt.xlabel('Rho (%)')
plt.ylabel('Value (in thousands EUR)')
plt.legend()

# Portfolio I - ES
plt.subplot(4, 1, 4)
plt.plot(rho_values, portfolio_I_es_90, marker='o', label='ES at 90%')
plt.plot(rho_values, portfolio_I_es_995, marker='o', label='ES at 99.5%')
plt.title('Portfolio I - Expected Shortfall vs Rho')
plt.xlabel('Rho (%)')
plt.ylabel('Value (in thousands EUR)')
plt.legend()

plt.tight_layout()
plt.show()

# Plotting for Portfolio II
plt.figure(figsize=(12, 16))

# Portfolio II - Expected Values
plt.subplot(4, 1, 1)
plt.plot(rho_values, portfolio_II_expected_values, marker='o', label='Expected Value')
plt.title('Portfolio II - Expected Values vs Rho')
plt.xlabel('Rho (%)')
plt.ylabel('Value (in thousands EUR)')
plt.legend()

# Portfolio II - VaR at 90%
plt.subplot(4, 1, 2)
plt.plot(rho_values, portfolio_II_var_90, marker='o', label='VaR at 90%')
plt.title('Portfolio II - VaR at 90% vs Rho')
plt.xlabel('Rho (%)')
plt.ylabel('Value (in thousands EUR)')
plt.legend()

# Portfolio II - VaR at 99.5%
plt.subplot(4, 1, 3)
plt.plot(rho_values, portfolio_II_var_995, marker='o', label='VaR at 99.5%')
plt.title('Portfolio II - VaR at 99.5% vs Rho')
plt.xlabel('Rho (%)')
plt.ylabel('Value (in thousands EUR)')
plt.legend()

# Portfolio II - ES
plt.subplot(4, 1, 4)
plt.plot(rho_values, portfolio_II_es_90, marker='o', label='ES at 90%')
plt.plot(rho_values, portfolio_II_es_995, marker='o', label='ES at 99.5%')
plt.title('Portfolio II - Expected Shortfall vs Rho')
plt.xlabel('Rho (%)')
plt.ylabel('Value (in thousands EUR)')
plt.legend()

plt.tight_layout()
plt.show()


