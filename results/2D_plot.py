import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Read the data from the text file
with open('2D_data/TP_T2_100.rtf', 'r') as file:
    data = file.read()

# Step 2: Parse the data to extract T_P, T_2, and Fidelity values
task_pattern = re.compile(r'Task \d+: T_P = ([\de.-]+), T_2 = ([\de.-]+), Fidelity = ([\de.-]+)')

tasks = []
for match in task_pattern.finditer(data):
    T_P = float(match.group(1))
    T_2 = float(match.group(2))
    Fidelity = float(match.group(3))
    tasks.append((T_P, T_2, Fidelity))

# Step 3: Create a DataFrame for easier handling and sorting
df = pd.DataFrame(tasks, columns=['T_P', 'T_2', 'Fidelity'])

# Create pivot table for 2D plotting
pivot_table = df.pivot_table(index='T_P', columns='T_2', values='Fidelity')

# Step 4: Generate a 2D colormesh plot
T_P_vals = pivot_table.index.values
T_2_vals = pivot_table.columns.values
Fidelity_vals = pivot_table.values

plt.figure(figsize=(8, 6))
plt.pcolormesh(T_2_vals, T_P_vals, Fidelity_vals, shading='auto', cmap='viridis')
plt.colorbar(label='Fidelity')
plt.xlabel('T_2')
plt.ylabel('T_P')
plt.yscale('log')
plt.xscale('log')
plt.show()
