import re
import matplotlib.pyplot as plt

def extract_data(file_path):
    T2_values = []
    fidelity_values = []
    key_rate_values = []
    
    with open(file_path, 'r') as file:
        content = file.read()
        tasks = re.findall(r'Task \d+: T_2 = ([\d.e-]+), Fidelity = ([\d.e-]+), Key rate per time = ([\d.e-]+)', content)
        for task in tasks:
            T2 = float(task[0])
            fidelity = float(task[1])
            key_rate = float(task[2])
            T2_values.append(T2)
            fidelity_values.append(fidelity)
            key_rate_values.append(key_rate)
    
    return T2_values, fidelity_values, key_rate_values

# File paths
file1 = '1D_data/Ber_Schäp_Kö_T2.rtf'
file1a = '1D_data/Ber_Schäp_Kö_T2_cut0.2.rtf'
file2 = '1D_data/Kö_Eul_Erf_T2.rtf'
file2a = '1D_data/Kö_Eul_Erf_T2_cut0.2.rtf'
file3 = '1D_data/Erf_Wal_Eit_T2.rtf'
file3a = '1D_data/Erf_Wal_Eit_T2_cut0.2.rtf'
file4 = '1D_data/Eit_Schü_DEC_T2.rtf'
file4a = '1D_data/Eit_Schü_DEC_T2_cut0.2.rtf'


# Extract data
T2_values_1, fidelity_values_1, key_rate_values_1 = extract_data(file1)
T2_values_1a, fidelity_values_1a, key_rate_values_1a = extract_data(file1a)
T2_values_2, fidelity_values_2, key_rate_values_2 = extract_data(file2)
T2_values_2a, fidelity_values_2a, key_rate_values_2a = extract_data(file2a)
T2_values_3, fidelity_values_3, key_rate_values_3 = extract_data(file3)
T2_values_3a, fidelity_values_3a, key_rate_values_3a = extract_data(file3a)
T2_values_4, fidelity_values_4, key_rate_values_4 = extract_data(file4)
T2_values_4a, fidelity_values_4a, key_rate_values_4a = extract_data(file4a)


# Plot data
plt.figure(figsize=(10, 6))
plt.scatter(T2_values_1, fidelity_values_1, s=12, color='blue', marker='o')
plt.scatter(T2_values_1a, fidelity_values_1a, s=12, color='blue', marker='*')
plt.scatter(T2_values_2, fidelity_values_2, s=12, color='red', marker='o')
plt.scatter(T2_values_2a, fidelity_values_2a, s=12, color='red', marker='*')
plt.scatter(T2_values_3, fidelity_values_3, s=12, color='green', marker='o')
plt.scatter(T2_values_3a, fidelity_values_3a, s=12, color='green', marker='*')
plt.scatter(T2_values_4, fidelity_values_4, s=12, color='orange', marker='o')
plt.scatter(T2_values_4a, fidelity_values_4a, s=12, color='orange', marker='*')

custom_lines = [
    plt.Line2D([0], [0], color='blue', lw=2, label='Berlin-Schäpe-Köckern'),
    plt.Line2D([0], [0], color='red', lw=2, label='Köckern-Eulau-Erfurt'),
    plt.Line2D([0], [0], color='green', lw=2, label='Erfurt-Waltershausen-Eiterfeld'),
    plt.Line2D([0], [0], color='orange', lw=2, label='Eiterfeld-Schüchtern-DeCIX'),
    plt.Line2D([0], [0], color='black', marker='*', linestyle='None', label='cutoff 0.2')
]

plt.legend(handles=custom_lines, loc='lower right')

plt.xlabel('T2')
plt.ylabel('Fidelity')
plt.grid(True)
plt.xscale('log')
#plt.yscale('log')
plt.show()
