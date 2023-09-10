import numpy as np
import pandas as pd

# Load the bus data
bus_data = pd.read_csv('bus.csv')
# Load the branch data
branch_data = pd.read_csv('branch_data.csv')
# print("branch data", len(branch_data))
# Number of buses and branches
num_buses = len(bus_data)
num_branches = len(branch_data)

# print("number of branches", len(branch_data))

# Initialize an empty Y matrix
Y_matrix = np.zeros((num_buses, num_buses), dtype=complex)
# print("y matrix", Y_matrix)
# Populate the Y matrix with admittances (Y)
for i in range(num_branches):
#     print("=================",branch_data['FromBus'][i])
#     if i==1: break

    from_bus = branch_data['FromBus'][i] - 1  # Subtract 1 to convert to 0-based indexing
    to_bus = branch_data['ToBus'][i] - 1
    
#     print("from bus=========", from_bus)
#     print("to bus=========", to_bus)
    
    # Calculate admittance for the branch
    R = branch_data['R (p.u)'][i]
    X = branch_data['X (p.u)'][i]
    Z = complex(R, X)
    Y = 1 / Z
#     print("Y admittance ", Y)
    # Add admittances to the Y matrix
    Y_matrix[from_bus, from_bus] += Y
#     Y_matrix[to_bus, to_bus] += Y
    Y_matrix[from_bus, to_bus] -= Y
#     Y_matrix[to_bus, from_bus] += Y

# Print the Y matrix (if desired)
print("Y Matrix:")
print(Y_matrix)
