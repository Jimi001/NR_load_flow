import numpy as np
import pandas as pd

# Load the bus, generator, and branch data
bus_data = pd.read_csv('bus_data.csv')
gen_data = pd.read_csv('generator_data.csv')
branch_data = pd.read_csv('branch_data.csv')

# Define power base values
BASE_MVA = 100.0  # Base MVA for the system

# Number of buses and branches
NUM_BUSES = len(bus_data)
NUM_BRANCHES = len(branch_data)

# Initialize an empty Y matrix
Y_matrix = np.zeros((NUM_BUSES, NUM_BUSES), dtype=complex)

def get_y_matrix() -> np.ndarray:
    # Populate the Y matrix with admittances (Y)
    for i in range(NUM_BRANCHES):
        from_bus = branch_data['FromBus'][i] - 1  # Subtract 1 to convert to 0-based indexing
        to_bus = branch_data['ToBus'][i] - 1
        
        # Calculate admittance for the branch
        R = branch_data['R (p.u)'][i]
        X = branch_data['X (p.u)'][i]
        Z = complex(R, X)
        Y = 1 / Z
        
        # Add admittances to the Y matrix
        Y_matrix[from_bus, from_bus] += Y
        Y_matrix[from_bus, to_bus] -= Y
    return Y_matrix

Y_matrix = get_y_matrix()

def pow_inj():
    # Create 'P_inj (p.u.)' and 'Q_inj (p.u.)' columns in the DataFrame
    bus_data['P_inj (p.u.)'] = 0.0
    bus_data['Q_inj (p.u.)'] = 0.0

    # Calculate real and reactive power injections for each bus
    for index, row in bus_data.iterrows():
        # Find the corresponding generator data by matching bus number
        matching_generator = gen_data[gen_data['Bus No'] == row['Bus No']]
        # print("matching generator", matching_generator)
        if not matching_generator.empty:
            generator_row = matching_generator.iloc[0]
            
            P_gen = generator_row['Pg (MW)'] / BASE_MVA  # Convert MW to per unit (p.u.)
            Q_gen = generator_row['Qg (Mvar)'] / BASE_MVA  # Convert Mvar to per unit (p.u.)
        else:
            # If there is no matching generator data, assume zero generation
            P_gen = 0.0
            Q_gen = 0.0

        P_load = row['Pd (MW)'] / BASE_MVA  # Convert MW to per unit (p.u.)
        Q_load = row['Qd(Mvar)'] / BASE_MVA  # Convert Mvar to per unit (p.u.)
        
        # Calculate real and reactive power injections
        P_inj = P_gen - P_load
        Q_inj = Q_gen - Q_load
        
        # Update the 'P_inj (p.u.)' and 'Q_inj (p.u.)' columns in the DataFrame
        bus_data.at[index, 'P_inj (p.u.)'] = P_inj
        bus_data.at[index, 'Q_inj (p.u.)'] = Q_inj

    # Display the updated bus_data DataFrame with power injections
    print("Bus Data with Power Injections:")
    print(bus_data)

# Initialize an empty Jacobian matrix with the correct dimensions
Jacobian = np.zeros((2 * NUM_BUSES, 2 * NUM_BUSES))

# Define a function to compute the power flow equations (P and Q)
def power_flow(Vm, Va):
    P_flows = np.zeros(NUM_BUSES)
    Q_flows = np.zeros(NUM_BUSES)
    
    for i in range(NUM_BUSES):
        # Calculate P and Q for each bus using the power flow equations
        if i == 0: continue
        # If it's a generator bus (voltage-controlled bus)
        if i in gen_data:
            # Keep Vm constant (known) and calculate Va
            # P_flows[i] = Vm[i] * sum(Vm * (Y_matrix[i, :] * np.cos(Va - Va[i]))) - bus_data.at[i, 'P_inj (p.u.)']
            Q_flows[i] = Vm[i] * sum(Vm * (Y_matrix[i, :] * np.sin(Va - Va[i]))) - bus_data.at[i, 'Q_inj (p.u.)']
        else:
            # For other non-slack buses, calculate both Vm and Va
            P_flows[i] = Vm[i] * sum(Vm * (Y_matrix[i, :] * np.cos(Va - Va[i]))) - bus_data.at[i, 'P_inj (p.u.)']
            Q_flows[i] = Vm[i] * sum(Vm * (Y_matrix[i, :] * np.sin(Va - Va[i]))) - bus_data.at[i, 'Q_inj (p.u.)']
    
    return P_flows, Q_flows


