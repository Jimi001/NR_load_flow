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

# Create a list of generator buses
generator_buses = gen_data['Bus No'].tolist()

def pow_inj():
    # Create 'P_inj (p.u.)' and 'Q_inj (p.u.)' columns in the DataFrame
    bus_data['P_inj (p.u.)'] = 0.0
    bus_data['Q_inj (p.u.)'] = 0.0

    # Calculate real and reactive power injections for each bus
    for index, row in bus_data.iterrows():
        # Skip calculations for the slack bus (bus 1)
        if index == 0:
            continue
        
        # Find the corresponding generator data by matching bus number
        matching_generator = gen_data[gen_data['Bus No'] == row['Bus No']]
        
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

def power_flow(Vm, Va):
    P_flows = np.zeros(NUM_BUSES)
    Q_flows = np.zeros(NUM_BUSES)
    
    pow_inj()

    for i in range(NUM_BUSES):
        # Skip calculations for the slack bus (bus 1)
        if i == 0:
            continue
        
        # If it's a generator bus (voltage-controlled bus)
        if i in generator_buses:
            # Keep Vm constant (known) and calculate Q
            Q_flows[i] = Vm[i] * sum(Vm * (Y_matrix[i, :] * np.sin(Va - Va[i]))) - bus_data.at[i, 'Q_inj (p.u.)']
        else:
            # For other non-slack buses, calculate both Vm and Va
            P_flows[i] = Vm[i] * sum(Vm * (Y_matrix[i, :] * np.cos(Va - Va[i]))) - bus_data.at[i, 'P_inj (p.u.)']
            Q_flows[i] = Vm[i] * sum(Vm * (Y_matrix[i, :] * np.sin(Va - Va[i]))) - bus_data.at[i, 'Q_inj (p.u.)']
    
    return P_flows, Q_flows

def newton_raphson_load_flow():
    # Define convergence tolerance
    tolerance = 1e-6
    
    # Initialize voltage magnitudes (Vm) and voltage angles (Va)
    Vm = np.ones(NUM_BUSES)
    Va = np.zeros(NUM_BUSES)
    
    # Perform the Newton-Raphson load flow
    while True:
        # Calculate the power flow equations and Jacobian matrix
        P_mismatches, Q_mismatches = power_flow(Vm, Va)
        Jacobian = build_jacobian(Vm, Va)
        
        # Solve for the voltage increments
        delta = np.linalg.solve(Jacobian, -np.concatenate((P_mismatches, Q_mismatches)))
        
        # Update voltage angles
        Va += delta[:NUM_BUSES]
        
        # Check for convergence
        if max(abs(delta)) < tolerance:
            break
    
    return Vm, Va

def build_jacobian(Vm, Va):
    # Initialize an empty Jacobian matrix
    Jacobian = np.zeros((2 * NUM_BUSES, 2 * NUM_BUSES))
    
    for i in range(NUM_BUSES):
        for j in range(NUM_BUSES):
            if i == j:
                continue
            
            dP_dVa, dP_dVm, dQ_dVa, dQ_dVm = calculate_jacobian_elements(i, j, Vm, Va)
            
            Jacobian[i, j] = dP_dVa / BASE_MVA
            Jacobian[i, j + NUM_BUSES] = dP_dVm / BASE_MVA
            Jacobian[i + NUM_BUSES, j] = dQ_dVa / BASE_MVA
            Jacobian[i + NUM_BUSES, j + NUM_BUSES] = dQ_dVm / BASE_MVA
    
    return Jacobian

def calculate_jacobian_elements(i, j, Vm, Va):
    # Define some constants
    Vmi = Vm[i]
    Vmj = Vm[j]
    Yij = Y_matrix[i, j]
    delta_ij = Va[i] - Va[j]
    
    # Calculate derivatives
    dP_dVa = Vmi * Vmj * abs(Yij) * Vmi * np.sin(delta_ij)
    dP_dVm = Vmj * abs(Yij) * Vmi * np.cos(delta_ij)
    dQ_dVa = -Vmi * Vmj * abs(Yij) * Vmi * np.cos(delta_ij)
    dQ_dVm = Vmi * abs(Yij) * Vmj * np.sin(delta_ij)
    
    return dP_dVa, dP_dVm, dQ_dVa, dQ_dVm

# Perform the Newton-Raphson load flow
final_Vm, final_Va = newton_raphson_load_flow()

# Print the final voltage magnitudes (Vm) and angles (Va) for all buses
print("Final Voltage Magnitudes (Vm):", final_Vm)
print("Final Voltage Angles (Va):", final_Va)
