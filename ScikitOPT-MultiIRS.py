import math
import numpy as np
import pyswarms as ps
import copy

### Distance (in m) ###
D_11_I1 = 20
D_12_I1 = 30
D_21_I2 = 15
D_22_I2 = 21
D_1_IB = 5
D_2_IB = 9
D_I1_1 = 4
D_I2_2 = 4
D_IB_B = 10
D_2_I1 = 8
D_1_I2 = 8
D_12_1 = 34
D_11_1 = 24
D_22_2 = 24
D_21_2 = 19
D_2_B = 19
D_1_B = 15

M = 1  # No. of antennas on BS
N = N1 = N2 = NB  = 8  # No. IRS elements
K = 1  # No.of transmit antennas at the relay device
K_dash = 1 # No. of transmit antennas at the redcap (IoT) device
q = 1 # No. of Quantization bits. 1,2,..., Q
m = 1 # m=1,2,..., 2^n-1
rho = 0.1

n_particles = 50
m_iterations = 200
inertia= 0.7

cognitive = 1.4
social = 1.4
### Indicator for Direct Link
gamma = 1

### Indicator for no-IRS
mu = 1

### Indicator for IRS in Relay to Device
alpha = 1

## Indicator for IRS in Relay to BS
beta = 1



def find_min(a, b):
  if(a > b):
    return b
  return a


def generate_channel(N, K, path_loss):
    h = (1/np.sqrt(2)) * (np.random.randn(N, K) + 1j * np.random.randn(N, K))
    h_normalized = h / np.linalg.norm(h, axis=0)
    h = path_loss * h_normalized
    return h


# def generate_channel (N, K, path_loss):
#     msr = 2  ## M=1 Rayleigh
#     var_sr = path_loss
#     x1 = np.random.gamma(msr,(var_sr)/msr,N*K)
#     y1 = 2 * math.pi * np.random.randn(N, K)
#     zsr = x1**(1j * y1)
#     h_sr = math.sqrt(zsr)
#     return h_sr


def get_pathloss_direct(d):
   ### QUES: Is there an indirect path loss as well? #TODO find any other 3GPP compliant model

   pl = 0
   if(d != 0):
    32.6 + 36.7 * math.log10(d)
   ### Convert to Transmit SINR ###
   return math.sqrt(10**((- pl) / 10))

### Calculate Channel ###
# Ques: Do all IRS have equal number of reflecting elements?
h_I1_1 = generate_channel(N, 1, get_pathloss_direct(D_I1_1))
h_11_I1 = generate_channel(1, N, get_pathloss_direct(D_11_I1))
h_12_I1 = generate_channel(1, N, get_pathloss_direct(D_12_I1))
h_12_1 = generate_channel(1, 1, get_pathloss_direct(D_12_1))
h_11_1 = generate_channel(1, 1, get_pathloss_direct(D_11_1))

h_I2_2 = generate_channel(N, 1, get_pathloss_direct(D_I2_2))
h_21_I2 = generate_channel(1, N, get_pathloss_direct(D_21_I2))
h_22_I2 = generate_channel(1, N, get_pathloss_direct(D_22_I2))
h_21_2 = generate_channel(1, 1, get_pathloss_direct(D_21_2))
h_22_2 = generate_channel(1, 1, get_pathloss_direct(D_22_2))

h_IB_B = generate_channel(N, 1, get_pathloss_direct(D_IB_B))
h_2_IB = generate_channel(1, N, get_pathloss_direct(D_2_IB))
h_1_IB = generate_channel(1, N, get_pathloss_direct(D_1_IB))
h_2_B = generate_channel(1, 1, get_pathloss_direct(D_2_B))
h_1_B = generate_channel(1, 1, get_pathloss_direct(D_1_B))

### Interference link path loss

### Relay 1 ###
h_IB_1 = generate_channel(N, 1, get_pathloss_direct(D_1_IB)) ### Channel is reciprocal ###
# ### Relay 2 ###
h_IB_2 = generate_channel(N, 1, get_pathloss_direct(D_2_IB))

### Interference  from IRS 1 (I1) to Relay 2 (R2)
h_I1_2 = generate_channel(N, 1, get_pathloss_direct(D_2_I1))
### Interference  from IRS 2 (I2) to Relay 1 (R1)
h_I2_1 = generate_channel(N, 1, get_pathloss_direct(D_1_I2))


## Duplexing interference ###
h_1_1 =  generate_channel(1, 1, 1)
h_2_2 =  generate_channel(1, 1, 1)

sigma_1 = sigma_2 = sigma_b = math.sqrt(10**((- 120) / 10))

lower_bound = ([0]*3*N + [0]*3*N + [0]*4 + [0]*4)
upper_bound = [1]*3*N + [2*math.pi]*3*N + [0.5]*4 + [1]*4


def objective_function(par):
    obj_func_vals = init(par)
    return obj_func_vals



def init(par):
    obj_func_vals = []
    parameters = copy.deepcopy(par)
    row_length = parameters.shape[0]
    for i in range(row_length):
        par = parameters[i]
        phi_I1_value = [math.sqrt(zeta) * math.exp(theta) for zeta, theta in zip(par[0:N-1], par[3*N: 3*N+N-1])]
        ### Generate a complex matrix ###
        phi_I1 = np.zeros((N, N), dtype=complex)
        np.fill_diagonal(phi_I1, phi_I1_value)
        ####################### Reflection co-efficient matrix for IRS_2 (I2) #########################
        ### Only one phase coefficient matrix is assumed as it is NOT STAR IRS ###
        phi_I2_value = [math.sqrt(zeta) * math.exp(theta) for zeta, theta in zip(par[N:2*N-1], par[3*N+N: 3*N+2*N-1])]
        ### Generate a complex matrix ###
        phi_I2 = np.zeros((N, N), dtype=complex)
        np.fill_diagonal(phi_I2, phi_I2_value)
        ####################### Reflection co-efficient matrix for IRS_B (IB) #########################
        ### Only one phase coefficient matrix is assumed as it is NOT STAR IRS ###
        phi_IB_value = [math.sqrt(zeta) * math.exp(theta) for zeta, theta in zip(par[2*N:3*N-1], par[3*N+2*N : 3*N+3*N-1])]
        ### Generate a complex matrix ###
        phi_IB = np.zeros((N, N), dtype=complex)
        np.fill_diagonal(phi_IB, phi_IB_value)
        ### Calculate Channel Gain ###
        H_1_B = gamma * h_1_B + beta * mu * np.dot(np.dot(np.conjugate(h_IB_B).transpose(), phi_IB), h_1_IB.transpose())
        H_2_B = gamma * h_2_B + beta * mu * np.dot(np.dot(np.conjugate(h_IB_B).transpose(), phi_IB), h_2_IB.transpose())
        H_11_1 = gamma * h_11_1 + alpha * mu * np.dot(np.dot(np.conjugate(h_I1_1).transpose(), phi_I1), h_11_I1.transpose())
        H_12_1 = gamma * h_12_1 + alpha * mu * np.dot(np.dot(np.conjugate(h_I1_1).transpose(), phi_I1), h_12_I1.transpose())
        H_21_2 = gamma * h_21_2 + alpha * mu * np.dot(np.dot(np.conjugate(h_I2_2).transpose(), phi_I2), h_21_I2.transpose())
        H_22_2 = gamma * h_22_2 + alpha * mu * np.dot(np.dot(np.conjugate(h_I2_2).transpose(), phi_I2), h_22_I2.transpose())
        ### Interference Link Channel Gain ###
        H_2_1 = np.dot(np.dot(np.conjugate(h_IB_1).transpose(), phi_IB), h_2_IB.transpose())
        H_1_2 = np.dot(np.dot(np.conjugate(h_IB_2).transpose(), phi_IB), h_1_IB.transpose())
        H_22_1 = np.dot(np.dot(np.conjugate(h_I2_1).transpose(), phi_I2), h_22_I2.transpose())
        H_21_1 = np.dot(np.dot(np.conjugate(h_I2_1).transpose(), phi_I2), h_21_I2.transpose())
        H_12_2 = np.dot(np.dot(np.conjugate(h_I1_2).transpose(), phi_I1), h_12_I1.transpose())
        H_11_2 = np.dot(np.dot(np.conjugate(h_I1_2).transpose(), phi_I1), h_11_I1.transpose())
        ### Calculate SINR Values ###

        P_11=par[6*N]
        P_12=par[6*N+1]
        P_21=par[6*N+2]
        P_22=par[6*N+3]
        P_dash_11=par[6*N+4]
        P_dash_12=par[6*N+5]
        P_dash_21=par[6*N+6]
        P_dash_22=par[6*N+7]
        F_1 = (P_21 * np.abs(H_22_1.item()) ** 2) + (
                (P_dash_21 + P_dash_22) * np.abs(H_2_1.item()) ** 2 + np.abs(h_1_1.item()) ** 2 * rho ** 2 * (
                P_dash_11 + P_dash_12)) + P_22 * np.abs(H_22_1) ** 2
        # SINR at R1 to detect symbol of D11 considering D12 as interference assuming H11_1 > H12_1
        SINR_R1_D11 = (P_11 * np.abs(H_11_1.item()) ** 2) / (P_12 * np.abs(H_12_1.item()) ** 2 + F_1 + sigma_1 ** 2)
        # SINR at R1 to detect symbol of D12 considering D11 as interference assuming D11 has already been decoded
        SINR_R1_D12 = (P_12 * np.abs(H_12_1.item()) ** 2) / (F_1 + sigma_1 ** 2)
        # SINR at R2
        F_2 = P_11 * np.abs(H_11_2.item()) ** 2 + P_12 * np.abs(H_12_2.item()) ** 2 + (P_dash_11 + P_dash_12) * np.abs(
            H_1_2.item()) ** 2 + np.abs(h_2_2.item()) ** 2 * rho ** 2 * (P_dash_21 + P_dash_12) + P_dash_22
        SINR_R2_D21 = P_21 * np.abs(H_21_2.item()) ** 2 / (P_22 * np.abs(H_22_2.item()) ** 2 + F_2 + sigma_2 ** 2)
        SINR_R2_D22 = P_22 * np.abs(H_22_2.item()) ** 2 / (F_2 + sigma_2 ** 2)
        # SINR at Base Station B
        SINR_B_D11 = P_dash_11 * np.abs(H_1_B.item()) ** 2 / (
                P_dash_12 * np.abs(H_1_B.item()) ** 2 + (P_dash_21 + P_dash_22) * np.abs(
            H_2_B.item()) ** 2 + sigma_b ** 2)
        SINR_B_D12 = P_dash_12 * np.abs(H_1_B.item()) ** 2 / (
                (P_dash_21 + P_dash_22) * np.abs(H_2_B.item()) ** 2 + sigma_b ** 2)
        #### Debug ###
        SINR_B_D21 = P_dash_21 * np.abs(H_2_B.item()) ** 2 / ((P_dash_22) * np.abs(H_2_B.item()) ** 2 + sigma_b ** 2)
        SINR_B_D22 = P_dash_22 * np.abs(H_2_B.item()) ** 2 / sigma_b ** 2
        ### Get SINR ###
        ### Rate of D11 ###
        rate_D11 = math.log2(1 + find_min(SINR_R1_D11, SINR_B_D11))
        ### Rate of D12 ###
        rate_D12 = math.log2(1 + find_min(SINR_R1_D12, SINR_B_D12))
        ### Rate of D21 ###
        rate_D21 = math.log2(1 + find_min(SINR_R2_D21, SINR_B_D21))
        ### Rate of D22 ###
        rate_D22 = math.log2(1 + find_min(SINR_R2_D22, SINR_B_D22))
    # return rate_D11 + rate_D11 + rate_D21 + rate_D22
        obj_func_vals.append(-(rate_D11 + rate_D11 + rate_D21 + rate_D22))
    return obj_func_vals





bounds = (lower_bound, upper_bound)

# %% Do PSO
from sko.PSO import PSO

# pso = PSO(func=objective_function, n_dim=6*N+8, pop=50, max_iter=150, lb=lower_bound, ub=upper_bound, w=0.7, c1=1.4, c2=1.4, verbose=True)
# pso.run()
# pso.precision = 4
# pso.record_mode = True
# print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y_hist)
#
# # %% Plot the result
# import matplotlib.pyplot as plt
#
# plt.plot(pso.gbest_y_hist)
# plt.show()

# # Initialize swarm
options = {'c1': 1.4, 'c2': 1.4, 'w':0.7}

# Call instance of PSO with bounds argument
optimizer = ps.single.GlobalBestPSO(n_particles=500, dimensions=6*N+8, options=options, bounds=bounds)

# Perform optimization
cost, pos = optimizer.optimize(objective_function, iters=1000)

# optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=56, options=options, bounds=bounds)