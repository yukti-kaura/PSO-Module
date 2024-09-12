import copy

import numpy as np
import math
import random
import matplotlib.pyplot as plt
import time
plt.style.use('fivethirtyeight')

### Distance (in m) ###
D_11_I1 = 10
D_12_I1 = 13
D_21_I2 = 15
D_22_I2 = 18



D_1_IB = 5
D_2_IB = 9

D_I1_1 = 4
D_I2_1 = 8

D_I2_2 = 6
D_I1_2 = 10

D_IB_B = 10






D_12_1 = 40
D_11_1 = 30
D_22_2 = 25
D_21_2 = 31
D_2_B = 19
D_1_B = 15

M = 1  # No. of antennas on BS
N = N1 = N2 = NB  = 16  # No. IRS elements
K = 1  # No.of transmit antennas at the relay device
K_dash = 1 # No. of transmit antennas at the redcap (IoT) device
q = 1 # No. of Quantization bits. 1,2,..., Q
m = 1 # m=1,2,..., 2^n-1
rho = 0.1

n_particles = 200

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

channels=[]

def find_min(a, b):
  if(a > b):
    return b
  return a


# def generate_channel(N, K, path_loss, M=1):
#     h = (1/np.sqrt(2)) * (np.random.randn(N, K) + 1j * np.random.randn(N, K))
#     h_normalized = h / np.linalg.norm(h, axis=0)
#     h = path_loss * h_normalized
#     return h

#M is the number of transmitter's antennas
#N is the number of receiver' antennas
#dist is the distance between the transmitter and the receiver
#pl is path-loss exponent
#Kdb is the Rician factor in dB

def Rician_Fading_Channels(M,N,dist,pl, Kdb):
    K = 10**(Kdb/10) #dB to mW
    mu = np.sqrt( K/((K+1))) #direct path
    s = np.sqrt( 1/(2*(K+1))) #scattered paths
    Hw = mu + s*(np.random.randn(M,N)+1j*np.random.randn(M,N)) #Rician channel
    H = np.sqrt(1/(dist**pl))*Hw  #Rician channel with pathloss
    return H


def generate_channel(trans, receive, path_loss, M=2):
    msr = M
    var_sr = path_loss
    x1 = np.random.gamma(msr,(var_sr)/msr,size=(trans,receive))
    y1 = 2 * math.pi * np.random.randn(trans, receive)
    zsr = x1**(1j * y1)
    h_sr = np.sqrt(zsr)
    return h_sr


def get_pathloss(d, los=True, type="DH"):
   # InF - SL Indoor Factory  with Sparse clutter and Low base station height (both Tx and Rx are below the average height of the clutter)
   # InF - DL Indoor Factory with Dense clutter and Low base station height (both Tx and Rx are below the average height of the clutter)
   # InF - SH Indoor Factory with Sparse clutter and High base station height (Tx or Rx elevated above the clutter)
   # InF - DH Indoor Factory with Dense clutter and High base station height (Tx or Rx elevated above the clutter)
   # InF - HH Indoor Factory with High Tx and High Rx (both elevated above the clutter)
   ### 3GPP 38.901
   # PL_LOS=31.84+21.50 log_10(d_3D )+19.00 log_10(f_c )
   # InF-DH: PL=33.63+21.9 log_10(d_3D )+20 log_10(f_c )
   # PL_NLOS=max(PL,PL_LOS)
   f = 3.5*math.e**9 ###  carrier frequency in Hz
   pl_LOS = 31.84 + 21.50 * math.log10(d) + 19.00 * math.log10(f)
   pl_LOS = math.sqrt(10 ** ((- pl_LOS) / 10))
   # return pl_LOS
   if(los):
    pl_LOS = math.sqrt(10 ** ((- pl_LOS) / 10))
    return pl_LOS
   else:
    PL_INFSL = 33 + 25.5 * math.log10(d) + 20 * math.log10(f)
    PL_INFDL = 18.6 + 35.7 * math.log10(d) + 20 * math.log10(f)
    PL_INFDH = 33.63 + 21.9 * math.log10(d) + 20 * math.log(f)
    PL_INFSH = 32.4 + 23.0 * math.log10(d) + 20 * math.log10(f)
    if(type == "DH"):
        pl_NLOS = max(PL_INFDH, pl_LOS)
    elif(type == "DL"):
        pl_NLOS = max(PL_INFDL, pl_LOS, PL_INFSL)
    elif(type == "SH"):
        pl_NLOS = max(PL_INFSH, pl_LOS)
    pl_NLOS = math.sqrt(10 ** ((- pl_NLOS) / 10))
    return pl_NLOS
   # if(los):
   #     pl = 0
   #     if(d != 0):
   #      pl = 32.6 + 36.7 * math.log10(d)
   #     ### Convert to Transmit SINR ###
   #      return math.sqrt(10**((- pl) / 10))
   # InF - SL: PL = 33 + 25.5log_10⁡(d_3D) + 20log_10⁡(f_c)
   # 〖PL〗_NLOS = max⁡(PL,〖PL〗_LOS)
   #
   # InF - DL: PL = 18.6 + 35.7
   # log_10⁡(d_3D) + 20   log_10⁡(f_c)
   # 〖PL〗_NLOS = max⁡(PL,〖PL〗_LOS, 〖PL〗_(InF-SL))
   #
   # InF - SH
   # PL = 32.4 + 23.0log_10⁡(d_3D) + 20log_10⁡(f_c)
   # 〖PL〗_NLOS = max⁡(PL,〖PL〗_LOS)


def init_channel():
    ### Calculate Channel ###
    # Ques: Do all IRS have equal number of reflecting elements?
    h_I1_1 = generate_channel(N, 1, get_pathloss(D_I1_1))
    channels.append(h_I1_1)
    h_11_I1 = generate_channel(1, N, get_pathloss(D_11_I1))
    channels.append(h_11_I1)
    h_12_I1 = generate_channel(1, N, get_pathloss(D_12_I1))
    channels.append(h_12_I1)
    h_12_1 = generate_channel(1, 1, get_pathloss(D_12_1, False, type="DL"), 1)
    channels.append(h_12_1)
    h_11_1 = generate_channel(1, 1, get_pathloss(D_11_1, False, type="DL"), 1)
    channels.append(h_11_1)

    h_I2_2 = generate_channel(N, 1, get_pathloss(D_I2_2))
    channels.append(h_I2_2)
    h_21_I2 = generate_channel(1, N, get_pathloss(D_21_I2))
    channels.append(h_21_I2)
    h_22_I2 = generate_channel(1, N, get_pathloss(D_22_I2))
    channels.append(h_22_I2)
    h_21_2 = generate_channel(1, 1, get_pathloss(D_21_2, False, type="DL"), 1)
    channels.append(h_21_2)
    h_22_2 = generate_channel(1, 1, get_pathloss(D_22_2, False, type="DL"), 1)
    channels.append(h_22_2)

    h_IB_B = generate_channel(N, 1, get_pathloss(D_IB_B))
    channels.append(h_IB_B)
    h_2_IB = generate_channel(1, N, get_pathloss(D_2_IB))
    channels.append(h_2_IB)
    h_1_IB = generate_channel(1, N, get_pathloss(D_1_IB))
    channels.append(h_1_IB)
    h_2_B = generate_channel(1, 1, get_pathloss(D_2_B, False, type="SH"), 1)
    channels.append(h_2_B)
    h_1_B = generate_channel(1, 1, get_pathloss(D_1_B, False, type="SH"), 1)
    channels.append(h_1_B)

    ### Interference link path loss

    ### Relay 1 ###
    h_IB_1 = generate_channel(N, 1, get_pathloss(D_1_IB)) ### Channel is reciprocal ###
    channels.append(h_IB_1)
    # ### Relay 2 ###
    h_IB_2 = generate_channel(N, 1, get_pathloss(D_2_IB))
    channels.append(h_IB_2)

    ### Interference  from IRS 1 (I1) to Relay 2 (R2)
    h_I1_2 = generate_channel(N, 1, get_pathloss(D_I1_2))
    channels.append(h_I1_2)
    ### Interference  from IRS 2 (I2) to Relay 1 (R1)
    h_I2_1 = generate_channel(N, 1, get_pathloss(D_I2_1))
    channels.append(h_I2_1)

    ## Duplexing interference ###
    h_1_1 =  generate_channel(1, 1, 1)
    channels.append(h_1_1)
    h_2_2 =  generate_channel(1, 1, 1)
    channels.append(h_2_2)
    np.savez('channels', channels)
    return channels

sigma_1 = sigma_2 = sigma_b = math.sqrt(10**((- 120) / 10))

lower_bound = ([0]*3*N + [0]*3*N + [0]*4 + [0]*4)
upper_bound = [1]*3*N + [2*math.pi]*3*N + [0.5]*4 + [1]*4

### Rate Thresholds ###
R_11_th = R_12_th = R_21_th = R_22_th = 0.05

### Power Thresholds ###
P_11_max = P_12_max = P_21_max = P_22_max = 0.5
P_dash_11_max = P_dash_12_max = P_dash_21_max = P_dash_22_max = 1

death_penalty = False

def objective_function(par):

    H_11_1, H_12_1, H_1_B, H_21_2, H_22_2, H_2_B, P_11, P_12, P_21, P_22, P_dash_11, P_dash_12, P_dash_21, P_dash_22, rate_D11, rate_D12, rate_D21, rate_D22 = evaluate_objective(
        par)
    penalty = 0.0
    penalty = evaluate_conditions(H_11_1, H_12_1, H_1_B, H_21_2, H_22_2, H_2_B, P_11, P_12, P_21, P_22, P_dash_11, P_dash_12,
                        P_dash_21, P_dash_22, rate_D11, rate_D12, rate_D21, rate_D22)

    rate = (rate_D11 + rate_D12 + rate_D21 + rate_D22)
    if(death_penalty):
        return rate
    else:
        return rate - penalty


def evaluate_objective(par):
    phi_I1_value = [math.sqrt(zeta) * math.exp(theta) for zeta, theta in zip(par[0:N - 1], par[3 * N: 3 * N + N - 1])]
    ### Generate a complex matrix ###
    phi_I1 = np.zeros((N, N), dtype=complex)
    np.fill_diagonal(phi_I1, phi_I1_value)
    ####################### Reflection co-efficient matrix for IRS_2 (I2) #########################
    ### Only one phase coeffi  cient matrix is assumed as it is NOT STAR IRS ###
    phi_I2_value = [math.sqrt(zeta) * math.exp(theta) for zeta, theta in
                    zip(par[N:2 * N - 1], par[3 * N + N: 3 * N + 2 * N - 1])]
    ### Generate a complex matrix ###
    phi_I2 = np.zeros((N, N), dtype=complex)
    np.fill_diagonal(phi_I2, phi_I2_value)
    ####################### Reflection co-efficient matrix for IRS_B (IB) #########################
    ### Only one phase coefficient matrix is assumed as it is NOT STAR IRS ###
    phi_IB_value = [math.sqrt(zeta) * math.exp(theta) for zeta, theta in
                    zip(par[2 * N:3 * N - 1], par[3 * N + 2 * N: 3 * N + 3 * N - 1])]
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
    P_11 = par[6 * N]
    P_12 = par[6 * N + 1]
    P_21 = par[6 * N + 2]
    P_22 = par[6 * N + 3]
    P_dash_11 = par[6 * N + 4]
    P_dash_12 = par[6 * N + 5]
    P_dash_21 = par[6 * N + 6]
    P_dash_22 = par[6 * N + 7]
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
    return H_11_1, H_12_1, H_1_B, H_21_2, H_22_2, H_2_B, P_11, P_12, P_21, P_22, P_dash_11, P_dash_12, P_dash_21, P_dash_22, rate_D11, rate_D12, rate_D21, rate_D22


def evaluate_conditions(H_11_1, H_12_1, H_1_B, H_21_2, H_22_2, H_2_B, P_11, P_12, P_21, P_22, P_dash_11, P_dash_12,
                        P_dash_21, P_dash_22, rate_D11, rate_D12, rate_D21, rate_D22):
    penalty = 0.0
    # Recalculate rates based on these parameters
    condition_1 = (rate_D11 >= R_11_th)
    condition_2 = (rate_D12 >= R_12_th)
    condition_3 = (rate_D21 >= R_21_th)
    condition_4 = (rate_D22 >= R_22_th)
    ### Test absolute value of channels ###
    condition_5 = np.abs(H_1_B).item() > np.abs(H_2_B).item()
    condition_6 = np.abs(H_11_1).item() > np.abs(H_12_1).item()
    condition_7 = np.abs(H_21_2).item() > np.abs(H_22_2).item()
    condition_8 = P_11 <= P_11_max
    condition_9 = P_22 <= P_22_max
    condition_10 = P_21 <= P_21_max
    condition_11 = P_12 <= P_12_max
    condition_12 = P_dash_11 <= P_dash_11_max
    condition_13 = P_dash_22 <= P_dash_22_max
    condition_14 = P_dash_21 <= P_dash_21_max
    condition_15 = P_dash_12 <= P_dash_12_max
    if not condition_1:
        penalty = penalty + 0 if rate_D11 - R_11_th >= 0 else abs(rate_D11 - R_11_th)
    if not condition_2:
        penalty = penalty + 0 if rate_D12 - R_12_th >= 0 else abs(rate_D12 - R_12_th)
    if not condition_3:
        penalty = penalty + 0 if rate_D21 - R_21_th >= 0 else abs(rate_D21 - R_21_th)
    if not condition_4:
        penalty = penalty + 0 if rate_D22 - R_22_th >= 0 else abs(rate_D22 - R_22_th)
    if not condition_5:
        penalty = penalty + 0 if np.abs(H_1_B).item() - np.abs(H_2_B).item() >= 0 else abs(
            np.abs(H_1_B).item() - np.abs(H_2_B).item())
    if not condition_6:
        penalty = penalty + 0 if np.abs(H_11_1).item() - np.abs(H_12_1).item() >= 0 else abs(
            np.abs(H_11_1).item() - np.abs(H_12_1).item())
    if not condition_7:
        penalty = penalty + 0 if np.abs(H_21_2).item() - np.abs(H_22_2).item() >= 0 else abs(
            np.abs(H_21_2).item() - np.abs(H_22_2).item())
    if not condition_8:
        penalty = penalty + 0 if P_11 - P_11_max >= 0 else abs(P_11 - P_11_max)
    if not condition_9:
        penalty = penalty + 0 if P_12 - P_12_max >= 0 else abs(P_12 - P_12_max)
    if not condition_10:
        penalty = penalty + 0 if P_21 - P_21_max >= 0 else abs(P_21 - P_21_max)
    if not condition_11:
        penalty = penalty + 0 if P_22 - P_22_max >= 0 else abs(P_22 - P_22_max)
    if not condition_12:
        penalty = penalty + 0 if P_dash_11 - P_dash_11_max >= 0 else abs(P_dash_11 - P_dash_11_max)
    if not condition_13:
        penalty = penalty + 0 if P_dash_12 - P_dash_12_max >= 0 else abs(P_dash_12 - P_dash_12_max)
    if not condition_14:
        penalty = penalty + 0 if P_dash_21 - P_dash_21_max <= 0 else abs(P_dash_21 - P_dash_21_max)
    if not condition_15:
        penalty = penalty + 0 if P_dash_22 - P_dash_22_max <= 0 else abs(P_dash_22 - P_dash_22_max)
    return penalty


bounds = [(0,1)]*(3*N) + [(0,2*math.pi)]*(3*N) +  [(0,0.5)]*4 + [(0,1)]*4
n_dimension = 6*N+8

w1 = 0.9
w2 = 0.4
w = 0.7
c1 = 1.4
c2 = 1.4
m_iterations = 250
n_particles = 100
mc_count = 20
r1 = []
r2 = []


def init_particles():
    particles = []
    velocities = []
    pbest = []
    for _ in range(n_particles):
        particle = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(n_dimension)]
        particles.append(particle)
        velocities.append([0] * n_dimension)
        pbest.append(particle[:])
        np.savez(f'particles_{m_iterations}_{mc_count}', particles)
        np.savez(f'velocities_{m_iterations}_{mc_count}', velocities)
        np.savez(f'pbest_{m_iterations}_{mc_count}', velocities)
    r1 = [np.random.rand(m_iterations, n_particles, n_dimension) for _ in range(mc_count)]
    np.savez('r1', r1)
    r2 = [np.random.rand(m_iterations, n_particles, n_dimension) for _ in range(mc_count)]
    np.savez('r2', r2)
    return particles, velocities, pbest


use_random = True

def penalty_pso_function(particles, pbest, velocities, r1, r2):
    particles, velocities, pbest = copy.deepcopy(particles), copy.deepcopy(velocities), copy.deepcopy(pbest)
    gbest_value = -float('inf')
    iteration_best_values = []
    for iter in range(m_iterations):
        iteration_best_value = -float('inf')
        for i in range(n_particles):
            current_position = particles[i]
            # temp = unflatten(current_position, parameter_values)
            fitness = objective_function(current_position)

            # Update personal best
            # temp = unflatten(pbest[i], parameter_values)
            if fitness > objective_function(pbest[i]):
                pbest[i] = current_position[:]

            # Update global best
            if fitness > gbest_value:
                gbest = current_position[:]
                gbest_value = fitness

            # Update iteration best value
            if fitness > iteration_best_value:
                iteration_best_value = fitness

            # Record the best value found in` this iteration
        iteration_best_values.append(iteration_best_value)


        # Update velocities and particles
        for i in range(n_particles):
            for j in range(n_dimension):
                w = (w1 - w2) * ((m_iterations - (iter + 1)) / (m_iterations)) + w2
                if(use_random):
                    new_velocity = (w * velocities[i][j] +
                                        c1  * r1[iter][i][j]*(pbest[i][j] - particles[i][j]) +
                                        c2  * r2[iter][i][j]*(gbest[j] - particles[i][j]))
                else:
                    new_velocity = (w * velocities[i][j] +
                                    c1 * random.random() * (pbest[i][j] - particles[i][j]) +
                                    c2 * random.random() * (gbest[j] - particles[i][j]))
                # new_velocity = 4 if new_velocity >= 4 else new_velocity

                old_position = particles[i][j]
                old_velocity = velocities[i][j]
                new_position = particles[i][j] + new_velocity
                new_position = max(min(new_position, bounds[j][1]), bounds[j][0])
                particles[i][j] = new_position
                velocities[i][j] = new_velocity
                if(death_penalty):
                    H_11_1, H_12_1, H_1_B, H_21_2, H_22_2, H_2_B, P_11, P_12, P_21, P_22, P_dash_11, P_dash_12, P_dash_21, P_dash_22, rate_D11, rate_D12, rate_D21, rate_D22 =     evaluate_objective(particles[i])
                    if (evaluate_conditions(H_11_1, H_12_1, H_1_B, H_21_2, H_22_2, H_2_B, P_11, P_12, P_21, P_22, P_dash_11, P_dash_12, P_dash_21, P_dash_22, rate_D11, rate_D12, rate_D21, rate_D22) != 0.0):
                        particles[i][j] = old_position
                        velocities[i][j] = old_velocity

        #Printing
        # if (iter % 100 == 0) or iter == (m_iterations-1):
                # print(f"Iteration {iter}: Value = {iteration_best_values[iter]}")

    return iteration_best_values, iteration_best_values[m_iterations-1]

# particles, velocities, pbest = init_particles()
# init_channel()

# r1 = [np.random.rand(m_iterations, n_particles, n_dimension) for _ in range(mc_count)]
# r2 = [np.random.rand(m_iterations, n_particles, n_dimension) for _ in range(mc_count)]
# #
channels = np.load('channels.npz',allow_pickle=True)
channels = channels[channels.files[0]].tolist()
velocities = np.load('FinalArrays/velocities.npz')
velocities = velocities[velocities.files[0]].tolist()
particles = np.load('FinalArrays/particles.npz')
particles  = particles [particles.files[0]].tolist()
pbest = np.load('FinalArrays/pbest.npz')
pbest = pbest [pbest.files[0]].tolist()
r1 = np.load('r1.npz')
r1 = r1 [r1.files[0]].tolist()
r2 = np.load('r2.npz')
r2 = r2 [r2.files[0]].tolist()
h_I1_1, h_11_I1, h_12_I1, h_12_1, h_11_1, h_I2_2, h_21_I2, h_22_I2, h_21_2, h_22_2, h_IB_B, h_2_IB, h_1_IB, h_2_B, h_1_B, h_IB_1, h_IB_2, h_I1_2, h_I2_1, h_1_1, h_2_2 = channels

pso_convergence_list=[]
x =  range(1,m_iterations+1)
def calc_sum_rate(type):
    print(f"P_max:{P_12_max}")
    bval_list = []
    best_val_iter_list = []
    for i in range(mc_count):
        best_val_iter, best_val = penalty_pso_function(particles, velocities, pbest, r1[i], r2[i])
        bval_list.append(best_val)
        best_val_iter_list.append(best_val_iter)
    pso_convergence = np.mean(best_val_iter_list, axis=0, dtype=np.float64)
    pso_convergence_list.append(pso_convergence)
    plt.plot(x, pso_convergence, marker="o", markevery=5,label=f'{type}')
    print('Mean Sum Rate:', np.mean(bval_list))

print("====Full IRS========")
gamma = 1
mu = 1
alpha = 1
beta = 1
type="IRS between Device-Relay and Relay-BS"

# # R_11_th = R_12_th = R_21_th = R_22_th = 2
# # P_dash_11_max = P_dash_12_max = P_dash_21_max = P_dash_22_max = 2x`
# P_11_max = P_12_max = P_21_max = P_22_max = 2
# calc_sum_rate()
#
# R_11_th = R_12_th = R_21_th = R_22_th = 1
# # P_dash_11_max = P_dash_12_max = P_dash_21_max = P_dash_22_max = 1
# P_11_max = P_12_max = P_21_max = P_22_max = 1
# calc_sum_rate()

# P_dash_11_max = P_dash_12_max = P_dash_21_max = P_dash_22_max = 0.5
# R_11_th = R_12_th = R_21_th = R_22_th = 0.5
# P_11_max = P_12_max = P_21_max = P_22_max = 0.5
# calc_sum_rate()
#
# # P_dash_11_max = P_dash_12_max = P_dash_21_max = P_dash_22_max = 0.25
# # R_11_th = R_12_th = R_21_th = R_22_th = 0.25
# P_11_max = P_12_max = P_21_max = P_22_max = 0.25
# calc_sum_rate()
#
# # P_dash_11_max = P_dash_12_max = P_dash_21_max = P_dash_22_max = 0.1
# # R_11_th = R_12_th = R_21_th = R_22_th = 0.1
# P_11_max = P_12_max = P_21_max = P_22_max = 0.1
# calc_sum_rate()
#
# # P_dash_11_max = P_dash_12_max = P_dash_21_max = P_dash_22_max = 0.05
# # R_11_th = R_12_th = R_21_th = R_22_th = 0.05
# P_11_max = P_12_max = P_21_max = P_22_max = 0.05
death_penalty = True
calc_sum_rate("PSO with Death Penalty")

death_penalty = False
calc_sum_rate("PSO with Dynamic Penalty")




# print("==========No IRS between Device and Relay============")
# gamma = 1
# mu = 1
# alpha = 0
# beta = 1
# type="IRS between Relay-BS"
# #
# # # # R_11_th = R_12_th = R_21_th = R_22_th = 2
# # # # P_dash_11_max = P_dash_12_max = P_dash_21_max = P_dash_22_max = 2
# # # P_11_max = P_12_max = P_21_max = P_22_max = 2
# # # calc_sum_rate()
# # #
# # # # P_dash_11_max = P_dash_12_max = P_dash_21_max = P_dash_22_max = 1
# # # # R_11_th = R_12_th = R_21_th = R_22_th = 1
# # # P_11_max = P_12_max = P_21_max = P_22_max = 1
# # # calc_sum_rate()
# #
# # # P_dash_11_max = P_dash_12_max = P_dash_21_max = P_dash_22_max = 0.5
# # # R_11_th = R_12_th = R_21_th = R_22_th = 0.5
# # P_11_max = P_12_max = P_21_max = P_22_max = 0.5
# # calc_sum_rate()
# #
# # # P_dash_11_max = P_dash_12_max = P_dash_21_max = P_dash_22_max = 0.25
# # # R_11_th = R_12_th = R_21_th = R_22_th = 0.25
# # P_11_max = P_12_max = P_21_max = P_22_max = 0.25
# # calc_sum_rate()
# #
# # # P_dash_11_max = P_dash_12_max = P_dash_21_max = P_dash_22_max = 0.1
# # # R_11_th = R_12_th = R_21_th = R_22_th = 0.1
# # P_11_max = P_12_max = P_21_max = P_22_max = 0.1
# # calc_sum_rate()
# #
# # # P_dash_11_max = P_dash_12_max = P_dash_21_max = P_dash_22_max = 0.05
# # # R_11_th = R_12_th = R_21_th = R_22_th = 0.05
# # P_11_max = P_12_max = P_21_max = P_22_max = 0.05
# calc_sum_rate(type)
#
# print("========== No IRS between Relay and BS ===========")
# gamma = 1
# mu = 1
# alpha = 1
# beta = 0
# type="IRS between Device-Relay"
# # ### Rate Thresholds ###
# # # R_11_th = R_12_th = R_21_th = R_22_th = 2
# # # P_dash_11_max = P_dash_12_max = P_dash_21_max = P_dash_22_max = 2
# # P_11_max = P_12_max = P_21_max = P_22_max = 2
# # calc_sum_rate()
# #
# # # P_dash_11_max = P_dash_12_max = P_dash_21_max = P_dash_22_max = 1
# # # R_11_th = R_12_th = R_21_th = R_22_th = 1
# # P_11_max = P_12_max = P_21_max = P_22_max = 1
# # calc_sum_rate()
#
# # P_dash_11_max = P_dash_12_max = P_dash_21_max = P_dash_22_max = 0.5
# # R_11_th = R_12_th = R_21_th = R_22_th = 0.5
# # P_11_max = P_12_max = P_21_max = P_22_max = 0.5
# calc_sum_rate(type)
# #
# # # P_dash_11_max = P_dash_12_max = P_dash_21_max = P_dash_22_max = 0.25
# # # R_11_th = R_12_th = R_21_th = R_22_th = 0.25
# # P_11_max = P_12_max = P_21_max = P_22_max = 0.25
# # calc_sum_rate()
# #
# # # P_dash_11_max = P_dash_12_max = P_dash_21_max = P_dash_22_max = 0.1
# # # R_11_th = R_12_th = R_21_th = R_22_th = 0.1
# # P_11_max = P_12_max = P_21_max = P_22_max = 0.1
# # calc_sum_rate()
# #
# # # P_dash_11_max = P_dash_12_max = P_dash_21_max = P_dash_22_max = 0.05
# # # R_11_th = R_12_th = R_21_th = R_22_th = 0.05
# # P_11_max = P_12_max = P_21_max = P_22_max = 0.05
# # calc_sum_rate()
#
# print("========== No IRS =========")
# gamma = 1
# mu = 0
# alpha = 1
# beta = 1
# type = "No IRS"
#
# # # R_11_th = R_12_th = R_21_th = R_22_th = 2
# # # P_dash_11_max = P_dash_12_max = P_dash_21_max = P_dash_22_max = 2
# # P_11_max = P_12_max = P_21_max = P_22_max = 2
# # calc_sum_rate()
# #
# # # P_dash_11_max = P_dash_12_max = P_dash_21_max = P_dash_22_max = 1
# # # R_11_th = R_12_th = R_21_th = R_22_th = 1
# # P_11_max = P_12_max = P_21_max = P_22_max = 1
# # calc_sum_rate()
#
# # P_dash_11_max = P_dash_12_max = P_dash_21_max = P_dash_22_max = 0.5
# # R_11_th = R_12_th = R_21_th = R_22_th = 0.5
# # P_11_max = P_12_max = P_21_max = P_22_max = 0.5
# # calc_sum_rate()
# #
# # # P_dash_11_max = P_dash_12_max = P_dash_21_max = P_dash_22_max = 0.25
# # # R_11_th = R_12_th = R_21_th = R_22_th = 0.25
# # P_11_max = P_12_max = P_21_max = P_22_max = 0.25
# # calc_sum_rate()
# #
# # # P_dash_11_max = P_dash_12_max = P_dash_21_max = P_dash_22_max = 0.1
# # # R_11_th = R_12_th = R_21_th = R_22_th = 0.1
# # P_11_max = P_12_max = P_21_max = P_22_max = 0.1
# # calc_sum_rate()
# #
# # # P_dash_11_max = P_dash_12_max = P_dash_21_max = P_dash_22_max = 0.05
# # # R_11_th = R_12_th = R_21_th = R_22_th = 0.05
# # P_11_max = P_12_max = P_21_max = P_22_max = 0.05
# calc_sum_rate(type)
#
# np.savez('pso_convergence_list.npz',type=pso_convergence_list)
plt.xlabel("No. of Iterations")
plt.ylabel("Sum Rate (Mbps)")
plt.legend()


current_timestamp = time.time()
plt.savefig(f"PSO{current_timestamp}.svg")
plt.show()

plt.show()