import math
import random
import numpy as np
import random
import math
import matplotlib.pyplot as plt

import math
import random
import copy

def flatten(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list

def unflatten(flat_list, structure):
    flat_iter = iter(flat_list)
    def helper(struct):
        result = []
        for elem in struct:
            if isinstance(elem, list):
                result.append(helper(elem))
            else:
                result.append(next(flat_iter))
        return result
    return helper(structure)


def conditions(para):
    global zeta_I1, zeta_I2, zeta_IB, theta_I1, theta_I2, theta_IB, P_11, P_12, P_21, P_22, P_dash_11, P_dash_12, P_dash_21, P_dash_22
    zeta_I1, zeta_I2, zeta_IB, theta_I1, theta_I2, theta_IB, P_11, P_12, P_21, P_22 = para
    # P_dash_11, P_dash_12, P_dash_21, P_dash_22
    reinit()
    # Recalculate rates based on these parameters
    # condition_1 = (rate_D11 >= R_11_th) and (rate_D12 >= R_12_th) and (rate_D21 >= R_21_th) and (
    #             rate_D22 >= R_22_th)
    ### Test absolute value of channels ###
    condition_2 = np.abs(H_1_B).item() > np.abs(H_2_B).item()
    condition_3 = np.abs(H_11_1).item() > np.abs(H_12_1).item()
    condition_4 = np.abs(H_21_2).item() > np.abs(H_22_2).item()
    condition_5 = P_11 <= P_11_max and P_22 <= P_22_max and P_21 <= P_21_max and P_12 <= P_12_max
    condition_6 = P_dash_11 <= P_dash_11_max and P_dash_22 <= P_dash_22_max and P_21 <= P_dash_21_max and P_dash_12 <= P_dash_12_max
    # print(condition_1 , condition_2 , condition_5 , condition_6)
    if (condition_2 and condition_3 and condition_4 and condition_5 and condition_6):
        return True
    return False


particles_init = []
velocities_init = []
pbest_init = []

def init(parameter_values, bounds, n_particles):
    para = flatten(parameter_values)
    len_para = len(para)
    for _ in range(n_particles):
        particle = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(len_para)]
        temp_con = unflatten(particle, par)
        while (conditions(temp_con) == False):
            particle = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(len_para)]
            temp_con = unflatten(particle, par)
        particles_init.append(particle)
        velocities_init.append([0] * (len_para))
        pbest_init.append(particle[:])
    # Save the array to an NPZ file
    np.savez('particles_init', particles_init)
    np.savez('velocities_init', velocities_init)
    np.savez('pbest_init', pbest_init)

def penalty_pso_function(parameter_values, bounds, n_particles, m_iterations, inertia, cognitive, social, iter_no):
    global particles_init, velocities_init, pbest_init
    # print("PSO Algorithm Started", iter_no+1)

    num_particles = n_particles
    max_iterations = m_iterations
    w = inertia  # inertia weight
    c1 = cognitive  # cognitive constant
    c2 = social  # social constant

    para = flatten(parameter_values)
    len_para = len(para)
    update_bounds = flatten(bounds)


    particles = copy.deepcopy(particles_init)
    velocities = copy.deepcopy(velocities_init)
    pbest = copy.deepcopy(pbest_init)
    gbest = None
    gbest_value = -float('inf')
    iteration_best_values = []

    # PSO loop
    for iter in range(max_iterations):
        iteration_best_value = -float('inf')
        for i in range(num_particles):
            current_position = particles[i]
            temp = unflatten(current_position, parameter_values)
            fitness = objective_function(temp, iter)

            # Update personal best
            temp = unflatten(pbest[i], parameter_values)
            if fitness > objective_function(temp, iter):
                pbest[i] = current_position[:]

            # Update global best
            if fitness > gbest_value:
                gbest = current_position[:]
                gbest_value = fitness

            # Update iteration best value
            if fitness > iteration_best_value:
                iteration_best_value = fitness

        # Record the best value found in this iteration
        iteration_best_values.append(iteration_best_value)

        # Update velocities and particles
        for i in range(num_particles):
            for j in range(len_para):
                #TODO set annealing for inertia
                w = (w1 - w2) * ((m_iterations - (iter + 1)) / (m_iterations)) + w2
                #TODO set velocity constraint
                new_velocity = (w * velocities[i][j] +
                                    c1  * random.random()*(pbest[i][j] - particles[i][j]) +
                                    c2  * random.random()*(gbest[j] - particles[i][j]))
                new_velocity = 4 if new_velocity >=4 else new_velocity
                new_position = particles[i][j] + new_velocity

                new_position = max(min(new_position, update_bounds[j][1]), update_bounds[j][0])

                particles[i][j] = new_position
                velocities[i][j] = new_velocity

        #Printing
        # if (iter % 100 == 0) or iter == (max_iterations-1):
        #         print(f"Iteration {iter}: Value = {iteration_best_values[iter]}")

    return iteration_best_values

M = 1  # No. of antennas on BS
N = N1 = N2 = NB  = 16  # No. IRS elements
K = 1  # No.of transmit antennas at the relay device
K_dash = 1 # No. of transmit antennas at the redcap (IoT) device
q = 1 # No. of Quantization bits. 1,2,..., Q
m = 1 # m=1,2,..., 2^n-1
R_min = 5

####################### Reflection co-efficient matrix for IRS_1 (I1) #########################
zeta_I1 = [random.uniform(0, 1) for _ in range(N)]
theta_I1 = [random.uniform(0, 2*math.pi) for _ in range(N)]
zeta_I2 = [random.uniform(0, 1) for _ in range(N)]
theta_I2 = [random.uniform(0, 2*math.pi) for _ in range(N)]
zeta_IB = [random.uniform(0, 1) for _ in range(N)]
theta_IB = [random.uniform(0, 2*math.pi) for _ in range(N)]



bounds = [(0,1)]*(3*N) + [(0,2*math.pi)]*(3*N) +  [(0,0.5)]*4 \
         # + [(0,1)]*4
# print(len(bounds))

### Rate Thresholds ###
R_11_th = R_12_th = R_21_th = R_22_th = 0.05

### Power Thresholds ###
P_11_max = P_12_max = P_21_max = P_22_max = 0.5
P_dash_11_max = P_dash_12_max = P_dash_21_max = P_dash_22_max = 1

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


### Noise  ###
# -120db :
sigma_1 = sigma_2 = sigma_b = math.sqrt(10**((- 120) / 10))

### Power at devices D11, D12, D21, D22 and R1, R2 ###

P_11 = random.uniform(0, 0.5)
P_12 = random.uniform(0, 0.5)
P_21 = random.uniform(0, 0.5)
P_22 = random.uniform(0, 0.5)

### Check with Justin ###
P_dash_11 = random.uniform(0, 1)
P_dash_12 = random.uniform(0, 1)
P_dash_21 = random.uniform(0, 1)
P_dash_22 = random.uniform(0, 1)

H_1_B = H_2_B = H_11_1 = H_12_1 = H_21_2 = H_22_2 = 0.0
SINR_B_D22 = SINR_B_D21 = SINR_B_D12 = SINR_B_D12 = SINR_R2_D22 = SINR_R2_D21 = SINR_R1_D11 = SINR_R1_D12 = 0.0
rate_D11 =  rate_D12 = rate_D21 = rate_D22 = 0


rho = 0.1
### We have Power and IRS Phase shifts to optimize ###
par =  zeta_I1, zeta_I2,  zeta_IB, theta_I1,theta_I2, theta_IB, P_11, P_12, P_21,P_22

       # P_dash_11, P_dash_12, P_dash_21, P_dash_22


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

penalty_factor = 0.9



def reinit():
    global rate_D21, rate_D11, rate_D22, rate_D12
    global zeta_I1, zeta_I2,  zeta_IB, theta_I1,theta_I2, theta_IB, P_11, P_12, P_21,P_22, P_dash_11, P_dash_12, P_dash_21, P_dash_22
    global H_1_B , H_2_B , H_11_1 , H_12_1 , H_21_2 , H_22_2
    global SINR_B_D22 , SINR_B_D21 , SINR_B_D12 , SINR_B_D12 , SINR_R2_D22 , SINR_R2_D21 , SINR_R1_D11 , SINR_R1_D12
    phi_I1_value = [math.sqrt(zeta) * math.exp(theta) for zeta, theta in zip(zeta_I1, theta_I1)]

    ### Generate a complex matrix ###
    phi_I1 = np.zeros((N, N), dtype=complex)
    np.fill_diagonal(phi_I1, phi_I1_value)

    ####################### Reflection co-efficient matrix for IRS_2 (I2) #########################

    ### Only one phase coefficient matrix is assumed as it is NOT STAR IRS ###
    phi_I2_value = [math.sqrt(zeta) * math.exp(theta) for zeta, theta in zip(zeta_I2, theta_I2)]

    ### Generate a complex matrix ###
    phi_I2 = np.zeros((N, N), dtype=complex)
    np.fill_diagonal(phi_I2, phi_I2_value)

    ####################### Reflection co-efficient matrix for IRS_B (IB) #########################

    ### Only one phase coefficient matrix is assumed as it is NOT STAR IRS ###
    phi_IB_value = [math.sqrt(zeta) * math.exp(theta) for zeta, theta in zip(zeta_IB, theta_IB)]

    ### Generate a complex matrix ###
    phi_IB = np.zeros((N, N), dtype=complex)
    np.fill_diagonal(phi_IB, phi_IB_value)

    ### Calculate Channel Gain ###
    H_1_B = gamma*h_1_B + beta*mu*np.dot(np.dot(np.conjugate(h_IB_B).transpose(), phi_IB), h_1_IB.transpose())
    H_2_B = gamma*h_2_B + beta*mu*np.dot(np.dot(np.conjugate(h_IB_B).transpose(), phi_IB), h_2_IB.transpose())
    H_11_1 = gamma*h_11_1 + alpha*mu*np.dot(np.dot(np.conjugate(h_I1_1).transpose(), phi_I1), h_11_I1.transpose())
    H_12_1 = gamma*h_12_1 + alpha*mu*np.dot(np.dot(np.conjugate(h_I1_1).transpose(), phi_I1), h_12_I1.transpose())
    H_21_2 = gamma*h_21_2 + alpha*mu*np.dot(np.dot(np.conjugate(h_I2_2).transpose(), phi_I2), h_21_I2.transpose())
    H_22_2 = gamma*h_22_2 + alpha*mu*np.dot(np.dot(np.conjugate(h_I2_2).transpose(), phi_I2), h_22_I2.transpose())

    ### Interference Link Channel Gain ###
    H_2_1 =  np.dot(np.dot(np.conjugate(h_IB_1).transpose(), phi_IB), h_2_IB.transpose())
    H_1_2 = np.dot(np.dot(np.conjugate(h_IB_2).transpose(), phi_IB), h_1_IB.transpose())
    H_22_1 =  np.dot(np.dot(np.conjugate(h_I2_1).transpose(), phi_I2), h_22_I2.transpose())
    H_21_1 = np.dot(np.dot(np.conjugate(h_I2_1).transpose(), phi_I2), h_21_I2.transpose())
    H_12_2 = np.dot(np.dot(np.conjugate(h_I1_2).transpose(), phi_I1), h_12_I1.transpose())
    H_11_2 = np.dot(np.dot(np.conjugate(h_I1_2).transpose(), phi_I1), h_11_I1.transpose())

    ### Calculate SINR Values ###

    F_1 = (P_21 * np.abs(H_22_1.item()) ** 2) + (
                (P_dash_21 + P_dash_22) * np.abs(H_2_1.item()) ** 2 + np.abs(h_1_1.item()) ** 2 * rho ** 2 * (
                    P_dash_11 + P_dash_12)) + P_22*np.abs(H_22_1)**2

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





def objective_function(para, iter):
   global zeta_I1, zeta_I2, zeta_IB, theta_I1, theta_I2, theta_IB, P_11, P_12, P_21, P_22, P_dash_11, P_dash_12, P_dash_21, P_dash_22
   zeta_I1, zeta_I2, zeta_IB, theta_I1, theta_I2, theta_IB, P_11, P_12, P_21, P_22 = para
   # , P_dash_11, P_dash_12, P_dash_21, P_dash_22
   reinit()
   penalty = 0
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
   condition_16 = rate_D11+rate_D12+rate_D22+rate_D21 >=R_min

   if not condition_1:
       penalty = penalty + 0 if rate_D11 - R_11_th >= 0 else abs(rate_D11 - R_11_th)
   elif not condition_2:
       penalty = penalty + 0 if rate_D12 - R_12_th >= 0 else abs(rate_D12 - R_12_th)
   elif not condition_3:
       penalty = penalty + 0 if rate_D21 - R_21_th >= 0 else abs(rate_D21 - R_21_th)
   elif not condition_4:
       penalty = penalty + 0 if rate_D22 - R_22_th >= 0 else abs(rate_D22 - R_22_th)
   elif not condition_5:
       penalty = penalty + 0 if np.abs(H_1_B).item() - np.abs(H_2_B).item() >= 0 else abs(
           np.abs(H_1_B).item() - np.abs(H_2_B).item())
   elif not condition_6:
       penalty = penalty + 0 if np.abs(H_11_1).item() - np.abs(H_12_1).item() >= 0 else abs(
           np.abs(H_11_1).item() - np.abs(H_12_1).item())
   elif not condition_7:
       penalty = penalty + 0 if np.abs(H_21_2).item() - np.abs(H_22_2).item() >= 0 else abs(
           np.abs(H_21_2).item() - np.abs(H_22_2).item())
   elif not condition_8:
       penalty = penalty + 0 if P_11 - P_11_max <= 0 else abs(P_11 - P_11_max)
   elif not condition_9:
       penalty = penalty + 0 if P_12 - P_12_max <= 0 else abs(P_12 - P_12_max)
   elif not condition_10:
       penalty = penalty + 0 if P_21 - P_21_max <= 0 else abs(P_21 - P_21_max)
   elif not condition_11:
       penalty = penalty + 0 if P_22 - P_22_max <= 0 else abs(P_22 - P_22_max)
   # elif not condition_12:
   #     penalty = penalty + 0 if P_dash_11 - P_dash_11_max <= 0 else abs(P_dash_11 - P_dash_11_max)
   # elif not condition_13:
   #     penalty = penalty + 0 if P_dash_12_max - P_dash_12_max <= 0 else abs(P_dash_12_max - P_dash_12_max)
   # elif not condition_14:
   #     penalty = penalty + 0 if P_dash_21_max - P_dash_21_max <= 0 else abs(P_dash_21_max - P_dash_21_max)
   # elif not condition_15:
   #     penalty = penalty + 0 if P_dash_22_max - P_dash_22_max <= 0 else abs(P_dash_22_max - P_dash_22_max)
   ## Recalculate rates based on these values.
   return rate_D11 + rate_D11 + rate_D21 + rate_D22 - penalty


#### Loop through to average ####
def avg_pso_vals(n_particles, inertia, cognitive, social):
    global gamma, m_iterations
    pso_mean_vals = []
    pso_max_vals = []
    pso_convergence = []
    best_val_list = []
    # pso_convergence =np.array([], dtype=float)
   #### Invoke PSO in loop
    for i in range(total_runs):
        best_val = penalty_pso_function(par, bounds, n_particles, m_iterations, inertia, cognitive, social, i)
        best_val_list.append(best_val)
        # if (type(pso_convergence) is list and pso_convergence) or type(pso_convergence) is not list:
        #     pso_convergence = np.mean([best_val,pso_convergence], axis=0, dtype=np.float64)
        # else:
        #     pso_convergence = best_val
        max_val = np.max(best_val)
        # print(max_val)

        pso_max_vals.append(max_val)
    # print(best_val_list)
    pso_convergence = np.mean(best_val_list, axis=0, dtype=np.float64)
    # print("Max of Mean",np.max(pso_convergence))
    return pso_max_vals, pso_mean_vals, pso_convergence


total_runs = 20
m_iterations = 200

### Indicator for Direct Link
gamma = 1

### Indicator for no-IRS
mu = 1

### Indicator for IRS in Relay to Device
alpha = 1

## Indicator for IRS in Relay to BS
beta = 1

n_particles = 100
inertia= 0.7
w1 = .9
w2 = .4
cognitive = 1.4
social = 1.4
### Initialize the particles
init(par, bounds, n_particles)
# velocities_init = np.load('velocities_init.npz')
# particles_init = np.load('particles_init.npz')
# pbest_init = np.load('pbest_init.npz')
# reinit()

print(f"Parameters:n_particles={n_particles}, inertia={inertia}, cognitive={cognitive}, social={social}, miterations={m_iterations}, totalrun={total_runs}")
print('### Full IRS ###')
beta = 1
alpha = 1
gamma = 1
mu = 1

P_11_max = P_12_max = P_21_max = P_22_max = 0.1

pso_max_vals, pso_mean_vals, pso_convergence = avg_pso_vals(n_particles, inertia, cognitive, social)
# print(len(pso_max_vals), np.max(pso_max_vals), f"n_particles={n_particles}, inertia={inertia}, cognitive={cognitive}, social={social}")
print( 'mean max val' , np.mean(pso_max_vals))
# plt.plot(np.arange(len(pso_convergence)), pso_convergence,  label="No IRS between Relay and BS", linestyle = 'dashed')

P_11_max = P_12_max = P_21_max = P_22_max = 0.2

pso_max_vals, pso_mean_vals, pso_convergence = avg_pso_vals(n_particles, inertia, cognitive, social)
# print(len(pso_max_vals), np.max(pso_max_vals), f"n_particles={n_particles}, inertia={inertia}, cognitive={cognitive}, social={social}")
print( 'mean max val' , np.mean(pso_max_vals))
# plt.plot(np.arange(len(pso_convergence)), pso_convergence,  label="No IRS between Relay and BS", linestyle = 'dashed')

P_11_max = P_12_max = P_21_max = P_22_max = 0.5
pso_max_vals, pso_mean_vals, pso_convergence = avg_pso_vals(n_particles, inertia, cognitive, social)
# print(len(pso_max_vals), np.max(pso_max_vals), f"n_particles={n_particles}, inertia={inertia}, cognitive={cognitive}, social={social}")
print( 'mean max val' , np.mean(pso_max_vals))
# plt.plot(np.arange(len(pso_convergence)), pso_convergence,  label="No IRS between Device and Relay", linestyle = 'dashed')

P_11_max = P_12_max = P_21_max = P_22_max = 1
pso_max_vals, pso_mean_vals, pso_convergence = avg_pso_vals(n_particles, inertia, cognitive, social)
# print(len(pso_max_vals), np.max(pso_max_vals), f"n_particles={n_particles}, inertia={inertia}, cognitive={cognitive}, social={social}")
print( 'mean max val' , np.mean(pso_max_vals))
# plt.plot(np.arange(len(pso_convergence)), pso_convergence,  label="No IRS between Device and Relay", linestyle = 'dashed')

P_11_max = P_12_max = P_21_max = P_22_max = 2
pso_max_vals, pso_mean_vals, pso_convergence = avg_pso_vals(n_particles, inertia, cognitive, social)
# print(len(pso_max_vals), np.max(pso_max_vals), f"n_particles={n_particles}, inertia={inertia}, cognitive={cognitive}, social={social}")
print( 'mean max val' , np.mean(pso_max_vals))

# pso_max_vals, pso_mean_vals, pso_convergence = avg_pso_vals(n_particles, inertia, cognitive, social)
# # print(len(pso_max_vals), np.max(pso_max_vals), f"n_particles={n_particles}, inertia={inertia}, cognitive={cognitive}, social={social}")
# print( 'mean max val' , np.mean(pso_max_vals))
# plt.plot(np.arange(len(pso_convergence)), pso_convergence,  label="No IRS between Device and Relay", linestyle = 'dashed')


# pso_max_vals, pso_mean_vals, pso_convergence = avg_pso_vals(n_particles, inertia, cognitive, social)
# # print(len(pso_max_vals), np.max(pso_max_vals), f"n_particles={n_particles}, inertia={inertia}, cognitive={cognitive}, social={social}")
# print( 'mean max val' , np.mean(pso_max_vals))
# plt.plot(np.arange(len(pso_convergence)), pso_convergence,  label="No IRS between Device and Relay", linestyle = 'dashed')

print('### No IRS between Device and Relay ###')
beta = 1
alpha = 0
gamma = 1
mu = 1

P_11_max = P_12_max = P_21_max = P_22_max = 0.1

pso_max_vals, pso_mean_vals, pso_convergence = avg_pso_vals(n_particles, inertia, cognitive, social)
# print(len(pso_max_vals), np.max(pso_max_vals), f"n_particles={n_particles}, inertia={inertia}, cognitive={cognitive}, social={social}")
print( 'mean max val' , np.mean(pso_max_vals))
# plt.plot(np.arange(len(pso_convergence)), pso_convergence,  label="No IRS between Relay and BS", linestyle = 'dashed')

P_11_max = P_12_max = P_21_max = P_22_max = 0.2

pso_max_vals, pso_mean_vals, pso_convergence = avg_pso_vals(n_particles, inertia, cognitive, social)
# print(len(pso_max_vals), np.max(pso_max_vals), f"n_particles={n_particles}, inertia={inertia}, cognitive={cognitive}, social={social}")
print( 'mean max val' , np.mean(pso_max_vals))
# plt.plot(np.arange(len(pso_convergence)), pso_convergence,  label="No IRS between Relay and BS", linestyle = 'dashed')

P_11_max = P_12_max = P_21_max = P_22_max = 0.5
pso_max_vals, pso_mean_vals, pso_convergence = avg_pso_vals(n_particles, inertia, cognitive, social)
# print(len(pso_max_vals), np.max(pso_max_vals), f"n_particles={n_particles}, inertia={inertia}, cognitive={cognitive}, social={social}")
print( 'mean max val' , np.mean(pso_max_vals))
# plt.plot(np.arange(len(pso_convergence)), pso_convergence,  label="No IRS between Device and Relay", linestyle = 'dashed')

P_11_max = P_12_max = P_21_max = P_22_max = 1
pso_max_vals, pso_mean_vals, pso_convergence = avg_pso_vals(n_particles, inertia, cognitive, social)
# print(len(pso_max_vals), np.max(pso_max_vals), f"n_particles={n_particles}, inertia={inertia}, cognitive={cognitive}, social={social}")
print( 'mean max val' , np.mean(pso_max_vals))
# plt.plot(np.arange(len(pso_convergence)), pso_convergence,  label="No IRS between Device and Relay", linestyle = 'dashed')

P_11_max = P_12_max = P_21_max = P_22_max = 2
pso_max_vals, pso_mean_vals, pso_convergence = avg_pso_vals(n_particles, inertia, cognitive, social)
# print(len(pso_max_vals), np.max(pso_max_vals), f"n_particles={n_particles}, inertia={inertia}, cognitive={cognitive}, social={social}")
print( 'mean max val' , np.mean(pso_max_vals))

# pso_max_vals, pso_mean_vals, pso_convergence = avg_pso_vals(n_particles, inertia, cognitive, social)
# # print(len(pso_max_vals), np.max(pso_max_vals), f"n_particles={n_particles}, inertia={inertia}, cognitive={cognitive}, social={social}")
# print( 'mean max val' , np.mean(pso_max_vals))
# plt.plot(np.arange(len(pso_convergence)), pso_convergence,  label="No IRS between Device and Relay", linestyle = 'dashed')


# pso_max_vals, pso_mean_vals, pso_convergence = avg_pso_vals(n_particles, inertia, cognitive, social)
# # print(len(pso_max_vals), np.max(pso_max_vals), f"n_particles={n_particles}, inertia={inertia}, cognitive={cognitive}, social={social}")
# print( 'mean max val' , np.mean(pso_max_vals))
# plt.plot(np.arange(len(pso_convergence)), pso_convergence,  label="No IRS between Device and Relay", linestyle = 'dashed')


# # print('### No IRS Between Relay and BS ###')
# # beta = 0
# # alpha = 1
# # gamma = 1
# # mu = 1
# #
# # R_11_th = R_12_th = R_21_th = R_22_th = 0.05
# #
# # pso_max_vals, pso_mean_vals, pso_convergence = avg_pso_vals(n_particles, inertia, cognitive, social)
# # # print(len(pso_max_vals), np.max(pso_max_vals), f"n_particles={n_particles}, inertia={inertia}, cognitive={cognitive}, social={social}")
# # print( 'mean max val' , np.mean(pso_max_vals))
# # # plt.plot(np.arange(len(pso_convergence)), pso_convergence,  label="No IRS between Relay and BS", linestyle = 'dashed')
# #
# # R_11_th = R_12_th = R_21_th = R_22_th = 0.1
# #
# # pso_max_vals, pso_mean_vals, pso_convergence = avg_pso_vals(n_particles, inertia, cognitive, social)
# # # print(len(pso_max_vals), np.max(pso_max_vals), f"n_particles={n_particles}, inertia={inertia}, cognitive={cognitive}, social={social}")
# print( 'mean max val' , np.mean(pso_max_vals))
# # plt.plot(np.arange(len(pso_convergence)), pso_convergence,  label="No IRS between Relay and BS", linestyle = 'dashed')
#
# R_11_th = R_12_th = R_21_th = R_22_th = 0.5
# pso_max_vals, pso_mean_vals, pso_convergence = avg_pso_vals(n_particles, inertia, cognitive, social)
# # print(len(pso_max_vals), np.max(pso_max_vals), f"n_particles={n_particles}, inertia={inertia}, cognitive={cognitive}, social={social}")
# print( 'mean max val' , np.mean(pso_max_vals))
# # plt.plot(np.arange(len(pso_convergence)), pso_convergence,  label="No IRS between Device and Relay", linestyle = 'dashed')
#
# R_11_th = R_12_th = R_21_th = R_22_th = 1
# pso_max_vals, pso_mean_vals, pso_convergence = avg_pso_vals(n_particles, inertia, cognitive, social)
# # print(len(pso_max_vals), np.max(pso_max_vals), f"n_particles={n_particles}, inertia={inertia}, cognitive={cognitive}, social={social}")
# print( 'mean max val' , np.mean(pso_max_vals))
# # plt.plot(np.arange(len(pso_convergence)), pso_convergence,  label="No IRS between Device and Relay", linestyle = 'dashed')
#
# R_11_th = R_12_th = R_21_th = R_22_th = 2
# pso_max_vals, pso_mean_vals, pso_convergence = avg_pso_vals(n_particles, inertia, cognitive, social)
# # print(len(pso_max_vals), np.max(pso_max_vals), f"n_particles={n_particles}, inertia={inertia}, cognitive={cognitive}, social={social}")
# print( 'mean max val' , np.mean(pso_max_vals))
#
# # pso_max_vals, pso_mean_vals, pso_convergence = avg_pso_vals(n_particles, inertia, cognitive, social)
# # # print(len(pso_max_vals), np.max(pso_max_vals), f"n_particles={n_particles}, inertia={inertia}, cognitive={cognitive}, social={social}")
# # print( 'mean max val' , np.mean(pso_max_vals))
# # plt.plot(np.arange(len(pso_convergence)), pso_convergence,  label="No IRS between Device and Relay", linestyle = 'dashed')
#
#
# # pso_max_vals, pso_mean_vals, pso_convergence = avg_pso_vals(n_particles, inertia, cognitive, social)
# # # print(len(pso_max_vals), np.max(pso_max_vals), f"n_particles={n_particles}, inertia={inertia}, cognitive={cognitive}, social={social}")
# # print( 'mean max val' , np.mean(pso_max_vals))
# # plt.plot(np.arange(len(pso_convergence)), pso_convergence,  label="No IRS between Device and Relay", linestyle = 'dashed')
#
#
# print('### No IRS ###')
# beta = 1
# alpha = 1
# gamma = 1
# mu = 0
#
# R_11_th = R_12_th = R_21_th = R_22_th = 0.05
#
# pso_max_vals, pso_mean_vals, pso_convergence = avg_pso_vals(n_particles, inertia, cognitive, social)
# # print(len(pso_max_vals), np.max(pso_max_vals), f"n_particles={n_particles}, inertia={inertia}, cognitive={cognitive}, social={social}")
# print( 'mean max val' , np.mean(pso_max_vals))
# # plt.plot(np.arange(len(pso_convergence)), pso_convergence,  label="No IRS between Relay and BS", linestyle = 'dashed')
#
# R_11_th = R_12_th = R_21_th = R_22_th = 0.1
#
# pso_max_vals, pso_mean_vals, pso_convergence = avg_pso_vals(n_particles, inertia, cognitive, social)
# # print(len(pso_max_vals), np.max(pso_max_vals), f"n_particles={n_particles}, inertia={inertia}, cognitive={cognitive}, social={social}")
# print( 'mean max val' , np.mean(pso_max_vals))
# # plt.plot(np.arange(len(pso_convergence)), pso_convergence,  label="No IRS between Relay and BS", linestyle = 'dashed')
#
# R_11_th = R_12_th = R_21_th = R_22_th = 0.5
# pso_max_vals, pso_mean_vals, pso_convergence = avg_pso_vals(n_particles, inertia, cognitive, social)
# # print(len(pso_max_vals), np.max(pso_max_vals), f"n_particles={n_particles}, inertia={inertia}, cognitive={cognitive}, social={social}")
# print( 'mean max val' , np.mean(pso_max_vals))
# # plt.plot(np.arange(len(pso_convergence)), pso_convergence,  label="No IRS between Device and Relay", linestyle = 'dashed')
#
# R_11_th = R_12_th = R_21_th = R_22_th = 1
# pso_max_vals, pso_mean_vals, pso_convergence = avg_pso_vals(n_particles, inertia, cognitive, social)
# # print(len(pso_max_vals), np.max(pso_max_vals), f"n_particles={n_particles}, inertia={inertia}, cognitive={cognitive}, social={social}")
# print( 'mean max val' , np.mean(pso_max_vals))
# # plt.plot(np.arange(len(pso_convergence)), pso_convergence,  label="No IRS between Device and Relay", linestyle = 'dashed')
#
# R_11_th = R_12_th = R_21_th = R_22_th = 2
# pso_max_vals, pso_mean_vals, pso_convergence = avg_pso_vals(n_particles, inertia, cognitive, social)
# # print(len(pso_max_vals), np.max(pso_max_vals), f"n_particles={n_particles}, inertia={inertia}, cognitive={cognitive}, social={social}")
# print( 'mean max val' , np.mean(pso_max_vals))
#
# # pso_max_vals, pso_mean_vals, pso_convergence = avg_pso_vals(n_particles, inertia, cognitive, social)
# # # print(len(pso_max_vals), np.max(pso_max_vals), f"n_particles={n_particles}, inertia={inertia}, cognitive={cognitive}, social={social}")
# # print( 'mean max val' , np.mean(pso_max_vals))
# # plt.plot(np.arange(len(pso_convergence)), pso_convergence,  label="No IRS between Device and Relay", linestyle = 'dashed')
#
#
# # pso_max_vals, pso_mean_vals, pso_convergence = avg_pso_vals(n_particles, inertia, cognitive, social)
# # # print(len(pso_max_vals), np.max(pso_max_vals), f"n_particles={n_particles}, inertia={inertia}, cognitive={cognitive}, social={social}")
# # print( 'mean max val' , np.mean(pso_max_vals))
# # plt.plot(np.arange(len(pso_convergence)), pso_convergence,  label="No IRS between Device and Relay", linestyle = 'dashed')
#
#
# # n_particles = 50
# # inertia= 0.7
# # cognitive = 2
# # social = 2
# #
# # avg_pso_vals(n_particles, inertia, cognitive, social)
# #
# #
# # n_particles = 20
# # inertia= 0.7
# # cognitive = 1.4
# # social = 1.4
# # avg_pso_vals(n_particles, inertia, cognitive, social)
# #
# #
# #
# # n_particles = 50
# # inertia= 0.7
# # cognitive = 1.4
# # social = 1.4
# #
# # avg_pso_vals(n_particles, inertia, cognitive, social)
# #
# #
# # n_particles = 20
# # inertia= 0.9
# # cognitive = .5
# # social = .3
# #
# # avg_pso_vals(n_particles, inertia, cognitive, social)
# #
# #
# # n_particles = 50
# # inertia= 0.9
# # cognitive = .5
# # social = .3
# #
# # avg_pso_vals(n_particles, inertia, cognitive, social)
#
#
#
#
#
#
#
#
#
#
#
# # plt.title(f'Optimization, penalty factor: {penalty_factor}')
# # plt.xlabel('Iteration')
# # plt.ylabel('Sum Rate')
# # plt.legend()
# # plt.grid(True)
# # plt.tight_layout()
# # import time
# # current_timestamp = time.time()
# # plt.savefig(f"PSO{current_timestamp}.svg")
# # plt.show()