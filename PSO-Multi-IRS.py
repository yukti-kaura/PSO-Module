import math
import random
import numpy as np
import random
import math
import matplotlib.pyplot as plt

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

def pso_function(parameter_values, bounds, n_particles, m_iterations, inertia, cognitive, social):

    print("PSO Algorithm Started")

    num_particles = n_particles
    max_iterations = m_iterations
    w = inertia  # inertia weight
    c1 = cognitive  # cognitive constant
    c2 = social  # social constant

    para = flatten(parameter_values)
    len_para = len(para)
    update_bounds = flatten(bounds)


    particles = []
    velocities = []
    pbest = []
    gbest = None
    gbest_value = -float('inf')
    iteration_best_values = []

    for _ in range(num_particles):
        particle = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(len_para)]
        temp_con = unflatten(particle, parameter_values)
        while (conditions(temp_con) == False):
            particle = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(len_para)]
            temp_con = unflatten(particle, parameter_values)
        particles.append(particle)
        velocities.append([0] * (len_para))
        pbest.append(particle[:])

    # PSO loop
    for iter in range(max_iterations):
        iteration_best_value = -float('inf')
        for i in range(num_particles):
            current_position = particles[i]
            temp = unflatten(current_position, parameter_values)
            fitness = objective_function(temp)

            # Update personal best
            temp = unflatten(pbest[i], parameter_values)
            if fitness > objective_function(temp):
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
                new_velocity = (w * velocities[i][j] +
                                    c1 * random.random() * (pbest[i][j] - particles[i][j]) +
                                    c2 * random.random() * (gbest[j] - particles[i][j]))
                new_position = particles[i][j] + new_velocity

                new_position = max(min(new_position, update_bounds[j][1]), update_bounds[j][0])

                # Update only if the new position satisfies the condition
                particles[i][j] = new_position
                velocities[i][j] = new_velocity

                temp_con = unflatten(particles[i], parameter_values)
                if not conditions(temp_con):
                    particles[i][j] -= new_velocity
                    velocities[i][j] = 0

   # if (iter % 100 == 0) or iter == (max_iterations-1):
        print(f"Iteration {iter}: Value = {iteration_best_values[iter]}")

    return iteration_best_values

M = 1  # No. of antennas on BS
N = N1 = N2 = NB  = 8  # No. IRS elements
K = 1  # No.of transmit antennas at the relay device
K_dash = 1 # No. of transmit antennas at the redcap (IoT) device
q = 1 # No. of Quantization bits. 1,2,..., Q
m = 1 # m=1,2,..., 2^n-1

####################### Reflection co-efficient matrix for IRS_1 (I1) #########################
zeta_I1 = [random.uniform(0, 1) for _ in range(N)]
theta_I1 = [random.uniform(0, 2*math.pi) for _ in range(N)]
zeta_I2 = [random.uniform(0, 1) for _ in range(N)]
theta_I2 = [random.uniform(0, 2*math.pi) for _ in range(N)]
zeta_IB = [random.uniform(0, 1) for _ in range(N)]
theta_IB = [random.uniform(0, 2*math.pi) for _ in range(N)]

#### Initialize PSO parameters ####
n_particles = 30
m_iterations = 200
inertia= 0.5
cognitive = 0.5
social = 0.5

### Bounds ###
# bounds = [(0,1)]*(2*M+1)+[(0,1)]*K+[(0,1)]*N+[(0,2*math.pi)]*(2*N)

# Create bounds
# max_bound = 5.12 * np.ones(2)
# min_bound = - max_bound
# bounds = (min_bound, max_bound)

bounds = [(0,1)]*(3*N) + [(0,2*math.pi)]*(3*N) +  [(0,0.5)]*4 + [(0,1)]*4
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

### Noise  ###
# -120db : TODO Assign linear value
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



### Check with Justin ###
rho = 0.1
### We have Power and IRS Phase shifts to optimize ###
par =  zeta_I1, zeta_I2,  zeta_IB, theta_I1,theta_I2, theta_IB, P_11, P_12, P_21,P_22, P_dash_11, P_dash_12, P_dash_21, P_dash_22



def find_min(a, b):
  if(a > b):
    return a
  return b

def generate_channel(N, K, path_loss):
    h = (1/np.sqrt(2)) * (np.random.randn(N, K) + 1j * np.random.randn(N, K))
    h_normalized = h / np.linalg.norm(h, axis=0)
    h = path_loss * h_normalized
    return h

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

h_I2_2 = generate_channel(N, 1, get_pathloss_direct(D_I2_2))
h_21_I2 = generate_channel(1, N, get_pathloss_direct(D_21_I2))
h_22_I2 = generate_channel(1, N, get_pathloss_direct(D_22_I2))

h_IB_B = generate_channel(N, 1, get_pathloss_direct(D_IB_B))
h_2_IB = generate_channel(1, N, get_pathloss_direct(D_2_IB))
h_1_IB = generate_channel(1, N, get_pathloss_direct(D_1_IB))


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

def reinit():
    global rate_D21, rate_D11, rate_D22, rate_D12
    global zeta_I1, zeta_I2,  zeta_IB, theta_I1,theta_I2, theta_IB, P_11, P_12, P_21,P_22, P_dash_11, P_dash_12, P_dash_21, P_dash_22
    global H_1_B , H_2_B , H_11_1 , H_12_1 , H_21_2 , H_22_2
    global SINR_B_D22 , SINR_B_D21 , SINR_B_D12 , SINR_B_D12 , SINR_R2_D22 , SINR_R2_D21 , SINR_R1_D11 , SINR_R1_D12
    ### Only one phase coefficient matrix is assumed as it is NOT STAR IRS ###
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
    H_1_B = np.dot(np.dot(np.conjugate(h_IB_B).transpose(), phi_IB), h_1_IB.transpose())
    H_2_B = np.dot(np.dot(np.conjugate(h_IB_B).transpose(), phi_IB), h_2_IB.transpose())
    H_11_1 = np.dot(np.dot(np.conjugate(h_I1_1).transpose(), phi_I1), h_11_I1.transpose())
    H_12_1 = np.dot(np.dot(np.conjugate(h_I1_1).transpose(), phi_I1), h_12_I1.transpose())
    H_21_2 = np.dot(np.dot(np.conjugate(h_I2_2).transpose(), phi_I2), h_21_I2.transpose())
    H_22_2 = np.dot(np.dot(np.conjugate(h_I2_2).transpose(), phi_I2), h_22_I2.transpose())

    ### Interference Link Channel Gain ###
    H_2_1 = np.dot(np.dot(np.conjugate(h_IB_1).transpose(), phi_IB), h_2_IB.transpose())
    H_1_2 = np.dot(np.dot(np.conjugate(h_IB_2).transpose(), phi_IB), h_1_IB.transpose())
    H_22_1 = np.dot(np.dot(np.conjugate(h_I2_1).transpose(), phi_I2), h_22_I2.transpose())
    H_21_1 = np.dot(np.dot(np.conjugate(h_I2_1).transpose(), phi_I2), h_21_I2.transpose())
    H_12_2 = np.dot(np.dot(np.conjugate(h_I1_2).transpose(), phi_I1), h_12_I1.transpose())
    H_11_2 = np.dot(np.dot(np.conjugate(h_I1_2).transpose(), phi_I1), h_11_I1.transpose())

    ### Calculate SINR Values ###

    F_1 = (P_21 * np.abs(H_22_1.item()) ** 2) + (
                (P_dash_21 + P_dash_22) * np.abs(H_2_1.item()) ** 2 + np.abs(h_1_1.item()) ** 2 * rho ** 2 * (
                    P_dash_11 + P_dash_12))

    # SINR at R1 to detect symbol of D11 considering D12 as interference assuming H11_1 > H12_1
    SINR_R1_D11 = (P_11 * np.abs(H_11_1.item()) ** 2) / (P_12 * np.abs(H_12_1.item()) ** 2 + F_1 + sigma_1 ** 2)

    # SINR at R1 to detect symbol of D12 considering D11 as interference assuming D11 has already been decoded
    SINR_R1_D12 = (P_12 * np.abs(H_12_1.item()) ** 2) / (F_1 + sigma_1 ** 2)

    # SINR at R2
    F_2 = P_11 * np.abs(H_11_2.item()) ** 2 + P_12 * np.abs(H_12_2.item()) ** 2 + (P_dash_11 + P_dash_12) * np.abs(
        H_1_2.item()) ** 2 + np.abs(h_2_2.item()) ** 2 * rho ** 2 * (P_dash_21 + P_dash_12)

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

### Initialize the initial Global vars
reinit()

def conditions(para):
    global zeta_I1, zeta_I2, zeta_IB, theta_I1, theta_I2, theta_IB, P_11, P_12, P_21, P_22, P_dash_11, P_dash_12, P_dash_21, P_dash_22
    zeta_I1, zeta_I2, zeta_IB, theta_I1, theta_I2, theta_IB, P_11, P_12, P_21, P_22, P_dash_11, P_dash_12, P_dash_21, P_dash_22 = para
    reinit()
    #Recalculate rates based on these parameters
    condition_1 = (rate_D11 >= R_11_th) and (rate_D12 >= R_12_th) and (rate_D21 >= R_21_th) and (rate_D22 >= R_22_th)
    ### Test absolute value of channels ###
    condition_2 = np.abs(H_1_B).item() > np.abs(H_2_B).item()
    condition_3 = np.abs(H_11_1).item() > np.abs(H_12_1).item()
    condition_4 = np.abs(H_21_2).item() > np.abs(H_22_2).item()
    condition_5 = P_11 <= P_11_max and P_22 <= P_22_max and P_21 <= P_21_max and P_12 <= P_12_max
    condition_6 = P_dash_11 <= P_dash_11_max and P_dash_22 <= P_dash_22_max and P_21 <= P_dash_21_max and P_dash_12 <= P_dash_12_max
    # print(condition_1 , condition_2 , condition_5 , condition_6)
    if (condition_1 and condition_2  and condition_3 and condition_4 and condition_5 and condition_6):
        return True
    return False

def objective_function(para):
   global zeta_I1, zeta_I2, zeta_IB, theta_I1, theta_I2, theta_IB, P_11, P_12, P_21, P_22, P_dash_11, P_dash_12, P_dash_21, P_dash_22
   zeta_I1, zeta_I2, zeta_IB, theta_I1, theta_I2, theta_IB, P_11, P_12, P_21, P_22, P_dash_11, P_dash_12, P_dash_21, P_dash_22 = para
   reinit()
   ## Recalculate rates based on these values.
   return rate_D11 + rate_D11 + rate_D21 +  rate_D22

a = pso_function(par, bounds, n_particles, m_iterations, inertia, cognitive, social)

plt.plot(np.arange(len(a)), a, label='PSO')
plt.title('Optimization')
plt.xlabel('Iteration')
plt.ylabel('Sum Rate')
plt.legend()
plt.tight_layout()
plt.show()