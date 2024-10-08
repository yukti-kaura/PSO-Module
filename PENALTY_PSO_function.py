import math
import random
import numpy as np

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
### Indicator for Direct Link
gamma = 1

### Indicator for no-IRS
mu = 1

### Indicator for IRS in Relay to Device
alpha = 1

## Indicator for IRS in Relay to BS
beta = 1

channels=[]
sigma_1 = sigma_2 = sigma_b = math.sqrt(10**((- 120) / 10))
### Rate Thresholds ###
R_11_th = R_12_th = R_21_th = R_22_th = 0.05

### Power Thresholds ###
P_11_max = P_12_max = P_21_max = P_22_max = 0.5
P_dash_11_max = P_dash_12_max = P_dash_21_max = P_dash_22_max = 1

death_penalty = False

def find_min(a, b):
  if(a > b):
    return b
  return a


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


def init_channel():
    ### Calculate Channel ###
    # Ques: Do all IRS have equal number of reflecting elements?
    h_I1_1 = generate_channel(N, 1, get_pathloss_direct(D_I1_1))
    channels.append(h_I1_1)
    h_11_I1 = generate_channel(1, N, get_pathloss_direct(D_11_I1))
    channels.append(h_11_I1)
    h_12_I1 = generate_channel(1, N, get_pathloss_direct(D_12_I1))
    channels.append(h_12_I1)
    h_12_1 = generate_channel(1, 1, get_pathloss_direct(D_12_1))
    channels.append(h_12_1)
    h_11_1 = generate_channel(1, 1, get_pathloss_direct(D_11_1))
    channels.append(h_11_1)

    h_I2_2 = generate_channel(N, 1, get_pathloss_direct(D_I2_2))
    channels.append(h_I2_2)
    h_21_I2 = generate_channel(1, N, get_pathloss_direct(D_21_I2))
    channels.append(h_21_I2)
    h_22_I2 = generate_channel(1, N, get_pathloss_direct(D_22_I2))
    channels.append(h_22_I2)
    h_21_2 = generate_channel(1, 1, get_pathloss_direct(D_21_2))
    channels.append(h_21_2)
    h_22_2 = generate_channel(1, 1, get_pathloss_direct(D_22_2))
    channels.append(h_22_2)

    h_IB_B = generate_channel(N, 1, get_pathloss_direct(D_IB_B))
    channels.append(h_IB_B)
    h_2_IB = generate_channel(1, N, get_pathloss_direct(D_2_IB))
    channels.append(h_2_IB)
    h_1_IB = generate_channel(1, N, get_pathloss_direct(D_1_IB))
    channels.append(h_1_IB)
    h_2_B = generate_channel(1, 1, get_pathloss_direct(D_2_B))
    channels.append(h_2_B)
    h_1_B = generate_channel(1, 1, get_pathloss_direct(D_1_B))
    channels.append(h_1_B)

    ### Interference link path loss

    ### Relay 1 ###
    h_IB_1 = generate_channel(N, 1, get_pathloss_direct(D_1_IB)) ### Channel is reciprocal ###
    channels.append(h_IB_1)
    # ### Relay 2 ###
    h_IB_2 = generate_channel(N, 1, get_pathloss_direct(D_2_IB))
    channels.append(h_IB_2)

    ### Interference  from IRS 1 (I1) to Relay 2 (R2)
    h_I1_2 = generate_channel(N, 1, get_pathloss_direct(D_2_I1))
    channels.append(h_I1_2)
    ### Interference  from IRS 2 (I2) to Relay 1 (R1)
    h_I2_1 = generate_channel(N, 1, get_pathloss_direct(D_1_I2))
    channels.append(h_I2_1)

    ## Duplexing interference ###
    h_1_1 =  generate_channel(1, 1, 1)
    channels.append(h_1_1)
    h_2_2 =  generate_channel(1, 1, 1)
    channels.append(h_2_2)
    np.savez('channels', channels)
    return channels
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

def init_particles():
    particles = []
    velocities = []
    pbest = []
    for _ in range(n_particles):
        particle = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(n_dimension)]
        particles.append(particle)
        velocities.append([0] * n_dimension)
        pbest.append(particle[:])
        np.savez('particles', particles)
        np.savez('velocities', velocities)
        np.savez('pbest', velocities)
    r1 = [np.random.rand(m_iterations, n_particles, n_dimension) for _ in range(mc_count)]
    np.savez('r1', r1)
    r2 = [np.random.rand(m_iterations, n_particles, n_dimension) for _ in range(mc_count)]
    np.savez('r2', r2)


def pso_function(n_dimension, bounds, n_particles, m_iterations, inertia, cognitive, social):

    print("PSO Algorithm Started")

    num_particles = n_particles
    max_iterations = m_iterations
    w = inertia  # inertia weight
    c1 = cognitive  # cognitive constant
    c2 = social  # social constant
    w1 = 0.9
    w2 = 0.4
    gbest = None
    gbest_value = -float('inf')
    iteration_best_values = []

    # PSO loop
    for iter in range(max_iterations):
        iteration_best_value = -float('inf')
        for i in range(num_particles):
            current_position = particles[i]
            fitness = objective_function(current_position)

            # Update personal best
            if fitness > objective_function(pbest[i]):
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
            for j in range(n_dimension):
                w = (w1 - w2) * ((m_iterations - (iter + 1)) / (m_iterations)) + w2
                new_velocity = (w * velocities[i][j] +
                                    c1 * r1[0][iter][i][j] * (pbest[i][j] - particles[i][j]) +
                                    c2 * r2[0][iter][i][j] * (gbest[j] - particles[i][j]))

                old_position = particles[i][j]
                old_velocity = velocities[i][j]
                new_position = particles[i][j] + new_velocity

                new_position = max(min(new_position, bounds[j][1]), bounds[j][0])

                particles[i][j] = new_position
                velocities[i][j] = new_velocity

                if(death_penalty):
                    if (evaluate_conditions() != 0.0):
                        particles[i][j] = old_position
                        velocities[i][j] = old_velocity
        #Printing
        if (iter % 100 == 0) or iter == (max_iterations-1):
                print(f"Iteration {iter}: Value = {iteration_best_values[iter]}")

    return iteration_best_values


bounds = [(0,1)] + [(-1,1)] + [(0.5, 1)]
def objective_function(x):
    x1, x2, x3 = x
    return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2

n_particles = 200
m_iterations = 200
inertia = 0.7
cognitive = 1.4
social = 1.4
n_dimension = 3
mc_count=1

# init_particles()
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

pso_function(n_dimension, bounds, n_particles, m_iterations, inertia, cognitive, social)