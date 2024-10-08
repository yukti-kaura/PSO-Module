import numpy as np
import random
import math
import matplotlib.pyplot as plt
def up_get_ini(alpha_K0, alpha_K_dash0, P_s0, W_0, h_as, h_ak, sigma_k, h_ac, h_sc, sigma_c, K_dash, sigma_k_dash, h_ak_dash, h_sk, h_ik, h_si, h_sk_dash, h_ik_dash, varphi_r, varphi_t):
    # initialize the rates

    SINRKK = 0.0
    beta_K0 = 0.0
    SINRKK_dash = 0.0
    beta_K_dash0 = 0.0
    SINRCC = 0.0

    k = 0
    k_dash = 0

    SINRKK = (alpha_K0 * P_s0 * (np.abs(h_sk[k][0] + np.dot(np.dot(h_ik[k],varphi_r),h_si).item()))**2) / (P_s0 * (np.abs(h_sk[k][0] + np.dot(np.dot(h_ik[k],varphi_r),h_si).item()))**2 * (alpha_K_dash0) + (np.abs(np.dot(W_0,h_ak[k])))**2 + sigma_k**2)
    beta_K0 = SINRKK

    SINRCC = (np.abs(np.dot(W_0,h_ac)))**2 / (P_s0 * np.abs(h_sc)**2 + sigma_c**2)

    SINRKK_dash = (alpha_K_dash0 * P_s0 * (np.abs(h_sk_dash[k_dash][0] + np.dot(np.dot(h_ik_dash[k_dash],varphi_t),h_si).item()))**2) / ((np.abs(np.dot(W_0,h_ak_dash[k_dash])))**2 + sigma_k_dash**2)
    beta_K_dash0 = SINRKK_dash

    rho = 0.8
    P_I =  np.abs(np.dot(W_0,h_as))**2 + (rho**2 * P_s0)

    #print(beta_K0, beta_K_dash0)
    return SINRCC, beta_K0, beta_K_dash0, P_I
def generate_channel(N, K, path_loss):

    h = (1/np.sqrt(2)) * (np.random.randn(N, K) + 1j * np.random.randn(N, K))
    h_normalized = h / np.linalg.norm(h, axis=0)
    h = path_loss * h_normalized

    return h

def up_GET_dist_pru(P_AP, P_DD, P_CC, P_KK, P_IRS, P_KK_dash, noise):
    relative_user_loc = [x - y for x, y in zip(P_AP, P_DD)]
    d_as = math.sqrt(relative_user_loc[0]**2 + relative_user_loc[1]**2)
    pathloss_direct = lambda d: 32.6 + 36.7 * math.log10(d)
    PL_as = pathloss_direct(d_as)
    PL_as = math.sqrt(10**((- noise - PL_as) / 10))

    relative_user_loc = [x - y for x, y in zip(P_AP, P_CC)]
    d_ac = math.sqrt(relative_user_loc[0]**2 + relative_user_loc[1]**2)
    PL_ac = pathloss_direct(d_ac)
    PL_ac = math.sqrt(10**((- noise - PL_ac) / 10))

    relative_user_loc = [x - y for x, y in zip(P_AP, P_KK)]
    d_ak = math.sqrt(relative_user_loc[0]**2 + relative_user_loc[1]**2)
    PL_ak = pathloss_direct(d_ak)
    PL_ak = math.sqrt(10**((- noise - PL_ak) / 10))

    relative_user_loc = [x - y for x, y in zip(P_DD, P_KK)]
    d_sk = math.sqrt(relative_user_loc[0]**2 + relative_user_loc[1]**2)
    PL_sk = pathloss_direct(d_sk)
    PL_sk = math.sqrt(10**((- noise - PL_sk) / 10))

    relative_user_loc = [x - y for x, y in zip(P_IRS, P_KK)]
    d_ik = math.sqrt(relative_user_loc[0]**2 + relative_user_loc[1]**2)
    PL_ik = pathloss_direct(d_ik)
    PL_ik = math.sqrt(10**((- noise - PL_ik) / 10))

    relative_user_loc = [x - y for x, y in zip(P_DD, P_IRS)]
    d_si = math.sqrt(relative_user_loc[0]**2 + relative_user_loc[1]**2)
    PL_si = pathloss_direct(d_si)
    PL_si = math.sqrt(10**((- noise - PL_si) / 10))

    relative_user_loc = [x - y for x, y in zip(P_DD, P_CC)]
    d_sc = math.sqrt(relative_user_loc[0]**2 + relative_user_loc[1]**2)
    PL_sc = pathloss_direct(d_sc)
    PL_sc = math.sqrt(10**((- noise - PL_sc) / 10))

    relative_user_loc = [x - y for x, y in zip(P_AP, P_KK_dash)]
    d_ak_dash = math.sqrt(relative_user_loc[0]**2 + relative_user_loc[1]**2)
    PL_ak_dash = pathloss_direct(d_ak_dash)
    PL_ak_dash = math.sqrt(10**((- noise - PL_ak_dash) / 10))

    relative_user_loc = [x - y for x, y in zip(P_DD, P_KK_dash)]
    d_sk_dash = math.sqrt(relative_user_loc[0]**2 + relative_user_loc[1]**2)
    PL_sk_dash = pathloss_direct(d_sk_dash)
    PL_sk_dash = math.sqrt(10**((- noise - PL_sk_dash) / 10))

    relative_user_loc = [x - y for x, y in zip(P_IRS, P_KK_dash)]
    d_ik_dash = math.sqrt(relative_user_loc[0]**2 + relative_user_loc[1]**2)
    PL_ik_dash = pathloss_direct(d_ik_dash)
    PL_ik_dash = math.sqrt(10**((- noise - PL_ik_dash) / 10))

    return (PL_as, PL_ac, PL_ak, PL_sk, PL_ik, PL_si, PL_sc, PL_ak_dash, PL_sk_dash, PL_ik_dash)
M = 2  # number of antennas on AP
N = 8  # number IRS elements
K = 1  # no.of user at receiver side
K_dash = 1  # no.of user at transmission side


P_AP = [0, 0]  # in meters
P_DD = [150, -10]
P_CC = [10, -150]
P_IRS = [175, -30]
P_KK = [160, -40]
P_KK_dash = [190, -20]
noise = -120

PL_as, PL_ac, PL_ak, PL_sk, PL_ik, PL_si, PL_sc, PL_ak_dash, PL_sk_dash, PL_ik_dash = up_GET_dist_pru(P_AP, P_DD, P_CC, P_KK, P_IRS, P_KK_dash, noise)

phi_ac = 1
phi_sc = 1
phi_sk = 0.2 * np.ones(K)
h_as= generate_channel(M,1,PL_as);
h_ac= generate_channel(M,1,PL_ac);
h_ak= generate_channel(M,K,PL_ak);
h_ak = h_ak.transpose() #For own Easy
h_sk = generate_channel(1,K,PL_sk);
h_sk = np.conjugate(h_sk.transpose())
h_ik= generate_channel(N,K,PL_ik);
h_ik = np.conjugate(h_ik.transpose())
h_si= generate_channel(1,N,PL_si);
h_si = h_si.transpose()
h_sc= generate_channel(1,1,PL_sc);
h_ak_dash= generate_channel(M,K_dash,PL_ak_dash);
h_ak_dash = h_ak_dash.transpose() #For own Easy
h_sk_dash= generate_channel(1,K_dash,PL_sk_dash);
h_sk_dash = np.conjugate(h_sk_dash.transpose())
h_ik_dash= generate_channel(N,K_dash,PL_ik_dash);
h_ik_dash = np.conjugate(h_ik_dash.transpose())
sigma_c = 1
sigma_k = 1
sigma_k_dash = 1
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 128),  # Increase capacity with more neurons
            nn.ReLU(),
            nn.Linear(128, 128),  # Additional layer for better representation
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                target = (reward + self.gamma * torch.max(self.model(next_state)).item())
            state = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state).detach().clone()
            target_f[0][action] = target
            self.model.zero_grad()
            criterion = nn.MSELoss()
            loss = criterion(self.model(state), target_f)
            loss.backward()
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)  # Adam optimizer for stability
            optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

import math
with open('rates_DQN.txt', 'w') as f:
    pass

def dqn_function_maximization(max_iterations=10000, batch_size=32):
    state_size = 2*M + 1 + K + 3*N  # size of the state vector
    action_size = 2*(1 + 2*M + K + 3*N)  # number of possible actions
    agent = DQNAgent(state_size, action_size)

    # Lists to store values for plotting
    iterations = []
    values = []
    avg = []

    W_0_real = [random.uniform(0, 1) for _ in range(M)]
    W_0_imaginary = [random.uniform(0, 1) for _ in range(M)]
    P_s0 = random.uniform(0, 1)
    alpha_K0 = np.array([random.uniform(0, 1)])
    alpha_K_dash0 = np.array([1 - alpha_K0.item()])
    zeta_r = [random.uniform(0, 1) for _ in range(N)]
    theta_r = [random.uniform(0, 2*math.pi) for _ in range(N)]
    theta_t = [random.uniform(0, 2*math.pi) for _ in range(N)]
    zeta_t = [1 - z for z in zeta_r]

    state = np.concatenate([W_0_real, W_0_imaginary, [P_s0], alpha_K0, zeta_r, theta_r, theta_t])

    iter = 0
    while iter < max_iterations:
        action = agent.act(state)

        if action < M:
            idx = action % M
            W_0_real[idx] = max(0, W_0_real[idx] + np.random.uniform(0, 0.1))
        elif action < 2*M:
            idx = (action-M) % M
            W_0_real[idx] = max(0, W_0_real[idx] + np.random.uniform(-0.1, 0))

        elif action < 2*M + M:
            idx = (action - 2*M) % M
            W_0_imaginary[idx] = max(0, W_0_imaginary[idx] + np.random.uniform(0, 0.1))
        elif action < 2*M + 2*M:
            idx = (action - 2*M + M) % M
            W_0_imaginary[idx] = max(0, W_0_imaginary[idx] + np.random.uniform(-0.1, 0))

        elif action < 1 + 2*M + 2*M:
            P_s0 = max(0, P_s0 + np.random.uniform(0, 0.1))
        elif action < 2 + 2*M + 2*M:
            P_s0 = max(0, P_s0 + np.random.uniform(-0.1, 0))

        elif action < 2 + 2*M + 2*M + K:
            alpha_K0[0] = max(0, min(1, alpha_K0[0] + np.random.uniform(0, 0.1)))
            alpha_K_dash0[0] = 1 - alpha_K0[0]
        elif action < 2 + 2*M + 2*M + 2*K:
            alpha_K0[0] = max(0, min(1, alpha_K0[0] + np.random.uniform(-0.1, 0)))
            alpha_K_dash0[0] = 1 - alpha_K0[0]

        elif action < 2 + 2*M + 2*M + 2*K + N:
            idx = (action - (2 + 2*M + 2*M + 2*K)) % N
            theta_r[idx] = min(2 * math.pi, theta_r[idx] + np.random.uniform(0, 0.1))
        elif action < 2 + 2*M + 2*M + 2*K + 2*N:
            idx = (action - (2 + 2*M + 2*M + 2*K + N)) % N
            theta_r[idx] = max(0, theta_r[idx] + np.random.uniform(-0.1, 0))

        elif action < 2 + 2*M + 2*M + 2*K + 2*N + N:
            idx = (action - (2 + 2*M + 2*M + 2*K + 2*N)) % N
            theta_t[idx] = min(2 * math.pi, theta_t[idx] + np.random.uniform(0, 0.1))
        elif action < 2 + 2*M + 2*M + 2*K + 2*N + 2*N:
            idx = (action - (2 + 2*M + 2*M + 2*K + 2*N + N)) % N
            theta_t[idx] = max(0, theta_t[idx] + np.random.uniform(-0.1, 0))

        elif action < 2 + 2*M + 2*M + 2*K + 2*N + 2*N + N:
            idx = (action - (2 + 2*M + 2*M + 2*K + 2*N + 2*N)) % N
            zeta_r[idx] = min(1, zeta_r[idx] + np.random.uniform(0, 0.1))
        elif action < 2 + 2*M + 2*M + 2*K + 2*N + 2*N + 2*N:
            idx = (action - (2 + 2*M + 2*M + 2*K + 2*N + 2*N + N)) % N
            zeta_r[idx] = max(0, zeta_r[idx] + np.random.uniform(-0.1, 0))

        zeta_t = [1 - z for z in zeta_r]

        W_0 = np.array([real + 1j * imaginary for real, imaginary in zip(W_0_real, W_0_imaginary)])

        varphi_r_value = [math.sqrt(zeta) * math.exp(theta) for zeta, theta in zip(zeta_r, theta_r)]
        varphi_r = np.zeros((N, N), dtype=complex)
        np.fill_diagonal(varphi_r, varphi_r_value)

        varphi_t_value = [math.sqrt(zeta) * math.exp(theta) for zeta, theta in zip(zeta_t, theta_t)]
        varphi_t = np.zeros((N, N), dtype=complex)
        np.fill_diagonal(varphi_t, varphi_t_value)

        SINRCC, SINRKK, SINRKK_dash, P_I = up_get_ini(alpha_K0, alpha_K_dash0, P_s0, W_0, h_as, h_ak, sigma_k, h_ac, h_sc, sigma_c, K_dash, sigma_k_dash, h_ak_dash, h_sk, h_ik, h_si, h_sk_dash, h_ik_dash, varphi_r, varphi_t)

        rateSINRCC = math.log2(SINRCC.item() + 1)
        rateSINRKK = math.log2(SINRKK.item() + 1)
        rateSINRKK_dash = math.log2(SINRKK_dash.item() + 1)

        R_min_k = 0.1
        R_min_k_dash = R_min_k
        R_min_c = 0.1
        P_max = 10**(20/10)
        P_I_min = 4
        P_I_max = 80
        eta = 0.5

        if P_I < P_I_min:
            P_sh = 0
        elif P_I >= P_I_min and P_I <= P_I_max:
            P_sh = eta * P_I
        elif P_I > P_I_max:
            P_sh = eta * P_I_max

        condition_1 = (rateSINRKK >= R_min_k) and (rateSINRKK_dash >= R_min_k_dash)
        condition_3 = rateSINRCC >= R_min_c
        condition_4 = all(np.abs(w)**2 <= P_max for w in W_0)
        condition_5 = P_s0 <= P_sh
        condition_6 = P_I >= P_I_min

        if not (condition_1 and condition_3 and condition_4 and condition_5 and condition_6):
            continue

        reward = (rateSINRKK + rateSINRKK_dash)
        next_state = np.concatenate([W_0_real, W_0_imaginary, [P_s0], alpha_K0, zeta_r, theta_r, theta_t])
        done = iter == max_iterations - 1

        agent.remember(state, action, reward, next_state, done)
        state = next_state

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        iterations.append(iter)
        values.append(reward)
        avg.append(reward)

        if iter % 100 == 0:
            print(f"Iteration {iter}: Best value = {max(values)}, Average = {sum(avg)/len(avg)}")
            avg = []
        iter += 1
        with open('rates_DQN.txt', 'a') as f:
            f.write(str(reward) + "\n")

    return iterations, values

# Perform DQN function maximization
iterations, values = dqn_function_maximization()

# Plot the optimization process
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(iterations)), values, label='Function Value')
plt.title('DQN Optimization of Random Function')
plt.xlabel('Iteration')
plt.ylabel('Function Value')
plt.legend()
plt.tight_layout()
plt.show()
window_size = 20
moving_average = np.convolve(values, np.ones(window_size)/window_size, mode='valid')

# Plot the original function values and the moving average
plt.figure(figsize=(10, 6))
plt.plot(iterations[window_size-1:], moving_average, label='Moving Average', color='grey')
plt.title(f'DQN Optimization (Moving average: {window_size})')
plt.xlabel('Iteration')
plt.ylabel('Function Value')
plt.legend()
plt.tight_layout()
plt.show()