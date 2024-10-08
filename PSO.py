
import random
import copy
import numpy as np

m_iterations = 250
n_particles = 200
n_dimension = 3
mc_count = 20
c1 = 1.4
c2 = 1.4
w = 0.7
w1 = 0.9
w2= 0.4



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
    return particles, velocities, pbest, r1, r2


use_random = False

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
                # gbest_value = fitness

            # Update iteration best value
            if fitness > iteration_best_value:
                iteration_best_value = fitness

            # Record the best value found in this iteration
        iteration_best_values.append(iteration_best_value)

        use_random = True
        # Update velocities and particles
        for i in range(n_particles):
            for j in range(n_dimension):
                # print(r1[iter][i][j], iter, i, j)
                # TODO set annealing for inertia
                # w = (w1 - w2) * ((m_iterations - (iter + 1)) / (m_iterations)) + w2
                if(use_random):
                    new_velocity = (w * velocities[i][j] +
                                        c1  * r1[iter][i][j]*(pbest[i][j] - particles[i][j]) +
                                        c2  * r2[iter][i][j]*(gbest[j] - particles[i][j]))
                else:
                    new_velocity = (w * velocities[i][j] +
                                    c1 * random.random() * (pbest[i][j] - particles[i][j]) +
                                    c2 * random.random() * (gbest[j] - particles[i][j]))
                # new_velocity = 4 if new_velocity >= 4 else new_velocity
                new_position = particles[i][j] + new_velocity
                new_position = max(min(new_position, bounds[j][1]), bounds[j][0])
                particles[i][j] = new_position
                velocities[i][j] = new_velocity

        #Printing
        if (iter % 100 == 0) or iter == (m_iterations-1):
            print(f"Iteration {iter}: Value = {iteration_best_values[iter]}")

    return iteration_best_values[m_iterations-1]

bounds = [(0,1)] + [(-1,1)] + [(0.5, 1)]


def objective_function(x):
    x1, x2, x3 = x
    return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2

particles, pbest, velocities, r1, r2 = init_particles()
penalty_pso_function(particles, pbest, velocities, r1, r2)