import math
import random

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

def funct(parameter, bounds, n_particles, m_iterations, inertia, cognitive, social):

    print("PSO Algorithm Started")

    num_particles = n_particles
    max_iterations = m_iterations
    w = inertia  # inertia weight
    c1 = cognitive  # cognitive constant
    c2 = social  # social constant

    para = flatten(parameter)
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
        temp_con = unflatten(particle, parameter)
        while (conditions(temp_con) == False):
            particle = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(len_para)]
            temp_con = unflatten(particle, parameter)
        particles.append(particle)
        velocities.append([0] * (len_para))
        pbest.append(particle[:])

    # PSO loop
    for iter in range(max_iterations):
        iteration_best_value = -float('inf')
        for i in range(num_particles):
            current_position = particles[i]
            temp = unflatten(current_position, parameter)
            fitness = objective_function(temp)

            # Update personal best
            temp = unflatten(pbest[i], parameter)
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

                temp_con = unflatten(particles[i], parameter)
                if not conditions(temp_con):
                    particles[i][j] -= new_velocity
                    velocities[i][j] = 0
        #Printing
        if (iter % 100 == 0) or iter == (max_iterations-1):
                print(f"Iteration {iter}: Value = {iteration_best_values[iter]}")

    return iteration_best_values