import numpy as np
import random
import math

N= 16
n_dimension = 6*N+8
n_particles = 1
bounds = [(0,1)]*(3*N) + [np.arange(0, np.pi, .15)]*(3*N) +  [(0,0.5)]*4 + [(0,1)]*4
print(len(bounds))
for _ in range(n_particles):
    particle1 = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(0, 3 * N)]
    particle2 = [random.choice(bounds[i]) for i in range(3*N, 2*3 * N)]
    particle3 = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(6*N, n_dimension)]
    particle = particle1 +  particle2 +  particle3
print(len(particle))

bounds = [(0,1)]*(3*N) +[(0,1)]*(3*N) +  [(0,0.5)]*4 + [(0,1)]*4
print(len(bounds))
for _ in range(n_particles):
    particle = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(n_dimension)]
print(len(particle))