import numpy as np

def build_A(n):
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = -4
        if i > 0:
            A[i, i - 1] = 1
        if i < n - 1:
            A[i, i + 1] = 1
        if i > 3:
            A[i, i - 4] = 1
        if i < n - 4:
            A[i, i + 4] = 1
    return A

def sample_galaxy(rate = 256, n_galaxies = 20, n_particles_per_galaxy=2000, seed=0):
    np.random.seed(seed)
    grid = np.zeros((rate, rate))

    particles = np.zeros((n_particles_per_galaxy * n_galaxies, 2))
    for i in range(n_galaxies):
        galaxy = np.random.normal(0, np.random.rand()*0.1+0.05, (n_particles_per_galaxy,2))
        galaxy += np.random.rand(2)- 0.5
        galaxy *= rate
        galaxy += rate/2
        particles[i*n_particles_per_galaxy : (i+1) * n_particles_per_galaxy] = galaxy
    
    # nearest grid point mass assignment
    for particle in particles:
        x = int(particle[0])
        y = int(particle[1])
        if x < 0 or x >= rate or y < 0 or y >= rate:
            continue
        grid[x, y] += 1

    return grid

galaxy = sample_galaxy()
A = build_A(galaxy.shape[0])