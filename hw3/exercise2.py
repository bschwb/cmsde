from math import sqrt, exp

import numpy as np
import matplotlib.pyplot as plt


T = 3/4
Y_0 = -1
S_0 = 100
K = 100

r = 0.04
alpha = 100
rho = -0.3

mc_samples = 10**4

n_timesteps = 1000
timestep = T / n_timesteps
dW = sqrt(timestep) * np.random.randn(n_timesteps, mc_samples)
dZ = sqrt(timestep) * np.random.randn(n_timesteps, mc_samples)
dZhat = rho * dW + sqrt(1 - rho**2) * dZ

Y = Y_0 * np.ones(mc_samples)
S = S_0 * np.ones(mc_samples)

for i in range(n_timesteps):
    print(i)
    S = S + r * S * timestep + np.exp(Y) * S * dW[i]
    Y = Y + (-alpha * (2 + Y) + 0.4 * sqrt(alpha) * sqrt(1-rho**2)) * timestep + 0.4*sqrt(alpha)*dZhat[i]

g = np.maximum(S - K, 0)
ev_g = sum(g) / mc_samples
f = exp(-r * T) * ev_g

print(f)
