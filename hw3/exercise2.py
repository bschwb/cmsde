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

mc_samples = 5*10**5
n_timesteps = 10000

timestep = T / n_timesteps

Y = Y_0 * np.ones(mc_samples)
S = S_0 * np.ones(mc_samples)

for i in range(n_timesteps):
    print(i)
    dW = sqrt(timestep) * np.random.randn(mc_samples)
    dZ = sqrt(timestep) * np.random.randn(mc_samples)
    S = S + r * S * timestep + np.exp(Y) * S * dW
    Y = Y + (-alpha * (2 + Y) + 0.4 * sqrt(alpha) * sqrt(1-rho**2)) * timestep + \
        0.4*sqrt(alpha)*(rho * dW + sqrt(1-rho**2) * dZ)

g = np.maximum(S - K, 0)
ev_g = sum(g) / mc_samples
f = exp(-r * T) * ev_g
sample_variance_f = exp(-2*r*T) * 1/(mc_samples - 1) * sum((g - ev_g)**2)

print(f)
print(sqrt(sample_variance_f))
