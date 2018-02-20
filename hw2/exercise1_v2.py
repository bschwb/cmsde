from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# GBM Data
mu = 1
sigma = 1

x0 = 1
endtime = 1

# Numerics
def forward_euler(x0, endtime, timesteps, samples):
    """Simulate strong approx error of forward Euler approx for SDE

    Employs the Monte Carlo method.
    """
    X = x0 * np.ones(samples)

    dt = endtime / timesteps
    t = np.linspace(0, endtime, endpoint=False, num=timesteps)
    dW = sqrt(dt) * np.random.randn(timesteps, samples)

    for tn, dw in zip(t, dW):
        X = X + mu * X * dt + sigma * X * dw

    return X, np.sum(dW, 0)


def sol(t, Wt):
    """Return the exact solution of the geometric Brownian motion at time t"""
    return x0 * np.exp((mu - sigma**2/2)*t + sigma * Wt)

sol = np.vectorize(sol)


# Simulation parameters
samples = 10000

n_partitions = np.logspace(1, 4, num=20, dtype=int)
er = np.zeros(len(n_partitions))
for i, n_timesteps in enumerate(n_partitions):
    print(i, n_timesteps)
    Xbar, Wt = forward_euler(x0, endtime, n_timesteps, samples)
    X = sol(endtime, Wt)
    er[i] = np.sqrt(sum((X - Xbar)**2)/samples)

dts = 1/n_partitions
p, _ = curve_fit(lambda x, p: p*np.sqrt(x), dts, er)
plt.semilogx(dts, er, dts, p*np.sqrt(dts))
plt.gca().invert_xaxis()
plt.show()
