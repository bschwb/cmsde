import numpy as np
from math import sqrt

import matplotlib.pyplot as plt


# SDE Data
a = np.vectorize(lambda t, x: 0)
b = np.vectorize(lambda t, x: 1)

def g(t, x):
    pass

x0 = 0
endtime = 1


# Numerics
def feuler_error(x0, endtime, timesteps, ref_timesteps, samples):
    """Simulate strong approx error of forward Euler approx for SDE

    Employs the Monte Carlo method.
    """
    assert(ref_timesteps > timesteps)

    X = x0 * np.ones(samples)
    Xbar = x0 * np.ones(samples)

    dt = endtime / ref_timesteps
    t = np.linspace(0, endtime, endpoint=False, num=ref_timesteps)
    dW = sqrt(dt) * np.random.randn(ref_timesteps, samples)

    dt_bar = endtime / timesteps
    tbar = np.linspace(0, endtime, endpoint=False, num=timesteps)
    dW_bar = np.sum(dW.reshape(-1,ref_timesteps//timesteps,samples), 1)

    for tn, dw in zip(t, dW):
        X = X + a(tn, X) * dt + b(tn, X) * dw

    for tn, dw in zip(tbar, dW_bar):
        Xbar = Xbar + a(tn, Xbar) * dt + b(tn, Xbar) * dw

    error = sqrt(np.sum((X-Xbar)**2/samples))
    return error


# Simulation parameters
samples = 1000
reftimesteps = 10000

N = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
er = np.zeros(len(N))
for i, timesteps in enumerate(N):
    er[i] = feuler_error(x0, endtime, timesteps, reftimesteps, samples)

plt.plot(er)
plt.show()
