from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# GBM Data
mu = 1
sigma = 1

x0 = 1
endtime = 1

# Functions for weak convergence
g1 = lambda x: np.exp(-x**2)
g2 = lambda x: x
g3 = lambda x: 1/np.sqrt(abs(x-sigma*5))

# Numerics
def forward_euler(x0, endtime, timesteps, samples):
    """Return forward Euler approx for SDE and Brownian Motion W_T"""
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

# Monte Carlo error approximations
n_partitions = np.logspace(1, 4, num=20, dtype=int)
err_strong = np.zeros(len(n_partitions))
err_weak1 = np.zeros(len(n_partitions))
err_weak2 = np.zeros(len(n_partitions))
err_weak3 = np.zeros(len(n_partitions))
for i, n_timesteps in enumerate(n_partitions):
    print(i, n_timesteps)
    Xbar, Wt = forward_euler(x0, endtime, n_timesteps, samples)
    X = sol(endtime, Wt)
    err_strong[i] = np.sqrt(sum((X - Xbar)**2)/samples)
    err_weak1[i] = sum(g1(X) - g1(Xbar))/samples
    err_weak2[i] = sum(g2(X) - g2(Xbar))/samples
    err_weak3[i] = sum(g3(X) - g3(Xbar))/samples


# Plot errors vs fitted curves
dts = 1/n_partitions
p, _ = curve_fit(lambda x, p: p*np.sqrt(x), dts, err_strong)
plt.semilogx(dts, err_strong, dts, p*np.sqrt(dts))
plt.gca().invert_xaxis()
plt.ylabel('strong error')
plt.xlabel('$\Delta t$')
plt.savefig('strong_error.pdf', bbox_inches='tight')
plt.cla()

k, d = np.polyfit(dts, err_weak1, 1)
plt.semilogx(dts, err_weak1, dts, k * dts + d)
plt.gca().invert_xaxis()
plt.ylabel('weak error g1')
plt.xlabel('$\Delta t$')
plt.savefig('weak_error1.pdf', bbox_inches='tight')
plt.clf()

k, d = np.polyfit(dts, err_weak2, 1)
plt.semilogx(dts, err_weak2, dts, k * dts + d)
plt.gca().invert_xaxis()
plt.ylabel('weak error g2')
plt.xlabel('$\Delta t$')
plt.savefig('weak_error2.pdf', bbox_inches='tight')
plt.clf()

k, d = np.polyfit(dts, err_weak3, 1)
plt.semilogx(dts, err_weak3, dts, k * dts + d)
plt.gca().invert_xaxis()
plt.ylabel('weak error g2')
plt.xlabel('$\Delta t$')
plt.savefig('weak_error3.pdf', bbox_inches='tight')
plt.clf()
