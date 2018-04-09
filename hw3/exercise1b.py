from math import sqrt, exp

import numpy as np
import matplotlib.pyplot as plt


T = 1/2
K = 35

r = 0.04
sigma = 0.2

mc_samples = 10**4

min_s = 0
max_s = 100
interval_splits = 200
delta_s = 1/interval_splits
start_vals = np.linspace(0, max_s, num=interval_splits+1, endpoint=True)
f = np.zeros(len(start_vals))

W_T = sqrt(T) * np.random.randn(mc_samples)
for i, S_0 in enumerate(start_vals):
    S_T = S_0 * np.exp((r-sigma**2/2)*T + sigma * W_T)
    g = np.maximum(S_T - K, 0)
    ev_g = sum(g) / mc_samples
    f[i] = exp(-r * T) * ev_g

plt.plot(start_vals, f)
plt.xlabel('start value $S_0$')
plt.ylabel('sensitivity $\Delta f$')
plt.savefig('delta_f.png')
plt.show()
