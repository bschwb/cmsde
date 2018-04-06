from math import sqrt, exp

import numpy as np
import matplotlib.pyplot as plt


T = 1/2
S_0 = 35
K = 35

r = 0.04
sigma = 0.2

samples_sequence = [10**i for i in range(2, 8)]
f = np.zeros(len(samples_sequence))
sample_variance_f = np.zeros(len(samples_sequence))
error_variance = np.zeros(len(samples_sequence))

for i, n_samples in enumerate(samples_sequence):
    W_T = sqrt(T) * np.random.randn(n_samples)
    S_T = S_0 * np.exp((r-sigma**2/2)*T + sigma * W_T)
    g = np.maximum(S_T - K, 0)
    ev_g = sum(g) / n_samples
    f[i] = exp(-r * T) * ev_g
    sample_variance_f[i] = exp(-2*r*T) * 1/(n_samples - 1) * sum((g - ev_g)**2)
    error_variance[i] = sample_variance_f[i] / n_samples

# plt.semilogx(samples_sequence, f)
plt.loglog(samples_sequence, error_variance)
plt.savefig('error_var.png')
plt.show()
