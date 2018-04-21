import numpy as np
import matplotlib.pyplot as plt

N = 40000
theta = np.zeros(N)
Y = np.random.randn(N-1)
theta[0] = 1

dt = 0.0001

for n, yn in enumerate(Y):
    theta[n+1] = theta[n] - 2 * dt * (theta[n] - yn)

plt.plot(theta)
plt.ylabel(r'$\theta$', rotation=0)
plt.xlabel('$n$')
plt.savefig('theta_conv.pdf', bbox_inches='tight')
plt.show()
