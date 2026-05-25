import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 1.0  # friction coefficient
D = 1.0  # diffusion coefficient (related to noise strength)
v0 = 5.0  # initial velocity

# Time values to plot the distribution at
time_points = [0.1, 0.5, 1.0, 2.0, 5.0]
v = np.linspace(-10, 10, 500)

# Plotting
plt.figure(figsize=(10, 6))

for t in time_points:
    sigma_squared = (D / alpha) * (1 - np.exp(-2 * alpha * t))
    mean = v0 * np.exp(-alpha * t)
    prefactor = 1 / np.sqrt(2 * np.pi * sigma_squared)
    f_vt = prefactor * np.exp(- (v - mean) ** 2 / (2 * sigma_squared))

    plt.plot(v, f_vt, label=f't = {t}')

plt.title('Non-equilibrium Velocity Distribution $f_{\\text{non-eq}}(v,t)$')
plt.xlabel('Velocity v')
plt.ylabel('f(v, t)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("fokker_planck_sol.pdf", format='pdf')
plt.show()