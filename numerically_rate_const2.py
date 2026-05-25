import numpy as np
import matplotlib.pyplot as plt

# Barrier height
E0 = 1.0

# Potential: symmetric double well
def V(x):
    return E0 * (x**2 - 1)**2

# Localized Boltzmann-like function around x_min
def f_relative(x, x_min, kBT):
    return np.exp(-(V(x) - V(x_min)) / kBT)

# x-range for integration
x = np.linspace(-2.5, 2.5, 1000)
x_min = -1.0
V_xmin = V(x_min)
xA_range = x[x > x_min]  # For flux denominator
x1, x2 = -1.5, -0.5      # Left well range
x_na_range = x[(x >= x1) & (x <= x2)]
f_xmin = 1.0  # Arbitrary, cancels in k = j / n_a

# Friction values
gamma_vals = np.linspace(0.1, 5, 50)

# Temperatures to analyze
kBT_values = [0.49, 0.5, 0.51]
k_results = []

for kBT in kBT_values:
    # Denominator of flux
    int_j_denom = np.trapezoid(np.exp(V(xA_range) / kBT), xA_range)

    # Population in left well
    int_na = np.trapezoid(f_relative(x_na_range, x_min, kBT), x_na_range)

    # Compute k(γ) for each γ
    k_vals = []
    for gamma in gamma_vals:
        j = (kBT / gamma) * f_xmin * np.exp(V_xmin / kBT) / int_j_denom
        na = f_xmin * int_na
        k = j / na
        k_vals.append(k)

    k_results.append(k_vals)

# Plotting only k vs γ
plt.figure(figsize=(6, 5))

for kBT, k_vals in zip(kBT_values, k_results):
    plt.plot(gamma_vals, k_vals, label=f'kBT = {kBT}')

plt.xlabel('γ (Friction coefficient)')
plt.ylabel('Rate constant k')
plt.title('k vs γ for different temperatures')
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()

