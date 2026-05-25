import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1.0
T = 0.1
nx = 200
D = 0.01
dx = L / (nx - 1)
x = np.linspace(0, L, nx)

# Time step based on stability
dt = 0.5 * dx**2 / D
nt = int(T / dt)
dt = T / nt

# Initial condition: delta-like at center
u = np.zeros(nx)
u[nx // 2] = 1.0 / dx  # approximate Dirac delta
u_all = [u.copy()]

# Right-hand side function for diffusion
def diffusion_rhs(u, D, dx):
    dudt = np.zeros_like(u)
    dudt[1:-1] = D * (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
    return dudt

# RK4 method
for n in range(nt):
    k1 = diffusion_rhs(u, D, dx)
    k2 = diffusion_rhs(u + 0.5 * dt * k1, D, dx)
    k3 = diffusion_rhs(u + 0.5 * dt * k2, D, dx)
    k4 = diffusion_rhs(u + dt * k3, D, dx)
    u = u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    if n % max(1, (nt // 5)) == 0:
        u_all.append(u.copy())

# Analytical solution function
def analytical_solution(x, t, D, x0):
    return (1 / np.sqrt(4 * np.pi * D * t)) * np.exp(-((x - x0) ** 2) / (4 * D * t))

# Plotting
plt.figure(figsize=(10, 6))
snap_times = [i * T / 5 for i in range(1, 6)]  # exclude t=0 for analytical (singularity)

for i, t_snap in enumerate(snap_times):
    u_num = u_all[i + 1]  # skip u_all[0] since t=0
    u_ana = analytical_solution(x, t_snap, D, L / 2)
    plt.plot(x, u_num, label=f'Numerical t={t_snap:.2f}', linestyle='-')
    plt.plot(x, u_ana, label=f'Analytical t={t_snap:.2f}', linestyle='--')

plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.title('1D Diffusion: Numerical (RK4) vs Analytical Solution')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save as PDF
plt.savefig("diffusion_plot.pdf", format='pdf')

plt.show()
