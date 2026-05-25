import numpy as np
import matplotlib.pyplot as plt

# Parameters
gamma = 0.5
kT_by_m = 2.0
D = gamma * kT_by_m
dt = 0.01
T_max = 20
N = int(T_max / dt)
t = np.linspace(0, T_max, N)
n_ensemble = 1000

# Initial velocities for both cases
v0_low = 0.0       # Case 1: <v^2(0)> < kT/m
v0_high = 3.0      # Case 2: <v^2(0)> > kT/m

def simulate(v0):
    all_traj = np.zeros((n_ensemble, N))
    for j in range(n_ensemble):
        v = np.zeros(N)
        v[0] = v0
        for i in range(1, N):
            noise = np.sqrt(2 * D * dt) * np.random.normal()
            v[i] = v[i - 1] - gamma * v[i - 1] * dt + noise
        all_traj[j, :] = v
    v2_avg = np.mean(all_traj**2, axis=0)
    return all_traj[0], v2_avg

# Run both cases
v_low_traj, v2_low_avg = simulate(v0_low)
v_high_traj, v2_high_avg = simulate(v0_high)

# Plotting
plt.figure(figsize=(12, 6))

# <v^2(t)> comparison
plt.subplot(1, 2, 1)
plt.plot(t, v2_low_avg, label=r'$\langle v^2(t) \rangle$, $v_0^2 < kT/m$', color='blue')
plt.plot(t, v2_high_avg, label=r'$\langle v^2(t) \rangle$, $v_0^2 > kT/m$', color='orange')
plt.axhline(kT_by_m, linestyle='--', color='red', label='Equilibrium: $kT/m$')
plt.title('Ensemble Average of $v^2(t)$')
plt.xlabel('Time')
plt.ylabel(r'$\langle v^2 \rangle$')
plt.grid(True)
plt.legend()

# Example trajectories
plt.subplot(1, 2, 2)
plt.plot(t, v_low_traj, label='Sample v(t), low v₀', color='blue')
plt.plot(t, v_high_traj, label='Sample v(t), high v₀', color='orange')
plt.title('Single Trajectories of $v(t)$')
plt.xlabel('Time')
plt.ylabel('v(t)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("lengevin_num_sol.pdf", format='pdf')
plt.show()