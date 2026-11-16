import numpy as np
import matplotlib.pyplot as plt

c = 0.3
# Define z up to just before the singularity at 1/c to avoid division by zero
z = np.linspace(0, 1/c, 1000, endpoint=False)

f_z = (1 - c * z) ** c
g_z = - (c ** 2) * (1 - c * z) ** (c - 1)

fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot f(z)
color = 'tab:blue'
ax1.set_xlabel('z')
ax1.set_ylabel('f(z)', color=color)
ax1.plot(z, f_z, color=color, linewidth=2, label=r'$f(z) = (1 - 0.3z)^{0.3}$')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(-0.1, 1.1)
ax1.grid(True, linestyle=':', alpha=0.6)

# Instantiate a second axes that shares the same x-axis for g(z)
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('g(z) = df/dz', color=color)
ax2.plot(z, g_z, color=color, linestyle='--', linewidth=2, label=r'$g(z) = -0.09(1 - 0.3z)^{-0.7}$')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(-5, 0.1) # Bound y-axis because g(z) approaches -infinity

plt.title('Real-valued Domain: $f(z)$ and $g(z)$ for $c = 0.3$')
fig.tight_layout()
plt.show()