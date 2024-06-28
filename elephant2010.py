import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

num = 101
k = np.arange(1, 6)
t = np.linspace(0, 2 * np.pi, num)

Ax = np.array([-60, 0, 0, 0, 0])
Bx = np.array([-30, 8, -10, 0, 0])
Ay = np.array([0, 0, -12, 0, 14])
By = np.array([-50, -18, 0, 0, 0])

x = Ax.reshape(-1, 1) * np.cos(k.reshape(-1, 1) * t) + Bx.reshape(-1, 1) * np.sin(k.reshape(-1, 1) * t)
y = Ay.reshape(-1, 1) * np.cos(k.reshape(-1, 1) * t) + By.reshape(-1, 1) * np.sin(k.reshape(-1, 1) * t)
x = x.sum(axis=0)
y = y.sum(axis=0)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig('elephant2010.pdf')
plt.show()