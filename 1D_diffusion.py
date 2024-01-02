# solving the extremely basic 1d diffusion equation 
# physics - represents a wave damped exponentially in time (diffusion) (if nu > 0)
# if nu < 0, represents an explosion 

import matplotlib.pyplot as plt
import numpy as np

i = 20 # no of grid points in the space coordinate
n = 50 # no of grid points in the time coordinate

nu = 0.1 # kinematic viscosity

dt = 0.01 # look into the stability of the solution
dx = 2 / (i - 1)

x = np.linspace(0, 2, i) # domain size = 2
ud = np.zeros_like(x) # diffusion 
uc = np.zeros_like(x) # for non-linear convection
ui = np.zeros_like(x)

# marching in space

for _ in range(i):
    if x[_] >= 0.5 and x[_] <= 1.0:
        ud[_] = 2.0
        ui[_] = 2.0
        uc[_] = 2.0
    else:
        ud[_] = 1.0
        ui[_] = 1.0
        uc[_] = 1.0
    
# marching in time

for _ in range(n):
    un = ud
    unc = uc

    for j in range(1, i-1):
       ud[j] = un[j] + nu * (dt / dx ** 2) * (un[j+1] - 2 * un[j] + un[j-1])
       uc[j] = unc[j] - unc[j] * (dt / dx) * (unc[j] - unc[j-1])


plt.plot(x, ui)
plt.plot(x, ud)
plt.plot(x, uc)
plt.show()







