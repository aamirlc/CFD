# solution to the 1D Burger's equation

import numpy as np
from matplotlib import pyplot as plt

i = 20
n = 50

dt = 0.01
dx = 2 / (i - 1)

im1 = [(i - 1) for i in range(i)]
ip1 = [(i + 1) for i in range(i)]
x = [i * dx for i in range(i)]

# enforcing the BC

ip1[19] = 0
im1[0] = 19

u = np.zeros_like(x) 

# initial conditions

def main():

    nu = 0.1

    def phi(x, nu=0.1, pi=np.pi):
        return np.exp(-x ** 2 / 4 * nu) + np.exp(-(x - 2 * pi) ** 2 / 4 * nu)

    def d_phi_d_x(x, nu=0.1, pi=np.pi):
        deriv = (- nu * x / 2) * np.exp(-x ** 2 / 4 * nu) 
        + (- nu * (x - 2 * pi) / 2) * np.exp(-(x - 2 * pi) ** 2 / 4 * nu)

        return deriv

    for _ in range(i):
        u[_] = 4 + (- 2 * nu) * d_phi_d_x(x[_]) / phi(x[_])

    for t in range(n):
        un = u

        for _ in range(i):
            u[_] = un[_] - un[_] * (dt / dx) * (un[_] - un[im1[_]]) 
            + (nu * (dt / dx ** 2) * (un[ip1[_]] - 2 * un[_] + un[im1[_]]))

    
    plt.plot(x, u)
    plt.show()

if __name__ == "__main__":
    main()
    