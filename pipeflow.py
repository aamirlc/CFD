import numpy as np
import matplotlib.pyplot as plt

# assumptions : no slip at the walls, periodic boundary condition assumed

# trying to solve the Hagen-Poiseulle 

i = 21 # grid points in each direction
n = 100

dt = 0.5

nu = 0.01

del_P_rho = np.array([-0.5, 0]) # constant pressure gradient 

def main():
    
    h = 3 / (i - 1)

    x = np.linspace(0, 3, i)
    y = np.linspace(0, 1, i) 

    X, Y = np.meshgrid(x, y)

    def central_diff_x(f):
        diff = (np.roll(f, shift=1, axis=1) - np.roll(f, shift=-1, axis=1)) / (2 * h)

        return diff

    def laplacian(f):
        diff = (np.roll(f, shift=1, axis=1) + np.roll(f, shift=1, axis=0)
                             + np.roll(f, shift=-1, axis=1) + np.roll(f, shift=-1, axis=0)
                             - 4 * f) / (h ** 2)

        return diff
    
    # setting IC & no-slip

    u0 = np.ones_like(X)
    u0[0, :] = 0.0
    u0[-1, :] = 0.0
    
    for _ in range(n):

        convection = u0 * central_diff_x(u0)
        diffusion = nu * laplacian(u0)

        additive = dt * ( - del_P_rho[0] + diffusion - convection)

        un = u0 + additive

        un[0, :] = 0.0
        un[-1, :] = 0.0

        u0 = un

        plt.contourf(X, Y, un, levels=20, cmap=plt.cm.bone)
        plt.colorbar()

        plt.quiver(X, Y, un, np.zeros_like(un))

        plt.twiny()
        plt.plot(un[:, 1], Y[:, 1], color="red")

        plt.draw()
        plt.pause(0.1)
        plt.clf()

    plt.show()

if __name__ == "__main__":
    main()

