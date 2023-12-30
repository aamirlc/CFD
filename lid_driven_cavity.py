import numpy as np
from matplotlib import pyplot as plt

n = 100 # discretizations 
box_size = 2.0 # domain size
n_iter = 500 # this and the dt term need to be studied more to ensure stability
dt = 0.0001

nu = 0.1 # kinematic viscosity
rho = 1.0 # density
u = 3.0 # lid velocity (horizontal)

p_poisson_iter = n_iter / 20

def main():
    h = box_size / (n - 1) # grid size
    x = np.linspace(0.0, box_size, n)
    y = np.linspace(0.0, box_size, n)

    X, Y = np.meshgrid(x, y)

    u0 = np.zeros_like(X)
    v0 = np.zeros_like(X)
    p0 = np.zeros_like(X)

    # using the central difference scheme for partial derivatives

    def central_diff_x(field):
        diff = np.zeros_like(field)
        diff[1:-1, 1:-1] = (field[1:-1, 2: ] - field[1:-1, 0:-2]) / (2 * h)

        return diff

    def central_diff_y(field):
        diff = np.zeros_like(field)
        diff[1:-1, 1:-1] = (field[2: , 1:-1] - field[0:-2, 1:-1]) / (2 * h)

        return diff
    
    # using the 5 point stencil for laplacian (check Wikipedia) 
    
    def laplacian(f):
        diff = np.zeros_like(f)
        diff[1:-1, 1:-1] = (f[0:-2, 1:-1] + f[2: , 1:-1]
                             + f[1:-1, 0:-2] + f[1:-1, 2: ] 
                             - 4 * f[1:-1, 1:-1]) / (h ** 2)

        return diff
    
    for t in range(n_iter):

        # Velocity: Dirichlet boundary condition except at the moving lid
        # Pressure: Neumann boundary condition except at the moving lid (p = 0)

        d_u0_d_x = central_diff_x(u0)
        d_u0_d_y = central_diff_y(u0)
        d_v0_d_x = central_diff_x(v0)
        d_v0_d_y = central_diff_y(v0)

        laplacian_u0 = laplacian(u0)
        laplacian_v0 = laplacian(v0)

        # solving the momentum equation neglecting the pressure gradient term in NS

        u_est = (u0 + dt * (nu * laplacian_u0 - (u0 * d_u0_d_x + v0 * d_u0_d_y) ))
        v_est = (v0 + dt * (nu * laplacian_v0 - (u0 * d_v0_d_x + v0 * d_v0_d_y) ))

        # enforce the boundary conditions

        u_est[0, :] = 0.0
        u_est[:, 0] = 0.0
        u_est[:, -1] = 0.0
        u_est[-1, :] = u

        v_est[0, :] = 0.0
        v_est[:, 0] = 0.0
        v_est[:, -1] = 0.0
        v_est[-1, :] = 0.0

        d_u_est_d_x = central_diff_x(u_est)
        d_u_est_d_y = central_diff_y(u_est)
        d_v_est_d_x = central_diff_x(v_est)
        d_v_est_d_y = central_diff_y(v_est)

        # divergence (Nabla operator)

        # solving the pressure-Poisson equation (check wikipedia for basics)

        rhs = (
            rho / dt
            * ( d_u_est_d_x + d_v_est_d_y )
        )

        for t in range(int(p_poisson_iter)):
            p_new = np.zeros_like(p0)
            p_new[1:-1, 1:-1] = 0.25 * (
                p0[1:-1, 0:-2] + p0[0:-2, 1:-1] + p0[1:-1, 2: ] + p0[2: , 1:-1] 
                - h ** 2 * rhs[1:-1, 1:-1]
            )

            # enforce the BC 

            p_new[-1, :] = 0.0
            p_new[:, -1] = p_new[:, -2]
            p_new[:, 0] = p_new[:, 1]
            p_new[0, :] = p_new[1, :]
            
            p0 = p_new

        d_p_new_d_x = central_diff_x(p_new)
        d_p_new_d_y = central_diff_y(p_new)

        # add pressure corrections

        u_new = u_est - dt / rho * d_p_new_d_x
        v_new = v_est - dt / rho * d_p_new_d_y

        u_new[0, :] = 0.0
        u_new[:, 0] = 0.0
        u_new[:, -1] = 0.0
        u_new[-1, :] = u

        v_new[-1, :] = 0.0
        v_new[0, :] = 0.0
        v_new[:, 0] = 0.0
        v_new[:, -1] = 0.0

        u0 = u_new
        v0 = v_new
        p0 = p_new

    plt.figure()
    plt.contourf(X, Y, p_new, cmap=plt.cm.bone)
    plt.colorbar()

    plt.streamplot(X, Y, u_new, v_new, color="black")
    plt.show()

if __name__ == "__main__":
    main()