# Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from typing import Callable, Tuple

# Defining parameters
L = 1.0  # Domain size
Nx = 50  # Number of grid points
Nt = 500  # Number of time steps
T = 0.1  # Final time
mu = 0.001  # Diffusion coefficient
a = 0.5 # Reaction coefficient

# Defining grid
x = np.linspace(0, L, Nx)
t = np.linspace(0, T, Nt)

def exact_solution_test_case(x: np.ndarray, t: float, mu: float, a: float, L: float) -> np.ndarray:
    """
    Exact solution for the test problem: u_t - mu u_xx = a u,
    with Dirichlet boundary conditions and initial condition u(x,0) = sin(pi*x/L).
    """
    return np.exp((a - mu * (np.pi**2 / L**2)) * t) * np.sin(np.pi * x / L)

# Reaction function
def linear_reaction_test(u):
    return a * u  # Linear reaction term

# Solver for reaction-diffusion equations
def solve_reaction_diffusion(Nx: int, Nt: int, L: float, T: float, mu: float, f: Callable[[np.ndarray], np.ndarray]) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Solves a 1D reaction-diffusion equation u_t = mu u_xx + f(u)
    using a modified Crank-Nicolson scheme.
    """
    dx = L / (Nx - 1)  # Spatial step size
    dt = T / Nt  # Time step size
    r = (mu * dt) / (dx**2)  # Stability parameter

    # Define grid
    x = np.linspace(0, L, Nx)
    u = np.sin(np.pi * x)  # Initial condition

    # Construct tridiagonal matrix for implicit system
    Nx_inner = Nx - 2  # Exclude boundary points
    diag_main = (1 + r) * np.ones(Nx_inner)
    diag_off = (-r / 2) * np.ones(Nx_inner - 1)

    # Tridiagonal matrix A
    A = np.diag(diag_main) + np.diag(diag_off, k=1) + np.diag(diag_off, k=-1)

    # Perform LU factorization once (for efficiency)
    LU = scipy.linalg.lu_factor(A)

    # Time stepping
    for _ in range(1, Nt):
        # Right-hand side
        rhs = u[1:-1] + (r / 2) * (u[:-2] - 2*u[1:-1] + u[2:]) + dt * f(u[1:-1])
        
        # Solve for u_star using LU decomposition
        u_star_inner = scipy.linalg.lu_solve(LU, rhs)

        # Compute u_new
        u_new_inner = u_star_inner + (dt / 2) * (f(u_star_inner) - f(u[1:-1]))

        # Update solution and enforce Dirichlet boundary conditions
        u[1:-1] = u_new_inner
        u[0] = u[-1] = 0  # Dirichlet BCs

    return x, u, dx, dt

# Solve the reaction-diffusion equation numerically for the test case
x, u_num, dx, dt = solve_reaction_diffusion(Nx, Nt, L, T, mu, f=linear_reaction_test)

# Compute the exact solution at final time T
u_exact = exact_solution_test_case(x, T, mu, a, L)

# Compute the L2 error
error = np.linalg.norm(u_num - u_exact, 2) / np.sqrt(Nx)

# Plot numerical vs exact solution
plt.figure(figsize=(8, 5))
plt.plot(x, u_num, label="Numerical Solution", linestyle="dashed")
plt.plot(x, u_exact, label="Exact Solution", linestyle="solid")
plt.xlabel("x")
plt.ylabel("u(x,T)")
plt.legend()
plt.title(f"Numerical vs Exact Solution (Error = {error:.6e})")
plt.show()

# Convergence Analysis: Log-Log Plots for Error vs h and k
Nx_values = [20, 40, 80, 160, 320]  # Different spatial resolutions
Nt_values = [100, 200, 400, 800, 1600]  # Different temporal resolutions

# Compute errors for different spatial resolutions (h-convergence)
errors_h = []
dx_values = []

for Nx in Nx_values:
    x, u_num, dx, _ = solve_reaction_diffusion(Nx, Nt, L=L, T=T, mu=mu, f=linear_reaction_test)
    u_exact = exact_solution_test_case(x, T, mu, a, L)
    
    # Compute L2 error
    error = np.linalg.norm(u_num - u_exact, 2) / np.sqrt(Nx)
    
    errors_h.append(error)
    dx_values.append(dx)

# Compute errors for different time step sizes (k-convergence)
errors_k = []
dt_values = []

for Nt in Nt_values:
    x, u_num, _, dt = solve_reaction_diffusion(Nx=320, Nt=Nt, L=L, T=T, mu=mu, f=linear_reaction_test)
    u_exact = exact_solution_test_case(x, T, mu, a, L)
    
    # Compute L2 error
    error = np.linalg.norm(u_num - u_exact, 2) / np.sqrt(Nx)
    
    errors_k.append(error)
    dt_values.append(dt)

# Compute order of accuracy
order_h = np.polyfit(np.log(dx_values), np.log(errors_h), 1)[0]
order_k = np.polyfit(np.log(dt_values), np.log(errors_k), 1)[0]

# Plot error vs. h (spatial resolution)
plt.figure(figsize=(8, 5))
plt.loglog(dx_values, errors_h, 'o-', label=f"Order ≈ {order_h:.2f}")
plt.xlabel("h (spatial step size)")
plt.ylabel("Global Error")
plt.legend()
plt.title("Convergence Plot: Error vs h")
plt.grid(True)
plt.show()

# Plot error vs. k (time step size)
plt.figure(figsize=(8, 5))
plt.loglog(dt_values, errors_k, 'o-', label=f"Order ≈ {order_k:.2f}")
plt.xlabel("k (time step size)")
plt.ylabel("Global Error")
plt.legend()
plt.title("Convergence Plot: Error vs k")
plt.grid(True)
plt.show()

# Print computed convergence rates
order_h, order_k