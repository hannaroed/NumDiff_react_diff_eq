# Imports
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple
from tqdm import trange
from scipy.linalg import lapack

# Solver
def solve_banded_lapack(diag: np.ndarray, sub_sup_diag: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """
    Factorize a symmetric positive definite tridiagonal matrix A.
    Return a function that will solve the system Ax = b for a given right-hand side b.
    """
    # From LAPACK docs:
    # DPTTRF computes the L*D*L**T factorization of a real symmetric
    # positive definite tridiagonal matrix A.  The factorization may also
    # be regarded as having the form A = U**T*D*U.

    diag_fact, sub_sup_fact, info = lapack.dpttrf(diag, sub_sup_diag, overwrite_d=0, overwrite_e=0)
    if info != 0:
        raise ValueError("LAPACK error in dpttrf")
    
    def tridiag_solve(b: np.ndarray) -> np.ndarray:
        # From LAPACK docs:
        # DPTTRS solves a tridiagonal system of the form
        #    A * X = B
        # using the L*D*L**T factorization of A computed by DPTTRF.  D is a
        # diagonal matrix specified in the vector D, L is a unit bidiagonal
        # matrix whose subdiagonal is specified in the vector E, and X and B
        # are N by NRHS matrices.

        x, info = lapack.dpttrs(diag_fact, sub_sup_fact, b)
        if info != 0:
            raise ValueError("LAPACK error in dpttrs")
        return x
    
    return tridiag_solve

# Defining parameters
L = 1.0 # Domain size
Nx = 40 # Number of grid points
Nt = 2000 # Number of time steps
T = 1.0 # Final time
mu = 0.001 # Diffusion coefficient
a = 0.5 # Reaction coefficient

def exact_solution_test_case(x: np.ndarray, t: np.ndarray, mu: float, a: float, L: float) -> np.ndarray:
    """
    Exact solution for the test problem: u_t - mu*u_xx = a*u,
    with Dirichlet boundary conditions and initial condition u(x,0) = sin(pi*x/L).
    """
    return np.exp((a - mu * (np.pi**2 / L**2)) * t) * np.sin(np.pi * x / L)

def linear_reaction_test(u):
    """
    Reaction function for the test problem: f(u) = a*u
    """
    return a * u # Linear reaction term

# Solver for reaction-diffusion equation
def solve_reaction_diffusion(Nx: int, Nt: int, L: float, T: float, mu: float, f: Callable[[np.ndarray], np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solves reaction-diffusion equation u_t = mu*u_xx + f(u) using a modified Crank-Nicolson scheme.
    """
    dx = L / Nx # Spatial step size
    dt = T / Nt # Temporal step size
    r = (mu * dt) / (dx**2) # Stability parameter

    x = np.linspace(0, L, Nx+1) # Spatial grid
    t = np.linspace(0, T, Nt+1) # Temporal grid
    u = np.sin(np.pi * x / L) # Initial condition
    
    x_mesh, t_mesh = np.meshgrid(x, t)

    # Construct tridiagonal matrix
    Nx_inner = Nx - 1 # Exclude boundaries
    diag_main = (1 + r) * np.ones(Nx_inner)
    diag_off = (-r / 2) * np.ones(Nx_inner - 1)

    # Perform LDLt factorization once (for efficiency)
    tridiag_solve = solve_banded_lapack(diag_main, diag_off)

    u_final = np.zeros((Nt+1, Nx+1))
    u_final[0] = u # Store initial condition

    # Time stepping
    for n in range(1, Nt+1):
        u_inner = u[1:-1] # Exclude boundaries

        # Predictor Step
        rhs = u_inner + (r / 2) * (u[:-2] - 2 * u_inner + u[2:]) + dt * f(u_inner)
        u_star_inner = tridiag_solve(rhs)

        # Corrector Step
        u_new_inner = u_star_inner + (dt / 2) * (f(u_star_inner) - f(u_inner))

        u[1:-1] = u_new_inner # Update solution
        u[0] = u[-1] = 0  # Dirichlet BCs
        u_final[n] = u # Store solution

    return u_final, x, t, x_mesh, t_mesh

# Solve the reaction-diffusion equation for the test case
u_num, x, t,x_mesh,t_mesh = solve_reaction_diffusion(Nx, Nt, L, T, mu, f=linear_reaction_test)

# Compute the exact solution for the test case
u_exact = exact_solution_test_case(x_mesh, t_mesh, mu, a, L)

# Compute the error
error = np.max(np.abs(u_num[-1] - u_exact[-1]))

# Plot numerical vs exact solution
plt.figure(figsize=(8, 5))
plt.plot(x, u_num[-1], label="Numerical Solution", linestyle="solid")
plt.plot(x, u_exact[-1], label="Exact Solution", linestyle="dashed")
plt.xlabel("x")
plt.ylabel("u(x,T)")
plt.legend()
plt.title(f"Numerical vs Exact Solution (Error = {error:.6e})")
plt.show()

### Convergence Analysis: h and k convergence

Nx_values = np.array([10, 50, 100, 500, 1000], dtype=int) # Different spatial resolutions

# Compute errors for different spatial resolutions (h-convergence)
errors_h = np.zeros_like(Nx_values, dtype=np.float32)

# Iterate over different spatial resolutions
for i in trange(len(Nx_values)):
    u_num1, _, _,x,t = solve_reaction_diffusion(Nx_values[i], Nt=10000, L=L, T=T, mu=mu, f=linear_reaction_test)
    u_exact = exact_solution_test_case(x, t, mu, a, L) 

    error = np.max(np.abs(u_num1 - u_exact)) # Compute max error across all time steps
    errors_h[i] = error

Nt_values = np.array([10, 50, 100, 500, 1000], dtype=int) # Different temporal resolutions

# Compute errors for different time step sizes (k-convergence)
errors_k = np.zeros_like(Nt_values, dtype=np.float32)

# Iterate over different temporal resolutions
for j in trange(len(Nt_values)):
    u_num2, _, _, x, t = solve_reaction_diffusion(Nx=5000, Nt=Nt_values[j], L=L, T=T, mu=mu, f=linear_reaction_test)
    u_exact = exact_solution_test_case(x, t, mu, a, L)

    error = np.max(np.abs(u_num2 - u_exact)) # Compute max error across all time steps
    errors_k[j] = error

# Compute order of accuracy
order_h = np.polyfit(np.log10(T/Nx_values), np.log10(errors_h), 1)[0]
order_k = np.polyfit(np.log10(T/Nt_values), np.log10(errors_k), 1)[0]

# Plot error vs. h (spatial resolution)
plt.figure(figsize=(8, 5))
plt.loglog(1/Nx_values, errors_h, 'o-', label=f"Order ≈ {order_h:.2f}")
plt.xlabel("h (spatial step size)")
plt.ylabel("Global Error")
plt.legend()
plt.title("Convergence Plot: Error vs h")
plt.grid(True)
plt.show()

# Plot error vs. k (time step size)
plt.figure(figsize=(8, 5))
plt.loglog(1/Nt_values, errors_k, 'o-', label=f"Order ≈ {order_k:.2f}")
plt.xlabel("k (time step size)")
plt.ylabel("Global Error")
plt.legend()
plt.title("Convergence Plot: Error vs k")
plt.grid(True)
plt.show()

# Print convergence rates
order_h, order_k