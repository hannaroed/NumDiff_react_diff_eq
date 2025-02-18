# Imports
import numpy as np
import matplotlib.pyplot as plt

# Defining parameters
L = 1.0 # Domain size
Nx = 50 # Number of grid points
Nt = 1000 # Number of time steps
T = 0.1 # Final time
mu = 0.1 # Diffusion coefficient

dx = L / (Nx - 1) # Spatial step size
dt = T / Nt # Time step size
r = mu * dt / dx**2 # Stability parameter

# Defining grid
x = np.linspace(0, L, Nx)
t = np.linspace(0, T, Nt)

# Initial condition u(x,0) = sin(pi*x)
u = np.sin(np.pi * x)

# Exact solution of the heat equation
def exact_solution(x: np.ndarray, t: float, mu: float) -> np.ndarray:
    return np.exp(-mu * np.pi**2 * t) * np.sin(np.pi * x)

# Time stepping using the numerical scheme (Implicit-Explicit Crank-Nicolson)
u_new = np.copy(u) # Temporary array for updated solution
for n in range(1, Nt):
    u_star = u + (r / 2) * (np.roll(u, -1) - 2*u + np.roll(u, 1))
    u_new = u_star + (dt / 2) * (np.sin(np.pi*x) - np.sin(np.pi*x))
    u_new[0] = u_new[-1] = 0 # Enforces boundary conditions
    u[:] = u_new # Updates solution

# Computing exact solution at final time
u_exact = exact_solution(x, T, mu)

# Computing the error
error = np.linalg.norm(u - u_exact, 2) # L2 norm

# Plotting results
plt.figure(figsize=(8,5))
plt.plot(x, u, label="Numerical Solution", linestyle="dashed")
plt.plot(x, u_exact, label="Exact Solution", linestyle="solid")
plt.xlabel("x")
plt.ylabel("u(x,T)")
plt.legend()
plt.title(f"Numerical vs Exact Solution (Error = {error:.6e})")
plt.show()

### Questions for student assistant:
# 1. Skal man verifisere bare convergence, eller heller vise consistency og stability og så si at det impliserer convergence (Lax equivalence)?
# 2. Evt om man skal plotte h, k -> 0 mot LTE -> 0 ?
# 3. Evt plotte noe med a og se når a er utenfor stabilitet?