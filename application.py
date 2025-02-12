import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Makes the plots visually better
plt.rcParams["figure.dpi"] = 120
plt.rcParams["figure.figsize"] = (8, 6)

# Setting parameters
L = 10.0  # Domain size
Nx = Ny = 50  # Number of grid points
dx = L / Nx  # Grid spacing
dy = dx # Creating square grid
T = 10  # Total time
dt = 0.01  # Time step
Nt = int(T / dt)  # Number of time steps

beta = 3.0  # Infection rate
gamma = 1.0  # Recovery rate
mu = 0.01  # Diffusion coefficient

# Creating spatial grid
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
X, Y = np.meshgrid(x, y)

# Initializing S, I, R fields
S = np.ones((Nx, Ny))  # Everyone starts susceptible
I = np.zeros((Nx, Ny))  # No initial infections
R = np.zeros((Nx, Ny))  # No initial recoveries

# Introducing initial infection at the center
I[Nx//2, Ny//2] = 0.1
S -= I  # Maintaining S + I + R = 1 constraint

# Creating the 2D Laplacian operator
r = mu * dt / dx**2
main_diag = (1 + 4 * r) * np.ones(Nx * Ny)  # Main diagonal
side_diag = -r * np.ones(Nx * Ny - 1)  # Left/right neighbors
up_down_diag = -r * np.ones(Nx * Ny - Nx)  # Up/down neighbors

laplacian_2D = diags(
    [side_diag, up_down_diag, main_diag, up_down_diag, side_diag], 
    [-1, -Nx, 0, Nx, 1],  # Adding up/down connectivity
    shape=(Nx * Ny, Nx * Ny),
    format="csr" # Compressed Sparse Row format
)

def laplacian(U: np.ndarray) -> np.ndarray:
    '''
    Apply the 2D Laplacian operator correctly.
    
    Parameters:
        U: 2D array, the field to apply the Laplacian to.
    '''
    U_flat = U.ravel()
    U_new = spsolve(laplacian_2D, U_flat)
    return U_new.reshape(Nx, Ny)

# Simulation loop
for t in range(Nt):
    # Compute reaction terms (explicit updates)
    dS = -beta * S * I * dt
    dI = (beta * S * I - gamma * I) * dt
    dR = gamma * I * dt
    
    # Updating values with reaction terms
    S += dS
    I += dI
    R += dR

    # Computing diffusion using Crank-Nicolson (solving linear system)
    I = laplacian(I)

    # Apply Neumann boundary conditions (zero-flux at edges)
    S[0, :], S[-1, :], S[:, 0], S[:, -1] = S[1, :], S[-2, :], S[:, 1], S[:, -2]
    I[0, :], I[-1, :], I[:, 0], I[:, -1] = I[1, :], I[-2, :], I[:, 1], I[:, -2]
    R[0, :], R[-1, :], R[:, 0], R[:, -1] = R[1, :], R[-2, :], R[:, 1], R[:, -2]

    # Visualization at certain time steps
    if t % (Nt // 10) == 0:
        plt.figure()
        plt.imshow(I, cmap="plasma", origin="lower", extent=[0, L, 0, L], interpolation="bicubic")
        plt.colorbar(label="Infected Fraction", shrink=0.8)
        plt.title(f"Infection Spread at t = {t * dt:.2f}", fontsize=14)
        plt.xlabel("x (space)", fontsize=12)
        plt.ylabel("y (space)", fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(False)
        plt.show()