import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Setting up high-quality plots
plt.rcParams["figure.dpi"] = 120
plt.rcParams["figure.figsize"] = (8, 6)

# Setting parameters
L = 10.0  # Domain size
Nx = Ny = 50  # Number of grid points
dx = L / Nx  # Grid spacing
dy = dx  # Creating square grid
T = 10  # Total time
dt = 0.01  # Time step
Nt = int(T / dt)  # Number of time steps

# Epidemiological parameters
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
    format="csr"  # Compressed Sparse Row format
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

# Prepare animation data storage
frames = []

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

    # Store frames for animation (every 10 time steps to reduce lag)
    if t % (Nt // 100) == 0:  # Store frames at regular intervals
        frames.append(I.copy())

# Create the interactive animation
fig, ax = plt.subplots()
cmap = "plasma"
img = ax.imshow(frames[0], cmap=cmap, origin="lower", extent=[0, L, 0, L], interpolation="bicubic")
plt.colorbar(img, ax=ax, label="Infected Fraction", shrink=0.8)
ax.set_title("Infection Spread Over Time")
ax.set_xlabel("x (space)")
ax.set_ylabel("y (space)")

def update(frame):
    img.set_array(frames[frame])
    ax.set_title(f"Infection Spread at t = {frame * (T / len(frames)):.2f}")
    return [img]

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=100, blit=False)
plt.show()