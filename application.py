import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from typing import List, Tuple
from tqdm import trange

# Making plots visually better
plt.rcParams["figure.dpi"] = 120 # High resolution display
plt.rcParams["figure.figsize"] = (8, 6) # Default plot size

# Setting parameters
L = 10.0 # Domain size
Nx = Ny = 50 # Number of grid points
dx = L / Nx # Grid spacing
dy = dx # Ensuring square grid
T = 10 # Total time
dt = 0.01 # Time step
Nt = int(T / dt) # Number of time steps

beta = 3.0 # Infection rate
gamma = 1.0 # Recovery rate
mu_S = 0.01 # Diffusion coefficient for S
mu_I = 0.02 # Diffusion coefficient for I

# Creating spatial grid
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
X, Y = np.meshgrid(x, y)

def initialize_simulation(initial_infections: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initializes the SIR model with given initial infection locations.
    """
    S = np.ones((Nx, Ny)) # Everyone starts susceptible
    I = np.zeros((Nx, Ny)) # No initial infections
    R = np.zeros((Nx, Ny)) # No initial recoveries

    # Introduce infections based on the provided locations
    for loc in initial_infections:
        I[loc] = 0.1 # Sets 10% of the population as infected
        S[loc] -= 0.1 # Ensure S + I + R = 1

    return S, I, R

def create_laplacian(mu: float) -> diags:
    """
    Creates a sparse 2D Laplacian operator for a given diffusion coefficient.
    """
    r = mu * dt / dx**2 # Diffusion coefficient scaling
    main_diag = (1 + 4 * r) * np.ones(Nx * Ny) # Main diagonal
    side_diag = -r * np.ones(Nx * Ny - 1) # Left/right neighbors
    up_down_diag = -r * np.ones(Nx * Ny - Nx) # Up/down neighbors

    return diags(
        [side_diag, up_down_diag, main_diag, up_down_diag, side_diag],
        [-1, -Nx, 0, Nx, 1], # Connectivity
        shape=(Nx * Ny, Nx * Ny),
        format="csr" # Compressed Sparse Row format for efficiency
    )

def laplacian(U: np.ndarray, laplacian_matrix) -> np.ndarray:
    """
    Apply the 2D Laplacian operator with the specified matrix.
    """
    U_flat = U.ravel()
    U_new = spsolve(laplacian_matrix, U_flat)
    return U_new.reshape(Nx, Ny)

def run_simulation(S: np.ndarray, I: np.ndarray, R: np.ndarray, moving_superspreader: bool = False) -> List[np.ndarray]:
    """
    Runs the SIR diffusion model, with separate diffusion for S and I.
    """

    # Create separate Laplacian operators for S and I
    laplacian_S = create_laplacian(mu_S)
    laplacian_I = create_laplacian(mu_I)

    frames = []
    circle_radius = Nx // 5 # Radius of movement
    center_x, center_y = Nx // 2, Ny // 2 # Center of circular path

    for t in trange(Nt):
        # Compute reaction terms (explicit updates)
        dS = -beta * S * I * dt
        dI = (beta * S * I - gamma * I) * dt
        dR = gamma * I * dt

        # Updating values with reaction terms
        S += dS
        I += dI
        R += dR

        if moving_superspreader and t % 10 == 0:
            theta = (t / 20) * (2 * np.pi / (Nt / 20)) # Angle for circular motion
            superspreader_x = int(center_x + circle_radius * np.cos(theta)) % Nx
            superspreader_y = int(center_y + circle_radius * np.sin(theta)) % Ny
            
            I[superspreader_x, superspreader_y] += 0.1
            S[superspreader_x, superspreader_y] -= 0.1

        # Diffusion step (implicit updates)
        S = laplacian(S, laplacian_S) # Apply diffusion for S
        I = laplacian(I, laplacian_I) # Apply diffusion for I

        # Apply Neumann boundary conditions (zero-flux at edges)
        S[0, :], S[-1, :], S[:, 0], S[:, -1] = S[1, :], S[-2, :], S[:, 1], S[:, -2]
        I[0, :], I[-1, :], I[:, 0], I[:, -1] = I[1, :], I[-2, :], I[:, 1], I[:, -2]
        R[0, :], R[-1, :], R[:, 0], R[:, -1] = R[1, :], R[-2, :], R[:, 1], R[:, -2]

        # Store frames for animation
        if t % (Nt // 100) == 0:
            frames.append(I.copy())

    return frames

def show_animation(frames: List[np.ndarray], title: str) -> None:
    """
    Creates an animation and displays it.
    """

    fig, ax = plt.subplots()
    cmap = "inferno"
    img = ax.imshow(frames[0], cmap=cmap, origin="lower", extent=[0, L, 0, L], interpolation="bicubic")
    plt.colorbar(img, ax=ax, label="Infected Fraction", shrink=0.8)
    ax.set_title(title)
    ax.set_xlabel("x (space)")
    ax.set_ylabel("y (space)")

    def update(frame: int):
        img.set_array(frames[frame])
        ax.set_title(f"{title} at t = {frame * (T / len(frames)):.2f}")
        return [img]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=100, blit=False)
    plt.show()

# Baseline model (single initial infection at the center)
S, I, R = initialize_simulation([(Nx//2, Ny//2)])
baseline_frames = run_simulation(S, I, R)
show_animation(baseline_frames, "Baseline Infection Spread (single source)")

# Multiple initial infection model
S, I, R = initialize_simulation([(Nx//4, Ny//4), (3*Nx//4, 3*Ny//4), (Nx//2, Ny//2)])
multi_frames = run_simulation(S, I, R)
show_animation(multi_frames, "Multiple Infection Sources")

# Moving superspreader model
S, I, R = initialize_simulation([])
superspreader_frames = run_simulation(S, I, R, moving_superspreader=True)
show_animation(superspreader_frames, "Moving Superspreader")