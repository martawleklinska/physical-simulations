import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

def get_coords(k, a, n=9):
    dx = 2 * a / (n - 1)
    return -a + dx * (k % n), -a + dx * (k // n)

def gaussian(x0, y0, x1, y1, alpha_x, alpha_y):
    normalization_factor = 1/(alpha_x * np.pi)**(1/4) * 1/(alpha_y * np.pi)**(1/4)
    exponent = np.exp(-(x1 - x0)**2/(2 * alpha_x) - (y1 - y0)**2/(2 * alpha_y))
    return normalization_factor * exponent

def overlap_matrix(n, a, alpha_x, alpha_y):
    coords = np.array([get_coords(i, a, n) for i in range(n**2)])
    X, Y = coords[:, 0], coords[:, 1]
    dX = X[:, None] - X[None, :]
    dY = Y[:, None] - Y[None, :]
    return np.exp(-(dX ** 2) / (4 * alpha_x) - (dY ** 2) / (4 * alpha_y))

def kinetic_energy_matrix(S, n, a, alpha_x, alpha_y, m):
    matrix = np.zeros((n**2, n**2))
    for i in range(n**2):
        for j in range(n**2):
            x_i, y_i = get_coords(i, a, n)
            x_j, y_j = get_coords(j, a, n)
            matrix[i, j] = -S[i, j] / (2 * m) * ((x_i - x_j) ** 2 / (4 * alpha_x ** 2) + (y_i - y_j) ** 2 / (4 * alpha_y ** 2)) 
    return matrix

def potential_energy_matrix(S, n, a, alpha_x, alpha_y, m, omega_x, omega_y):
    matrix = np.zeros((n**2, n**2))
    for i in range(n**2):
        for j in range(n**2):
            x_i, y_i = get_coords(i, a, n)
            x_j, y_j = get_coords(j, a, n)
            matrix[i, j] = S[i, j]* m /2 * (omega_x**2*((x_i + x_j)**2+(2 * alpha_x))/4+ omega_y**2*((y_i + y_j)**2+(2*alpha_y))/4)
    return matrix

def hamiltonian_matrix(S, n, a, alpha_x, alpha_y, m, omega_x, omega_y):
    return kinetic_energy_matrix(S, n, a, alpha_x, alpha_y, m) + potential_energy_matrix(S, n, a, alpha_x, alpha_y, m, omega_x, omega_y)

def compute_eigenvalues(H, S, low, high):
    return eigh(H, b=S, eigvals_only=True, subset_by_index=[low, high])

def compute_eigenvectors(H, S):
    return eigh(H, b=S)[1]

def plot_wavefunctions(vectors, n, a, alpha_x, alpha_y):
    coords = np.array([get_coords(i, a, n) for i in range(n**2)])
    X, Y = coords[:, 0], coords[:, 1]
    functions = np.zeros((6, n, n))
    for num in range(6):
        psi = vectors[:, num]
        for i in range(n):
            for j in range(n):
                x, y = get_coords(i * n + j, a, n)
                functions[num, i, j] = np.sum(psi * gaussian(X, Y, x, y, alpha_x, alpha_y))
    functions **= 2
    
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle("Wavefunctions", fontsize=16)
    for idx, ax in enumerate(axs.flat):
        ax.set_title(f"Square of {idx+1}th eigenfunction")
        ax.contourf(functions[idx].T, 100)
    # plt.show()
    plt.savefig("wavefunctions.png")

# Example parameters
n = 9
dx_values = np.linspace(0.6, 5, 21)
Eh = 27.211
omega_x, omega_y = 0.08 / Eh, 0.2 / Eh
m = 0.24

eigenvalues_dx = np.zeros((10, len(dx_values)))
for i, dx_nm in enumerate(dx_values):
    dx = dx_nm / 0.052918
    a = (n - 1) * dx / 2
    alpha_x, alpha_y = 1 / (m * omega_x), 1 / (m * omega_y)
    S = overlap_matrix(n, a, alpha_x, alpha_y)
    H = hamiltonian_matrix(S, n, a, alpha_x, alpha_y, m, omega_x, omega_y)
    eigenvalues_dx[:, i] = compute_eigenvalues(H, S, 0, 9)

eigenvalues_dx *= Eh
plt.figure(figsize=(13, 7))
for l in range(10):
    plt.plot(dx_values, eigenvalues_dx[l, :], '-o', label=f'n={l}')
plt.xlabel(r'$\Delta x$ [nm]')
plt.ylabel('E [eV]')
plt.title('Exercise 2')
plt.legend()
# plt.show()
plt.savefig("ex2.png")

omega_x_values = np.linspace(20, 500, 97) * 0.005 / Eh
eigenvalues_omega_x = np.zeros((10, len(omega_x_values)))
for i, omega_x in enumerate(omega_x_values):
    alpha_x = 1 / (m * omega_x)
    S = overlap_matrix(n, a, alpha_x, alpha_y)
    H = hamiltonian_matrix(S, n, a, alpha_x, alpha_y, m, omega_x, omega_y)
    eigenvalues_omega_x[:, i] = compute_eigenvalues(H, S, 0, 9)

eigenvalues_omega_x *= Eh
plt.figure(figsize=(13, 7))
for l in range(10):
    plt.plot(omega_x_values * Eh * 1000, eigenvalues_omega_x[l, :], label=f'n={l}')
plt.xlabel(r'$\hbar \omega_x$ [meV]')
plt.ylabel('E [eV]')
plt.title('Exercise 4')
plt.legend()
# plt.show()
plt.savefig("ex4.png")

vectors = compute_eigenvectors(H, S)
plot_wavefunctions(vectors, n, a, alpha_x, alpha_y)