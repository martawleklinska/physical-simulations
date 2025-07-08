import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

## consts
Energy_hartree = 27.211
bohr = 0.0529177
m = 0.24
n = 9


def get_k(i, j, n=9):
    return i * n + j


def get_i(k, n=9):
    return k // n


def get_j(k, n=9):
    return k % n


def get_x(i, a, n=9):
    dx = 2 * a / (n - 1)
    return -a + dx * i


def get_y(j, a, n=9):
    dy = 2 * a / (n - 1)
    return -a + dy * j


def gaussian(k1, k2, a1, n1, alpha_x, alpha_y):
    x0 = get_x(get_i(k1, n1), a1, n1)
    y0 = get_y(get_j(k1, n1), a1, n1)
    x1 = get_x(get_i(k2, n1), a1, n1)
    y1 = get_y(get_j(k2, n1), a1, n1)
    normalization_factor = (
        1 / (alpha_x * np.pi) ** (1 / 4) * 1 / (alpha_y * np.pi) ** (1 / 4)
    )
    exponent = np.exp(
        -((x1 - x0) ** 2) / (2 * alpha_x) - (y1 - y0) ** 2 / (2 * alpha_y)
    )
    return normalization_factor * exponent


def matrix_element_S(k_i, k_j, a_s, alpha_x, alpha_y, n=9):
    x_k = get_x(get_i(k_i, n), a_s, n)
    y_k = get_y(get_j(k_i, n), a_s, n)
    x_l = get_x(get_i(k_j, n), a_s, n)
    y_l = get_y(get_j(k_j, n), a_s, n)

    exponent = -((x_k - x_l) ** 2) / (4 * alpha_x) - (y_k - y_l) ** 2 / (4 * alpha_y)
    return np.exp(exponent)


def matrix_S(a, alpha_x, alpha_y, n=9):
    matrix = np.zeros((n**2, n**2))
    for i in range(n**2):
        for j in range(n**2):
            matrix[i, j] = matrix_element_S(i, j, a, alpha_x, alpha_y, n)
    return matrix


def matrix_kinetic(a, alpha_x, alpha_y, m, n=9):
    matrix = np.zeros((n**2, n**2))
    S_matrix = matrix_S(a, alpha_x, alpha_y, n)
    for i in range(n**2):
        for j in range(n**2):
            x_i = get_x(get_i(i, n), a, n)
            y_i = get_y(get_j(i, n), a, n)
            x_j = get_x(get_i(j, n), a, n)
            y_j = get_y(get_j(j, n), a, n)
            matrix[i, j] = (
                -S_matrix[i, j]
                / (2 * m)
                * (
                    ((x_i - x_j) ** 2 - (2 * alpha_x)) / (4 * alpha_x**2)
                    + ((y_i - y_j) ** 2 - (2 * alpha_y)) / (4 * alpha_y**2)
                )
            )
    return matrix


def matrix_potential(a, alpha_x, alpha_y, m, omega_x, omega_y, n=9):
    matrix = np.zeros((n**2, n**2))
    S_matrix = matrix_S(a, alpha_x, alpha_y, n)
    for i in range(n**2):
        for j in range(n**2):
            xi = get_x(get_i(i, n), a, n)
            yi = get_y(get_j(i, n), a, n)
            xj = get_x(get_i(j, n), a, n)
            yj = get_y(get_j(j, n), a, n)
            matrix[i, j] = (
                S_matrix[i, j]
                * m
                / 2
                * (
                    omega_x**2 * ((xi + xj) ** 2 + (2 * alpha_x)) / 4
                    + omega_y**2 * ((yi + yj) ** 2 + (2 * alpha_y)) / 4
                )
            )
    return matrix


def hamiltonian_matrix(a, alpha_x, alpha_y, m, omega_x, omega_y, n=9):
    term_matrix_V = matrix_potential(a, alpha_x, alpha_y, m, omega_x, omega_y, n)
    term_matrix_K = matrix_kinetic(a, alpha_x, alpha_y, m, n)
    return term_matrix_K + term_matrix_V


def eigen_values(H, S, l, h):
    vals = eigh(H, b=S, eigvals_only=True, subset_by_index=[l, h])
    return vals


def eigen_vectors(H, S):
    vals, vecs = eigh(H, b=S)
    return vecs


dx = 2
dx = dx / bohr
a = (n - 1) * dx / 2
omega_x = 0.08 / Energy_hartree
omega_y = 0.2 / Energy_hartree
alpha_x = 1 / (m * omega_x)
alpha_y = 1 / (m * omega_y)
N = int(n**2)
wf0 = np.zeros((n, n))
wf8 = np.zeros((n, n))
wf9 = np.zeros((n, n))


for k in range(N):
    wf0[get_i(k, n), get_j(k, n)] = gaussian(0, k, a, n, alpha_x, alpha_y)
    wf8[get_i(k, n), get_j(k, n)] = gaussian(8, k, a, n, alpha_x, alpha_y)
    wf9[get_i(k, n), get_j(k, n)] = gaussian(9, k, a, n, alpha_x, alpha_y)


# ex1
fig, axs = plt.subplots(3, 1)

plt.sca(axs[0])
plt.contourf(np.transpose(wf0), 100)
plt.colorbar()
plt.xlabel("$x$ [a.u.]")
plt.ylabel("$y$ [a.u.]")
plt.title("baza")

plt.sca(axs[1])
plt.contourf(np.transpose(wf8), 100)
plt.colorbar()
plt.xlabel("$x$ [a.u.]")
plt.ylabel("$y$ [a.u.]")

plt.sca(axs[2])
plt.contourf(np.transpose(wf9), 100)
plt.colorbar()
plt.xlabel("$x$ [a.u.]")
plt.ylabel("$y$ [a.u.]")
# plt.show()
plt.savefig("density_ex1.png")

omega_x = 0.08 / Energy_hartree
omega_y = 0.2 / Energy_hartree
alpha_x = 1 / (m * omega_x)
alpha_y = 1 / (m * omega_y)
N = int(n**2)
low = 0
high = 9

min_dx = 0.6
max_dx = 5
n_dx = 21
out_dx = np.zeros((10, n_dx))

steps = np.linspace(min_dx, max_dx, n_dx)
index = 0
for dx in steps:
    a = (n - 1) * (dx / 0.052918) / 2

    out_dx[:, index] = eigen_values(
        hamiltonian_matrix(a, alpha_x, alpha_y, m, omega_x, omega_y, n),
        matrix_S(a, alpha_x, alpha_y, n),
        low,
        high,
    )[:10]
    index += 1

out_dx *= Energy_hartree

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylabel("$E$ [eV]")
ax.set_xlabel(r"$\Delta x$ [nm]")
# ax.set_title(r"Exercise 2", fontsize=16)


for l in range(10):
    ax.plot(steps, out_dx[l, :], "-o", label=f"n = {l}")

plt.legend()
plt.savefig("exercise2.png")
# plt.show()

dx = 1
dx = dx / bohr
a = (n - 1) * dx / 2
omega_x = 0.08 / Energy_hartree
omega_y = 0.2 / Energy_hartree
m = 0.24
alpha_x = 1 / (m * omega_x)
alpha_y = 1 / (m * omega_y)
N = int(n**2)


vectors = np.zeros((N, N))

vectors = eigen_vectors(
    hamiltonian_matrix(a, alpha_x, alpha_y, m, omega_x, omega_y, n),
    matrix_S(a, alpha_x, alpha_y, n),
)

functions = np.zeros((6, n, n))
for num in range(6):
    for k_1 in range(N):
        for k_0 in range(N):
            functions[num, get_i(k_1, n), get_j(k_1, n)] += vectors[
                k_0, num
            ] * gaussian(k_0, k_1, a, n, alpha_x, alpha_y)
functions = functions**2


fig, axs = plt.subplots(2, 3, figsize=(13, 7))
fig.suptitle("Exercise 3", fontsize=16)

plt.sca(axs[0, 0])
plt.title("ground state")
plt.contourf(np.transpose(functions[0, :, :]), 100)

plt.sca(axs[0, 1])
plt.title("1st excited state")
plt.contourf(np.transpose(functions[1, :, :]), 100)

plt.sca(axs[0, 2])
plt.title("2nd excited state")
plt.contourf(np.transpose(functions[2, :, :]), 100)

plt.sca(axs[1, 0])
plt.title("3rd excited state")
plt.contourf(np.transpose(functions[3, :, :]), 100)

plt.sca(axs[1, 1])
plt.title(" 4th excited state")
plt.contourf(np.transpose(functions[4, :, :]), 100)

plt.sca(axs[1, 2])
plt.title("5th excited state")
plt.contourf(np.transpose(functions[5, :, :]), 100)

# plt.show()
plt.savefig("exercise3.png")

dx = 1
dx = dx / bohr
a = (n - 1) * dx / 2

omega_y = 0.2 / Energy_hartree
m = 0.24

alpha_y = 1 / (m * omega_y)
N = int(n**2)

out = np.zeros((10, 97))
for omega in range(4, 101):
    omega_x = omega * 0.005 / Energy_hartree
    alpha_x = 1 / (m * omega_x)

    out[:, omega - 4] = eigen_values(
        hamiltonian_matrix(a, alpha_x, alpha_y, m, omega_x, omega_y, n),
        matrix_S(a, alpha_x, alpha_y, n),
        low,
        high,
    )[:10]

out *= Energy_hartree

omegas = np.linspace(20, 500, 97)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylabel("$E$ [eV]")
ax.set_xlabel(r"$\hbar \omega_x$ [meV]")
ax.set_title(r"Exercise 4", fontsize=16)


for l in range(10):
    ax.plot(omegas, out[l, :], label=f"n = {l}")

tab_teor_0 = omegas * (1 / 2) / 1000 + 0.1
ax.plot(omegas, tab_teor_0, "--", label=f"Theoretical", c="k")

tab_teor_1 = omegas * (3 / 2) / 1000 + 0.1
ax.plot(omegas, tab_teor_1, "--", c="k")

tab_teor_2 = omegas * (5 / 2) / 1000 + 0.1
ax.plot(omegas, tab_teor_2, "--", c="k")

plt.legend()

# plt.show()
plt.savefig("exercise4.png")

dx = 0.6  # nm
dx = dx / bohr
a = (n - 1) * dx / 2

omega_x = 0.08 / Energy_hartree
# omega_y = 0.3177/Energy_hartree
omega_y = 0.5 / Energy_hartree
m = 0.24
alpha_x = 1 / (m * omega_x)
alpha_y = 1 / (m * omega_y)
N = int(n**2)


vectors = np.zeros((N, N))

vectors = eigen_vectors(
    hamiltonian_matrix(a, alpha_x, alpha_y, m, omega_x, omega_y, n),
    matrix_S(a, alpha_x, alpha_y, n),
)

functions = np.zeros((6, n, n))
for num in range(6):
    for k_1 in range(N):
        for k_0 in range(N):
            functions[num, get_i(k_1, n), get_j(k_1, n)] += vectors[
                k_0, num
            ] * gaussian(k_0, k_1, a, n, alpha_x, alpha_y)
functions = functions**2


fig, axs = plt.subplots(2, 3, figsize=(13, 7))

fig.suptitle(
    r"Exercise 5; $\hbar \omega_y = $" + f"{omega_y * Energy_hartree * 1000} meV",
    fontsize=16,
)

plt.sca(axs[0, 0])
plt.title("ground state")
plt.contourf(np.transpose(functions[0, :, :]), 100)

plt.sca(axs[0, 1])
plt.title("1st excited state")
plt.contourf(np.transpose(functions[1, :, :]), 100)

plt.sca(axs[0, 2])
plt.title("2nd excited state")
plt.contourf(functions[2, :, :].T, 100)

plt.sca(axs[1, 0])
plt.title("3rd excited state")
plt.contourf(functions[3, :, :].T, 100)

plt.sca(axs[1, 1])
plt.title("4th excited state")
plt.contourf(functions[4, :, :].T, 100)

plt.sca(axs[1, 2])
plt.title("5th excited state")
plt.contourf(functions[5, :, :].T, 100)

# plt.show()
plt.savefig("exercise5.pdf", format="pdf")
