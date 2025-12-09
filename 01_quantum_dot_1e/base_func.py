import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

def get_k(i, j, n = 9):
    return i * n + j

def get_i(k, n=9):
    return k // n

def get_j(k, n = 9):
    return k % n

def get_x(i, a, n = 9):
    dx = 2 * a/(n - 1)
    return -a + dx * i

def get_y(j, a, n = 9):
    dy = 2 * a / (n - 1)
    return -a + dy * j

def gaussian(k1, k2, a1, n1, alpha_x, alpha_y):
    x0 = get_x(get_i(k1, n1), a1, n1)
    y0 = get_y(get_j(k1, n1), a1, n1)
    x1 = get_x(get_i(k2, n1), a1, n1)
    y1 = get_y(get_j(k2, n1), a1, n1)
    normalization_factor = 1/(alpha_x * np.pi)**(1/4) * 1/(alpha_y * np.pi)**(1/4)
    exponent = np.exp(-(x1 - x0)**2/(2 * alpha_x) - (y1 - y0)**2/(2 * alpha_y))
    return normalization_factor*exponent

def matrix_element_S(k_i, k_j, a_s, alpha_x, alpha_y, n = 9):
    x_k = get_x(get_i(k_i, n), a_s, n)
    y_k = get_y(get_j(k_i, n), a_s, n)
    x_l = get_x(get_i(k_j, n), a_s, n)
    y_l = get_y(get_j(k_j, n), a_s, n)
    
    exponent = -(x_k - x_l)**2/(4 * alpha_x) - (y_k - y_l)**2/(4 * alpha_y)
    return np.exp(exponent)

def matrix_S(a, alpha_x, alpha_y, n = 9):
    matrix = np.zeros((n ** 2, n ** 2))
    for i in range(n ** 2):
        for j in range(n ** 2):
            matrix[i, j] = matrix_element_S(i, j, a, alpha_x, alpha_y, n)
    return matrix


def matrix_K(a, alpha_x, alpha_y, m, n = 9):
    matrix = np.zeros((n ** 2, n ** 2))
    S_matrix = matrix_S(a, alpha_x, alpha_y, n)
    for i in range(n ** 2):
        for j in range(n ** 2):
            x_k = get_x(get_i(i, n), a, n)
            y_k = get_y(get_j(i, n), a, n)
            y_l = get_x(get_j(j, n), a, n)
            x_l = get_x(get_i(j, n), a, n)
            matrix[i, j] = S_matrix[i, j] * m / 2 * (alpha_x * (x_k - x_l)**2 + alpha_y * (y_k - y_l)**2)
            
    return matrix

def matrix_V(a, alpha_x, alpha_y, m, omega_x, omega_y, n = 9):
    matrix = np.zeros((n ** 2, n ** 2))
    S_matrix = matrix_S(a, alpha_x, alpha_y, n)
    for i in range(n ** 2):
        for j in range(n ** 2):
            x_k = get_x(get_i(i, n), a, n)
            y_k = get_y(get_j(i, n), a, n)
            x_l = get_x(get_i(j, n), a, n)
            y_l = get_y(get_j(j, n), a, n)
            gaussian_part = (omega_x**2 * ((x_k + x_l)**2 + 2 * alpha_x) / 4 + omega_y**2 * ((y_k + y_l)**2 + 2 * alpha_y) / 4)
            matrix[i, j] = S_matrix[i, j] * m / 2 * gaussian_part
    return matrix

def hamiltonian_matrix(a, alpha_x, alpha_y, m, omega_x, omega_y, n = 9):
    term_matrix_V = matrix_V(a, alpha_x, alpha_y, m, omega_x, omega_y, n)
    term_matrix_K = matrix_K(a, alpha_x, alpha_y, m, n)
    return term_matrix_K + term_matrix_V


def eigen_values(H, S, l, h):
#   vals, vecs= eig(He,b=Se)
#   vals = np.sort(vals)
#   idx = np.argsort(vals)
#   vecs = vecs[:,idx]
  vals = eigh(H, b = S, eigvals_only=True, subset_by_index= [l, h])
  return vals

def eigen_vectors(H, S):
  vals, vecs = eigh(H, b = S)
  return vecs

n = 9
dx = 2 #nm
dx = dx/0.052918
a = (n - 1) * dx / 2
Eh = 27.211
omega_x = 0.08/Eh
omega_y = 0.2/Eh
m = 0.24
alpha_x = 1 / (m * omega_x)
alpha_y =1 / (m * omega_y)
N = int(n**2)
web0 = np.zeros((n,n))
web8 = np.zeros((n,n))
web9 = np.zeros((n,n))


for k in range(N):
  web0[get_i(k, n), get_j(k, n)] = gaussian(0, k, a, n, alpha_x, alpha_y)
  web8[get_i(k, n), get_j(k, n)] = gaussian(8, k, a, n, alpha_x, alpha_y)
  web9[get_i(k, n), get_j(k, n)] = gaussian(9, k, a, n, alpha_x, alpha_y)



fig, axs = plt.subplots(3,1)


plt.sca(axs[0])
plt.contourf(np.transpose(web0),100)
plt.colorbar()

plt.sca(axs[1])
plt.contourf(np.transpose(web8),100)
plt.colorbar()

plt.sca(axs[2])
plt.contourf(np.transpose(web9),100)
plt.colorbar()

plt.show()