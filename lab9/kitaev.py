import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt
import kwant
import numpy as np
import utils as ult


class Kitaev:
    u = ult.Utils()

    def __init__(self, dx=1.0, L=int(25), delta=1.0, t=1.0, mu=1.0):
        self.dx = dx
        self.L = int(L)
        self.delta = delta
        self.t = 1.0
        self.mu = mu

    def make_system(self):
        def onsite(site):
            return -self.mu * self.u.sigma_z

        def hopping(sitei, sitej):
            i, j = sitei.tag[0], sitej.tag[0]
            sign = np.sign(j - i)
            return -self.t * self.u.sigma_z + 1j * sign * self.delta * self.u.sigma_y

        sys = kwant.Builder()
        lat = kwant.lattice.chain(self.dx, norbs=2)

        sys[(lat(i) for i in range(self.L))] = onsite
        sys[lat.neighbors()] = hopping

        sys = sys.finalized()
        return sys

    def calculate_energies(self, mu_range):
        energies = []
        mu_values = []

        for mu in mu_range:
            self.mu = mu
            sys = self.make_system()
            ham_mat = sys.hamiltonian_submatrix()

            e, _ = np.linalg.eigh(ham_mat)
            e = np.real(e)
            e.sort()
            energies.append(e)
            mu_values.append(mu)

        return mu_values, energies

    def calculate_density(self, mu):
        self.mu = mu
        sys = self.make_system()
        ham_mat = sys.hamiltonian_submatrix()
        e, vecs = np.linalg.eigh(ham_mat)
        idx = np.argmin(np.abs(e))
        psi = vecs[:, idx]

        density = []
        for i in range(self.L):
            psi_e = psi[2 * i]
            psi_h = psi[2 * i + 1]
            density.append(np.abs(psi_e) ** 2 + np.abs(psi_h) ** 2)

        return np.arange(self.L), np.array(density)
