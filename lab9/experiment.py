import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt
import kwant
import numpy as np
import utils as ult


class Eksperyment:
    u = ult.Utils()

    def __init__(
        self, dx=20, L=int(300), delta=0.2e-03, mu=1.0e-03, Bx=0.0, By=0.0, Bz=0.0
    ):
        self.dx = self.u.nm2au(dx)
        self.mass = 0.014
        self.Bx, self.By, self.Bz = self.u.T2au(Bx), self.u.T2au(By), self.u.T2au(Bz)

        self.L = int(L)
        self.delta = self.u.eV2au(delta)
        self.t = 1.0
        self.mu = self.u.eV2au(mu)
        self.g = -50
        self.alpha = self.u.nm2au(self.u.eV2au(50e-03))

    def make_system(self):
        t0 = 1 / (2 * self.mass * self.dx**2)
        rashba_coeff = self.alpha / (2 * self.dx)
        mu_B = 0.5

        Bx, By, Bz = self.Bx, self.By, self.Bz

        lat = kwant.lattice.chain(self.dx, norbs=4)
        sys = kwant.Builder()

        def onsite(site):
            h_onsite = (2 * t0 - self.mu) * np.kron(self.u.sigma_z, self.u.sigma_0)

            zeeman = (
                0.5
                * self.g
                * mu_B
                * np.kron(
                    self.u.sigma_z,
                    Bx * self.u.sigma_x + By * self.u.sigma_y + Bz * self.u.sigma_z,
                )
            )

            pairing = -self.delta * np.kron(self.u.sigma_y, self.u.sigma_y)

            return h_onsite + zeeman + pairing

        def hopping(site1, site2):
            hop_kinetic = -t0 * np.kron(self.u.sigma_z, self.u.sigma_0)
            hop_soc = -1j * rashba_coeff * np.kron(self.u.sigma_z, self.u.sigma_y)
            return hop_kinetic + hop_soc

        sys[(lat(x) for x in range(self.L))] = onsite
        sys[lat.neighbors()] = hopping

        return sys.finalized()

    def calculate_energies_vs_Bx(self, Bx_values, n_eigs=100):
        energies = []

        for Bx in Bx_values:
            self.Bx = self.u.T2au(Bx)
            sys = self.make_system()
            ham = sys.hamiltonian_submatrix(sparse=True)
            eigs = sla.eigsh(ham, k=n_eigs, sigma=0, return_eigenvectors=False)
            eigs = np.sort(np.real(eigs))
            energies.append(eigs)

        return Bx_values, np.array(energies)

    def calculate_density(self, Bx=1.0):
        self.Bx = self.u.T2au(Bx)
        self.By = 0.0
        self.Bz = 0.0
        sys = self.make_system()
        ham = sys.hamiltonian_submatrix(sparse=True)
        vals, vecs = sla.eigsh(ham, k=20, sigma=0.0)
        idx = np.argmin(np.abs(vals))
        psi = vecs[:, idx]

        density = []
        for i in range(self.L):
            state = psi[i * 4 : (i + 1) * 4]
            prob = np.sum(np.abs(state) ** 2)
            density.append(prob)

        return np.arange(self.L), np.array(density)

    def calculate_energies_vs_theta_xy(self, B0=1.0, n_angles=100, n_eigs=40):
        thetas = np.linspace(0, 2 * np.pi, n_angles)
        B0_au = self.u.T2au(B0)
        energies = []

        for theta in thetas:
            self.Bx = B0_au * np.cos(theta)
            self.By = B0_au * np.sin(theta)
            self.Bz = 0.0
            sys = self.make_system()
            ham = sys.hamiltonian_submatrix(sparse=True)
            eigs = sla.eigsh(ham, k=n_eigs, which="SA", return_eigenvectors=False)
            energies.append(np.sort(np.real(eigs)))

        return thetas, np.array(energies)

    def calculate_energies_vs_theta_xz(self, B0=1.0, n_angles=100, n_eigs=200):
        thetas = np.linspace(0, 2 * np.pi, n_angles)
        B0_au = self.u.T2au(B0)
        energies = []

        for theta in thetas:
            self.Bx = B0_au * np.cos(theta)
            self.By = 0.0
            self.Bz = B0_au * np.sin(theta)
            sys = self.make_system()
            ham = sys.hamiltonian_submatrix(sparse=True)
            eigs = sla.eigsh(ham, k=n_eigs, which="SA", return_eigenvectors=False)
            energies.append(np.sort(np.real(eigs)))

        return thetas, np.array(energies)
