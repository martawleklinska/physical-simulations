from kitaev import Kitaev
import scipy.sparse.linalg as sla
from utils import Utils
import numpy as np
import matplotlib.pyplot as plt
import kwant
from experiment import Eksperyment

u = Utils()


class TaskManager:
    def __init__(self):
        self.system = Kitaev()

    def task1(self):

        mu_range = np.linspace(0.0, 4 * 1.0, 100)
        kitaev = Kitaev()
        mu_values, energies = kitaev.calculate_energies(mu_range)
        plt.figure(figsize=(8, 6))
        for e in np.array(energies).T:
            plt.plot(mu_values, e, "b-", alpha=0.5)

        plt.xlabel("$\mu/t$")
        plt.ylabel("$E/t$")
        plt.legend()
        plt.grid()
        # plt.show()
        plt.savefig("plots/ex1_spectrum.pdf")
        plt.close()

        ## //// density
        plt.figure()
        x, dens = self.system.calculate_density(mu=1.0)
        plt.plot(x, dens)
        plt.title(r"$|\psi|^2$ for $\mu = t$")
        plt.xlabel("$x$")
        plt.ylabel(r"$|\psi|^2$")
        # plt.show()
        plt.savefig("plots/ex1_density.pdf")

        plt.close()
        plt.figure()
        system = Kitaev(mu=4.0)
        x, dens = system.calculate_density(mu=4.0)

        plt.plot(x, dens)
        plt.title(r"$|\psi|^2$ for $\mu = t$")
        plt.xlabel("$x$")
        plt.ylabel(r"$|\psi|^2$")
        # plt.show()
        plt.savefig("plots/ex1_density4t.pdf")
        plt.close()

    def task2(self):
        exp = Eksperyment()
        Bx_vals = np.linspace(0, 2.0, 100)
        Bx, energies = exp.calculate_energies_vs_Bx(Bx_vals, n_eigs=200)

        plt.figure(figsize=(8, 6))
        for e in energies.T:
            e = u.au2eV(e)
            plt.plot(Bx, e * 1e03, "b-", alpha=0.5)
        plt.xlabel(r"$B_x$ [T]")
        plt.ylabel("$E$ [meV]")
        plt.title("task2")
        plt.grid()
        plt.ylim(-1.1, 1.1)
        plt.savefig("plots/task2_spectrum_vs_Bx.pdf")
        plt.close()

    def task3(self):
        exp = Eksperyment()
        x, dens = exp.calculate_density(Bx=1.0)

        plt.figure()
        plt.plot(x, dens)
        plt.title(r"$B_x = 1$ T")
        plt.xlabel("Site")
        plt.ylabel(r"$|\psi|^2$")
        plt.grid()
        plt.savefig("plots/task3_density_Bx1.pdf")
        plt.close()

    def task4(self):
        exp = Eksperyment()
        thetas = np.linspace(0, 2 * np.pi, 40)
        B0_au = exp.u.T2au(1.0)
        energies = []

        for theta in thetas:
            Bx = B0_au * np.cos(theta)
            By = B0_au * np.sin(theta)
            Bz = 0.0
            exp = Eksperyment(Bx=exp.u.au2T(Bx), By=exp.u.au2T(By), Bz=exp.u.au2T(Bz))
            sys = exp.make_system()
            ham = sys.hamiltonian_submatrix(sparse=True)
            eigs = sla.eigsh(ham, k=100, sigma=0.0, return_eigenvectors=False)
            eigs = np.sort(np.real(eigs)) * exp.u.au2eV(1) * 1e3
            energies.append(eigs)

        energies = np.array(energies)

        plt.figure(figsize=(8, 6))
        for e in energies.T:
            plt.plot(thetas, e, "b-", alpha=0.5)
        plt.xlabel(r"$\theta$ [rad]")
        plt.ylabel("$E$ [meV]")
        plt.ylim(-1, 1)
        plt.grid()
        plt.savefig("plots/task4_spectrum_vs_theta_xy.pdf")
        plt.close()

    def task5(self):
        thetas = np.linspace(0, 2 * np.pi, 40)
        B0_au = u.T2au(1.0)
        energies = []

        for theta in thetas:
            Bx = B0_au * np.cos(theta)
            By = 0.0
            Bz = B0_au * np.sin(theta)
            exp = Eksperyment(Bx=u.au2T(Bx), By=u.au2T(By), Bz=u.au2T(Bz))
            sys = exp.make_system()
            ham = sys.hamiltonian_submatrix(sparse=True)
            eigs = sla.eigsh(ham, k=100, sigma=0.0, return_eigenvectors=False)
            eigs = np.sort(np.real(eigs)) * u.au2eV(1) * 1e3
            energies.append(eigs)

        energies = np.array(energies)

        plt.figure(figsize=(8, 6))
        for e in energies.T:
            plt.plot(thetas, e, "b-", alpha=0.5)
        plt.xlabel(r"$\theta$ [rad]")
        plt.ylabel("$E$ [meV]")
        plt.ylim(-1, 1)
        plt.grid()
        plt.savefig("plots/task5_spectrum_vs_theta_xz.pdf")
        plt.close()
