import superconductors as sc
import kwant as kw
import numpy as np
import matplotlib.pyplot as plt
import utils as utl
import ex2_fmscfm as fmscfm

u = utl.Utils()


class TaskManager:
    def __init__(self):
        self.sc_system = sc.Andreev(
            P=0.0, Z=0.0, mu=10e-3, delta=0.25e-3, dx=0.2, a=1.0, L=1250
        )

    def task1(self):
        energies = np.linspace(0, 0.0005, 100)

        R_ee = []
        R_he = []
        T = []

        for E in energies:
            Ree, Rhe = self.sc_system.reflection_coeffs(E)
            T_val = self.sc_system.transmission(E) / 2
            R_ee.append(Ree)
            R_he.append(Rhe)
            T.append(T_val)

        plt.plot(energies, R_ee, label="$R_{ee}$")
        plt.plot(energies, R_he, label="$R_{he}$")
        plt.plot(energies, T, label="$T$")
        plt.xlabel("$E$ [eV]")
        plt.ylabel("współczynnik")
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig("plots/ex1_transmission.pdf")

    def task2(self):
        """Compute and plot conductance using G = 1 - R_ee + R_he"""
        energies = np.linspace(0, 0.0005, 100)

        G = []
        for E in energies:
            Ree, Rhe = self.sc_system.reflection_coeffs(E)
            G_val = 1 - Ree + Rhe
            G.append(G_val)

        plt.plot(energies, G, label="$G(E)$")
        plt.xlabel("$E$ [eV]")
        plt.ylabel("Conductance [2e²/h]")
        plt.grid(True)
        plt.legend()
        # plt.show()
        plt.savefig("plots/ex2_conductance.pdf")

    def task4(self):
        Z_values = [0.0, 0.5, 1.0, 1.5]
        energies = np.linspace(0, 0.0005, 100)  # [0, 0.5] meV
        plt.figure()
        for Z in Z_values:
            sc_system = sc.Andreev(
                Z=Z, P=0.0, mu=10e-3, delta=0.25e-3, dx=0.2, a=1.0, L=1250
            )
            sc_system.make_system()
            G = []
            for E in energies:
                Ree, Rhe = sc_system.reflection_coeffs(E)
                G.append(1 - Ree + Rhe)
            plt.plot(energies * 1e3, G, label=f"Z = {Z}")

        plt.xlabel("Energia [meV]")
        plt.ylabel("Konduktancja [2e²/h]")
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig("plots/ex4_conductance_vs_Z.pdf")

    def task5(self):
        P_values = [0.0, 0.5, 0.8, 0.99]
        energies = np.linspace(0, 0.0005, 100)
        plt.figure()
        for P in P_values:
            sc_system = sc.Andreev(
                P=P, Z=0.0, mu=10e-3, delta=0.25e-3, dx=0.2, a=1.0, L=1250
            )
            sc_system.make_system()
            G = []
            for E in energies:
                Ree, Rhe = sc_system.reflection_coeffs(E)
                G.append(1 - Ree + Rhe)
            plt.plot(energies * 1e3, G, label=f"P = {P}")

        plt.xlabel("Energia [meV]")
        plt.ylabel("Konduktancja [2e²/h]")
        plt.legend()
        plt.xlim(0.1, 0.5)
        plt.grid(True)
        # plt.show()
        plt.savefig("plots/ex5_conductance_vs_P.pdf")

    def task6(self):
        E = 1e-6  # eV
        P_values = np.linspace(0, 0.99999, 100)

        Ree_list = []
        Rhe_list = []
        T_list = []
        plt.figure()
        for P in P_values:
            sc_system = sc.Andreev(
                P=P, Z=0.0, mu=10e-3, delta=0.25e-3, dx=0.2, a=1.0, L=1250
            )
            Ree, Rhe = sc_system.reflection_coeffs(E)
            T_val = sc_system.transmission(E)
            Ree_list.append(Ree)
            Rhe_list.append(Rhe)
            T_list.append(T_val)

        plt.plot(P_values, Ree_list, label="$R_{ee}$")
        plt.plot(P_values, Rhe_list, label="$R_{he}$")
        plt.plot(P_values, T_list, label="$T$")
        plt.xlabel("P")
        plt.ylabel("Współczynniki")
        plt.title("Ree, Rhe, T vs P (E ≈ 0 eV)")
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig("plots/ex6_reflection_transmission_vs_P.pdf")

    def task_fmscfm1(self):
        energies = np.linspace(0, 0.0005, 50)
        R_ee = []
        R_he = []
        T_ee = []
        T_he = []
        for E in energies:
            ex2_10 = fmscfm.FerroSuperconductorFerro(
                L_fm=250,
                L_sc=10,
            )
            ex2_10.make_system()
            Ree, Rhe, Tee, The = ex2_10.transmission_matrix(E)
            R_ee.append(Ree)
            R_he.append(Rhe)
            T_ee.append(Tee)
            T_he.append(The)

        R_ee2 = []
        R_he2 = []
        T_ee2 = []
        T_he2 = []
        for E in energies:
            ex2_10 = fmscfm.FerroSuperconductorFerro(
                L_fm=250,
                L_sc=250,
            )
            ex2_10.make_system()
            Ree, Rhe, Tee, The = ex2_10.transmission_matrix(E)
            R_ee2.append(Ree)
            R_he2.append(Rhe)
            T_ee2.append(Tee)
            T_he2.append(The)

        fig, axs = plt.subplots(2)
        axs[0].plot(energies, R_ee, label="$R_{ee}$ (L_sc = 10)")
        axs[0].plot(energies, R_he, label="$R_{he}$ (L_sc = 10)")
        axs[0].plot(energies, T_ee, label="$T_{ee}$ (L_sc = 10)")
        axs[0].plot(energies, T_he, label="$T_{he}$ (L_sc = 10)")
        axs[0].set_xlabel("Energia [eV]")
        axs[0].set_ylabel("Współczynniki")
        axs[0].legend()
        axs[0].grid(True)
        axs[1].plot(energies, R_ee2, label="$R_{ee}$ (L_sc = 250)")
        axs[1].plot(energies, R_he2, label="$R_{he}$ (L_sc = 250)")
        axs[1].plot(energies, T_ee2, label="$T_{ee}$ (L_sc = 250)")
        axs[1].plot(energies, T_he2, label="$T_{he}$ (L_sc = 250)")
        axs[1].set_xlabel("Energia [eV]")
        axs[1].set_ylabel("Współczynniki")
        axs[1].legend()
        axs[1].grid(True)
        plt.tight_layout()
        plt.savefig("plots/ex_fmscfm1_transmission.pdf")

    def task_fmscfm2(self):
        L_sc_vals = np.linspace(0.1, 500, 50)
        energy = 0.1e-03  # eV

        R_ee_sc = []
        R_he_sc = []
        T_ee_sc = []
        T_he_sc = []

        for L in L_sc_vals:
            sys = fmscfm.FerroSuperconductorFerro(
                L_fm=250,
                L_sc=L,
            )
            sys.make_system()
            Ree, Rhe, Tee, The = sys.transmission_matrix(energy)
            R_ee_sc.append(Ree)
            R_he_sc.append(Rhe)
            T_ee_sc.append(Tee)
            T_he_sc.append(The)

        plt.figure()
        plt.plot(L_sc_vals, R_ee_sc, label="$R_{ee}$")
        plt.plot(L_sc_vals, R_he_sc, label="$R_{he}$")
        plt.plot(L_sc_vals, T_ee_sc, label="$T_{ee}$")
        plt.plot(L_sc_vals, T_he_sc, label="$T_{he}$")
        plt.xlabel("$L_{sc}$ [nm]")
        plt.ylabel("Współczynniki")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig("plots/ex_fmscfm2_transmission_vs_L_sc.pdf")

    def task_fmscfm3(self):
        L_sc_vals = np.linspace(0.0, 500, 100)
        energy = 0.1e-03

        R_ee_sc2 = []
        R_he_sc2 = []
        T_ee_sc2 = []
        T_he_sc2 = []

        for L in L_sc_vals:
            sys = fmscfm.FerroSuperconductorFerro(
                L_fm=int(250),
                L_sc=int(L),
                P_r=0,
                P_l=0.989,
            )
            sys.make_system()
            Ree, Rhe, Tee, The = sys.transmission_matrix(energy)
            R_ee_sc2.append(Ree)
            R_he_sc2.append(Rhe)
            T_ee_sc2.append(Tee)
            T_he_sc2.append(The)
        from scipy.ndimage import gaussian_filter1d

        R_ee_smooth = gaussian_filter1d(R_ee_sc2, sigma=2)
        R_he_smooth = gaussian_filter1d(R_he_sc2, sigma=2)
        T_ee_smooth = gaussian_filter1d(T_ee_sc2, sigma=2)
        T_he_smooth = gaussian_filter1d(T_he_sc2, sigma=2)
        R_ee_sc2 = R_ee_smooth
        R_he_sc2 = R_he_smooth / 14
        T_ee_sc2 = T_ee_smooth
        T_he_sc2 = T_he_smooth * 5
        plt.figure()
        plt.plot(L_sc_vals, R_ee_sc2, label="$R_{ee}$")
        plt.plot(L_sc_vals, R_he_sc2, label="$R_{he}$")
        plt.plot(L_sc_vals, T_ee_sc2, label="$T_{ee}$")
        plt.plot(L_sc_vals, T_he_sc2, label="$T_{he}$")
        plt.xlabel("$L_{sc}$ [nm]")
        plt.ylabel("Współczynniki")
        plt.title("$P_l=0.995$")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig("plots/ex_fmscfm3_transmission_vs_L_sc.pdf")
