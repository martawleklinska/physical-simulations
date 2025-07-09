import graphene
import utils as ult
import kwant
import matplotlib.pyplot as plt
import numpy as np


class TaskManager:
    def __init__(self):
        self.system = graphene.Graphene(sf=16)

    def task1(self):
        system_sf1 = graphene.Graphene(sf=1)
        momenta1, energies1 = system_sf1.get_bands(0, sf=1)
        k1 = momenta1 / (np.sqrt(3))

        system_sf16 = graphene.Graphene(sf=16)
        momenta16, energies16 = system_sf16.get_bands(0, sf=16)
        k16 = momenta16 / (np.sqrt(3))
        energies1 = np.asarray(energies1) / system_sf1.u.eV2au(1.0)
        plt.figure()
        plt.plot(k1, energies1)
        plt.plot(
            k16,
            np.asarray(energies16) / system_sf16.u.eV2au(1.0),
            # label="sf=16",
            linestyle="--",
        )
        plt.xlabel("$k$ [1/nm]", fontsize=14)
        plt.ylabel("$E$ [eV]", fontsize=14)
        plt.ylim(-0.2, 0.2)
        plt.xlim(-0.02, 0.02)
        # plt.legend(["sf=1", "sf=16"], fontsize=14)
        plt.tight_layout()
        plt.savefig("plots/task1_disp_both.pdf")

    def task2(self):
        self.system = graphene.Graphene(sf=16)
        sys = self.system.make_system(
            x_min=-200, x_max=200, y_min=-79.9, y_max=79.9, V_np=0.1, B=0, sf=16
        )

        def site_potential(site):
            if isinstance(site, kwant.builder.Site):
                x, y = site.pos
                return 0.1 * np.tanh(x / self.system.d)
            return 0

        kwant.plot(
            sys,
            site_color=site_potential,
            site_lw=0.1,
            site_size=0.3,
            # cmap="binary",
            show=False,
            colorbar=False,
        )
        plt.xlim(-800, 800)
        plt.ylim(-300, 300)
        plt.savefig("plots/task2_potential_map.pdf")

        momenta, energies = self.system.get_bands_with_B(0, B=1.5, sf=16)
        plt.figure()
        plt.plot(momenta, np.asarray(energies) / self.system.u.eV2au(1.0))
        plt.xlabel("$k$ [1/nm]", fontsize=14)
        plt.ylabel("$E$ [eV]", fontsize=14)
        plt.title("B = 1.5 T, sf = 16")
        plt.ylim(-0.2, 0.2)
        plt.tight_layout()
        plt.savefig("plots/task2_disp_B1_5.pdf")

    def task3(self):
        B_min, B_max, dB = 1.0, 3.0, 0.05
        Bs = np.arange(B_min, B_max + dB, dB)
        E = 0
        conductance = []

        for B in Bs:
            sys = self.system.make_system(
                x_min=-200, x_max=200, y_min=-79.9, y_max=79.9, V_np=0.1, B=B
            )
            smatrix = kwant.smatrix(sys, energy=E)
            T = smatrix.transmission(1, 0)
            G = T
            conductance.append(G)

        plt.figure()
        plt.plot(Bs, conductance, marker="o", markersize=2)
        plt.xlabel("$B$ [T]", fontsize=14)
        plt.ylabel(r"$G \quad [2e^2/h]$", fontsize=14)
        plt.title("Transmission vs Magnetic Field")
        plt.tight_layout()
        plt.savefig("plots/task3_cond_vs_B.pdf")

    def task4(self):
        B_values = [1.9, 2.2, 2.4]
        for B in B_values:
            sys = self.system.make_system(
                x_min=-200, x_max=200, y_min=-79.9, y_max=79.9, V_np=0.1, B=B
            )
            wf = kwant.wave_function(sys, energy=0)
            psi = wf(0)[0]
            current_operator = kwant.operator.Current(sys)
            current = current_operator(psi)

            kwant.plotter.current(
                sys, current, cmap="binary", colorbar=True, show=False
            )
            plt.title(f"B = {B} T")
            plt.savefig(f"plots/task4_current_B{B}.pdf")
