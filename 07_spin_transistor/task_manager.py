import spin_transistor as spin_system
import spin_transistor_ferro as spin_ferro_system

import kwant as kw
import numpy as np
import matplotlib.pyplot as plt
import utils as utl

import plotter as plot_utils

u = utl.Utils()
general_plotter = plot_utils.Plotter()


class TaskManager:
    def task1(self, *, do_plot=False):
        sp = spin_system.SpinSystem()
        sys = sp.make_system()
        kw.plot(sys, show=do_plot, file="plots/ex1_system.pdf")

        moments, enes = sp.dispersion(0, 0.05, 200)
        general_plotter.plot_dispersion(
            moments, enes, filename="ex1_disp.pdf", show=False
        )

        sp = spin_system.SpinSystem(Bx=1)
        moments, enes = sp.dispersion(0, 0.05, 200)
        general_plotter.plot_dispersion(
            moments, enes, filename="ex1_disp_bx_1.pdf", show=False
        )

        sp = spin_system.SpinSystem(By=1)
        moments, enes = sp.dispersion(0, 0.05, 200)
        general_plotter.plot_dispersion(
            moments, enes, filename="ex1_disp_by_1.pdf", show=False
        )

        sp = spin_system.SpinSystem(Bz=1)
        moments, enes = sp.dispersion(0, 0.05, 200)
        general_plotter.plot_dispersion(
            moments, enes, filename="ex1_disp_bz_1.pdf", show=False
        )

        enes, conds = sp.conductance(0.05, 50)
        general_plotter.plot_conductance(
            conds, enes, filename="ex1_conductance_bz_1.pdf", show=False
        )

        # to run for night
        t_bys_values = np.linspace(0, 1, 50)
        E = 0.005
        transmittances = np.zeros((4, len(t_bys_values)), dtype=np.float64)

        for idx, by in enumerate(t_bys_values):
            sp = spin_system.SpinSystem(
                Bz=0.1, additional_By=by, additional_By_boundaries=[0.2, 0.8]
            )
            t_upup = sp.transmission(E, [1, 0], [1, 1])
            t_updown = sp.transmission(E, [1, 0], [0, 1])
            t_downup = sp.transmission(E, [1, 0], [1, 0])
            t_downdown = sp.transmission(E, [1, 0], [0, 0])
            transmittances[0, idx] = t_upup
            transmittances[1, idx] = t_updown
            transmittances[2, idx] = t_downup
            transmittances[3, idx] = t_downdown

        general_plotter.plot_transmittance(
            t_bys_values,
            transmittances,
            show=False,
            filename="ex1_transmittances.pdf",
        )
        sp = spin_system.SpinSystem(
            Bz=0.1, additional_By=0.6, additional_By_boundaries=[0.2, 0.8]
        )
        sys = sp.make_system()
        E = 0.005
        up, down, both, sys = sp.wave_function(E)
        general_plotter.plot_wavefunction(
            sys, up, show=False, filename="ex1_density_up.pdf"
        )
        general_plotter.plot_wavefunction(
            sys, down, show=False, filename="ex1_density_down.pdf"
        )
        general_plotter.plot_wavefunction(
            sys, both, show=False, filename="ex1_density_both.pdf"
        )

        up, down, both, sys = sp.wave_function_spins(E)
        general_plotter.plot_wavefunction(
            sys, up, show=False, filename="ex1_density_x.pdf"
        )
        general_plotter.plot_wavefunction(
            sys, down, show=False, filename="ex1_density_y.pdf"
        )
        general_plotter.plot_wavefunction(
            sys, both, show=False, filename="ex1_density_z.pdf"
        )

    def task2(self, *, do_plot=False):
        sp = spin_system.SpinSystem(
            L=int(200), alpha=0.050, alpha_boundaries=[0.0, 1.0]
        )
        sys = sp.make_system()
        kw.plot(sys, show=do_plot, file="plots/ex2_system.pdf")
        print("calculating dispersion")

        moments, enes = sp.dispersion(0, 0.05, 400)
        general_plotter.plot_dispersion(
            moments, enes, filename="ex2_disp.pdf", show=False
        )
        #
        print("calculating conductance")
        enes, conds = sp.conductance(0.05, 50)
        general_plotter.plot_conductance(
            conds, enes, filename="ex2_conductance.pdf", show=False
        )

        print("calculating transmittance")
        ##to run for night
        alpha_values = np.linspace(0, 0.050, 150)
        E = 0.005
        transmittances = np.zeros((4, len(alpha_values)), dtype=np.float64)

        for idx, alpha_value in enumerate(alpha_values):
            sp = spin_system.SpinSystem(
                L=int(200), alpha=alpha_value, alpha_boundaries=[0.2, 0.8]
            )
            t_upup = sp.transmission(E, [1, 0], [1, 1])
            t_updown = sp.transmission(E, [1, 0], [0, 1])
            t_downup = sp.transmission(E, [1, 0], [1, 0])
            t_downdown = sp.transmission(E, [1, 0], [0, 0])
            transmittances[0, idx] = t_upup
            transmittances[1, idx] = t_updown
            transmittances[2, idx] = t_downup
            transmittances[3, idx] = t_downdown

        general_plotter.plot_transmittance_alpha(
            alpha_values,
            transmittances,
            show=False,
            filename="ex2_transmittances.pdf",
        )

        print("calculating transmission")
        # to run for night
        alpha_values = np.linspace(0, 0.050, 50)
        E = 0.005
        g = np.zeros((3, len(alpha_values)), dtype=np.float64)
        ps = [0.2, 0.4, 1]

        for p in ps:
            for idx, alpha_value in enumerate(alpha_values):
                sp = spin_system.SpinSystem(
                    L=int(200), alpha=alpha_value, alpha_boundaries=[0.2, 0.8]
                )
                t_upup, t_updown, t_downup, t_downdown = sp.transmission_all(E)

                g_up = ((1 + p) / 2) * t_upup + ((1 - p) / 2) * t_updown
                g_down = ((1 + p) / 2) * t_downup + ((1 - p) / 2) * t_downdown
                g[0, idx] = g_up
                g[1, idx] = g_down
                g[2, idx] = ((1 + p) / 2) * g_up + ((1 - p) / 2) * g_down

            general_plotter.plot_g_cond(
                alpha_values, g, show=False, filename=f"ex_2_cond_p={p}.pdf"
            )

        sp = spin_system.SpinSystem(
            L=int(200), alpha=0.018, alpha_boundaries=[0.2, 0.8]
        )
        E = 0.005
        up, down, both, sys = sp.wave_function(E)
        general_plotter.plot_wavefunction(
            sys, up, show=False, filename="ex2_density_up.pdf"
        )
        general_plotter.plot_wavefunction(
            sys, down, show=False, filename="ex2_density_down.pdf"
        )
        general_plotter.plot_wavefunction(
            sys, both, show=False, filename="ex2_density_both.pdf"
        )

        up, down, both, sys = sp.wave_function_spins(E)
        general_plotter.plot_wavefunction(
            sys, up, show=False, filename="ex2_density_x.pdf"
        )
        general_plotter.plot_wavefunction(
            sys, down, show=False, filename="ex2_density_y.pdf"
        )
        general_plotter.plot_wavefunction(
            sys, both, show=False, filename="ex2_density_z.pdf"
        )

    def task3(self, *, do_plot=False):
        # sp = spin_ferro_system.SpinSystem()
        # sys = sp.make_system()
        # kw.plot(sys, show=do_plot, file="plots/ex3_system.pdf")
        # print("calculating dispersion")

        # moments, enes = sp.dispersion(0, 0.4, 400)
        # general_plotter.plot_dispersion(
        #     moments, enes, filename="ex3_disp_04.pdf", show=False
        # )

        ## conductance
        Bext = np.linspace(0.0, 0.1, 50)  # T
        ene = 0.00405  # eV
        ene *= 0.03674932587122423

        cond = []

        for B in Bext:
            sp = spin_ferro_system.SpinSystem(Bext=B)
            sys = sp.make_system()
            smatrix = kw.smatrix(sys, ene)
            t = smatrix.transmission(0, 1)
            cond.append(t)
            print(t)
        plt.plot(Bext, cond)
        plt.xlabel(r"$B_{ext}$ [T]")
        plt.ylabel(r"$G$ [e^2/h]")
        plt.savefig("plots/ex3_conductance.pdf")
