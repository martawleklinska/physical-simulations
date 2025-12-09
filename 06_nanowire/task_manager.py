import lab6.nanowire_utils as nwut
import lab6.nanoring_utils as nrut
import kwant as kw
import numpy as np
import matplotlib.pyplot as plt


class TaskManager:
    def task1(self):
        print("Solving first task")
        nw = nwut.NanowireSystem()
        sys = nwut.make_system(nw)
        kw.plot(
            sys,
            site_color=lambda site: sys.hamiltonian(site, site),
            fig_size=(10, 5),
            colorbar=False,
            show=False,
            num_lead_cells=2,
        )
        plt.savefig("plots/kwant_system_with_gauss.pdf")
        # plot.show()

        print("Calculating conductance")
        ene, cond = nwut.conductance(nw, 0.2, 50)
        plt.figure()
        plt.plot(ene, cond, color="black")
        plt.xlabel("E [eV]")
        plt.ylabel("G [2e^2/h]")
        plt.savefig("plots/condutance.pdf")
        # plt.show()

        print("Plotting wavefunctions and currents")
        fig, axs = plt.subplots(2, 3, figsize=(20, 5), dpi=300)
        nwut.wave_function(nw, 0.03, 0, ax=axs[0, 0])
        axs[0, 0].set_title("Energy = 0.03 eV", fontsize=22)
        nwut.wave_function(nw, 0.05, 0, ax=axs[0, 1])
        axs[0, 1].set_title("Energy = 0.05 eV", fontsize=22)
        nwut.wave_function(nw, 0.1, 0, ax=axs[0, 2])
        axs[0, 2].set_title("Energy = 0.1 eV", fontsize=22)
        nwut.current(nw, 0.03, 0, 0, ax=axs[1, 0])
        nwut.current(nw, 0.05, 0, 0, ax=axs[1, 1])
        nwut.current(nw, 0.1, 0, 0, ax=axs[1, 2])
        plt.tight_layout()
        plt.savefig("plots/wavefunctions_currents.pdf")
        # plot.show()

    def task2(self):
        print("Solving second task")
        nw = nwut.NanowireSystem(V0=0.0, B=nwut.T2au(2), W=int(20))
        sys = nwut.make_system(nw)
        kw.plot(
            sys,
            site_color=lambda site: sys.hamiltonian(site, site),
            fig_size=(10, 5),
            colorbar=False,
            show=False,
            num_lead_cells=2,
        )
        plt.savefig("plots/kwant_ex2_system_without_gauss.pdf")
        # plot.show()

        moments, enes = nwut.disperssion(nw, 0, 0.1, 200)
        plt.figure()
        plt.plot(moments, np.asarray(enes) / nwut.eV2au(1.0), "k-")
        plt.tick_params(axis="both", which="major", labelsize=22)
        plt.ylim((0, 0.2))
        plt.xlim((-0.5, 0.5))
        plt.xlabel("k [1/nm]", fontsize=22)
        plt.ylabel("E [eV]", fontsize=22)
        plt.tight_layout()
        plt.savefig("plots/disp_ex2_B2_40nm.pdf")
        # plot.show()

        nw = nwut.NanowireSystem(V0=0.0, B=nwut.T2au(2), W=int(50))
        moments, enes = nwut.disperssion(nw, 0, 0.1, 200)
        plt.figure()
        plt.plot(moments, np.asarray(enes) / nwut.eV2au(1.0), "k-")
        plt.tick_params(axis="both", which="major", labelsize=22)
        plt.ylim((0, 0.2))
        plt.xlim((-0.5, 0.5))
        plt.xlabel("k [1/nm]", fontsize=22)
        plt.ylabel("E [eV]", fontsize=22)
        plt.tight_layout()
        plt.savefig("plots/disp_ex2_B2_100nm.pdf")
        # plt.show()

        nw = nwut.NanowireSystem(V0=0.0, B=nwut.T2au(2), W=int(50))
        ene, cond = nwut.conductance(nw, 0.1, 50)
        plt.figure()
        plt.plot(ene, cond, color="black")
        plt.xlabel("E [eV]")
        plt.ylabel("G [2e^2/h]")
        plt.tight_layout()
        plt.savefig("plots/ex2_condutance.pdf")
        # plt.show()

        # energy slightly above first step - 0.012 eV
        nw = nwut.NanowireSystem(V0=0.0, B=nwut.T2au(2), W=int(50))
        fig, ax = plt.subplots(1, 2, dpi=300)
        nwut.wave_function(nw, 0.012, 0, ax=ax[0])
        nwut.wave_function(nw, 0.012, 1, ax=ax[1])
        ax[0].set_xlabel("$x$ [a.u.]")
        ax[0].set_ylabel("$y$ [a.u.]")
        ax[1].set_xlabel("$x$ [a.u.]")
        ax[1].set_ylabel("$y$ [a.u.]")
        plt.tight_layout()
        plt.savefig("plots/ex2_wavefunction.pdf")
        # plt.show()

        ## ======= potencjal rozpraszania na brzegu =======
        nw = nwut.NanowireSystem(V0=nwut.eV2au(0.05), B=nwut.T2au(2), W=int(50))
        ene, cond = nwut.conductance(nw, 0.1, 50)
        plt.figure()
        plt.plot(ene, cond, color="black")
        plt.xlabel("E [eV]")
        plt.ylabel("G [2e^2/h]")
        plt.tight_layout()
        plt.savefig("plots/ex2_condutance_brzeg.pdf")
        # plt.show()
        # energy slightly above first step - 0.012 eV
        fig, ax = plt.subplots(2, 2, dpi=300)
        nwut.wave_function(nw, 0.012, 0, ax=ax[0, 0])
        nwut.wave_function(nw, 0.012, 1, ax=ax[1, 0])

        nwut.current(nw, 0.012, 0, 0, ax=ax[0, 1])
        nwut.current(nw, 0.012, 1, 0, ax=ax[1, 1])
        plt.tight_layout()
        plt.savefig("plots/ex2_wavefunction_brzeg.pdf")
        # plt.show()

    def task3(self):
        print("Solving Task 3")

    ##=== 1. Plot the system ===
    nr = nrut.NanoringSystem(V0=0, B=nrut.T2au(0), W=int(3), L=int(500))
    sys = nrut.make_system(nr)

    kw.plot(
        sys,
        # site_color=lambda site: sys.hamiltonian(site, site),
        fig_size=(10, 5),
        colorbar=False,
        show=False,
        num_lead_cells=2,
    )
    plt.tight_layout()
    plt.savefig("plots/ex3_ring_system.pdf")
    plt.close()

    # === 2. Dispersion relation ===
    nr_disp = nrut.NanoringSystem(V0=0, B=nrut.T2au(0), W=int(3), L=int(500))
    moments, enes = nrut.disperssion(nr_disp, nr_lead=0, k_max=0.1, nk=200)

    plt.figure()
    plt.plot(moments, np.asarray(enes) / nrut.eV2au(1.0), "k-")
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.ylim((0, 0.2))
    plt.xlim((-0.5, 0.5))
    plt.xlabel("$k$ [1/nm]", fontsize=16)
    plt.ylabel("$E$ [eV]", fontsize=16)
    plt.tight_layout()
    plt.savefig("plots/ex3_dispersion.pdf")
    plt.close()

    ene = nrut.eV2au(0.05)  # eV

    # # === 3. Conductance vs magnetic field ===
    Bs = np.linspace(nrut.T2au(0), nrut.T2au(0.01), 200)
    conds = np.zeros(len(Bs))

    for i, B in enumerate(Bs):
        nr = nrut.NanoringSystem(V0=0, B=B, W=int(6), L=int(1000))
        try:
            sys = nrut.make_system(nr)
            smatrix = kw.smatrix(sys, ene)
            conductance = smatrix.transmission(1, 0)
            print(f"B = {B:.2e} | G = {conductance}")
            conds[i] = conductance
        except Exception as e:
            print(f"B = {B:.2e} | Error: {e}")

    plt.figure()
    plt.plot(Bs / nrut.T2au(1.0), conds, color="blue", label="G(B)")
    plt.xlabel("Bz [T]")
    plt.ylabel("G [2e²/h]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/ex3_conductance.pdf")
    plt.close()
    plt.show()

    # # === 4. Wavefunction & current density at min and max conductance ===
    fig, axs = plt.subplots(2, 2, dpi=150)
    B = 0.0007
    nr = nrut.NanoringSystem(V0=0, B=nrut.T2au(B), W=int(6), L=int(1000))
    nrut.current(nr, 0.05, 0, 0, ax=axs[0, 0])
    axs[0, 0].set_title(f"B = {B} T")
    axs[0, 0].set_xlabel("x [a.u.]")
    axs[0, 0].set_ylabel("y [a.u.]")

    B = 0.0018
    nr = nrut.NanoringSystem(V0=0, B=nrut.T2au(B), W=int(6), L=int(1000))
    nrut.current(nr, 0.05, 0, 0, ax=axs[0, 1])
    axs[0, 1].set_title(f"B = {B} T")
    axs[0, 1].set_xlabel("x [a.u.]")
    axs[0, 1].set_ylabel("y [a.u.]")

    B = 0.0007
    nr = nrut.NanoringSystem(V0=0, B=nrut.T2au(B), W=int(6), L=int(1000))
    nrut.wave_function(nr, 0.05, 0, ax=axs[1, 0])
    axs[1, 0].set_title(f"B = {B} T")
    axs[1, 0].set_xlabel("x [a.u.]")
    axs[1, 0].set_ylabel("y [a.u.]")

    B = 0.0018
    nr = nrut.NanoringSystem(V0=0, B=nrut.T2au(B), W=int(6), L=int(1000))
    nrut.wave_function(nr, 0.05, 0, ax=axs[1, 1])
    axs[1, 1].set_title(f"B = {B} T")
    axs[1, 1].set_xlabel("x [a.u.]")
    axs[1, 1].set_ylabel("y [a.u.]")

    plt.tight_layout()
    plt.savefig("plots/current_ex3.pdf")
    # plt.show()
