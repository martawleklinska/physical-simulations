import nanowire_utils as nwut
import nanoring_utils as nrut
import kwant as kw
import numpy as np
import matplotlib.pyplot as plt
import os
os.makedirs("plots", exist_ok=True)

import matplotlib
matplotlib.use('Agg')


class TaskManager:
    def task1(self):
        print("Solving first task")
        nw = nwut.NanowireSystem();
        sys = nwut.make_system(nw)
        kw.plot(sys, site_color=lambda site: sys.hamiltonian(site,site), fig_size=(10,5), colorbar=False, show=False, num_lead_cells=2);
        plt.savefig("plots/kwant_system_with_gauss.pdf")
        #plot.show()

        print("Calculating conductance")
        ene, cond = nwut.conductance(nw, 0.2, 50)
        plt.figure()
        plt.plot(ene, cond, color = 'black')
        plt.xlabel("E [eV]")
        plt.ylabel("G [2e^2/h]")
        plt.savefig("plots/condutance.pdf")
        #plt.show()

        print("Plotting wavefunctions and currents")
        fig, axs = plt.subplots(2, 3, figsize=(20, 5), dpi = 300)
        nwut.wave_function(nw, 0.03, 0, ax = axs[0, 0])
        axs[0, 0].set_title("Energy = 0.03 eV", fontsize=22)
        nwut.wave_function(nw, 0.05, 0, ax = axs[0, 1])
        axs[0, 1].set_title("Energy = 0.05 eV", fontsize=22)
        nwut.wave_function(nw, 0.1, 0, ax = axs[0, 2])
        axs[0, 2].set_title("Energy = 0.1 eV", fontsize=22)
        nwut.current(nw, 0.03, 0, 0, ax = axs[1, 0])
        nwut.current(nw, 0.05, 0, 0, ax = axs[1, 1])
        nwut.current(nw, 0.1, 0, 0, ax = axs[1, 2])
        plt.tight_layout()
        plt.savefig("plots/wavefunctions_currents.pdf")
        #plot.show()

    def task2(self):
        print("Solving second task")
        nw = nwut.NanowireSystem(V0 = 0., B = nwut.T2au(2), W = int(20))
        sys = nwut.make_system(nw)
        kw.plot(sys, site_color=lambda site: sys.hamiltonian(site,site), fig_size=(10,5), colorbar=False, show=False, num_lead_cells=2);
        plt.savefig("plots/kwant_ex2_system_without_gauss.pdf")
        #plot.show()

        moments, enes = nwut.disperssion(nw, 0, .1, 200)
        plt.figure();
        plt.plot(moments, np.asarray(enes)/nwut.eV2au(1.0),'k-')
        plt.tick_params(axis='both', which='major', labelsize=22)
        plt.ylim((0,.2))  
        plt.xlim((-0.5,.5))  
        plt.xlabel("k [1/nm]",fontsize=22)
        plt.ylabel("E [eV]",fontsize=22)
        plt.tight_layout()
        plt.savefig("plots/disp_ex2_B2_40nm.pdf")
        #plot.show()

        nw = nwut.NanowireSystem(V0 = 0., B = nwut.T2au(2), W = int(50))
        moments, enes = nwut.disperssion(nw, 0, .1, 200)
        plt.figure();
        plt.plot(moments, np.asarray(enes)/nwut.eV2au(1.0),'k-')
        plt.tick_params(axis='both', which='major', labelsize=22)
        plt.ylim((0,.2))  
        plt.xlim((-0.5,.5))  
        plt.xlabel("k [1/nm]",fontsize=22)
        plt.ylabel("E [eV]",fontsize=22)
        plt.tight_layout()
        plt.savefig("plots/disp_ex2_B2_100nm.pdf")
        #plt.show()

        nw = nwut.NanowireSystem(V0 = 0., B = nwut.T2au(2), W = int(50))
        ene, cond = nwut.conductance(nw, 0.1, 50)
        plt.figure()
        plt.plot(ene, cond, color = "black")
        plt.xlabel("E [eV]")
        plt.ylabel("G [2e^2/h]")
        plt.tight_layout()
        plt.savefig("plots/ex2_condutance.pdf")
        #plt.show()

        #energy slightly above first step - 0.012 eV
        nw = nwut.NanowireSystem(V0 = 0., B = nwut.T2au(2), W = int(50))
        fig, ax = plt.subplots(1, 2, dpi = 300)
        nwut.wave_function(nw, 0.012, 0, ax = ax[0])
        nwut.wave_function(nw, 0.012, 1, ax = ax[1])
        ax[0].set_xlabel("x [a.u.]")
        ax[0].set_ylabel("y [a.u.]")
        ax[1].set_xlabel("x [a.u.]")
        ax[1].set_ylabel("y [a.u.]")
        plt.tight_layout()
        plt.savefig("plots/ex2_wavefunction.pdf")
        #plt.show()

    def task3(self):
        print("solving third task")
        nr = nrut.NanoringSystem(V0 = 0., B = nrut.T2au(0), W = int(15), L = int(50))
        sys = nrut.make_system(nr)
        kw.plot(sys, site_color=lambda site: sys.hamiltonian(site,site), fig_size=(10,5), colorbar=False, show=False, num_lead_cells=2);
        plt.tight_layout()
        plt.savefig("plots/kwant_ex3_ring_system.pdf")
        #plot.show()

        ## ================ ##

        nr = nrut.NanoringSystem(V0 = 0., B = nrut.T2au(0), W = int(15), L = int(50))
        sys = nrut.make_system(nr)
        moments, enes = nrut.disperssion(nr, 0, .1, 200)

        plt.plot(moments, np.asarray(enes)/nwut.eV2au(1.0),'k-')
        plt.tick_params(axis='both', which='major', labelsize=22)
        plt.ylim((0,.2))  
        plt.xlim((-0.5,.5))  
        plt.xlabel("k [1/nm]",fontsize=22)
        plt.ylabel("E [eV]",fontsize=22)
        plt.tight_layout()
        plt.savefig("plots/kwant_ex3_dispersion.pdf")
        #plot.show()

        ## ================ ##

        Bs = np.linspace(nrut.T2au(-10), nrut.T2au(10), 40)
        conds_up = np.zeros(len(Bs));
        conds_down = np.zeros(len(Bs));
        ene = 0.10

        for i in range(len(Bs)):
            nr = nrut.NanoringSystem(V0 = 0, B = Bs[i], W = int(15), L = int(50))
            cond_up = nrut.transmission(nr, ene, 1, 0)
            cond_down = nrut.transmission(nr, ene, 2, 0)
            conds_up[i] = cond_up
            conds_down[i] = cond_down
            
        plt.figure()
        plt.plot(Bs, conds_up, color = 'blue', label = "Lead out 1")
        plt.plot(Bs, conds_down, color = 'red', label = "Lead out 2")
        plt.xlabel("Bz [T]")
        plt.ylabel("G [2e^2/h]")
        plt.legend()
        plt.tight_layout()
        plt.savefig("plots/conductance_ex3.pdf")
        #plot.show()

        ## ================ ##

        fig, axs = plt.subplots(2, 2, dpi = 150)

        B = 5
        nr = nrut.NanoringSystem(V0 = 0, B = nrut.T2au(B), W = int(15), L = int(50))
        nrut.current(nr, 0.03, 0, 0, ax = axs[0, 0])
        axs[0,0].set_title(f"B = {B} T")
        axs[0,0].set_xlabel("x [a.u.]")
        axs[0,0].set_ylabel("y [a.u.]")

        B = 1
        nr = nrut.NanoringSystem(V0 = 0, B = nrut.T2au(B), W = int(15), L = int(50))
        nrut.current(nr, 0.03, 0, 0, ax = axs[0, 1])
        axs[0,1].set_title(f"B = {B} T")
        axs[0,1].set_xlabel("x [a.u.]")
        axs[0,1].set_ylabel("y [a.u.]")

        B = 0
        nr = nrut.NanoringSystem(V0 = 0, B = nrut.T2au(B), W = int(15), L = int(50))
        nrut.current(nr, 0.03, 0, 0, ax = axs[1, 0])
        axs[1,0].set_title(f"B = {B} T")
        axs[1,0].set_xlabel("x [a.u.]")
        axs[1,0].set_ylabel("y [a.u.]")

        B = -5
        nr = nrut.NanoringSystem(V0 = 0, B = nrut.T2au(B), W = int(15), L = int(50))
        nrut.current(nr, 0.03, 0, 0, ax = axs[1, 1])
        axs[1,1].set_title(f"B = {B} T")
        axs[1,1].set_xlabel("x [a.u.]")
        axs[1,1].set_ylabel("y [a.u.]")

        plt.tight_layout()
        plt.savefig("plots/current_ex3.pdf")
        #plot.show()