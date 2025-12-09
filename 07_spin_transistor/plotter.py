import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import get_cmap
import numpy as np
import utils as u
import kwant as kw


class Plotter:
    def __init__(self):
        ## set options inside plotter class

        # rcParams["text.usetex"] = True
        # rcParams["font.family"] = "serif"
        # rcParams["font.serif"] = "Computer Modern Roman"
        rcParams["font.size"] = 14
        # rcParams["text.latex.preamble"] = r"\input{~/Desktop/Studia/LaTeX/setup.tex}"
        ## Lines
        rcParams["lines.solid_joinstyle"] = "miter"  # other options: 'round' or 'bevel'
        rcParams["lines.antialiased"] = (
            True  # turning on/off of antialiasing for sharper edges
        )
        rcParams["lines.linewidth"] = 1.25

        ## Legend
        rcParams["legend.loc"] = "upper left"
        rcParams["legend.frameon"] = False

        ## Ticks
        rcParams["xtick.direction"] = "in"
        rcParams["ytick.direction"] = "in"
        rcParams["xtick.top"] = True
        rcParams["ytick.right"] = True

        rcParams["xtick.minor.visible"] = True
        rcParams["ytick.minor.visible"] = True

        ## Resolution
        rcParams["figure.dpi"] = 150

        ## Global color
        rcParams["image.cmap"] = "viridis"

        ## Colors
        ### cmaps
        self.cm_inferno = get_cmap("inferno")
        self.cm_viridis = get_cmap("viridis")
        self.cm_seismic = get_cmap("seismic")
        self.cm_jet = get_cmap("jet")
        self.cm_tab10 = get_cmap("tab10")

        ### Palettes from color-hex.com/
        self.c_google = [
            "#008744",
            "#0057e7",
            "#d62d20",
            "#ffa700",
        ]  # G, B, R, Y # https://www.color-hex.com/color-palette/1872
        self.c_twilight = [
            "#363b74",
            "#673888",
            "#ef4f91",
            "#c79dd7",
            "#4d1b7b",
        ]  # https://www.color-hex.com/color-palette/809

        ### unit hanlder for the project
        self.unit_handler = u.Utils()

    def plot_dispersion(self, moments, enes, *, filename=None, show=False):
        fig = plt.figure()
        plt.plot(moments, np.asarray(enes) / self.unit_handler.eV2au(1.0), "k-")
        plt.tick_params(axis="both", which="major", labelsize=22)
        plt.ylim((0, 0.05))
        plt.xlim((-0.4, 0.4))
        plt.xlabel(r"$k$ [nm$^{-1}$]", fontsize=22)
        plt.ylabel(r"$E$ [eV]", fontsize=22)
        plt.tight_layout()

        if filename != None:
            plt.savefig("plots/" + filename)

        if show:
            plt.show()

        plt.close(fig)

    def plot_conductance(self, conds, enes, *, filename=None, show=False):
        fig = plt.figure()
        plt.plot(enes, conds, color="black")
        plt.xlabel(r"$E$ [eV]")
        plt.ylabel(r"Conductance $[\frac{e^2}{h}]$")

        if filename != None:
            plt.savefig("plots/" + filename)

        if show:
            plt.show()

        plt.close(fig)

    def plot_transmittance(
        self, bys_values, transmittances, *, filename=None, show=False
    ):
        fig = plt.figure()
        plt.plot(
            bys_values,
            transmittances[0, :],
            color="blue",
            linestyle="-",
            label=r"$T_{up\rightarrow up}$",
        )
        plt.plot(
            bys_values,
            transmittances[1, :],
            color="orange",
            linestyle="--",
            label=r"$T_{up\rightarrow down}$",
        )
        plt.plot(
            bys_values,
            transmittances[2, :],
            color="green",
            linestyle="-",
            label=r"$T_{down\rightarrow up}$",
        )
        plt.plot(
            bys_values,
            transmittances[3, :],
            color="red",
            linestyle="--",
            label=r"$T_{down\rightarrow down}$",
        )
        plt.xlabel(r"$E$ [eV]")
        plt.ylabel(r"Transmittance")
        plt.legend()

        if filename != None:
            plt.savefig("plots/" + filename)

        if show:
            plt.show()

        plt.close(fig)

    def plot_transmittance_alpha(
        self, alpha_values, transmittances, *, filename=None, show=False
    ):
        fig = plt.figure()
        plt.plot(
            alpha_values,
            transmittances[0, :],
            color="blue",
            linestyle="-",
            label=r"$T_{up\rightarrow up}$",
        )
        plt.plot(
            alpha_values,
            transmittances[1, :],
            color="orange",
            linestyle="--",
            label=r"$T_{up\rightarrow down}$",
        )
        plt.plot(
            alpha_values,
            transmittances[2, :],
            color="green",
            linestyle="-",
            label=r"$T_{down\rightarrow up}$",
        )
        plt.plot(
            alpha_values,
            transmittances[3, :],
            color="red",
            linestyle="--",
            label=r"$T_{down\rightarrow down}$",
        )
        plt.xlabel(r"$\alpha$ [eVnm]")
        plt.ylabel(r"Transmittance")

        plt.tight_layout()
        plt.legend()

        if filename != None:
            plt.savefig("plots/" + filename)

        if show:
            plt.show()

        plt.close(fig)

    def plot_wavefunction(self, sys, dens, *, filename=None, show=False):
        fig = plt.figure()
        ax = plt.axes()
        kw.plotter.map(sys, dens, ax=ax, cmap="RdBu", colorbar=True)

        if filename != None:
            plt.savefig("plots/" + filename)

        if show:
            plt.show()

        plt.close(fig)

    def plot_current(self, sys, current, *, filename=None, show=False):
        fig = plt.figure()
        ax = plt.axes()
        kw.plotter.current(sys, current, ax=ax)

        if filename != None:
            plt.savefig("plots/" + filename)

        if show:
            plt.show()

        plt.close(fig)

    def plot_g_cond(self, bys_values, transmittances, *, filename=None, show=False):
        fig = plt.figure()
        plt.plot(
            bys_values,
            transmittances[0, :],
            color="blue",
            linestyle="-",
            label=r"$G_{up}$",
        )
        plt.plot(
            bys_values,
            transmittances[1, :],
            color="red",
            linestyle="-",
            label=r"$G_{down}$",
        )
        plt.plot(
            bys_values,
            transmittances[2, :],
            color="green",
            linestyle="-",
            label=r"$G_{total}$",
        )
        plt.xlabel(r"$\alpha$ [eVnm]")
        plt.ylabel(r"Conductance $[\frac{e^2}{h}]$")
        plt.legend()
        plt.tight_layout()

        if filename != None:
            plt.savefig("plots/" + filename)

        if show:
            plt.show()

        plt.close(fig)
