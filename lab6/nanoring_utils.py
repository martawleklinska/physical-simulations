import kwant
import numpy as np
import matplotlib.pyplot as plt


# the calculations are done in atomic units e=h=me=1. Here the conversion factors are defined.
def eV2au(energy):  # eV -> j.a
    return energy * 0.03674932587122423


def au2eV(energy):  # j.a -> eV
    return energy * 27.2117


def nm2au(length):  # nm -> j.a
    return length * 18.89726133921252


def T2au(length):  # Tesla -> j.a
    return length * 4.254382e-6


class NanoringSystem:
    def __init__(
        self,
        *,
        dx=nm2au(2),
        L=1250,
        W=7.5,
        m=0.014,
        V0=eV2au(0.05),
        sigma=nm2au(1),
        B=T2au(0),
        R1=nm2au(600),
        R2=nm2au(630)
    ):
        self.dx = dx
        self.L = int(L)
        self.W = int(W)
        self.m = m
        self.V0 = V0
        self.sigma = sigma
        self.B = B
        self.R1 = R1
        self.R2 = R2

        def ring_with_two_leads(pos):
            x, y = pos
            rsq = x**2 + y**2

            if R1**2 < rsq < R2**2:
                return True

            if (-L * dx / 2 < x < -R2 + 2 * dx) and (-W * dx < y < W * dx):
                return True

            if (R2 - 2 * dx < x < L / 2 * dx) and (-W * dx < y < W * dx):
                return True

            return False

        self.ring_function = ring_with_two_leads


def make_system(nr):
    dx = nr.dx
    m = nr.m
    V0 = nr.V0
    sigma = nr.sigma
    x0 = 0.0
    y0 = 0.0
    B = nr.B

    def pot(x, y):
        return V0 * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / sigma**2)

    def onsite(site):
        (x, y) = site.pos
        t = 1.0 / (2.0 * m * dx * dx)
        return 4 * t  # + pot(x, y)

    def hopping_x(sitei, sitej):
        (xi, yi) = sitei.pos
        (xj, yj) = sitej.pos
        t = 1.0 / (2.0 * m * dx * dx) * np.exp(-1.0j / 2 * B * (xi - xj) * (yi + yj))
        return -t

    def hopping_y(sitei, sitej):
        (xi, yi) = sitei.pos
        (xj, yj) = sitej.pos
        t = 1.0 / (2.0 * m * dx * dx) * np.exp(-1.0j / 2 * B * (xi - xj) * (yi + yj))
        return -t

    lat = kwant.lattice.square(dx, norbs=1)
    syst = kwant.Builder()

    start_point = ((nr.R1 + nr.R2) / 2, 0)
    syst[lat.shape(nr.ring_function, start_point)] = onsite

    syst[(kwant.builder.HoppingKind((-1, 0), lat, lat))] = hopping_x
    syst[(kwant.builder.HoppingKind((0, -1), lat, lat))] = hopping_y

    lead_left = kwant.Builder(kwant.TranslationalSymmetry((-dx, 0)))
    lead_left[(lat(0, j) for j in range(-nr.W, nr.W))] = onsite
    lead_left[kwant.builder.HoppingKind((-1, 0), lat, lat)] = hopping_x
    lead_left[kwant.builder.HoppingKind((0, -1), lat, lat)] = hopping_y

    syst.attach_lead(lead_left)
    lead_right = kwant.Builder(kwant.TranslationalSymmetry((dx, 0)))
    lead_right[(lat(0, j) for j in range(-nr.W, nr.W))] = onsite
    lead_right[kwant.builder.HoppingKind((-1, 0), lat, lat)] = hopping_x
    lead_right[kwant.builder.HoppingKind((0, -1), lat, lat)] = hopping_y
    syst.attach_lead(lead_right)

    return syst.finalized()


def disperssion(nr, nr_lead, k_max, nk):
    dx = nr.dx
    sys = make_system(nr)
    momenta = np.linspace(-k_max * dx, k_max * dx, nk)
    bands = kwant.physics.Bands(sys.leads[nr_lead])
    energies = [bands(k) for k in momenta]
    return (momenta / dx) * nm2au(1.0), energies


def transmission(nr, E, lead_in, lead_out):
    E = eV2au(E)
    sys = make_system(nr)
    smatrix = kwant.smatrix(sys, E)
    t = smatrix.transmission(lead_in, lead_out)
    return t


def conductance(nw, Emax, ne, lin, lout):
    energies = np.linspace(0, Emax, ne)
    cond = [transmission(nw, E, lin, lout) for E in energies]
    return energies, cond


def wave_function(nr, E, nr_lead, *, ax=None):
    E = eV2au(E)
    sys = make_system(nr)
    wave = kwant.wave_function(sys, E)
    density = (abs(wave(nr_lead)) ** 2).sum(axis=0)
    kwant.plotter.map(sys, density, ax=ax, dpi=300)


def current(nr, E, nr_lead, nr_mod, *, ax=None):
    E = eV2au(E)
    sys = make_system(nr)
    current = kwant.operator.Current(sys).bind()
    psi = kwant.wave_function(sys, E)(nr_lead)
    curr = current(psi[nr_mod])
    kwant.plotter.current(sys, curr, ax=ax)
