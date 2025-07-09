import kwant
import numpy as np
import matplotlib.pyplot as plt
import utils as ult


class Graphene:

    def __init__(self, a_nm=0.25, sf=1.0, t_eV=-3.0):
        self.u = ult.Utils()
        self.sf = sf

        self.a = self.u.nm2au(a_nm * sf)
        self.t0 = self.u.eV2au(t_eV) / sf

        sin_30 = np.sin(np.pi / 6)
        cos_30 = np.cos(np.pi / 6)
        self.d = self.u.nm2au(50)
        self.t = self.u.eV2au(-3)
        self.a0 = self.u.nm2au(0.25) * sf

        self.graphene = kwant.lattice.general(
            [(0, self.a), (cos_30 * self.a, sin_30 * self.a)],
            [(0, 0), (self.a / np.sqrt(3), 0)],
            norbs=1,
        )
        self.a_sub, self.b_sub = self.graphene.sublattices

    def make_system(
        self, x_min=-15, x_max=15, y_min=-12.9, y_max=12.9, V_np=0.0, B=0, sf=16
    ):

        y_min = self.u.nm2au(y_min)
        y_max = self.u.nm2au(y_max)
        x_min = self.u.nm2au(x_min)
        x_max = self.u.nm2au(x_max)
        t0 = self.t / sf
        B = self.u.T2au(B)
        V_np = self.u.eV2au(V_np)

        a0 = self.a0

        sys_gr1 = kwant.Builder()

        graphene = self.graphene

        def rect(pos):
            x, y = pos
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return True
            else:
                return False

        def lead_shape(pos):
            x, y = pos
            return y_min <= y <= y_max

        def potential(site):
            x, y = site.pos
            return V_np * np.tanh(x / self.d)

        def nn_hopping(site1, site2):
            x1, y1 = site1.pos
            x2, y2 = site2.pos
            flux = -B * (y1 + y2) * (x2 - x1) / 2
            return t0 * np.exp(1j * flux)

        sys_gr1[graphene.shape(rect, (0, 0))] = potential
        sys_gr1[graphene.neighbors()] = nn_hopping

        syml = kwant.TranslationalSymmetry([-np.sqrt(3) * a0, 0])
        symr = kwant.TranslationalSymmetry([np.sqrt(3) * a0, 0])

        leadl = kwant.Builder(syml)
        leadl[graphene.shape(lead_shape, (0, 0))] = V_np * np.tanh(x_min / self.d)
        leadl[graphene.neighbors()] = nn_hopping

        leadr = kwant.Builder(symr)
        leadr[graphene.shape(lead_shape, (0, 0))] = V_np * np.tanh(x_max / self.d)
        leadr[graphene.neighbors()] = nn_hopping

        sys_gr1.attach_lead(leadl)
        sys_gr1.attach_lead(leadr)
        sysf = sys_gr1.finalized()
        return sysf

    def get_bands(self, nr_lead=0, sf=1):
        sys = self.make_system(sf=sf)
        dx = np.sqrt(3) * self.a0
        k_max = np.pi / dx
        momenta = np.linspace(-k_max * dx, k_max * dx, 200)
        bands = kwant.physics.Bands(sys.leads[nr_lead])
        energies = [bands(k) for k in momenta]
        return (momenta / dx), energies

    def get_bands_with_B(self, nr_lead, B, sf):
        a0 = self.u.nm2au(0.25) * sf
        dx = np.sqrt(3) * a0
        k_max = np.pi / dx
        sys = self.make_system(
            x_min=-200, x_max=200, y_min=-79.9, y_max=79.9, V_np=0.1, B=B, sf=sf
        )
        momenta = np.linspace(-k_max * dx, k_max * dx, 200)
        bands = kwant.physics.Bands(sys.leads[nr_lead])
        energies = [bands(k) for k in momenta]
        return (momenta / dx), energies
