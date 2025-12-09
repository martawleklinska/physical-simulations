import kwant
import numpy as np
import utils as utl


class SpinSystem:
    u = utl.Utils()

    def __init__(
        self, *, dx=4, L=int(500), W=int(25), m=0.014, Bx=0, By=1, Bz=0.1, alpha=0.000
    ):
        self.dx = self.u.nm2au(dx)
        self.L = int(L)
        self.W = int(W)
        self.m = m
        self.Bx = self.u.T2au(Bx)
        self.By = self.u.T2au(By)
        self.Bz = self.u.T2au(Bz)
        self.alpha = self.u.nm2au(self.u.eV2au(alpha))

        self.precalc_mag_term = (
            0.5
            * self.u.bohr_magneton_au
            * self.u.lande_g
            * (
                self.Bx * self.u.sigma_x
                + self.By * self.u.sigma_y
                + self.Bz * self.u.sigma_z
            )
        )
        self.precalc_mag_term_By = (
            0.5
            * self.u.bohr_magneton_au
            * self.u.lande_g
            * (self.Bx * self.u.sigma_x + self.Bz * self.u.sigma_z)
        )
        self.precalc_t = 1.0 / (2.0 * self.m * self.dx * self.dx)
        self.precalc_tso = self.alpha / (2 * self.dx)

    def make_system(self):
        def onsite(site):
            (xi, yi) = site.pos
            if xi < self.L * self.dx / 5 or xi > self.L * self.dx * 4 / 5:
                return 4 * self.precalc_t * np.identity(2) + self.precalc_mag_term_By
            return 4 * self.precalc_t * np.identity(2) + self.precalc_mag_term

        def hopping_x(sitei, sitej):
            (xi, yi) = sitei.pos
            (xj, yj) = sitej.pos
            return (
                -self.precalc_t * np.identity(2)
                + 1j * self.precalc_tso * self.u.sigma_y
            )

        def hopping_y(sitei, sitej):
            (xi, yi) = sitei.pos
            (xj, yj) = sitej.pos
            if (xi < self.L * self.dx / 5 or xi > self.L * self.dx * 4 / 5) and (
                xj < self.L * self.dx / 5 or xj > self.L * self.dx * 4 / 5
            ):
                return -self.precalc_t * np.identity(2)
            return (
                -self.precalc_t * np.identity(2)
                - 1j * self.precalc_tso * self.u.sigma_x
            )

        sys = kwant.Builder()
        lat = kwant.lattice.square(self.dx, norbs=2)
        sys[(lat(i, j) for i in range(self.L) for j in range(0, self.W))] = onsite
        sys[(kwant.builder.HoppingKind((-1, 0), lat, lat))] = hopping_x
        sys[(kwant.builder.HoppingKind((0, -1), lat, lat))] = hopping_y

        sigma_law = np.matrix([[1, 0], [0, 2]])

        # attach the left contact to the system
        lead_left = kwant.Builder(
            kwant.TranslationalSymmetry((-self.dx, 0)), conservation_law=sigma_law
        )
        lead_left[(lat(0, j) for j in range(0, self.W))] = onsite
        lead_left[(kwant.builder.HoppingKind((-1, 0), lat, lat))] = hopping_x
        lead_left[(kwant.builder.HoppingKind((0, -1), lat, lat))] = hopping_y
        sys.attach_lead(lead_left)

        lead_right = kwant.Builder(
            kwant.TranslationalSymmetry((self.dx, 0)), conservation_law=sigma_law
        )
        lead_right[(lat(0, j) for j in range(0, self.W))] = onsite
        lead_right[(kwant.builder.HoppingKind((-1, 0), lat, lat))] = hopping_x
        lead_right[(kwant.builder.HoppingKind((0, -1), lat, lat))] = hopping_y
        sys.attach_lead(lead_right)

        sys = sys.finalized()
        return sys

    def dispersion(self, nr_lead, k_max, nk):
        sys = self.make_system()
        momenta = np.linspace(-k_max * self.dx, k_max * self.dx, nk)
        bands = kwant.physics.Bands(sys.leads[nr_lead])
        energies = [bands(k) for k in momenta]
        return (momenta / self.dx) * self.u.nm2au(1.0), energies

    def transmission(self, E):
        E = self.u.eV2au(E)
        sys = self.make_system()
        smatrix = kwant.smatrix(sys, E)
        t = smatrix.transmission(1, 0)
        return t

    def conductance(self, Emax, ne):
        energies = np.linspace(0, Emax, ne)
        cond = [self.transmission(E) for E in energies]
        return energies, cond

    # def transmission_matrix(self, E, orb_num):
    #     E = self.u.eV2au(E)
    #     sys = self.make_system()
    #     smatrix = kwant.smatrix(sys, E)
    #     tup_down = smatrix.transmission((0, 1), (1, 0))
    #     tup_up = smatrix.transmission((0, 1), (1, 1))
    #     tdown_down = smatrix.transmission((0, 1), (1, 1))
    #     tdown_up = smatrix.transmission((0, 1), (1, 0))
    #     return tup_down, tup_up, tdown_down, tdown_up

    def transmission_matrix(self, E):
        E = self.u.eV2au(E)
        sys = self.make_system()
        smatrix = kwant.smatrix(sys, E)

        tup_down = smatrix.transmission((0, 0), (1, 1))
        tup_up = smatrix.transmission((0, 0), (1, 0))
        tdown_down = smatrix.transmission((0, 1), (1, 1))
        tdown_up = smatrix.transmission((0, 1), (1, 0))

        return tup_down, tup_up, tdown_down, tdown_up
