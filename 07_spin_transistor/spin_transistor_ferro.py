import kwant
import tinyarray
import numpy as np
import utils as utl


class SpinSystem:
    u = utl.Utils()

    def __init__(
        self,
        *,
        dx=1,
        L=int(1000),
        W=int(30),
        m=0.1,
        Bext=0.0,
        g=200,
    ):
        self.dx = self.u.nm2au(dx)
        self.L = int(L)
        self.W = int(W)
        self.m = m
        self.Bext = self.u.T2au(Bext)
        self.g = g

        self.Bh = self.u.T2au(0.05)
        self.a = L * self.dx
        self.x0 = L * self.dx / 2

        self.precalc_mag_term = (
            0.5 * self.u.bohr_magneton_au * self.g * self.Bext * self.u.sigma_z
        )

        self.precalc_t = 1.0 / (2.0 * self.m * self.dx * self.dx)

    def make_system(self):
        def onsite(site):
            (x1, y1) = site.pos
            value = 4 * self.precalc_t * np.identity(2) + self.precalc_mag_term

            magnetic_term = (
                self.Bh * np.sin(2 * np.pi * (x1 - self.x0) / self.a) * self.u.sigma_x
            ) + self.Bh * np.cos(2 * np.pi * (x1 - self.x0) / self.a) * self.u.sigma_z
            value = value + magnetic_term * 0.5 * self.u.bohr_magneton_au * self.g

            return value

        def hopping_x(sitei, sitej):
            value = -self.precalc_t * np.identity(2)

            (xi, yi) = sitei.pos
            (xj, yj) = sitej.pos

            return value

        def hopping_y(sitei, sitej):
            (xi, yi) = sitei.pos
            (xj, yj) = sitej.pos

            value = -self.precalc_t * np.identity(2)

            return value

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

        # attach the right contact to the system
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

    def _transmission(self, E):
        sys = self.make_system()
        smatrix = kwant.smatrix(sys, E)
        t = smatrix.transmission(1, 0)
        return t

    def conductance(self, Emax, ne):
        energies = np.linspace(0, Emax, ne)
        cond = [self._transmission(self.u.eV2au(E)) for E in energies]
        return energies, cond

    def transmission(self, E, leads, spins):
        E = self.u.eV2au(E)
        sys = self.make_system()
        smatrix = kwant.smatrix(sys, E)
        t = smatrix.transmission((leads[0], spins[0]), (leads[1], spins[1]))
        return t

    def transmission_all(self, E):
        E = self.u.eV2au(E)
        sys = self.make_system()
        smatrix = kwant.smatrix(sys, E)

        t_upup = smatrix.transmission((1, 1), (0, 1))
        t_updown = smatrix.transmission((1, 0), (0, 1))
        t_downup = smatrix.transmission((1, 1), (0, 0))
        t_downdown = smatrix.transmission((1, 0), (0, 0))

        return t_upup, t_updown, t_downup, t_downdown

    density_up = tinyarray.array([[1, 0], [0, 0]])
    density_down = tinyarray.array([[0, 0], [0, 1]])
    density_both = tinyarray.array([[1, 0], [0, 1]])

    def wave_function(self, E):
        E = self.u.eV2au(E)
        sys = self.make_system()
        wave_f = kwant.wave_function(sys, E)(0)

        density_up_op = kwant.operator.Density(sys, self.density_up)
        density_down_op = kwant.operator.Density(sys, self.density_down)
        density_both_op = kwant.operator.Density(sys, self.density_both)

        density_up_map = density_up_op(wave_f[0])
        density_down_map = density_down_op(wave_f[0])
        density_both_map = density_both_op(wave_f[0])

        return (density_up_map, density_down_map, density_both_map, sys)

    def wave_function_spins(self, E):
        E = self.u.eV2au(E)
        sys = self.make_system()
        wave_f = kwant.wave_function(sys, E)(0)

        density_x_op = kwant.operator.Density(sys, self.u.sigma_x)
        density_y_op = kwant.operator.Density(sys, self.u.sigma_y)
        density_z_op = kwant.operator.Density(sys, self.u.sigma_z)

        density_x_map = density_x_op(wave_f[0])
        density_y_map = density_y_op(wave_f[0])
        density_z_map = density_z_op(wave_f[0])

        return (density_x_map, density_y_map, density_z_map, sys)
