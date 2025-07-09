import kwant
import tinyarray
import numpy as np
import utils as utl


class SpinSystem:
    u = utl.Utils()

    def __init__(
        self,
        *,
        dx=4,
        L=int(500),
        W=int(25),
        m=0.014,
        Bx=0.0,
        By=0.0,
        Bz=0.0,
        alpha=0.000,
        additional_By_boundaries=None,
        additional_By=0.0,
        alpha_boundaries=None
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

        self.precalc_t = 1.0 / (2.0 * self.m * self.dx * self.dx)
        self.precalc_tso = self.alpha / (2 * self.dx)

        self.add_term = additional_By_boundaries
        self.additional_By = self.u.T2au(additional_By)
        self.alpha_boundaries = alpha_boundaries

    def make_system(self):
        def onsite(site):
            value = 4 * self.precalc_t * np.identity(2) + self.precalc_mag_term
            if self.add_term != None:
                if (site.pos[0] > self.add_term[0] * self.L * self.dx) and (
                    site.pos[0] < self.add_term[1] * self.L * self.dx
                ):
                    value += (
                        0.5
                        * self.u.bohr_magneton_au
                        * self.u.lande_g
                        * (self.additional_By * self.u.sigma_y)
                    )

            return value

        def hopping_x(sitei, sitej):
            value = -self.precalc_t * np.identity(2)

            (xi, yi) = sitei.pos
            (xj, yj) = sitej.pos

            if self.alpha_boundaries != None:
                if (xj > self.alpha_boundaries[0] * self.L * self.dx) and (
                    xi < self.alpha_boundaries[1] * self.L * self.dx
                ):
                    value = value + 1j * self.precalc_tso * self.u.sigma_y

            return value

        def hopping_y(sitei, sitej):
            (xi, yi) = sitei.pos
            (xj, yj) = sitej.pos

            value = -self.precalc_t * np.identity(2)
            if self.alpha_boundaries != None:
                if (xi > self.alpha_boundaries[0] * self.L * self.dx) and (
                    xj < self.alpha_boundaries[1] * self.L * self.dx
                ):

                    value = value - 1j * self.precalc_tso * self.u.sigma_x

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

        # finalize the system
        sys = sys.finalized()
        return sys

    def dispersion(self, nr_lead, k_max, nk):
        sys = self.make_system()
        momenta = np.linspace(-k_max * self.dx, k_max * self.dx, nk)
        bands = kwant.physics.Bands(sys.leads[nr_lead])
        energies = [bands(k) for k in momenta]
        return (momenta / self.dx) * self.u.nm2au(1.0), energies

    # calculates the transmission coefficient - input in a. u.
    def _transmission(self, E):
        sys = self.make_system()
        smatrix = kwant.smatrix(sys, E)
        t = smatrix.transmission(1, 0)  # total transmission from 0 to 1 (or reversed)
        return t

    # calculates the conductance - the Landauer formula is used
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
