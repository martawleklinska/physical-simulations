import kwant
import numpy as np
import utils as utl


class Andreev:
    u = utl.Utils()

    def __init__(
        self, *, dx=0.2, L=int(1250), delta=2.5e-04, a=1.0, P=0.0, mu=0.01, Z=0.0
    ):
        self.dx = self.u.nm2au(dx)
        self.L = int(L)
        self.delta_val = self.u.eV2au(delta)
        self.a = self.u.nm2au(a)
        self.P = P
        self.mu = self.u.eV2au(mu)
        self.Z = Z

        def h(x):
            if x < self.L * self.dx / 2:
                return P * self.mu
            else:
                return 0.0

        self.h = h

        def delta_func(x):
            if x < self.L * self.dx / 2:
                return 0.0
            else:
                return self.delta_val

        self.delta = delta_func

        def potential_term(x):
            center = self.L * self.dx / 2
            return self.Z * self.mu * np.exp(-((x - center) ** 2) / (2 * self.a**2))

        self.V = potential_term
        self.t = 1 / (2 * self.dx**2)

        """2t + V(x)-h(x)-mu      delta(x)
        delta(x)          -2t - V(x)-h(x)+mu

        t=hbar^2/(2m dx^2)
        to na gorze onsite

        hopping:
        -t 0 0 t"""

    def make_system(self):
        def onsite(site):
            (xi,) = site.pos
            t = self.t
            mu = self.mu
            h = self.h(xi)
            delta = self.delta(xi)
            V = self.V(xi)

            return np.array(
                [[2 * t - mu - h + V, delta], [delta, -(2 * t - mu + h + V)]]
            )

        def onsite_normal(site):
            (xi,) = site.pos
            t = self.t
            mu = self.mu
            h = self.h(xi)
            return np.array([[2 * t - mu - h, 0], [0, -(2 * t - mu + h)]])

        def hopping_x(sitei, sitej):
            t = 1 / (2 * 1 * self.dx**2)
            return np.matrix(
                [[-t, 0], [0, t]],
            )

        sys = kwant.Builder()
        lat = kwant.lattice.chain(self.dx, norbs=2)
        sys[(lat(i) for i in range(self.L))] = onsite
        sys[lat.neighbors()] = hopping_x

        sigma_law = np.array([[1, 0], [0, 2]])

        lead_left = kwant.Builder(
            kwant.TranslationalSymmetry((-self.dx,)), conservation_law=sigma_law
        )
        lead_left[lat(0)] = onsite_normal
        lead_left[lat.neighbors()] = hopping_x
        sys.attach_lead(lead_left)

        def onsite_sc(site):
            (xi,) = site.pos
            t = self.t
            mu = self.mu
            delta = self.delta_val

            return np.array([[2 * t - mu, delta], [delta, -(2 * t - mu)]])

        def hopping_sc(sitei, sitej):
            t = 1 / (2 * 1 * self.dx**2)
            return np.matrix([[-t, 0], [0, t]])

        lead_right = kwant.Builder(kwant.TranslationalSymmetry((self.dx,)))
        lead_right[lat(0)] = onsite_sc
        lead_right[lat.neighbors()] = hopping_sc
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

    def transmission_matrix(self, E):
        E = self.u.eV2au(E)
        sys = self.make_system()
        smatrix = kwant.smatrix(sys, E)

        tup_down = smatrix.transmission((0, 0), (1, 1))
        tup_up = smatrix.transmission((0, 0), (1, 0))
        tdown_down = smatrix.transmission((0, 1), (1, 1))
        tdown_up = smatrix.transmission((0, 1), (1, 0))

        return tup_down, tup_up, tdown_down, tdown_up

    def reflection_coeffs(self, E):
        E = self.u.eV2au(E)
        sys = self.make_system()
        smatrix = kwant.smatrix(sys, E)

        Ree = smatrix.transmission((0, 0), (0, 0))
        Rhe = smatrix.transmission((0, 1), (0, 0))
        return Ree, Rhe
