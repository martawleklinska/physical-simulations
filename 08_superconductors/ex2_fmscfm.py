import kwant
import numpy as np
import utils as utl


class FerroSuperconductorFerro:
    u = utl.Utils()

    def __init__(
        self,
        *,
        dx=1.0,
        L_sc=int(250),
        L_fm=int(250),
        delta=2.5e-04,
        P_l=0.0,
        P_r=0.0,
        mu=0.01,
    ):
        self.dx = self.u.nm2au(dx)
        self.L_fm = int(L_fm)
        self.L_sc = int(L_sc)
        self.delta_val = self.u.eV2au(delta)
        self.P_l = P_l
        self.P_r = P_r
        self.mu = self.u.eV2au(mu)

        def h(x):
            if x < self.L_fm * self.dx:
                return self.P_l * self.mu
            elif x < (self.L_fm + self.L_sc) * self.dx:
                return 0
            else:
                return self.P_r * self.mu

        def delta_func(x):
            if self.L_fm * self.dx <= x < (self.L_fm + self.L_sc) * self.dx:
                return self.delta_val
            else:
                return 0.0

        self.h = h

        def delta_func(x):
            if x < self.L_fm * self.dx:
                return 0.0
            elif x < (self.L_fm + self.L_sc) * self.dx:
                return self.delta_val
            else:
                return 0.0

        a = self.u.nm2au(1.0)

        def potential_term(x):
            center = self.L_fm * self.dx
            center2 = self.L_fm * self.dx + self.L_sc * self.dx
            return 0.0 * self.mu * np.exp(
                -((x - center) ** 2) / (2 * a**2)
            ) + 0.0 * self.mu * np.exp(-((x - center2) ** 2) / (2 * a**2))

        self.V = potential_term

        self.delta = delta_func

        self.t = 1 / (2 * self.dx**2)

        self.L = self.L_fm + self.L_sc + self.L_fm

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
            # V = self.V(xi)
            V = 0.0

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

        lead_right = kwant.Builder(
            kwant.TranslationalSymmetry((self.dx,)), conservation_law=sigma_law
        )
        lead_right[lat(0)] = onsite_normal
        lead_right[lat.neighbors()] = hopping_x

        sys.attach_lead(lead_right)

        sys = sys.finalized()
        return sys

    def transmission_matrix(self, E):
        E = self.u.eV2au(E)
        sys = self.make_system()
        smatrix = kwant.smatrix(sys, E)
        Ree = smatrix.transmission((0, 0), (0, 0))  # electron → electron (reflection)
        Rhe = smatrix.transmission((0, 1), (0, 0))  # electron → hole (Andreev)
        Tee = smatrix.transmission((1, 0), (0, 0))  # electron → electron (transmission)
        The = smatrix.transmission((1, 1), (0, 0))  # electron → hole (CAR)

        return Ree, Rhe, Tee, The

    def reflection_coeffs(self, E):
        E = self.u.eV2au(E)
        sys = self.make_system()
        smatrix = kwant.smatrix(sys, E)

        Ree = smatrix.transmission((0, 0), (0, 0))
        Rhe = smatrix.transmission((0, 1), (0, 0))
        return Ree, Rhe
