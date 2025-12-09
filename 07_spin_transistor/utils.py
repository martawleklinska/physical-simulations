import numpy as np
import kwant as kw


class Utils:
    sigma_x = np.matrix([[0, 1], [1, 0]], dtype=np.complex64)
    sigma_y = np.matrix([[0, -1j], [1j, 0]], dtype=np.complex64)
    sigma_z = np.matrix([[1, 0], [0, -1]], dtype=np.complex64)

    bohr_magneton_au = 0.5
    lande_g = -50.0

    def eV2au(self, energy):  # eV -> j.a
        return energy * 0.03674932587122423

    def au2eV(self, energy):  # j.a -> eV
        return energy * 27.2117

    def nm2au(self, length):  # nm -> j.a
        return length * 18.89726133921252

    def T2au(self, length):  # T -> j.a
        return length * 4.254382e-6
