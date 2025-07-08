import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

energy_hartree = 27.211
MASS = 0.067


def amplitude_electric_to_au(F):
    return F / energy_hartree * 0.052918e-04


def amplitude_to_kv_per_cm(amplitude):
    return amplitude / 0.052918e-04 * energy_hartree


def omega_at_au(omega):
    time_bohr = 2.418884e-17
    return omega * time_bohr


def hamiltonian_matrix(potential, amplitude, alpha, x):
    matrix = np.zeros((len(potential), len(potential)))
    for i in range(len(potential)):
        for j in range(len(potential)):
            if np.abs(i - j) == 1:
                matrix[i, j] = -alpha
            elif i == j:
                matrix[i, j] = 2 * alpha + potential[i] + amplitude * x[i]
    return matrix


def eigen_problem(hamiltonian):
    values, vectors = eigh(hamiltonian, subset_by_index=[0, 3])
    return values, vectors


def time_dependent_potential(amplitide, x, omega, time):
    output = amplitide * x * np.sin(omega * time)
    return output


def cranck_nicolson_prime(wavefunction, potential_bound, potential_time, alpha):
    n = len(wavefunction)
    output = np.zeros(n, dtype="complex")
    output[0] = (
        -alpha * (wavefunction[1] - 2 * wavefunction[0])
        + potential_bound[0] * wavefunction[0]
        + potential_time[0] * wavefunction[0]
    )
    output[n - 1] = (
        -alpha * (wavefunction[n - 2] - 2 * wavefunction[n - 1])
        + potential_bound[n - 1] * wavefunction[n - 1]
        + potential_time[n - 1] * wavefunction[n - 1]
    )
    for i in range(1, n - 1):
        output[i] = (
            -alpha * (wavefunction[i + 1] + wavefunction[i - 1] - 2 * wavefunction[i])
            + potential_bound[i] * wavefunction[i]
            + potential_time[i] * wavefunction[i]
        )
    return output


def cranck_nicolson(wavefuncion, dt, omega, x, amplitude, alpha, potential_bound):
    wavefunction_next = wavefuncion + 0j
    potential_time = time_dependent_potential(amplitude, x, omega, dt)
    potential_time_at_0 = time_dependent_potential(amplitude, x, omega, 0)
    wavefuncion_prime_at_0 = cranck_nicolson_prime(
        wavefuncion, potential_bound, potential_time_at_0, alpha
    )

    for i in range(10):
        wavefuncion_prime = wavefuncion_prime_at_0 + cranck_nicolson_prime(
            wavefunction_next, potential_bound, potential_time, alpha
        )
        wavefunction_next = wavefuncion + wavefuncion_prime * dt / (2 * complex(0, 1))

    return wavefunction_next


def askara_cakmaka(
    wavefunction_at_0,
    dt,
    omega,
    x,
    amplitude,
    alpha,
    potential_bound,
    N,
    wavefunction1_1,
    dx,
):
    wavefunction1 = wavefunction_at_0 + 0j
    wavefunction2 = cranck_nicolson(
        wavefunction_at_0, dt, omega, x, amplitude, alpha, potential_bound
    )
    M = int(1e04)
    output = np.zeros((3, int(N // M)))

    for i in range(N):
        potential_time = time_dependent_potential(amplitude, x, omega, dt * (i + 2))
        wavefunction_prime = cranck_nicolson_prime(
            wavefunction2, potential_bound, potential_time, alpha
        )
        waveduncion_temp = wavefunction2
        wavefunction2 = wavefunction1 + 2 * dt * wavefunction_prime / (complex(0, 1))

        wavefunction1 = waveduncion_temp

        if i % M == 0:
            wavefunction2_0 = (
                np.abs(sum(np.conj(wavefunction2) * wavefunction_at_0 * dx)) ** 2
            )
            wavefunction2_1 = (
                np.abs(sum(np.conj(wavefunction2) * wavefunction1_1 * dx)) ** 2
            )
            output[0, i // M] = wavefunction2_0
            output[1, i // M] = wavefunction2_1
            output[2, i // M] = wavefunction2_0 + wavefunction2_1

    return output


##################### exercise 1 #####################
MASS = 0.067
bohr_length = 0.052918
a = 25 / bohr_length
n = 99
dx = 2 * a / (n + 1)
d1 = 12 / bohr_length
d2 = 4 / bohr_length
energy_hartree = 27.2114
V2 = 0.2 / energy_hartree
V1 = 0.25 / energy_hartree
alpha = 1 / (2 * MASS * dx**2)
x = np.linspace(-a, a, n + 2)
potential_bound = np.zeros(n + 2)


def exercise1():
    N = 21
    amplitudes = np.linspace(
        amplitude_electric_to_au(-2), amplitude_electric_to_au(2), N
    )

    for i in range(n + 2):
        if np.abs(x[i]) >= d1:
            potential_bound[i] = V1
        elif np.abs(x[i]) < d2:
            potential_bound[i] = V2
    ex1 = np.zeros((4, N))

    for F in amplitudes:
        hamiltonian = hamiltonian_matrix(potential_bound, F, alpha, x)
        ex1[:, np.where(amplitudes == F)[0][0]], vectors = eigen_problem(hamiltonian)

    ex1 *= energy_hartree
    F_kVcm = np.linspace(-2, 2, N)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    plt.sca(axs[0])
    plt.plot(F_kVcm, ex1[0, :], label="$n = 0$")
    plt.plot(F_kVcm, ex1[1, :], label="$n = 1$")
    plt.plot(F_kVcm, ex1[2, :], label="$n = 2$")
    plt.plot(F_kVcm, ex1[3, :], label="$n = 3$")
    plt.xlabel("$F$ [kV/cm]", fontsize=18)
    plt.ylabel("$E$ [eV]", fontsize=18)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.title("first 4 energies", fontsize=18)

    plt.sca(axs[1])
    plt.title("first 2 energies", fontsize=18)
    plt.plot(F_kVcm, ex1[0, :], label="$n = 0$")
    plt.plot(F_kVcm, ex1[1, :], label="$n = 1$")
    plt.legend(fontsize=16)
    plt.xlabel("$F$ [kV/cm]", fontsize=18)
    plt.ylabel("$E$ [eV]", fontsize=18)
    plt.tight_layout()
    plt.savefig("ex1.pdf")
    # plt.show()


# exercise1()

####################### exercise 2 #####################
potential_bound = np.zeros(n + 2)
amplitude = 0


def exercise2():
    for i in range(n + 2):
        if np.abs(x[i]) >= d1:
            potential_bound[i] = V1
        elif np.abs(x[i]) < d2:
            potential_bound[i] = V2
    hamiltonian = hamiltonian_matrix(potential_bound, amplitude, alpha, x)
    values, vectors = eigen_problem(hamiltonian)
    wavefunction_at_0 = vectors.T[0, :]
    wavefunction_at_1 = vectors.T[1, :]

    print((values[1] - values[0]) * energy_hartree)
    wavefunction_at_0 = wavefunction_at_0 / (np.sqrt(sum(wavefunction_at_0**2) * dx))
    wavefunction_at_1 = wavefunction_at_1 / (np.sqrt(sum(wavefunction_at_1**2) * dx))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_xlabel("$x$ [nm]", fontsize=18)
    ax.set_ylabel("$V$ [eV], $\\psi$ [a.u.]", fontsize=18)
    plt.plot(
        x * bohr_length,
        potential_bound * energy_hartree - V1 * energy_hartree,
        label="$V_w$",
    )
    plt.plot(x * bohr_length, wavefunction_at_0, label="$\\psi_0$")
    plt.plot(x * bohr_length, wavefunction_at_1, label="$\\psi_1$")
    plt.legend(fontsize=18)
    # plt.show()
    plt.savefig("ex2.pdf")


# exercise2()

################### exercise 3 #####################
potential_bound = np.zeros(n + 2)


def exercise3():
    for i in range(n + 2):
        if np.abs(x[i]) >= d1:
            potential_bound[i] = V1
        elif np.abs(x[i]) < d2:
            potential_bound[i] = V2

    amplitude = amplitude_electric_to_au(0.08)
    dt = 1
    Nsteps = int(3e06)
    omega = 104.22e-7 * 2.418884
    # omega = 95e-7 * 2.418884
    hamiltonian = hamiltonian_matrix(potential_bound, amplitude, alpha, x)

    _, vectors = eigen_problem(hamiltonian)
    wavefunction0 = vectors.T[0, :]
    wavefunction1 = vectors.T[1, :]

    wavefunction0 = wavefunction0 / np.sqrt(sum(np.abs(wavefunction0) ** 2 * dx))

    wavefunction1 = wavefunction1 / np.sqrt(sum(np.abs(wavefunction1) ** 2 * dx))
    askara_cakmaka_output = askara_cakmaka(
        wavefunction0,
        dt,
        omega,
        x,
        amplitude,
        alpha,
        potential_bound,
        Nsteps,
        wavefunction1,
        dx,
    )

    length = len(askara_cakmaka_output[0, :])

    x_axis = np.linspace(0, (10000 * (length - 1)) * 2.418884e-8, length)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    plt.plot(x_axis, askara_cakmaka_output[0, :], label=r"$|<\psi|0>^2$")
    plt.plot(x_axis, askara_cakmaka_output[1, :], label=r"$|<\psi|1>^2$")
    plt.plot(
        x_axis, askara_cakmaka_output[2, :], label=r"$|<\psi|0>|^2 + |<\psi|1>|^2$"
    )
    plt.xlabel("$t$ [ns]", fontsize=18)
    plt.ylabel("probability", fontsize=18)
    plt.legend(fontsize=12)
    # plt.show()
    plt.savefig("ex3.pdf")


exercise3()

