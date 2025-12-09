import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform
from numba import njit

@njit
def electron_interaction(x_values, i, j, dielectric_const, wire_thickness):
    r_squared = (x_values[i] - x_values[j]) ** 2 + wire_thickness**2

    if r_squared < 1e-6: 
        r_squared = 1e-6

    return 1 / (dielectric_const * np.sqrt(r_squared))



def compute_hamiltonian(
    mass, dx, wavefunction, x_values, dielectric_const, wire_thickness
):
    hamiltonian = np.zeros(np.shape(wavefunction))
    for i in range(1, len(wavefunction[:, 0]) - 1):
        for j in range(1, len(wavefunction[0, :]) - 1):
            hamiltonian[i, j] = -(
                wavefunction[i + 1, j]
                + wavefunction[i - 1, j]
                + wavefunction[i, j + 1]
                + wavefunction[i, j - 1]
                - 4 * wavefunction[i, j]
            ) / (2 * mass * dx**2) + wavefunction[i, j] * electron_interaction(
                x_values, i, j, dielectric_const, wire_thickness
            )
    return hamiltonian


def update_wavefunction(wavefunction, hamiltonian, time_step):
    return wavefunction - time_step * hamiltonian


def normalize_wavefunction(wavefunction, dx):
    norm_factor = np.sum(wavefunction**2 * dx**2)
    return wavefunction / np.sqrt(norm_factor)


def compute_energy(wavefunction, hamiltonian, dx):
    return np.sum(wavefunction * hamiltonian * dx**2)


def convergence_test(previous_energy, current_energy, tolerance):
    return np.abs((current_energy - previous_energy) / current_energy) < tolerance


def imaginary_time_evolution(
    grid_size, box_size, electron_mass, dielectric_const, wire_thickness, tolerance, ex1 = True
):
    dx = 2 * box_size / (grid_size - 1)
    x_values = np.linspace(-box_size, box_size, grid_size)
    time_step = electron_mass * dx**2 * 0.4

    wavefunction = np.zeros((grid_size, grid_size))
    wavefunction[1 : grid_size - 1, 1 : grid_size - 1] = uniform(
        -1, 1, size=(grid_size - 2, grid_size - 2)
    )

    energy = 0
    iteration = 0
    iterations = []
    energies = []

    while True:
        hamiltonian = compute_hamiltonian(
            electron_mass, dx, wavefunction, x_values, dielectric_const, wire_thickness
        )
        updated_wavefunction = update_wavefunction(wavefunction, hamiltonian, time_step)
        normalized_wavefunction = normalize_wavefunction(updated_wavefunction, dx)
        new_energy = compute_energy(normalized_wavefunction, hamiltonian, dx)

        if convergence_test(energy, new_energy, tolerance):
            break

        wavefunction = normalized_wavefunction
        energy = new_energy
        iteration += 1

        if iteration % 100 == 0:
            iterations.append(iteration)
            energies.append(energy)
    
    if ex1:
        return iterations, np.array(energies) * 27.211
    else:
        return wavefunction



def plot_energy_convergence(iterations, energies):
    plt.figure(figsize=(8, 6))
    plt.scatter(iterations, energies)
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Energy [eV]", fontsize = 14)
    plt.title("Energy Convergence", fontsize = 14)
    # plt.show()
    plt.savefig("ex1_energy_convergence.png")


grid_size = 41
wire_thickness = 10 / 0.052918
electron_mass = 0.067
dielectric_const = 12.5
box_size = 30 / 0.052918
tolerance = 1e-9

iterations, energies = imaginary_time_evolution(grid_size, box_size, electron_mass, dielectric_const, wire_thickness, tolerance, ex1=True)
plot_energy_convergence(iterations, energies)

##########################################
# exercise 2
##########################################


def plot_wavefunction_density(grid_points, wavefunction, box_size, title):
    x_values = np.linspace(-box_size, box_size, grid_points) * 0.052918

    plt.figure(figsize=(8, 6))
    plt.contourf(x_values, x_values, wavefunction**2, 100, cmap="seismic")
    plt.colorbar()
    plt.xlabel("$x$ [nm]", fontsize = 18)
    plt.ylabel("$y$ [nm]", fontsize = 18)
    plt.title(title, fontsize=20)
    # plt.show()
    plt.savefig(f"ex2_{title}.png")


grid_size = 41
wire_thickness = 10 / 0.052918
electron_mass = 0.067
dielectric_const = 12.5
tolerance = 1e-9

box_size_30nm = 30 / 0.052918
wavefunction_30nm = imaginary_time_evolution(
    grid_size, box_size_30nm, electron_mass, dielectric_const, wire_thickness, tolerance, ex1=False
)
plot_wavefunction_density(
    grid_size, wavefunction_30nm, box_size_30nm, "$a = 30$ nm"
)

box_size_60nm = 60 / 0.052918
wavefunction_60nm = imaginary_time_evolution(
    grid_size, box_size_60nm, electron_mass, dielectric_const, wire_thickness, tolerance, ex1=False
)
plot_wavefunction_density(
    grid_size, wavefunction_60nm, box_size_60nm, "$a = 60$ nm"
)

##############################################
# exercise 2, 4
##############################################

HARTREE_ENERGY = 27.211  
GRID_POINTS = 41 
LENGTH_SCALE = 10 / 0.052918
EFFECTIVE_MASS = 0.067  
DIELECTRIC_CONSTANT = 12.5
CONVERGENCE_TOL = 1e-9
DOT_SIZES = np.linspace(30 / 0.052918, 60 / 0.052918, 7) 

def compute_imaginary_time_energies():
    energies_it = np.zeros(len(DOT_SIZES))
    
    for i, dot_size in enumerate(DOT_SIZES):
        dx = 2 * dot_size / (GRID_POINTS - 1)
        x_values = np.linspace(-dot_size, dot_size, GRID_POINTS) 
        time_step = EFFECTIVE_MASS * dx**2 * 0.4
        wave_function = np.zeros((GRID_POINTS, GRID_POINTS))
        wave_function[1:GRID_POINTS-1, 1:GRID_POINTS-1] = np.random.uniform(-1, 1, size=(GRID_POINTS-2, GRID_POINTS-2))

        converged, energy = False, 0
        iteration = 0
        MAX_ITER = 100_000 

        while not converged and iteration < MAX_ITER:
            hamiltonian = compute_hamiltonian(EFFECTIVE_MASS, dx, wave_function, x_values, DIELECTRIC_CONSTANT, LENGTH_SCALE)
            next_wave_function = update_wavefunction(wave_function, hamiltonian, time_step)
            next_wave_function = normalize_wavefunction(next_wave_function, dx)
            next_energy = compute_energy(next_wave_function, hamiltonian, dx)
            
            converged = convergence_test(energy, next_energy, CONVERGENCE_TOL)

            if not converged:
                wave_function = next_wave_function
            energy = next_energy
            iteration += 1
        
        if iteration == MAX_ITER:
            print(f"Warning: Imaginary time evolution did not converge for dot size {dot_size:.2f} a.u.")

        energies_it[i] = energy

    return energies_it * HARTREE_ENERGY


@njit(nopython=True)
def hartree_fock_potential(wave_function, grid, dx, dielectric_const, wire_thickness):
    potential = np.zeros(len(wave_function))
    for i in range(len(wave_function)):
        for j in range(len(wave_function)):
            if i != j:
                potential[i] += dx * wave_function[j] ** 2 * electron_interaction(grid, i, j, dielectric_const, wire_thickness) * wave_function[j]
    return potential

def energy_sum(wave_function, fock, dx):
    return np.sum(wave_function * fock * dx)

def hartree_fock_hamiltonian(wave_function, mass, dx):
    hamiltonian = np.zeros(np.shape(wave_function))
    
    for i in range(1, len(wave_function) - 1):
        hamiltonian[i] = -(wave_function[i-1] + wave_function[i + 1] - 2 * wave_function[i]) / (2 * mass * dx ** 2)
    return hamiltonian

def hartree_fock_iteration(wave_function, grid, tolerance, potential):
    mass = EFFECTIVE_MASS
    convergence = 0
    dx = 2 * grid / (GRID_POINTS - 1)
    dtau = mass * dx ** 2 * 0.4
    energy = 0
    
    while not convergence:
        fock = hartree_fock_hamiltonian(wave_function, mass, dx) + potential * wave_function
        next_wave_function = update_wavefunction(wave_function, fock, dtau)
        next_wave_function = normalize_wavefunction(next_wave_function, dx)
        next_energy = energy_sum(next_wave_function, fock, dx)
        convergence = convergence_test(energy, next_energy, tolerance)
        if not convergence:
            wave_function = next_wave_function
        energy = next_energy
    return wave_function, energy

def compute_hartree_fock_energies():
    energies = np.zeros(len(DOT_SIZES)) 
    
    for idx, dot_size in enumerate(DOT_SIZES):
        dx = 2 * dot_size / (GRID_POINTS - 1)
        x_values = np.linspace(-dot_size, dot_size, GRID_POINTS) 

        wave_function = np.zeros(GRID_POINTS)
        wave_function[1:GRID_POINTS - 1] = np.random.uniform(-1, 1, size=(GRID_POINTS - 2))

        convergence = False
        energy = 0
        iteration = 0
        MAX_ITER = 10_000_000 

        while not convergence:# and iteration < MAX_ITER: 
            potential = hartree_fock_potential(wave_function, x_values, dx, DIELECTRIC_CONSTANT, LENGTH_SCALE)

            wave_function, energy = hartree_fock_iteration(wave_function, dot_size, CONVERGENCE_TOL, potential)
            energy_next = 2 * energy - dx * np.sum(wave_function ** 2 * potential)

            convergence = convergence_test(energy, energy_next, CONVERGENCE_TOL)

            if not convergence:
                energy = energy_next
            iteration += 1

        # if iteration == MAX_ITER:
        #     print(f"Hartree-Fock method did not converge for dot size {dot_size:.2f} a.u.")

        energies[idx] = energy 
        print(energies[idx])

    return energies * HARTREE_ENERGY 


    
energies_it = compute_imaginary_time_energies()
energies_hf = compute_hartree_fock_energies()
dot_sizes_nm = DOT_SIZES * 0.052918

fig = plt.figure(figsize=(10, 7))  
ax = fig.add_subplot(111)
ax.set_ylabel("energia [eV]", fontsize = 18)
ax.set_xlabel(r"$a$ [nm]", fontsize = 18)
ax.set_title(r"Energia od rozmiaru układu", fontsize=20)

ax.plot(dot_sizes_nm, energies_hf, label="H-F method")  
ax.plot(dot_sizes_nm, energies_it, label="IT method") 

plt.legend()
# plt.show()
plt.savefig("ex2_4_energies.pdf")