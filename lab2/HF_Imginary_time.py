import numpy
import numpy as np
import matplotlib.pyplot as plt
# import numba as nb
# ============ for latex fonts ============
from matplotlib import rc #, font_manager
rc('text.latex', preamble=r'\usepackage{lmodern}')# this helps use the plots in tex files
plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'xtick.labelsize': 14,
		  'ytick.labelsize': 14,
		  'xtick.major.pad': 6,
		  'ytick.major.pad': 6,
		  'font.serif': 'Computer Modern Roman',
		  'axes.formatter.use_mathtext': True,
		  'axes.labelpad': 6.0 }) 
# ==========================================

# plt.figure(figsize=(8, 6), dpi=80)
plt.figure()
Ha = 27.211 # 1 Hartree in eV
a0 = 0.05292 # Bohr radius in nm
m = 0.067 # effective mass for GaAs
eps = 12.5 # dielectric constant for GaAs

def plot_hamiltonian_spectrum(alphas, spectrum, E_range=(-7, 0)):
    plt.plot(alphas, spectrum, 'co' , linewidth=1, markersize=4.5)


# @nb.njit
def HF_imaginary_time_evolution(Nx, Ny, Lx, Ly, writing=False):
    dx = Lx / (Nx-1) # the grid spacing
    dy = Ly / (Ny-1) # the grid spacing in y
    alphax = 0.50 / m / dx**2 # in atomic units
    alphay = 0.50 / m / dy**2 # in atomic units
    dt = 0.39 * m * 1./(1/dx**2 + 1/dy**2)
        
    # confinement potential
    V = numpy.zeros((Nx, Ny)) # potential energy 
    for i in range(0, Nx): 
        for j in range(0, Ny):
            x = (i - Nx//2) * dx
            y = (j - Ny//2) * dx
            V[i, j] = -0.5/Ha * np.exp((-x**2/(Lx/2)**2-y**2/(Ly/3)**2))
            
    # save 1/r_ij to array for speed up
    interact = np.zeros((Nx, Ny)) 
    for i in np.arange(0, Nx):
        for j in np.arange(0, Ny):
            if(i != j):
                interact[i, j] = 1./ np.sqrt( (i * dx)**2 +  (j * dy)**2) * dx*dy / eps
            else: # the case r_ij=0
                interact[i, j] = dx * 4 * np.log(2.**0.5 + 1) / eps # only works for dx=dy
    
    # if writing:
    #     f = open("E_iteration.txt", 'w')
    
    # fill the wave function with random numbers
    psi = np.random.random((Nx, Ny)) * 2 - 1
    
    # boundary condition: 
    psi[0, :] = 0
    psi[Nx - 1, :] = 0
    psi[:, 0] = 0
    psi[:, Ny - 1] = 0
    # initialising loop variables
    diffpsi = 10
    psi_old = np.zeros((Nx, Ny))
    it_HF = 0
    
    J = np.zeros((Nx, Ny)) # Will be useful later to calculate the interaction energy
    while ( np.abs(diffpsi) > 1e-7):
        it_HF += 1
        if (it_HF % 10 == 0):
            print(f"iteracja HF, L, E", it_HF, Lx*a0, diffpsi)
        diffpsi = np.sum( np.abs(psi-psi_old) )
        psi_old = psi
        
        # calculate the mean field potential, to be used in the imaginary time method
        J = np.zeros((Nx, Ny))
        for i1 in np.arange(0, Nx):
            for j1 in np.arange(0, Ny):
                for i2 in np.arange(0, Nx):
                    for j2 in np.arange(0, Ny):
                        J[i1, j1] += np.abs(psi[i2, j2])**2 * interact[np.abs(i2 - i1), np.abs(j2 - j1)] 
        
        # solve single-electron problem with imaginary time method
        Ek_n = 1
        Ek_old = 2
        iteration = 0
        while ( np.abs((Ek_n - Ek_old)/Ek_old) > 1e-8):
            iteration = iteration + 1
            Ek_old = Ek_n
            
    		# calculate F = H \psi 
            F = np.zeros((Nx, Ny))
            for i in np.arange(1, Nx-1):
                for j in np.arange(1, Ny-1):
                    F[i, j] += -(psi[i + 1, j] + psi[i - 1, j] - 2 * psi[i, j]) * alphax - (psi[i, j + 1] + psi[i, j - 1] - 2 * psi[i, j]) * alphay
                    F[i, j] += psi[i, j] * V[i, j]
                    F[i, j] += psi[i, j] * J[i, j]
                    
    		# new approximation
            psi = psi - dt * F
            
    		# normalize
            norm = np.sum(np.abs(psi**2)) *dx*dy
            psi = psi / np.sqrt(norm)
	    
	    # mix with solution from the previous iteration for stability
            psi = psi * 0.25 + psi_old * 0.75
        
    		# calculate the one-electron energy for check
            Ek_n = np.sum( np.conjugate(psi)*F )*dx*dy
        
    # interaction energy
    F = np.zeros((Nx, Ny))
    for i in np.arange(1, Nx-1):
        for j in np.arange(1, Ny-1):
            F[i, j] += J[i, j] * psi[i, j]
    E_int = np.sum( np.conjugate(F) * psi ) *dx*dy
    # total energy of the system
    E_tot = 2 * Ek_n - E_int
    return E_tot


def find_spectrum_L(Nx, Ny, L_values):        
    spectrum = [HF_imaginary_time_evolution(Nx, Ny, L, L, writing=True) for L in L_values]
    return np.array(spectrum)


Nx = 20
Ny = 20

# analytical energies -- denser x axis
Ls = np.linspace(5, 25, 80) # in nm
Ls = Ls / a0 # in atomic units
Ly = Ls

# Ls = np.linspace(5, 25, 5) # in nm
#Ls = np.linspace(50, 50, 1) # in nm
Ls = np.linspace(20, 50, 7) # in nm
Ls = Ls / a0 # in atomic units
spectrum = find_spectrum_L(Nx, Ny, Ls)

plot_hamiltonian_spectrum(Ls * a0, spectrum * Ha)

# save to file to plot together with CI
f = open("E_HF_L.txt", 'w')
for i in np.arange(len(Ls)):
    Lx = Ls[i]
    f.write('%f\t %f\t %e\t \n' % (Lx*a0, np.pi**2 / (Lx**2 * 2 * m) * Ha, spectrum[i] * Ha))
f.close()

plt.title(f"Basis state for $N_x$={Nx}", usetex=True)
plt.xlabel("$L\ (\mathrm{nm})$", usetex=True)
plt.ylabel("$E_{n}\ (\mathrm{eV})$", usetex=True)
plt.savefig("E_L0.pdf",bbox_inches='tight', transparent=True)

plt.show()



