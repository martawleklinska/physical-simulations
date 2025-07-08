from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
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
Np = 7 # no Gaussian functions in a basis
Ha = 27.211 # 1 Hartree in eV
a0 = 0.0529 # Bohr radius in nm
m = 1 #0.067 # effective mass for GaAs
omegax = 0.05/Ha

# gaussian functions -- needed for plotting the wave function
def fm(i, x, alfaxa0, dx):
    xmin = -dx * (Np // 2)
    xi = xmin + i * dx
    return np.exp( -0.5/alfax * (x - xi)**2) * (1 / alfax / np.pi)**0.25

# overlap integrals
def O_value(i, j, alfax, dx):
    xmin = -dx * (Np // 2)
    xi = xmin + i * dx
    xj = xmin + j * dx
    return np.exp(-(xi - xj)**2 / 4 / alfax)

# Kij integrals 
def Ek_value(i, j, alfax, dx):
    xmin = -dx * (Np // 2)
    xi = xmin + i * dx
    xj = xmin + j * dx
    return ( 2*alfax - (xi - xj)**2 ) / ( alfax * 2 )**2 * O_value(i, j, alfax, dx)* 0.5/m

# Vij integrals 
def Ep_value(i, j, a, dx):
    xmin = -dx * (Np // 2)
    xi = xmin + i * dx
    xj = xmin + j * dx
    return (a / 2 + (xi + xj)**2 / 4) * O_value(i, j, a, dx) * 0.5 * m * (omegax)**2 

def plot_hamiltonian_spectrum(alphas, spectrum, E_range=(-7, 0)):
    plt.plot(alphas, spectrum[:, :10], 'co' , linewidth=1, markersize=4.5)


#generate a Hamiltonian as a function of L
def make_H_S(alfax, dx):
    H = np.zeros((Np, Np))
    S = np.zeros((Np, Np)) 
    for i in np.arange(0, Np):
        for j in np.arange(0, Np):
            S[i, j] = O_value(i, j, alfax, dx)
            H[i, j] = Ek_value(i, j, alfax, dx) + Ep_value(i, j, alfax, dx) 
    return (H, S)

def energies(alfax, dx):
    H, S = make_H_S(alfax, dx)
    eigvals = eigh(H, S, eigvals_only=True)
    return eigvals

def find_spectrum_L(alfax_values, dx):
    spectrum = [energies(alfax, dx) for alfax in alfax_values]
    return np.array(spectrum)

def find_spectrum_dx(alfax, dx_values):
    spectrum = [energies(alfax, dx) for dx in dx_values]
    return np.array(np.real(spectrum))
    
    

dxs = np.linspace(0.5, 2, 40) / a0 # xmin has to be calculated again
# alfax = 1 / m / omegax / a0
alfax = 10/a0

factors = np.arange(0, 5, 1) 
E_analytical = (omegax * Ha) * np.ones(len(dxs)) 
plt.plot(dxs * a0, np.outer(E_analytical, factors) + 0.5 * (omegax * Ha), 'b-' , linewidth=1)
spectrum = find_spectrum_dx(alfax, dxs)

plot_hamiltonian_spectrum(dxs * a0, spectrum * Ha) 
plt.title(rf"Spectrum for $\alpha_x$={alfax*a0}, $N_p$={Np}", usetex=True)
plt.xlabel(r"$\Delta x\ (\mathrm{nm})$", usetex=True)
plt.ylabel("$E_{n}\ (\mathrm{eV})$", usetex=True)
plt.savefig("E_dx.pdf",bbox_inches='tight', transparent=True)



dx = 0.7 / a0
xmin = -dx * (Np // 2)
plt. figure()
# spectrum as a function of alfa
alfas = np.linspace(2, 61, 60) # in nm
alfas = alfas / a0

factors = np.arange(0, 5, 1) 
E_analytical = (omegax * Ha) * np.ones(len(alfas)) 
plt.plot(alfas * a0, np.outer(E_analytical, factors) + 0.5 * (omegax * Ha), 'b-' , linewidth=1)

spectrum = find_spectrum_L(alfas, dx)

plot_hamiltonian_spectrum(alfas * a0, spectrum * Ha)
plt.title(rf"Spectrum for $\Delta x$={dx*a0}, $N_p$={Np}", usetex=True)
# plt.xlim(10,50)
plt.xlabel(r"$\alpha_x\ (\mathrm{nm})$", usetex=True)
plt.ylabel("$E_{n}\ (\mathrm{eV})$", usetex=True)
plt.savefig("E_alfa.pdf",bbox_inches='tight', transparent=True)


fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
# plotting a selected wave function
alfax = 20 / a0

H, S = make_H_S(alfax, dx)
eigvals, eigvecs = eigh(H, S, eigvals_only=False)
npoint = 100
xs = np.linspace(-dx * 10, dx * 10, npoint)
psi = np.zeros(npoint) * 1j
gausses = np.zeros((npoint, Np))
for i in np.arange(Np):
    for j in np.arange(npoint):
        psi[j] = psi[j] + fm(i, xs[j], alfax, dx) * eigvecs[i, 1]
        gausses[j, i] = fm(i, xs[j], alfax, dx) * eigvecs[i, 1]

plt.title(rf"Wave function for $\Delta x$={dx*a0}, $N_p$={Np}, $\alpha=${alfax*a0}", usetex=True)
ax1.plot(xs * a0, psi, 'b-')
ax2.plot(xs * a0, 0.5 * xs**2 * m * (omegax)**2 * Ha)
ax1.set_xlabel(r"$x\ (\mathrm{nm})$", usetex=True)
ax1.set_ylabel(r"$\psi\ (\mathrm{a.u.})$", usetex=True)
ax2.set_ylabel(r"$V\ (\mathrm{eV})$", usetex=True)
ax1.plot(xs * a0, gausses, 'gray', linestyle='dashed')
plt.savefig("wavefunc.pdf",bbox_inches='tight', transparent=True)

'''
alfax = 10 / a0
dx = 1. / a0
H, S = make_H_S(alfax, dx)
eigvals, eigvecs = eigh(H, S, eigvals_only=False)
psi = np.zeros(npoint) * 1j
for i in np.arange(Np):
    for j in np.arange(npoint):
        psi[j] = psi[j] + fm(i, xs[j], alfax, dx) * eigvecs[i, 1]
ax1.plot(xs * a0, -psi, 'r--')
'''

plt.show()


