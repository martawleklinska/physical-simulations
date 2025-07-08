import numpy
from scipy import linalg
import numpy as np
import time
import numba as nb

import matplotlib.pyplot as plt
# ============ for latex fonts ============
from matplotlib import rc #, font_manager
rc('text.latex', preamble=r'\usepackage{lmodern}')# this helps use the plots in tex files
plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'xtick.labelsize': 14,
		  'ytick.labelsize': 14,
		  'xtick.major.pad': 6,
		  'ytick.major.pad': 6,
          'axes.titlesize': 14,
		  'font.serif': 'Computer Modern Roman',
		  'axes.formatter.use_mathtext': True,
		  'axes.labelpad': 6.0 }) 
# ==========================================

# bledy:
# - macierz 2N x 2N nie jest hermitowska, trzeba uzyc linalg.eig
# - poza tym od razu trzeba zrobic macierze A i B w powyzszym zespolone (dodac 0j)
# - w macierzy blokowej Ef*I-H0, przy Ef brakowalo np.eye :(
# - przy wyliczaniu tego co nazwalam alfa, brakowalo podwojnej sumy.
# - wspolczynniki odbicia nie chcialy dzialac: znaki w wyrazie wolnym na odwrot. T wychodzilo dobrze mimo wszystko

# plt.figure(figsize=(8, 6), dpi=80)
Ha = 27.211 # 1 Hartree in eV
a0 = 0.0529177249 # Bohr radius in nm
m = 0.017 # effective mass for InSb

@nb.njit
def idx1D(i, j, Ny):
    return i * Ny + j

#generate a Hamiltonian as a function of L
@nb.njit
def make_H(Nx, Ny, Lx, Ly, Vsg):
    print("Lx, Ly=", Lx*a0, Ly*a0)
    dx = Lx / (Nx+1) # the grid spacing
    dy = Ly / (Ny+1) # the grid spacing
    alpha1 = 0.50 / m / dx**2 # in atomic units
    alpha2 = 0.50 / m / dy**2 # in atomic units
    
    # H = numpy.zeros((Nx * Ny * 2, Nx * Ny * 2))
    H = numpy.zeros((Nx * Ny, Nx * Ny)) + 0j
    V = numpy.zeros((Nx, Ny)) # here potential energy is zero, but it can be changed easily
    for i in range(0, Nx): #diagonal elements
        for j in range(0, Ny):
            x = (i+1) * dx - Lx/2
            y = (j+1) * dx - Ly/2
            
            V[i, j] = -0.035/Ha * Vsg * np.exp(-(x**2/(300/a0)**2+(y+Ly/2)**2/(300/a0)**2)**2.) # to bylo duze
            V[i, j] +=-0.035/Ha * Vsg * np.exp(-(x**2/(300/a0)**2+(y-Ly/2)**2/(300/a0)**2)**2.)
    
    for i in range(0, Nx): #diagonal elements
        for j in range(0, Ny):
            H[idx1D(i, j, Ny), idx1D(i, j, Ny)] = 2*alpha1 + 2*alpha2 + V[i, j]
            
    for i in range(1, Nx): #non-diagonal elements, below diagonal (left neighbor)
        for j in range(0, Ny):
            H[ idx1D(i, j, Ny), idx1D(i - 1, j, Ny)] = -alpha1
    
    for i in range(0, Nx-1): #non-diagonal elements, above diagonal (right neighbor)
        for j in range(0, Ny):
            H[idx1D(i, j, Ny), idx1D(i + 1, j, Ny)] = -alpha1
           
    for i in range(0, Nx): #non-diagonal elements, (lower neighbor)
        for j in range(1, Ny):
            H[ idx1D(i, j, Ny), idx1D(i, j - 1, Ny)] = -alpha2
    
    for i in range(0, Nx): #non-diagonal elements, (upper neighbor)
        for j in range(0, Ny-1):
            H[idx1D(i, j, Ny), idx1D(i, j + 1, Ny)] = -alpha2
            
    return H, V


#generate a matrix for the modes in a lead
@nb.njit
def make_lead_H(Nx, Ny, Lx, Ly, Ef):
    dx = Lx / (Nx+1) # the grid spacing
    dy = Ly / (Ny+1) # the grid spacing
    alpha1 = 0.50 / m / dx**2 # in atomic units
    alpha2 = 0.50 / m / dy**2 # in atomic units
    
    A = numpy.zeros((Ny * 2, Ny * 2)) + 0j
    B = numpy.zeros((Ny * 2, Ny * 2)) + 0j
    
    H0 = numpy.zeros((Ny, Ny))
    tau = numpy.zeros((Ny, Ny)) 
    # V = numpy.zeros(Ny) # here potential energy is zero, but it can be changed easily
    # for j in range(0, Ny):
    #      y = (j+1) * dx - Ly/2
    #      V[j] = 0.05/Ha * 1 * np.exp(-((y+Ly/2)**2/(300/a0)**2)**2.)
    #      V[j] +=0.05/Ha * 1 * np.exp(-((y-Ly/2)**2/(300/a0)**2)**2.)
    
    for i in np.arange(0, Ny): #tau intercell hoppings, only on the diagonal
        tau[i, i] = -alpha1
        
    for i in np.arange(0, Ny): # H0 diagonal elements
        H0[i, i] = 2 * alpha1 + 2 * alpha2 #+ V[i]
           
    for i in np.arange(1, Ny): # H0 intracell hoppings
        H0[i, i-1] = -alpha2
        H0[i-1, i] = -alpha2
    
    #assembling the matrix for the eigenproblem
    A[:Ny, Ny:] = np.eye(Ny, Ny)
    A[Ny:, :Ny] = -tau
    A[Ny:, Ny:] = Ef*np.eye(Ny, Ny) - H0
    
    B[:Ny, :Ny] = np.eye(Ny, Ny)
    B[Ny:, Ny:] = tau # should be hermitian conjugate but here it is real and diagonal
    
    return A, B, tau 


def calc_lead_modes(Nx, Ny, Lx, Ly, Ef):
    A, B, tau = make_lead_H(Nx, Ny, Lx, Ly, Ef)
    eigvals, eigvecs = linalg.eig(a=A, b=B)
    
    # we choose the propagating modes
    val_prop = eigvals[np.abs(np.abs(eigvals)-1)<1e-2]
    vec_prop = eigvecs[:Ny, np.abs(np.abs(eigvals)-1)<1e-2]
    
    # unormowac mody poprzeczne (2 poniewaz rozmiar macierzy jest 2Ny x 2Ny)
    vec_prop = vec_prop * 2**0.5 #/ dx**0.5 
    
    # find the direction of propagation
    v = -np.imag(val_prop * np.einsum('ij,ji->i', np.conjugate(vec_prop.T), np.matmul(np.conjugate(tau.T), vec_prop)))
    return val_prop[v>0], vec_prop[:, v>0], val_prop[v<0], vec_prop[:, v>0], v[v>0], v[v<0]
    
    
def calc_bandstruct(Nx, Ny, Lx, Ly, ks):
    dx = Lx / (Nx+1) # the grid spacing
    dy = Ly / (Ny+1) # the grid spacing
    alpha1 = 0.50 / m / dx**2 # in atomic units
    alpha2 = 0.50 / m / dy**2 # in atomic units
        
    H0 = numpy.zeros((Ny, Ny))
    tau = numpy.zeros((Ny, Ny)) 
    
    for i in np.arange(0, Ny): #tau elements, only on the diagonal
        tau[i, i] = -alpha1
        
    for i in np.arange(0, Ny): # H0 diagonal elements
        H0[i, i] = 2 * alpha1 + 2 * alpha2
           
    for i in np.arange(1, Ny): # H0 hopping elements
        H0[i, i-1] = -alpha2
        H0[i-1, i] = -alpha2
        
	#assembling the matrix for the eigenproblem	
    def Ek(k):
        A = H0 + tau * np.exp(1j * k * dx) + tau * np.exp(-1j * k * dx) 
        return linalg.eigh(A, eigvals_only=True)
    
    spectrum = [Ek(k)[:] for k in ks]
    return np.array(spectrum)
    

Nx = 9
Ny = 19
Ly = 20 / a0
dx = Ly / (Ny + 1) 
#Lx = 100 / a0
Lx = dx * (Nx + 1)
print('dx=', dx)

for Ef in np.linspace(200 / 1000. / Ha, 400 / 1000. / Ha, 2):
    # we start with the modes in the leads.
    eikap, up, eikam, um, vp, vm = calc_lead_modes(Nx, Ny, Lx, Ly, Ef)
    
    plt.figure()
    
    kmodp = np.imag(np.log(eikap)) / dx / a0 
    kmodm = np.imag(np.log(eikam)) / dx / a0 
    plt.plot(kmodp, np.ones(len(kmodp))*Ef*Ha, 'bo' )
    plt.plot(kmodm, np.ones(len(kmodm))*Ef*Ha, 'ro' )

    no_modes = len(eikap)

    # sprawdzenie, czy znalezione k leza na rel. dysp
    ks = np.linspace(-np.pi/dx, np.pi/dx, 50)
    bandstruct =  calc_bandstruct(Nx, Ny, Lx, Ly, ks)
    plt.plot(ks/a0, bandstruct*Ha, 'b-' , linewidth=1) 
    plt.plot(ks/a0, np.ones(len(ks))*Ef*Ha, 'gray', linestyle='dashed' )
#    plt.ylim(0,5)
#    plt.xlim(-2,2)

    plt.xlabel("$k_x\ (1/\mathrm{nm})$", usetex=True)
    plt.ylabel("$E\ (\mathrm{eV})$", usetex=True)
    plt.savefig(f"E_k_{Ef*Ha}.pdf",bbox_inches='tight', transparent=True)


#%% Starting QTBM 

def calc_transmission(Nx, Ny, Lx, Ly, Vsg, Ef, plot_dens=False):
        # we start with the modes in the leads.
    eikap, up, eikam, um, vp, vm = calc_lead_modes(Nx, Ny, Lx, Ly, Ef)
    no_modes = len(eikap)
   
    # the matrix for the transport system of equations
    A, _ = make_H(Nx, Ny, Lx, Ly, Vsg) 
    
    tau = -0.50 / m / dx**2 * np.eye(Ny, Ny) + 0j
    A -= Ef * np.eye(Nx * Ny, Nx * Ny)
    B = np.zeros(Nx * Ny) + 0j
    
    # adding the right lead -- 
    # 1. overlap matrix
    #X = ( eikap[:, None]**(Nx-1) ) * up.T
    #Sp = X.conj() @ X.T
    #for i in np.arange(no_modes):
        #for j in np.arange(no_modes): 
    #    print(i, Sp[i, :])       
    #Sp = np.linalg.inv(Sp)
    
    # 2. the term originating from the backscattered modes
    beta = np.zeros((Ny, Ny)) + 0j
    for i in np.arange(no_modes):
        #beta += np.outer(np.conjugate(eikap[i]**(Nx-1) * up[:, i].T), eikap[i]**(Nx-1) * up[:, i]) * (1 - eikap[i]) 
        beta += np.outer(np.conjugate(up[:, i].T), up[:, i]) * (1 - eikap[i]) 
    
    # 3. assemble into the matrix
    A[-Ny:, -Ny:] += tau # the coupling matrix from the term u_Nx (b.c.)
    A[-Ny:, -Ny:] -= np.matmul(tau, beta)
    
    
    # adding the left lead    
    # 1. modes' overlap matrices
    #Bb = np.matmul(np.conjugate(um.T), up)
    #for i in np.arange(no_modes):
    #    print(i, Bb[i,:])
    
    # 2. the term originating from the backscattered modes
    alfa = np.zeros((Ny, Ny)) + 0j
    for i in np.arange(no_modes):
        alfa += np.outer(np.conjugate(um[:, i].T), um[:, i]) * (1 - 1/eikam[i])
    
    # 3. assemble into the matrix
    A[:Ny, :Ny] += tau # the coupling matrix from the term u_-1 (b.c.)
    A[:Ny, :Ny] -= np.matmul(tau, alfa)
    
    
    # test of the matrix with Dirichlet b.c. on the other end
    # testing the left lead
    # A[:Ny, :] = 0
    # A[:Ny, :Ny] = np.eye(Ny, Ny)
    # B[:Ny] = up[:, 0]
    
    # testing the right lead
    # A[-Ny:, :] = 0
    # A[-Ny:, -Ny:] = np.eye(Ny, Ny)
    # B[-Ny:] = (eikap**(Nx - 1) * up)[:, ind]
    
    # 4. the free terms on the right side, denpending on the mode injected
    t = 0
    r = 0
    psi_total = np.zeros(Ny*Nx) + 0j
    # idx_in = 0
    for idx_in in np.arange(no_modes):
    # if(idx_in is not None):
        cin = np.zeros(no_modes) + 0j#, dtype=int)
        cin[idx_in] = 1
        cin2 = cin
        
        #B[:Ny] = np.matmul(tau, up[:, idx_in]) * (1 - 1/eikap[idx_in])  # tu byly zle znaki kiedys.
        #B[:Ny] -= np.matmul(tau, um[:, idx_in]) * (1 - 1/eikam[idx_in])  # tu byly zle znaki kiedys.
        B[:Ny] = tau[0,0]*up[:, idx_in] * (1 - 1/eikap[idx_in])  # tu byly zle znaki kiedys.
        B[:Ny] -= tau[0,0]*um[:, idx_in] * (1 - 1/eikam[idx_in])  # tu byly zle znaki kiedys.

        ##%% Solving QTBM 
        # solving the system of equations
        psi = np.linalg.solve(A, B)
        
        ##%% calculating the transmission amplitudes
        psi_in = psi[:Ny]
        psi_out = psi[-Ny:]
        
        cin = np.abs(vp[idx_in])**0.5
        coutt = np.matmul(np.conjugate(um).T, psi_in) 
        cout = (coutt - cin2) * np.abs(vm)**0.5
        #X = ( eikap[:, None]**(Nx-1) ) * up.T
        X = up.T
        dout = np.matmul(np.conjugate(X), psi_out) 
        dout = dout * np.abs(vp)**0.5
        
        t += np.sum(np.abs(dout/cin)**2)
        r += np.sum(np.abs(cout/cin)**2)
        
        psi_total += np.abs(psi)**2
    print('amplitues', t, r, t+r, no_modes)#, cin2[idx_in])
	
    if(plot_dens == True):
        plt.figure()
        y = (np.arange(Ny)+1) * dx * a0 - Ly/2 * a0 
        plt.plot(y, np.real(um[:,0]), '-', label=r'$Re(u_{-,0})$')
        plt.plot(y, np.imag(um[:,0]), '-', label=r'$Im(u_{-,0})$')
            
        if(no_modes>1):
            plt.plot(y, np.real(um[:,1]), '-', label=r'$Re(u_{-,1})$')
            plt.plot(y, np.imag(um[:,1]), '-', label=r'$Im(u_{-,1})$')

        plt.legend(frameon=False, handlelength=1, loc='upper right')
        plt.xlabel("$y\ (\mathrm{nm})$", usetex=True)
        plt.ylabel("$u_{-,n}\ (\mathrm{a.u.})$", usetex=True)
            
        plt.savefig(f"mody_V={Vsg}_E={np.ceil(Ef*Ha*1000)/1000}.pdf",bbox_inches='tight', transparent=True)
            
        # plot the obtained phi
        plt.figure()
        plt.axes().set_aspect('equal')
            
        x = (np.arange(Nx)+1) * dx * a0 - Lx/2 * a0 
        y = (np.arange(Ny)+1) * dx * a0 - Ly/2 * a0 
        plt.imshow(np.abs(psi_total.reshape(Nx, -1)).T)
        
        plt.title(fr"$E=${np.ceil(Ef*Ha*1000)/1000} eV, $V=${Vsg} V", usetex=True)
        plt.xlabel("$x\ (\mathrm{nm})$", usetex=True)
        plt.ylabel("$y\ (\mathrm{nm})$", usetex=True)
        plt.gca().set_xticks(range(0,len(x), 20))
        plt.gca().set_yticks(range(0,len(y), 20))
        plt.gca().set_xticklabels(np.floor(x[::20]))
        plt.gca().set_yticklabels(np.floor(-y[::20]))
        # colorbar dla wykresu
        cbar = plt.colorbar()
        
        plt.savefig(f"psi_V={Vsg}_E={np.ceil(Ef*Ha*1000)/1000}.pdf",bbox_inches='tight', transparent=True)

    return t, r

Nx = 40
Ny = 19
Ly = 20 / a0
dx = Ly / (Ny + 1) 
Lx = dx * (Nx + 1)
print('dx=', dx)

# probne dla malego ukladu
# for Ef in np.linspace(200 / 1000. / Ha, 900 / 1000. / Ha, 8):
for Ef in np.linspace(200 / 1000. / Ha, 400 / 1000. / Ha, 2):
    calc_transmission(Nx, Ny, Lx, Ly, Vsg=-0.80, Ef=Ef, plot_dens=True)


#%% large system 

Nx = 49
Ny = 34
Ly = 700 / a0
dx = Ly / (Ny + 1) 
Lx = dx * (Nx + 1)


# test relacji dyspersji
plt.figure()
ks = np.linspace(-np.pi/dx, np.pi/dx, 50)
bandstruct =  calc_bandstruct(Nx, Ny, Lx, Ly, ks)
plt.plot(ks/a0, bandstruct*Ha, 'b-' , linewidth=1)
plt.plot(ks/a0, np.ones(len(ks))*0.015, 'gray', linestyle='dashed' )

plt.figure()
Vsg = -1
A, V = make_H(Nx, Ny, Lx, Ly, Vsg) 
plt.imshow(V.T) 
plt.xlabel("$x\ (\mathrm{nm})$", usetex=True)
plt.ylabel("$y\ (\mathrm{nm})$", usetex=True)

x = (np.arange(Nx)+1) * dx * a0 - Lx/2 * a0 
y = (np.arange(Ny)+1) * dx * a0 - Ly/2 * a0 
plt.gca().set_xticks(range(0,len(x), 20))
plt.gca().set_yticks(range(0,len(y), 20))
plt.gca().set_xticklabels(np.floor(x[::20]))
plt.gca().set_yticklabels(np.floor(-y[::20]))
plt.savefig("V.pdf",bbox_inches='tight', transparent=True)


Vsg_vals = np.arange(-1.3, -0.7, 0.025) # do tego wiekszego
Np = len(Vsg_vals)
T = np.zeros(Np)
R = np.zeros(Np)
idx = 0
# for Vsg in Vsg_vals:
for Vsg in Vsg_vals:
    t,r = calc_transmission(Nx, Ny, Lx, Ly, Vsg=Vsg, Ef=0.015 / Ha, plot_dens=False)
    T[idx] = t
    R[idx] = r
    print(f"for Vsg={Vsg}, T={t}, R={r}")
    idx += 1
    
plt.figure()
plt.plot(Vsg_vals, T)
plt.grid(axis = 'y')
plt.yticks([0, 1, 2, 3])
plt.xlabel("$V_{gates}\ (\mathrm{V})$", usetex=True)
plt.ylabel("$G\ (2e^2/h)$", usetex=True)
plt.savefig("T_V.pdf",bbox_inches='tight', transparent=True)
plt.figure()
plt.plot(Vsg_vals, R)
plt.xlabel("$V_{gates}\ (\mathrm{V})$", usetex=True)
plt.ylabel("$R$", usetex=True)
plt.savefig("R_V.pdf",bbox_inches='tight', transparent=True)
plt.figure()
plt.plot(Vsg_vals, T+R)
plt.xlabel("$V_{gates}\ (\mathrm{V})$", usetex=True)
plt.ylabel("$T+R$", usetex=True)
plt.ylim(20.99, 21.01)

plt.savefig("TR_V.pdf",bbox_inches='tight', transparent=True)


#%% large system densities
# kilka gęstości 
calc_transmission(Nx, Ny, Lx, Ly, Vsg=-1.3, Ef=0.015 / Ha, plot_dens=True)
calc_transmission(Nx, Ny, Lx, Ly, Vsg=-1.1, Ef=0.015 / Ha, plot_dens=True)
calc_transmission(Nx, Ny, Lx, Ly, Vsg=-0.85, Ef=0.015 / Ha, plot_dens=True)

#plt.show()
