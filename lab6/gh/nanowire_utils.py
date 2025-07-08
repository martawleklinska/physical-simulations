#imports all necessary libraries including Kwant
import kwant
import numpy as np
import matplotlib.pyplot as plt
       
#the calculations are done in atomic units e=h=me=1. Here the conversion factors are defined.
def eV2au(energy): #eV -> j.a
    return energy*0.03674932587122423
def au2eV(energy): #j.a -> eV
    return energy*27.2117
def nm2au(length): #nm -> j.a
    return length*18.89726133921252
def T2au(length):  #nm -> j.a
    return length*4.254382E-6

def make_system(nw):
    m = nw.m
    dx= nw.dx
    L = nw.L
    W = nw.W
    V0 = nw.V0
    sigma = nw.sigma
    x0 = nm2au(L)
    y0 = 0.
    B = nw.B

    def pot(x, y):
        return V0 * np.exp( (- (x - x0)**2 -  (y - y0)**2) / sigma**2)

    def onsite(site):
        (x, y) = site.pos
        t=1.0/(2.0*m*dx*dx)
        return 4*t + pot(x,y)

    def hopping_x(sitei, sitej):
        (xi, yi) = sitei.pos
        (xj, yj) = sitej.pos
        t = 1.0/(2.0*m*dx*dx) * np.exp(-1.0j/2 * B * (xi - xj) * (yi + yj))
        return -t
    
    def hopping_y(sitei, sitej):
         (xi, yi) = sitei.pos
         (xj, yj) = sitej.pos
         t=1.0/(2.0*m*dx*dx) * np.exp(-1.0j/2 * B * (xi - xj) * (yi + yj))
         return -t

    # We subsequently define:
    # 1. Define the system - function kwant.Builder()  
    # 2. Describe the geometry of the grid - kwant.lattice.square(dx, norbs=1)
    # 3. Fill the Hamiltonian matrix
    sys = kwant.Builder()  
    lat = kwant.lattice.square(dx, norbs=1)
    sys[(lat(i,j) for i in range(L) for j in range(-W,W))]=onsite
    sys[(kwant.builder.HoppingKind((-1,0), lat, lat))] = hopping_x
    sys[(kwant.builder.HoppingKind((0,-1), lat, lat))] = hopping_y
    
    
    #attach the left contact to the system
    lead_left = kwant.Builder(kwant.TranslationalSymmetry((-dx, 0)))    
    lead_left[(lat(0,j) for j in range(-W,W))]=onsite
    lead_left[(kwant.builder.HoppingKind((-1,0), lat, lat))] = hopping_x
    lead_left[(kwant.builder.HoppingKind((0,-1), lat, lat))] = hopping_y
    sys.attach_lead(lead_left)
    
    
    #attach the right contact to the system
    lead_right = kwant.Builder(kwant.TranslationalSymmetry((dx, 0)))    
    lead_right[(lat(0,j) for j in range(-W,W))]=onsite
    lead_right[(kwant.builder.HoppingKind((-1,0), lat, lat))] = hopping_x
    lead_right[(kwant.builder.HoppingKind((0,-1), lat, lat))] = hopping_y
    sys.attach_lead(lead_right)
    
    #finalize the system
    sys = sys.finalized()
    return sys

############################################################################################
##       Define various functions calculating the basic physical properties               ##
############################################################################################

#calculates the dispersion relation in the contact nr_lead in the range [-k_max,k_max] with nk points
def disperssion(nw, nr_lead, k_max, nk):
    dx=nw.dx
    sys=make_system(nw)
    momenta = np.linspace(-k_max*dx,k_max*dx,nk)
    bands=kwant.physics.Bands(sys.leads[nr_lead])
    energies=[bands(k) for k in momenta]
    return (momenta/dx)*nm2au(1.0), energies

#calculates the reflection and transmission coefficient
def transmission_reflection(nw, E):
    E=eV2au(E)
    sys=make_system(nw)
    smatrix=kwant.smatrix(sys,E)
    r=smatrix.transmission(0,0)
    t=smatrix.transmission(1,0)
    return r, t

#calculates the transmission coefficient
def transmission(nw, E):
    E=eV2au(E)
    sys=make_system(nw)
    smatrix=kwant.smatrix(sys,E)
    t=smatrix.transmission(1,0)
    return t

#calculates the conductance - the Landauer formula is used
def conductance(nw, Emax, ne):
    energies=np.linspace(0,Emax,ne)
    cond=[transmission(nw, E) for E in energies]
    return energies, cond

#plots the wave function of an electron with energy E incident in the contact nr_lead
def wave_function(nw, E, nr_lead, *, ax = None):
    E=eV2au(E)
    sys=make_system(nw)
    wave=kwant.wave_function(sys, E)
    density=(abs(wave(nr_lead))**2).sum(axis=0)
    kwant.plotter.map(sys,density, ax = ax, dpi = 300)

#fplots the dos of an electron with energy E
def dos(nw, E):
    E=eV2au(E)
    sys=make_system(nw)
    dos=kwant.ldos(sys, E)
    f = kwant.plotter.map(sys,dos)
    return f

#plots the current of an electron with energy E incident in the contact nr_lead in the state nr_mod
def current(nw, E, nr_lead, nr_mod, *, ax = None):
    E=eV2au(E)
    sys=make_system(nw)
    current = kwant.operator.Current(sys).bind()
    psi=kwant.wave_function(sys, E)(nr_lead)
    curr=current(psi[nr_mod])
    kwant.plotter.current(sys,curr, ax = ax)

class NanowireSystem():
    def __init__(self, *, dx = nm2au(2), L = int(100), W = int(15), m = 0.014
                        , V0 = eV2au(0.05), sigma = nm2au(10), B = T2au(0)):
        self.dx = dx
        self.L = int(L)
        self.W = int(W)
        self.m = m
        self.V0 = V0
        self.sigma = sigma
        self.B = B