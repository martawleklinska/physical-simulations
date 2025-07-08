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
def T2au(length):  #Tesla -> j.a
    return length*4.254382E-6

# rather Y-junction, but based on ring
class NanoringSystem():
    def __init__(self, *, dx = nm2au(2), L = int(100), W = int(15), m = 0.014
                        , V0 = eV2au(0.05), sigma = nm2au(10), B = T2au(0)
                        , R1 = nm2au(60), R2 = nm2au(120)):
        self.dx = dx
        self.L = int(L)
        self.W = int(W)
        self.m = m
        self.V0 = V0
        self.sigma = sigma
        self.B = B
        self.R1 = R1
        self.R2 = R2

        def y_ring(pos):
            (x, y) = pos
            rsq = x ** 2 + y ** 2

            if((x <= -R2 + 50) and (x > -R2 -self.L * dx / 2)):
                return - self.W * dx < y < self.W * dx

            if((x >= 0) and (x < self.L * dx / 2)):
                top_site = -self.W * dx < y + (self.R1 + self.R2)/2 < self.W * dx 
                bot_site = -self.W * dx < y - (self.R1 + self.R2)/2 < self.W * dx

                return (top_site or bot_site)

            return ((R1 ** 2 < rsq < R2 ** 2 and x < 0) )

        self.ring_function = y_ring

def make_system(nr):
    m = nr.m
    dx= nr.dx
    L = nr.L
    W = nr.W
    V0 = nr.V0
    sigma = nr.sigma
    x0 = nm2au(L)
    y0 = 0.
    B = nr.B

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
    lat = kwant.lattice.square(dx, norbs=1)
    sys = kwant.Builder()  

    sys[lat.shape(nr.ring_function, (-50, nr.R1 + 50))] = onsite

    #sys[(lat(i,j) for i in range(L) for j in range(-W,W))] = onsite
    sys[(kwant.builder.HoppingKind((-1,0), lat, lat))] = hopping_x
    sys[(kwant.builder.HoppingKind((0,-1), lat, lat))] = hopping_y

    #attach the left contact to the system
    lead_left = kwant.Builder(kwant.TranslationalSymmetry((-dx, 0)))    
    lead_left[(lat(0,j) for j in range(-W,W))]=onsite
    lead_left[(kwant.builder.HoppingKind((-1,0), lat, lat))] = hopping_x
    lead_left[(kwant.builder.HoppingKind((0,-1), lat, lat))] = hopping_y
    sys.attach_lead(lead_left)
    
    top_idx = np.round(((nr.R1 + nr.R2) / 2) / nr.dx)

    # attach the right top contact to the system
    lead_top_right = kwant.Builder(kwant.TranslationalSymmetry((dx, 0)))    
    lead_top_right[(lat(0,j + top_idx) for j in range(-W,W))]=onsite
    lead_top_right[(kwant.builder.HoppingKind((-1,0), lat, lat))] = hopping_x
    lead_top_right[(kwant.builder.HoppingKind((0,-1), lat, lat))] = hopping_y
    sys.attach_lead(lead_top_right)

    # attach the right bot contant to the system
    lead_bot_right = kwant.Builder(kwant.TranslationalSymmetry((dx, 0)))    
    lead_bot_right[(lat(0,j - top_idx + 1) for j in range(-W,W))]=onsite
    lead_bot_right[(kwant.builder.HoppingKind((-1,0), lat, lat))] = hopping_x
    lead_bot_right[(kwant.builder.HoppingKind((0,-1), lat, lat))] = hopping_y
    sys.attach_lead(lead_bot_right)
    
    #finalize the system
    sys = sys.finalized()
    return sys

def disperssion(nr, nr_lead, k_max, nk):
    dx=nr.dx
    sys=make_system(nr)
    momenta = np.linspace(-k_max*dx,k_max*dx,nk)
    bands=kwant.physics.Bands(sys.leads[nr_lead])
    energies=[bands(k) for k in momenta]
    return (momenta/dx)*nm2au(1.0), energies

def transmission(nr, E, lead_in, lead_out):
    E=eV2au(E)
    sys=make_system(nr)
    smatrix=kwant.smatrix(sys,E)
    t=smatrix.transmission(lead_in, lead_out)
    return t

def conductance(nw, Emax, ne, lin, lout):
    energies=np.linspace(0,Emax,ne)
    cond=[transmission(nw, E, lin, lout) for E in energies]
    return energies, cond

#plots the current of an electron with energy E incident in the contact nr_lead in the state nr_mod
def current(nr, E, nr_lead, nr_mod, *, ax = None):
    E=eV2au(E)
    sys=make_system(nr)
    current = kwant.operator.Current(sys).bind()
    psi=kwant.wave_function(sys, E)(nr_lead)
    curr=current(psi[nr_mod])
    kwant.plotter.current(sys,curr, ax = ax)

