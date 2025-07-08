import kwant

import numpy as np
import matplotlib.pyplot as plt
def onsite(site):
    x, y = site.pos
    return 4 

def hopping(site_0, site_1, B):
    x0, y0 = site_0.pos
    x1, y1 = site_1.pos
    
    C = 2 * np.pi * B 
    
    return - np.exp(-0.5j * C * (y0 + y1) * (x0 - x1))

Rmax = 30
Rmin = 20

Wmin = -5
Wmax = 5

def shape(pos):
    x, y = pos
    return ((Rmin) ** 2 < (x ** 2 + y ** 2) < (Rmax) ** 2)

def shape_lead(pos):
    x, y = pos
    return (Wmin < y < Wmax)

sys = kwant.Builder()
lat = kwant.lattice.square(1, norbs =1)
sys[lat.shape(shape, (0, Rmin+1))] = onsite
sys[lat.neighbors()] = hopping

leadl = kwant.Builder(kwant.TranslationalSymmetry((1*9)))
leadl[lat.shape(shape_lead((0,0)))] = onsite
leadl[lat.neighbors()] = hopping

leadr = leadl.reversed()
sys.attach_lead(leadl)
sys.attach_lead(leadr)

sysf = sys.finalized()

kwant.plot(sysf)