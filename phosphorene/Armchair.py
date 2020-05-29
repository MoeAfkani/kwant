# Physics background
import kwant
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sla
from tqdm import  tqdm
import scipy
from IPython.display import set_matplotlib_formats

syst = kwant.Builder()

a ,b ,c = 4.43, 3.27, 1

lat = kwant.lattice.Polyatomic([[a,0],[0,b]],#bravis vectores
                               [
                                #basis
                                [0.00*a , 0.00*b ],
                                [0.25*a , 0.50*b ],
                                [0.50*a , 0.50*b ],
                                [0.75*a , 0.00*b ]],
                                       norbs=1)

t1 = -1.220
t2 =  3.665
t3 = -0.205
t4 = -0.105
t5 = -0.055

W = 11 #Q = 21
L = 16 #N = 32
# Define the scattering region
sub_a, sub_b, sub_c, sub_d = lat.sublattices

base = 0.0

for i in range(L):
    for j in range(W):
# On-site Hamiltonian
        syst[sub_a(i, j)] = base
        syst[sub_b(i, j)] = base
        syst[sub_c(i, j)] = base
        syst[sub_d(i, j)] = base
#hopings
########### t1 ########### 
        syst[sub_a(i, j),sub_b(i, j)] = t1  
        syst[sub_d(i, j),sub_c(i, j)] = t1  
        if j >0:
            syst[sub_a(i, j),sub_b(i, j-1)] = t1  
            syst[sub_d(i, j),sub_c(i, j-1)] = t1  
########### t1 ########### 
########### t2 ########### 
        syst[sub_c(i, j),sub_b(i, j)] = t2
        if i >0:
            syst[sub_a(i, j),sub_d(i-1, j)] = t2
########### t2 ########### 
########### t3 ########### 
        if i >0 and j>0:
            syst[sub_a(i, j),sub_b(i-1,j-1)] = t3
        if i >0:
            syst[sub_a(i, j),sub_b(i-1,j)] = t3
        if i >0 and j>0:
            syst[sub_d(i-1,j),sub_c(i,j-1)] = t3            
        if i >0:
            syst[sub_c(i, j),sub_d(i-1,j)] = t3
########### t3 ########### 
########### t4 ###########
        syst[sub_b(i, j),sub_d(i, j)] = t4
        syst[sub_c(i, j),sub_a(i, j)] = t4
        if j >0:
            syst[sub_c(i, j-1),sub_a(i, j)] = t4
            syst[sub_b(i, j-1),sub_d(i, j)] = t4
        if i >0:
            syst[sub_c(i-1, j),sub_a(i, j)] = t4
            syst[sub_d(i-1, j),sub_b(i, j)] = t4
        if i*j >0:
            syst[sub_a(i-1, j-1),sub_c(i, j)] = t4
            syst[sub_d(i-1, j),sub_b(i, j-1)] = t4
########### t4 ########### 
########### t5 ########### 
        syst[sub_d(i, j),sub_a(i, j)] = t5
        if i>0:
            syst[sub_b(i, j),sub_c(i-1, j)] = t5
########### t5 ########### 

# First the lead to the left
sym_left_lead = kwant.TranslationalSymmetry((-a, 0))
left_lead = kwant.Builder(sym_left_lead)

for i in range(2):
    for j in range(W):
# On-site Hamiltonian
        left_lead[sub_a(i, j)] = base
        left_lead[sub_b(i, j)] = base
        left_lead[sub_c(i, j)] = base
        left_lead[sub_d(i, j)] = base
#hopings
########### t1 ########### 
        left_lead[sub_a(i, j),sub_b(i, j)] = t1  
        left_lead[sub_d(i, j),sub_c(i, j)] = t1  
        if j >0:
            left_lead[sub_a(i, j),sub_b(i, j-1)] = t1  
            left_lead[sub_d(i, j),sub_c(i, j-1)] = t1  
########### t1 ########### 
########### t2 ########### 
        left_lead[sub_c(i, j),sub_b(i, j)] = t2
        if i >0:
            left_lead[sub_a(i, j),sub_d(i-1, j)] = t2
########### t2 ########### 
########### t3 ########### 
        if i >0 and j>0:
            left_lead[sub_a(i, j),sub_b(i-1,j-1)] = t3
        if i >0:
            left_lead[sub_a(i, j),sub_b(i-1,j)] = t3
        if i >0 and j>0:
            left_lead[sub_d(i-1,j),sub_c(i,j-1)] = t3            
        if i >0:
            left_lead[sub_c(i, j),sub_d(i-1,j)] = t3
########### t3 ########### 
########### t3 ########### 
########### t4 ###########
        left_lead[sub_b(i, j),sub_d(i, j)] = t4
        left_lead[sub_c(i, j),sub_a(i, j)] = t4
        if j >0:
            left_lead[sub_c(i, j-1),sub_a(i, j)] = t4
            left_lead[sub_b(i, j-1),sub_d(i, j)] = t4
        if i >0:
            left_lead[sub_c(i-1, j),sub_a(i, j)] = t4
            left_lead[sub_d(i-1, j),sub_b(i, j)] = t4
        if i*j >0:
            left_lead[sub_a(i-1, j-1),sub_c(i, j)] = t4
            left_lead[sub_d(i-1, j),sub_b(i, j-1)] = t4
########### t4 ########### 
########### t5 ########### 
        left_lead[sub_d(i, j),sub_a(i, j)] = t5
        if i>0:
            left_lead[sub_b(i, j),sub_c(i-1, j)] = t5
########### t5 ########### 


syst.attach_lead(left_lead)

# Then the lead to the right
sym_right_lead = kwant.TranslationalSymmetry((a, 0))
right_lead = kwant.Builder(sym_right_lead)
for i in range(2):
    for j in range(W):
# On-site Hamiltonian
        right_lead[sub_a(i, j)] = base
        right_lead[sub_b(i, j)] = base
        right_lead[sub_c(i, j)] = base
        right_lead[sub_d(i, j)] = base
#hopings
########### t1 ########### 
        right_lead[sub_a(i, j),sub_b(i, j)] = t1  
        right_lead[sub_d(i, j),sub_c(i, j)] = t1  
        if j >0:
            right_lead[sub_a(i, j),sub_b(i, j-1)] = t1  
            right_lead[sub_d(i, j),sub_c(i, j-1)] = t1  
########### t1 ########### 
########### t2 ########### 
        right_lead[sub_c(i, j),sub_b(i, j)] = t2
        if i >0:
            right_lead[sub_a(i, j),sub_d(i-1, j)] = t2
########### t2 ########### 
########### t3 ########### 
        if i >0 and j>0:
            right_lead[sub_a(i, j),sub_b(i-1,j-1)] = t3
        if i >0:
            right_lead[sub_a(i, j),sub_b(i-1,j)] = t3
        if i >0 and j>0:
            right_lead[sub_d(i-1,j),sub_c(i,j-1)] = t3            
        if i >0:
            right_lead[sub_c(i, j),sub_d(i-1,j)] = t3
########### t3 ########### 
########### t4 ###########
        right_lead[sub_b(i, j),sub_d(i, j)] = t4
        right_lead[sub_c(i, j),sub_a(i, j)] = t4
        if j >0:
            right_lead[sub_c(i, j-1),sub_a(i, j)] = t4
            right_lead[sub_b(i, j-1),sub_d(i, j)] = t4
        if i >0:
            right_lead[sub_c(i-1, j),sub_a(i, j)] = t4
            right_lead[sub_d(i-1, j),sub_b(i, j)] = t4
        if i*j >0:
            right_lead[sub_a(i-1, j-1),sub_c(i, j)] = t4
            right_lead[sub_d(i-1, j),sub_b(i, j-1)] = t4
########### t4 ########### 
########### t5 ########### 
        right_lead[sub_d(i, j),sub_a(i, j)] = t5
        if i>0:
            right_lead[sub_b(i, j),sub_c(i-1, j)] = t5
########### t5 ########### 

        
syst.attach_lead(right_lead)
# Plot it, to make sure it's OK
def family_colors(site):
    return 'r' if (site.family == sub_a or site.family == sub_b)  else 'g'
kwant.plot(syst,site_size=0.12,site_color=family_colors)
kwant.plotter.sys_leads_sites(syst)
# Finalize the system

syst = syst.finalized()

def plot_wave_function(sys):
    # Calculate the wave functions in the system.
    ham_mat = sys.hamiltonian_submatrix(sparse=True)
    evecs = sla.eigsh(ham_mat, k=10, which='SM')[1]

    # Plot the probability density of the 10th eigenmode.
    kwant.plotter.map(sys, np.abs(evecs[:, 9])**2,
                       oversampling=1)
    
#plot_wave_function(syst)

fig, (ax1, ax0) = plt.subplots(ncols=2, figsize=[6, 10])

# Now that we have the system, we can compute conductance
energies = []
data = []
for energy in tqdm(np.linspace(-3,3,10)):
    # compute the scattering matrix at a given energy
    smatrix = kwant.smatrix(syst, energy)

    # compute the transmission probability from lead 0 to
    # lead 1
    energies.append(energy)
    data.append(smatrix.transmission(1, 0))
# Use matplotlib to write output
# We should see conductance steps
#plt.figure()
ax0.plot(data , energies, "r")
#ax0.set_ylabel("energy [t]")
ax0.set_xlabel("conductance [e^2/h]")
ax0.set_ylim(-3,3)
ax0.set_xlim(-0.1,10)
##plt.show()
momenta = np.linspace(-np.pi, np.pi, 101)
bands = kwant.physics.Bands(syst.leads[0])
energies = [bands(k) for k in momenta]
#plt.figure(figsize=[2.0, 5])
momentaPI = [mom/np.pi for mom in momenta]
ax1.plot(momentaPI, energies, "b")
ax1.set_xlabel("Wavevctor [(lattice constant)^-1]")
ax1.set_ylabel("energy [t]")
ax1.set_ylim(-3,3)
ax1.set_xlim(-1,1)
plt.savefig("ArmChair.png", dpi=300)
#plt.show()
    


'''
import pandas as pd
## convert your array into a dataframe
df = pd.DataFrame (syst.hamiltonian_submatrix())
## save to xlsx file
filepath = 'hamiltoni.xlsx'
df.to_excel(filepath, index=False)
'''
spectrum = kwant.kpm.SpectralDensity(syst, rng=0)
energies, densities = spectrum()
energy_subset = np.linspace(-7,-1)
density_subset = spectrum(energy_subset)
plt.plot(energy_subset,density_subset)

# Fermi energy 0.1 and temperature 0.2
fermi = lambda E: 1 / (np.exp((E - 0.1) / 0.2) + 1)

print('number of filled states:', spectrum.integrate(fermi))

def plot_dos(labels_to_data):
    plt.figure()
    for label, (x, y) in labels_to_data:
        plt.plot(x, y.real, label=label, linewidth=2)
    plt.legend(loc=2, framealpha=0.5)
    plt.xlabel("energy [t]")
    plt.ylabel("DoS [a.u.]")
    plt.savefig("ArmchairDos.png", dpi=300)   
    #plt.show()

# Plot fill density of states plus curves on the same axes.
def plot_dos_and_curves(dos, labels_to_data):
    plt.figure()
    plt.fill_between(dos[0], dos[1], label="DoS [a.u.]",
                     alpha=0.5, color='gray')
    for label, (x, y) in labels_to_data:
        plt.plot(x, y, label=label, linewidth=2)
    plt.legend(loc=2, framealpha=0.5)
    plt.xlabel("energy [t]")
    plt.ylabel("$σ [e^2/h]$")
    plt.savefig("ArmchairDosIANDcure.png", dpi=300)
    #plt.show()


def site_size_conversion(densities):
    return 3 * np.abs(densities) / max(densities)


# Plot several local density of states maps in different subplots
def plot_ldos(syst, densities):
    fig, axes = plt.subplots(1, len(densities), figsize=(7*len(densities), 7))
    for ax, (title, rho) in zip(axes, densities):
        kwant.plotter.density(syst, rho.real, ax=ax)
        ax.set_title(title)
        ax.set(adjustable='box', aspect='equal')
    plt.savefig("ArmchairLDos.png", dpi=300)   
    #plt.show()
    
    
spectrum = kwant.kpm.SpectralDensity(syst, rng=0)

energies, densities = spectrum()

energy_subset = np.linspace(-3 ,3)
density_subset = spectrum(energy_subset)

plot_dos([
    ('densities', (energies, densities)),
    ('density subset', (energy_subset, density_subset)),
])

print('identity resolution:', spectrum.integrate())

# Fermi energy 0.1 and temperature 0.2
fermi = lambda E: 1 / (np.exp((E - 0.1) / 0.2) + 1)

print('number of filled states:', spectrum.integrate(fermi))

def make_syst_staggered(r=30, t=-1, a=1, m=0.1):
    syst = kwant.Builder()
    lat = kwant.lattice.honeycomb(a, norbs=1)

    def circle(pos):
        x, y = pos
        return x ** 2 + y ** 2 < r ** 2

    syst[lat.a.shape(circle, (0, 0))] = m
    syst[lat.b.shape(circle, (0, 0))] = -m
    syst[lat.neighbors()] = t
    syst.eradicate_dangling()

    return syst

fsyst_staggered = make_syst_staggered().finalized()
# find 'A' and 'B' sites in the unit cell at the center of the disk
center_tag = np.array([0, 0])
where = lambda s: s.tag == center_tag
# make local vectors
vector_factory = kwant.kpm.LocalVectors(fsyst_staggered, where)

# 'num_vectors' can be unspecified when using 'LocalVectors'
local_dos = kwant.kpm.SpectralDensity(fsyst_staggered, num_vectors=None,
                                      vector_factory=vector_factory,
                                      mean=False,
                                      rng=0)
energies, densities = local_dos()

plot_dos([
    ('A sublattice', (energies, densities[:, 0])),
    ('B sublattice', (energies, densities[:, 1])),
])

spectrum = kwant.kpm.SpectralDensity(syst, rng=0)
original_dos = spectrum()

spectrum.add_moments(energy_resolution=0.03)

spectrum.add_moments(100)
spectrum.add_vectors(5)

increased_moments_dos = spectrum()
plot_dos([
    ('density', original_dos),
    ('higher number of moments', increased_moments_dos),
])

# identity matrix
matrix_op = scipy.sparse.eye(len(syst.sites))
matrix_spectrum = kwant.kpm.SpectralDensity(syst, operator=matrix_op, rng=0)

# 'sum=True' means we sum over all the sites
kwant_op = kwant.operator.Density(syst, sum=True)
operator_spectrum = kwant.kpm.SpectralDensity(syst, operator=kwant_op, rng=0)

# 'sum=False' is the default, but we include it explicitly here for clarity.
kwant_op = kwant.operator.Density(syst, sum=False)
kwant_op = kwant.operator.Density(syst, sum=False)
local_dos = kwant.kpm.SpectralDensity(syst, operator=kwant_op, rng=0)

zero_energy_ldos = local_dos(energy=0)
finite_energy_ldos = local_dos(energy=1)

plot_ldos(syst, [
    ('energy = 0', zero_energy_ldos),
    ('energy = 1', finite_energy_ldos)
])

def make_syst_topo(r=30, a=1, t=1, t2=0.5):
    syst = kwant.Builder()
    lat = kwant.lattice.honeycomb(a, norbs=1, name=['a', 'b'])

    def circle(pos):
        x, y = pos
        return x ** 2 + y ** 2 < r ** 2

    syst[lat.shape(circle, (0, 0))] = 0.
    syst[lat.neighbors()] = t
    # add second neighbours hoppings
    syst[lat.a.neighbors()] = 1j * t2
    syst[lat.b.neighbors()] = -1j * t2
    syst.eradicate_dangling()

    return lat, syst.finalized()

# construct the Haldane model
lat, fsyst_topo = make_syst_topo()
# find 'A' and 'B' sites in the unit cell at the center of the disk
where = lambda s: np.linalg.norm(s.pos) < 1

# component 'xx'
s_factory = kwant.kpm.LocalVectors(fsyst_topo, where)
cond_xx = kwant.kpm.conductivity(fsyst_topo, alpha='x', beta='x', mean=True,
                                 num_vectors=None, vector_factory=s_factory,
                                 rng=0)
# component 'xy'
s_factory = kwant.kpm.LocalVectors(fsyst_topo, where)
cond_xy = kwant.kpm.conductivity(fsyst_topo, alpha='x', beta='y', mean=True,
                                 num_vectors=None, vector_factory=s_factory,
                                 rng=0)

energies = cond_xx.energies
cond_array_xx = np.array([cond_xx(e, temperature=0.01) for e in energies])
cond_array_xy = np.array([cond_xy(e, temperature=0.01) for e in energies])

# area of the unit cell per site
area_per_site = np.abs(np.cross(*lat.prim_vecs)) / len(lat.sublattices)
cond_array_xx /= area_per_site
cond_array_xy /= area_per_site

s_factory = kwant.kpm.LocalVectors(fsyst_topo, where)
spectrum = kwant.kpm.SpectralDensity(fsyst_topo, num_vectors=None,
                                     vector_factory=s_factory,
                                     rng=0)


plot_dos_and_curves(
(spectrum.energies, spectrum.densities * 8),
[
    (r'Longitudinal conductivity $\sigma_{xx} / 4$',
     (energies, cond_array_xx.real / 4)),
    (r'Hall conductivity $\sigma_{xy}$',
     (energies, cond_array_xy.real))],
)


# construct a generator of vectors with n random elements -1 or +1.
n = syst.hamiltonian_submatrix(sparse=True).shape[0]
def binary_vectors():
    while True:
        yield np.rint(np.random.random_sample(n)) * 2 - 1

custom_factory = kwant.kpm.SpectralDensity(syst,
                                           vector_factory=binary_vectors(),
                                           rng=0)

rho = kwant.operator.Density(syst, sum=True)

# sesquilinear map that does the same thing as `rho`
def rho_alt(bra, ket):
    return np.vdot(bra, ket)

rho_spectrum = kwant.kpm.SpectralDensity(syst, operator=rho, rng=0)
rho_alt_spectrum = kwant.kpm.SpectralDensity(syst, operator=rho_alt, rng=0)
