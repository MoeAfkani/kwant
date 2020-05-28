# Physics background
import kwant
import numpy as np
from matplotlib import pyplot
import scipy.sparse.linalg as sla
from tqdm import  tqdm

syst = kwant.Builder()

b ,a ,c = 4.43, 3.27, 1

lat = kwant.lattice.Polyatomic([[a,0],[0,b]],#bravis vectores
                               [
                                #basis
                                [0.00*a , 0.00*b ],
                                [0.50*a , 0.25*b ],
                                [0.50*a , 0.50*b ],
                                [0.00*a , 0.75*b ]])

t1 = -1.220
t2 =  3.665
t3 = -0.205
t4 = -0.105
t5 = -0.055

L = 15  #Q=33 Ly=5.2nm
W = 11 #N=20 Lx=4.4nm
# Define the scattering region
sub_a, sub_b, sub_c, sub_d = lat.sublattices

base = 0

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
        if i >0:
            syst[sub_a(i, j),sub_b(i-1, j)] = t1  
            syst[sub_d(i, j),sub_c(i-1, j)] = t1  
########### t1 ########### 
########### t2 ########### 
        syst[sub_c(i, j),sub_b(i, j)] = t2
        if j >0:
            syst[sub_a(i, j),sub_d(i, j-1)] = t2
########### t2 ########### 
########### t3 ########### 
        if i >0 and j>0:
            syst[sub_a(i, j),sub_b(i-1,j-1)] = t3
        if j >0:
            syst[sub_a(i, j),sub_b(i,j-1)] = t3
        if i >0 and j>0:
            syst[sub_d(i,j-1),sub_c(i-1,j)] = t3            
        if j >0:
            syst[sub_c(i, j),sub_d(i,j-1)] = t3
########### t3 ########### 
########### t4 ###########
        syst[sub_b(i, j),sub_d(i, j)] = t4
        syst[sub_c(i, j),sub_a(i, j)] = t4
        if i >0:
            syst[sub_c(i-1, j),sub_a(i, j)] = t4
            syst[sub_b(i-1, j),sub_d(i, j)] = t4
        if j >0:
            syst[sub_c(i, j-1),sub_a(i, j)] = t4
            syst[sub_d(i, j-1),sub_b(i, j)] = t4
        if i*j >0:
            syst[sub_a(i-1, j-1),sub_c(i, j)] = t4
            syst[sub_d(i, j-1),sub_b(i-1, j)] = t4
########### t4 ########### 
########### t5 ########### 
        syst[sub_d(i, j),sub_a(i, j)] = t5
        if j>0:
            syst[sub_b(i, j),sub_c(i, j-1)] = t5
########### t5 ########### 


# First the lead to the left
sym_left_lead = kwant.TranslationalSymmetry((-a, 0))
left_lead = kwant.Builder(sym_left_lead)

for i in range(2):
    for j in tqdm(range(W)):
# On-site Hamiltonian
        left_lead[sub_a(i, j)] = base
        left_lead[sub_b(i, j)] = base
        left_lead[sub_c(i, j)] = base
        left_lead[sub_d(i, j)] = base
#hopings
########### t1 ########### 
        left_lead[sub_a(i, j),sub_b(i, j)] = t1  
        left_lead[sub_d(i, j),sub_c(i, j)] = t1  
        if i >0:
            left_lead[sub_a(i, j),sub_b(i-1, j)] = t1  
            left_lead[sub_d(i, j),sub_c(i-1, j)] = t1  
########### t1 ########### 
########### t2 ########### 
        left_lead[sub_c(i, j),sub_b(i, j)] = t2
        if j >0:
            left_lead[sub_a(i, j),sub_d(i, j-1)] = t2
########### t2 ########### 
########### t3 ########### 
        if i >0 and j>0:
            left_lead[sub_a(i, j),sub_b(i-1,j-1)] = t3
        if j >0:
            left_lead[sub_a(i, j),sub_b(i,j-1)] = t3
        if i >0 and j>0:
            left_lead[sub_d(i,j-1),sub_c(i-1,j)] = t3            
        if j >0:
            left_lead[sub_c(i, j),sub_d(i,j-1)] = t3
########### t3 ########### 
########### t4 ###########
        left_lead[sub_b(i, j),sub_d(i, j)] = t4
        left_lead[sub_c(i, j),sub_a(i, j)] = t4
        if i >0:
            left_lead[sub_c(i-1, j),sub_a(i, j)] = t4
            left_lead[sub_b(i-1, j),sub_d(i, j)] = t4
        if j >0:
            left_lead[sub_c(i, j-1),sub_a(i, j)] = t4
            left_lead[sub_d(i, j-1),sub_b(i, j)] = t4
        if i*j >0:
            left_lead[sub_a(i-1, j-1),sub_c(i, j)] = t4
            left_lead[sub_d(i, j-1),sub_b(i-1, j)] = t4
########### t4 ########### 
########### t5 ########### 
        left_lead[sub_d(i, j),sub_a(i, j)] = t5
        if j>0:
            left_lead[sub_b(i, j),sub_c(i, j-1)] = t5
########### t5 ########### 
syst.attach_lead(left_lead)
right_lead=left_lead.reversed()
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


fig, (ax1, ax0) = pyplot.subplots(ncols=2, figsize=[6, 10])

# Now that we have the system, we can compute conductance
energies = []
data = []
for energy in tqdm(np.linspace(-3,3,1000)):
    # compute the scattering matrix at a given energy
    smatrix = kwant.smatrix(syst, energy)

    # compute the transmission probability from lead 0 to
    # lead 1
    energies.append(energy)
    data.append(smatrix.transmission(1, 0))
# Use matplotlib to write output
# We should see conductance steps
#pyplot.figure()
ax0.plot(data , energies, "r")
#ax0.set_ylabel("energy [t]")
ax0.set_xlabel("conductance [e^2/h]")
ax0.set_ylim(-3,3)
ax0.set_xlim(-0.1,6.1)
#pyplot.show()
momenta = np.linspace(-np.pi, np.pi, 101)
bands = kwant.physics.Bands(syst.leads[0])
energies = [bands(k) for k in momenta]
#pyplot.figure(figsize=[2.0, 5])
momentaPI = [mom/np.pi for mom in momenta]
ax1.plot(momentaPI, energies, "b")
ax1.set_xlabel("Wavevctor [(lattice constant)^-1]")
ax1.set_ylabel("energy [t]")
ax1.set_ylim(-3,3)
ax1.set_xlim(-1,1)
pyplot.savefig("ZigZag.png", dpi=300)

pyplot.show()
    




'''
import pandas as pd
## convert your array into a dataframe
df = pd.DataFrame (syst.hamiltonian_submatrix())

## save to xlsx file

filepath = 'hamiltoni.xlsx'

df.to_excel(filepath, index=False)
'''