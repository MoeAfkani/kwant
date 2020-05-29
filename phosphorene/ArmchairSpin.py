# Physics background
import kwant
import numpy as np
from matplotlib import pyplot
import scipy.sparse.linalg as sla
from tqdm import  tqdm
systUp = kwant.Builder()
systDw = kwant.Builder()
a ,b ,c = 4.43, 3.27, 1

lat = kwant.lattice.Polyatomic([[a,0],[0,b]],#bravis vectores
                               [
                                #basis
                                [0.00*a , 0.00*b ],
                                [0.25*a , 0.50*b ],
                                [0.50*a , 0.50*b ],
                                [0.75*a , 0.00*b ]],
                                       norbs=1)


W = 11 #Q = 21
L = 16 #N = 32

e_z=0.184 #184
base = 0

t1 = -1.220
t2 =  3.665
t3 = -0.205
t4 = -0.105
t5 = -0.055


OnSiteUp = base + e_z
OnSiteDw = base - e_z
# Define the scattering region
sub_a, sub_b, sub_c, sub_d = lat.sublattices


for i in range(L):
    for j in range(W):
# On-site Hamiltonian
        systUp[sub_a(i, j)] = OnSiteUp
        systUp[sub_b(i, j)] = OnSiteUp
        systUp[sub_c(i, j)] = OnSiteUp
        systUp[sub_d(i, j)] = OnSiteUp
        
        systDw[sub_a(i, j)] = OnSiteDw
        systDw[sub_b(i, j)] = OnSiteDw
        systDw[sub_c(i, j)] = OnSiteDw
        systDw[sub_d(i, j)] = OnSiteDw
#hopings
########### t1 ########### 
        systUp[sub_a(i, j),sub_b(i, j)] = t1
        systDw[sub_a(i, j),sub_b(i, j)] = t1
        systUp[sub_d(i, j),sub_c(i, j)] = t1
        systDw[sub_d(i, j),sub_c(i, j)] = t1
        if j >0:
            systUp[sub_a(i, j),sub_b(i, j-1)] = t1
            systDw[sub_a(i, j),sub_b(i, j-1)] = t1
            systUp[sub_d(i, j),sub_c(i, j-1)] = t1
            systDw[sub_d(i, j),sub_c(i, j-1)] = t1
########### t1 ########### 
########### t2 ########### 
        systUp[sub_c(i, j),sub_b(i, j)] = t2
        systDw[sub_c(i, j),sub_b(i, j)] = t2
        if i >0:
            systUp[sub_a(i, j),sub_d(i-1, j)] = t2
            systDw[sub_a(i, j),sub_d(i-1, j)] = t2
########### t2 ########### 
########### t3 ########### 
        if i >0 and j>0:
            systUp[sub_a(i, j),sub_b(i-1,j-1)] = t3
            systDw[sub_a(i, j),sub_b(i-1,j-1)] = t3
            systUp[sub_a(i, j),sub_b(i-1,j-1)] = t3
            systDw[sub_a(i, j),sub_b(i-1,j-1)] = t3
        if i >0:
            systUp[sub_a(i, j),sub_b(i-1,j)] = t3
            systDw[sub_a(i, j),sub_b(i-1,j)] = t3
        if i >0 and j>0:
            systUp[sub_d(i-1,j),sub_c(i,j-1)] = t3    
            systDw[sub_d(i-1,j),sub_c(i,j-1)] = t3    
        if i >0:
            systUp[sub_c(i, j),sub_d(i-1,j)] = t3
            systDw[sub_c(i, j),sub_d(i-1,j)] = t3
########### t3 ########### 
########### t4 ###########
        systUp[sub_b(i, j),sub_d(i, j)] = t4
        systDw[sub_b(i, j),sub_d(i, j)] = t4
        systUp[sub_c(i, j),sub_a(i, j)] = t4
        systDw[sub_c(i, j),sub_a(i, j)] = t4
        if j >0:
            systUp[sub_c(i, j-1),sub_a(i, j)] = t4
            systDw[sub_c(i, j-1),sub_a(i, j)] = t4
            systUp[sub_b(i, j-1),sub_d(i, j)] = t4
            systDw[sub_b(i, j-1),sub_d(i, j)] = t4
        if i >0:
            systUp[sub_c(i-1, j),sub_a(i, j)] = t4
            systDw[sub_c(i-1, j),sub_a(i, j)] = t4
            systUp[sub_d(i-1, j),sub_b(i, j)] = t4
            systDw[sub_d(i-1, j),sub_b(i, j)] = t4
        if i*j >0:
            systUp[sub_a(i-1, j-1),sub_c(i, j)] = t4
            systDw[sub_a(i-1, j-1),sub_c(i, j)] = t4
            systUp[sub_d(i-1, j),sub_b(i, j-1)] = t4
            systDw[sub_d(i-1, j),sub_b(i, j-1)] = t4
########### t4 ########### 
########### t5 ########### 
        systUp[sub_d(i, j),sub_a(i, j)] = t5
        systDw[sub_d(i, j),sub_a(i, j)] = t5
        if i>0:
            systUp[sub_b(i, j),sub_c(i-1, j)] = t5
            systDw[sub_b(i, j),sub_c(i-1, j)] = t5
########### t5 ########### 

# First the lead to the left
symUp_left_lead = kwant.TranslationalSymmetry((-a, 0))
left_leadUp = kwant.Builder(symUp_left_lead)
symDw_left_lead = kwant.TranslationalSymmetry((-a, 0))
left_leadDw = kwant.Builder(symDw_left_lead)

for i in range(2):
    for j in range(W):
# On-site Hamiltonian
        left_leadUp[sub_a(i, j)] = OnSiteUp
        left_leadUp[sub_b(i, j)] = OnSiteUp
        left_leadUp[sub_c(i, j)] = OnSiteUp
        left_leadUp[sub_d(i, j)] = OnSiteUp

        left_leadDw[sub_a(i, j)] = OnSiteDw
        left_leadDw[sub_b(i, j)] = OnSiteDw
        left_leadDw[sub_c(i, j)] = OnSiteDw
        left_leadDw[sub_d(i, j)] = OnSiteDw
#hopings
########### t1 ########### 
        left_leadUp[sub_a(i, j),sub_b(i, j)] = t1
        left_leadDw[sub_a(i, j),sub_b(i, j)] = t1
        left_leadUp[sub_d(i, j),sub_c(i, j)] = t1
        left_leadDw[sub_d(i, j),sub_c(i, j)] = t1
        if j >0:
            left_leadUp[sub_a(i, j),sub_b(i, j-1)] = t1
            left_leadDw[sub_d(i, j),sub_c(i, j-1)] = t1
########### t1 ########### 
########### t2 ########### 
        left_leadUp[sub_c(i, j),sub_b(i, j)] = t2
        left_leadDw[sub_c(i, j),sub_b(i, j)] = t2
        if i >0:
            left_leadUp[sub_a(i, j),sub_d(i-1, j)] = t2
            left_leadDw[sub_a(i, j),sub_d(i-1, j)] = t2
########### t2 ########### 
########### t3 ########### 
        if i >0 and j>0:
            left_leadUp[sub_a(i, j),sub_b(i-1,j-1)] = t3
            left_leadDw[sub_a(i, j),sub_b(i-1,j-1)] = t3
        if i >0:
            left_leadUp[sub_a(i, j),sub_b(i-1,j)] = t3
            left_leadDw[sub_a(i, j),sub_b(i-1,j)] = t3
        if i >0 and j>0:
            left_leadUp[sub_d(i-1,j),sub_c(i,j-1)] = t3
            left_leadDw[sub_d(i-1,j),sub_c(i,j-1)] = t3
        if i >0:
            left_leadUp[sub_c(i, j),sub_d(i-1,j)] = t3
            left_leadDw[sub_c(i, j),sub_d(i-1,j)] = t3
########### t3 ########### 
########### t4 ###########
        left_leadUp[sub_b(i, j),sub_d(i, j)] = t4
        left_leadDw[sub_b(i, j),sub_d(i, j)] = t4
        left_leadUp[sub_c(i, j),sub_a(i, j)] = t4
        left_leadDw[sub_c(i, j),sub_a(i, j)] = t4
        if j >0:
            left_leadUp[sub_c(i, j-1),sub_a(i, j)] = t4
            left_leadDw[sub_c(i, j-1),sub_a(i, j)] = t4
            left_leadUp[sub_b(i, j-1),sub_d(i, j)] = t4
            left_leadDw[sub_b(i, j-1),sub_d(i, j)] = t4
        if i >0:
            left_leadUp[sub_c(i-1, j),sub_a(i, j)] = t4
            left_leadDw[sub_c(i-1, j),sub_a(i, j)] = t4
            left_leadUp[sub_d(i-1, j),sub_b(i, j)] = t4
            left_leadDw[sub_d(i-1, j),sub_b(i, j)] = t4
        if i*j >0:
            left_leadUp[sub_a(i-1, j-1),sub_c(i, j)] = t4
            left_leadDw[sub_a(i-1, j-1),sub_c(i, j)] = t4
            left_leadUp[sub_d(i-1, j),sub_b(i, j-1)] = t4
            left_leadDw[sub_d(i-1, j),sub_b(i, j-1)] = t4
########### t4 ########### 
########### t5 ########### 
        left_leadUp[sub_d(i, j),sub_a(i, j)] = t5
        left_leadDw[sub_d(i, j),sub_a(i, j)] = t5
        if i>0:
            left_leadUp[sub_b(i, j),sub_c(i-1, j)] = t5
            left_leadDw[sub_b(i, j),sub_c(i-1, j)] = t5
########### t5 ########### 

systUp.attach_lead(left_leadUp)
right_leadUp=left_leadUp.reversed()
systUp.attach_lead(right_leadUp)
systDw.attach_lead(left_leadDw)
right_leadDw=left_leadDw.reversed()
systDw.attach_lead(right_leadDw)

# Plot it, to make sure it's OK
def family_colors(site):
    return 'r' if (site.family == sub_a or site.family == sub_b)  else 'g'
kwant.plot(systUp,site_size=0.12,site_color=family_colors)
kwant.plotter.sys_leads_sites(systUp)
# Finalize the system
systUp = systUp.finalized()
systDw = systDw.finalized()

def plot_wave_function(sys):
    # Calculate the wave functions in the system.
    ham_mat = sys.hamiltonian_submatrix(sparse=True)
    evecs = sla.eigsh(ham_mat, k=100, which='SM')[1]

    # Plot the probability density of the 10th eigenmode.
    kwant.plotter.map(sys, np.abs(evecs[:, 9])**2,
                       oversampling=1)
    
#plot_wave_function(syst)

fig, (ax1, ax0) = pyplot.subplots(ncols=2, figsize=[6, 10])

# Now that we have the system, we can compute conductance
energiesUp = []
energiesDw = []
dataUp = []
dataDw = []
for energy in tqdm(np.linspace(-3,3,1000)):
    # compute the scattering matrix at a given energy
    smatrixUp = kwant.smatrix(systUp, energy)
    smatrixDw = kwant.smatrix(systDw, energy)

    # compute the transmission probability from lead 0 to
    # lead 1
    energiesUp.append(energy)
    energiesDw.append(energy)
    dataUp.append(smatrixUp.transmission(1, 0))
    dataDw.append(smatrixDw.transmission(1, 0))
# Use matplotlib to write output
# We should see conductance steps
#pyplot.figure()
#ax0.plot(data , energies, "r")
ax0.plot(dataUp , energiesUp, "r")
ax0.plot(dataDw , energiesDw, "b")
#ax0.set_ylabel("energy [t]")
ax0.set_xlabel("conductance [e^2/h]")
ax0.set_ylim(-3,3)
#ax0.set_xlim(0,10)
#pyplot.show()
momenta = np.linspace(-np.pi, np.pi, 101)
bandsUp = kwant.physics.Bands(systUp.leads[0])
bandsDw = kwant.physics.Bands(systDw.leads[0])
energiesUp = [bandsUp(k) for k in momenta]
energiesDw = [bandsDw(k) for k in momenta]
#pyplot.figure(figsize=[2.0, 5])
momentaPI = [mom/np.pi for mom in momenta]
ax1.plot(momentaPI, energiesUp, "r")
ax1.plot(momentaPI, energiesDw, "b")
ax1.set_xlabel("Wavevctor [(lattice constant)^-1]")
ax1.set_ylabel("energy [t]")
ax1.set_ylim(-3,3)
ax1.set_xlim(-1,1)
pyplot.savefig("ArmChairSpin.png", dpi=300)
pyplot.show()
    

#pd.DataFrame (syst.hamiltonian_submatrix()).to_excel('hamiltoni.xlsx', index=False)
#pd.DataFrame(kwant.solvers.default.greens_function(syst).data).to_excel('Green.xlsx', index=False)
#pd.DataFrame(kwant.solvers.default.smatrix(syst).data).to_excel('SMatrix.xlsx', index=False)
#pyplot.plot(range(len(kwant.solvers.default.ldos(syst, energy=-3))),kwant.solvers.default.ldos(syst, energy=-3))



#ldos = kwant.ldos(syst)
#kwant.plotter.map(syst, ldos, num_lead_cells=0)
