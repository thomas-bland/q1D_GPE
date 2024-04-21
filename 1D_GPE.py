# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 15:10:13 2024

@author: Thomas Bland

Finds the ground state of the quasi-one-dimensional (q1D) Gross-Pitaevskii equation,
and finds the excitation spectrum through diagonalizing the Bogoliubov-de Gennes
equations
"""

######################## Ground state part ############################


import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
import scipy.sparse.linalg as la

"""
Recommended reading: Carlo F Barenghi and Nick. G .Parker, A primer on quantum fluids, appendix A
https://arxiv.org/abs/1605.09580
"""

# Constants
hbar = 1.05457182e-34 #Reduced Planck's constant
m = 39*1.66054e-27 #39K mass
a0 = 5.29177210903e-11 #Bohr radius

#Units - in micrometres
l = 1e-6 #Length unit
w0 = hbar/(m*l**2) #Frequency unit (energy is hbar*w0)

# Parameters
L = 100.0  # Length of the space domain
N = 2048  # Number of spatial grid points
dx = L / N  # Spatial grid spacing
dt = -0.001j  # Imaginary time step

a_s = 100*a0 # Scattering length

om = 2*np.pi*100/w0  #Axial trap frequency
om_rho = 2*np.pi*500/w0 # Radial trap frequency

#Corresponding length scales from H.O. frequencies
l_rho = np.sqrt(hbar/(m*om_rho))
l_x = np.sqrt(hbar/(m*om))

Norm = 30000 # Atom number

g = 2*a_s*Norm/l_rho # Dimensionless q1D interaction

tol = 1e-15 #Imainary time tolerance

num_eigenvalues = 1024  # Number of eigenvalues to compute

# Initial wave function parameters
x0 = 0.0  # Initial position of the wave packet
sigma = 100.0  # Width of the wave packet

#Set up for simulation
E_old = 1
E_err = 1
mu = 0 # Chemical potential
index = 0

# Initialize spatial grid
x = np.linspace(-L/2, L/2, N, endpoint=False)

# Initialize wave function
if om==0:
    psi = np.ones(N) #Homogeneous initial condition
else:
    psi = np.exp(-(x - x0)**2 / (2 * sigma**2))
        
psi *= 1/np.sqrt(np.trapz(np.abs(psi)**2, x)) #Renormalize to 1
    
# Initialize potential
V = 1/2 * om**2 * x**2

# Fourier space variables
k = fftfreq(N, d=dx) * 2 * np.pi #FFT-shifted
k_norm = np.fft.fftshift(k) #Normal ordering

############# Stationary solution finder ####################
np.disp('Finding stationary solution....')

# Imaginary time evolution
while E_err > tol:
    # Split-step Fourier method
    # Step 1: Apply half of the kinetic energy operator
    psi_k = fft(psi)
    psi_k *= np.exp(-0.25j * (k**2) * dt)
    psi = ifft(psi_k)
    
    # Step 2: Apply potential energy operator
    psi *= np.exp(-1j * (V + g*np.abs(psi)**2 - mu)  * dt)
    
    # Step 3: Apply another half of the kinetic energy operator
    psi_k = fft(psi)
    psi_k *= np.exp(-0.25j * (k**2) * dt)
    psi = ifft(psi_k)
        
    # Normalize wave function
    psi *= 1/np.sqrt(np.trapz(np.abs(psi)**2, x))
        
    # Calculate energy and chemical potential
    E = np.trapz(-0.5*np.real(np.conj(psi)*ifft(-k**2*fft(psi)))+V*np.abs(psi)**2+g/2*np.abs(psi)**4,x)
    mu = np.trapz(-0.5*np.real(np.conj(psi)*ifft(-k**2*fft(psi)))+V*np.abs(psi)**2+g*np.abs(psi)**4,x)/np.trapz(np.abs(psi)**2, x)

    #Calculate change in energy
    E_err = np.abs(E-E_old)/np.abs(E)
    E_old = E
    
    #Iterate
    index +=1
    
    # Plot results
    if index % 2000 == 0: #plot every 2000 steps
        plt.plot(x, np.abs(psi)**2)
        plt.xlabel('x')
        plt.ylabel('|Psi|^2')
        plt.title(f'Energy error E_err = {E_err/tol:.2f}')
        plt.show()

#Remove small imaginary parts from fft        
psi = abs(psi)

#Ensure mu is correct
mu = np.trapz(-0.5*np.real(np.conj(psi)*ifft(-k**2*fft(psi)))+V*np.abs(psi)**2+g*np.abs(psi)**4,x)/np.trapz(np.abs(psi)**2, x)

#Plot final result
plt.plot(x, np.abs(psi)**2)
plt.xlabel('x')
plt.ylabel('|Psi|^2')
plt.title('Stationary solution')
plt.show()
np.disp('.... found!')

######################## BdG part ############################
"""
Recommended reading: Au-Chen Lee, Dipolar Bose-Einstein Condensate with a Vortex
Chapters 2.5, 3.1.2, 3.3.2
https://ourarchive.otago.ac.nz/handle/10523/9254

Equation references below refer to this thesis
"""
# Define a function representing the kinetic part for an arbitrary function
def kin(f):
    H_kin = -0.5*np.real(ifft(-k**2*fft(f)))
    return H_kin

#Exhcange operator
def X_op(f):
    X = (g*np.abs(psi)**2)*f
    return X

#GPE operator
def L_GP_op(f):
    L_GP = kin(f) + V*f +(g*np.abs(psi)**2)*f - mu*f
    return L_GP


#Construct BdG matrix acting on eigenfunction f [Eq. (3.127)]
def BdG_fun(f):
    BdG = L_GP_op(L_GP_op(f)) + 2*X_op(L_GP_op(f))
    return BdG

#A is then the operator, means that we don't have to construct the large matrix
A = la.LinearOperator((N,N), BdG_fun)

# Compute eigenvalues and eigenvectors
np.disp('Diagonalizing the BdG matrix....')
eigenvalues, eigenvectors = la.eigs(A, k=num_eigenvalues, v0=psi, which='SM',tol=0)
np.disp('... Eigenvalues found!')

# Sort eigenvalues and eigenvectors
idx = eigenvalues.argsort()
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# This returns e^2, so we need to take the square root
eigenvalues = np.lib.scimath.sqrt(eigenvalues)

#Remove the Goldstone mode
if np.abs(eigenvalues[0])==0:
    eigenvectors = eigenvectors[:,range(1,num_eigenvalues)]
    eigenvalues = eigenvalues[range(1,num_eigenvalues)]
    num_eigenvalues += -1

# The eigenvectors are of the form u \pm v, so we need to reconstruct u and v
fplus = eigenvectors
fminus = fplus

for jj in range(num_eigenvalues):
    fminus[:,jj] = (L_GP_op(fplus[:,jj]))/eigenvalues[jj]
    
u = (fplus + fminus)/2
v = (fplus - fminus)/2

# Renormalize to |u|^2 - |v|^2 = 1
for ii in range(num_eigenvalues):
    utemp = u[:,ii]
    u[:,ii] = u[:,ii]/np.sqrt(np.trapz(np.abs(u[:,ii])**2 - np.abs(v[:,ii])**2,x))
    v[:,ii] = v[:,ii]/np.sqrt(np.trapz(np.abs(utemp)**2 - np.abs(v[:,ii])**2,x))

# This is the wavefunction perturbation
dpsi = u-np.conj(v)

# Print the eigenvalues
print("Eigenvalues:", eigenvalues[range(10)])

# Plot some of the eigenfunctions
for i in range(min(num_eigenvalues, 5)):
    plt.plot(x, np.abs(dpsi[:, i])**2, label=f"Eigenvector {i+1}")
plt.xlabel('x')
plt.ylabel('|dPsi|^2')
plt.title('Eigenfunctions')
plt.legend()
plt.show()

################## Structure factor ##########################
"""
Recommended reading: T. Bland et al., Two-dimensional supersolid formation in dipolar condensates
Phys. Rev. Lett. 128, 195302 (2022)
https://arxiv.org/abs/2107.06680
Equation references below refer to this paper
"""
# energy array
omega = np.linspace(0, 20, 1024, endpoint=False)

#Energy braodening
sig = 0.05

#Construct 2D matrices
[K,OM] = np.meshgrid(k,omega)

#Calculate Structure factor [Eq. (S3)]
np.disp('Calculating the structure factor....')
S = np.zeros(OM.shape)
for ii in range(num_eigenvalues):
    S = S + np.matlib.repmat(np.abs(np.fft.ifftshift(fft(np.fft.fftshift(np.conj(u[:,ii] + v[:,ii])*psi)))*dx)**2,len(omega),1)*np.exp(-(OM-np.real(eigenvalues[ii]))**2/(2*sig**2))/np.sqrt(2*np.pi*sig**2)

np.disp('.... done!')

#Plot the 2D structure factor
plt.pcolormesh(k_norm,omega,S,vmin=0,vmax=0.5)
plt.xlabel('k')
plt.ylabel('Energy')
plt.title('Dynamic structure factor')
plt.xlim(-10, 10)
plt.ylim(0, max(omega))


#If there is no harmonic trap, this has an analytic form
if om==0:
    plt.plot(k_norm,np.sqrt(k_norm**2/2*(k_norm**2/2+2*g*max(np.abs(psi)**2))))
    
plt.show()

# For trapped systems, k is not a good quantum number, so here we estimate each eigenvalue's corresponding k value
avg_k = np.zeros((num_eigenvalues))
for ii in range(num_eigenvalues):
    uk = np.fft.fftshift(fft(u[:,ii]))
    vk = np.fft.fftshift(fft(v[:,ii]))
    func_k = np.abs(uk)**2 + np.abs(vk)**2
    avg_k[ii] = np.sqrt(np.trapz(k_norm**2*func_k)/np.trapz(func_k))

plt.plot(avg_k,np.real(eigenvalues))
plt.xlabel('k')
plt.ylabel('Energy')
plt.xlim(0, 10)
plt.ylim(0, max(omega))
plt.title('Average k vs e')
plt.show()

#Calculate the static structure factor and
#first moment of structure factor (f-sum rule)
statS = np.zeros(k_norm.shape)
m1 = np.zeros(k_norm.shape)

for ii in range(num_eigenvalues):
    statS = statS + np.abs(np.fft.ifftshift(fft(np.fft.fftshift(np.conj(u[:,ii] + v[:,ii])*psi)))*dx)**2
    m1 = m1 + eigenvalues[ii]*np.abs(np.fft.ifftshift(fft(np.fft.fftshift(np.conj(u[:,ii] + v[:,ii])*psi)))*dx)**2


#Plot the first moment and compare to k^2/2 (should overlap)
#This is the f-sum rule
plt.figure()
plt.plot(k_norm,m1,label='First moment of S')
plt.plot(k_norm,k_norm**2/2,'--',label='k^2/2')
plt.xlim(-30, 30)
plt.ylim(0, 30**2/2)
plt.xlabel('k')
plt.legend()
plt.title('f-sum rule')
plt.show()

#plot the static structure factor, should tend to 1 in large k limit
plt.figure()
plt.plot(k_norm,statS)
plt.plot(k_norm,np.ones(N),'--')
plt.xlim(-30, 30)
plt.ylim(0, 1.05)
plt.xlabel('k')
plt.ylabel('S(k)')
plt.title('Static structure factor')
plt.show() 