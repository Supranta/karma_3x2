import numpy as np
import h5py as h5
import time
import camb
from camb import model
from camb.sources import SplinedSourceWindow, GaussianSourceWindow
from math import sqrt

def get_spectra(N_Z_BINS, zs, n_zs, probe_list, Omega_m, A_s, h=0.7, ns=0.97, Omega_b=0.046):
    tic = time.time()
    lmax=8000    
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100 * h, ombh2=Omega_b * h * h, omch2=(Omega_m - Omega_b) * h * h)
    pars.InitPower.set_params(As=A_s, ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    pars.Want_CMB = False 
    pars.Want_CMB_lensing = False 
    pars.NonLinear = model.NonLinear_both
    SourceWindows = []
    for z, n_z, probe in zip(zs, n_zs, probe_list):
        if(probe=='lensing'):
            window = SplinedSourceWindow(source_type='lensing', z=z, W=n_z)
        elif(probe=='galaxy'):
            mu, std = get_mean_std(n_z, z)
            window = GaussianSourceWindow(redshift=mu, source_type='counts', bias=1., sigma=std)
        SourceWindows.append(window)        
        
    pars.SourceWindows = SourceWindows
    results = camb.get_results(pars)
    cls = results.get_source_cls_dict(raw_cl=True)
    
    lmax = len(cls['W1xW1'])
    
    Cl = np.zeros((N_Z_BINS, N_Z_BINS, lmax))

    for i in range(N_Z_BINS):
        for j in range(N_Z_BINS):
            cross_bin = 'W'+str(i+1)+'xW'+str(j+1)
            Cl[i,j] = cls[cross_bin]
    
    for i in range(N_Z_BINS):
        Cl[i,i,:2] = 1e-15                
    
    toc = time.time()
    print("Time taken for power spectrum calculations: %2.3f seconds"%(toc - tic))
    return Cl

def get_lensing_spectra(N_Z_BINS, zs, n_zs, Omega_m, A_s, h=0.7, ns=0.97, Omega_b=0.046):
    tic = time.time()
    lmax=8000    
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100 * h, ombh2=Omega_b * h * h, omch2=(Omega_m - Omega_b) * h * h)
    pars.InitPower.set_params(As=A_s, ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    pars.Want_CMB = False 
    pars.Want_CMB_lensing = False 
    pars.NonLinear = model.NonLinear_both
    SourceWindows = []
    for z, n_z in zip(zs, n_zs):
        SourceWindows.append(SplinedSourceWindow(source_type='lensing', z=z, W=n_z))
    pars.SourceWindows = SourceWindows
    results = camb.get_results(pars)
    cls = results.get_source_cls_dict(raw_cl=True)
    
    lmax = len(cls['W1xW1'])
    
    Cl = np.zeros((N_Z_BINS, N_Z_BINS, lmax))

    for i in range(N_Z_BINS):
        for j in range(N_Z_BINS):
            cross_bin = 'W'+str(i+1)+'xW'+str(j+1)
            Cl[i,j] = cls[cross_bin]
    
    for i in range(N_Z_BINS):
        Cl[i,i,:2] = 1e-15                
    
    toc = time.time()
    print("Time taken for power spectrum calculations: %2.3f seconds"%(toc - tic))
    return Cl

def get_mean_std(n_z, z):
    z_mean = np.trapz(n_z * z, z)
    z_var  = np.trapz(n_z * (z - z_mean)**2, z)
    z_std  = np.sqrt(z_var)
    return z_mean, z_std

def get_galaxy_spectra(N_Z_BINS, zs, n_zs, Omega_m, A_s, h=0.7, ns=0.97, Omega_b=0.046):
    tic = time.time()
    lmax=8000    
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100 * h, ombh2=Omega_b * h * h, omch2=(Omega_m - Omega_b) * h * h)
    pars.InitPower.set_params(As=A_s, ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    pars.Want_CMB = False 
    pars.Want_CMB_lensing = False 
    pars.NonLinear = model.NonLinear_both
    SourceWindows = []
    for z, n_z in zip(zs, n_zs):
        mu, std = get_mean_std(n_z, z)
        window = GaussianSourceWindow(redshift=mu, source_type='counts', bias=1., sigma=std)
#         window = SplinedSourceWindow(source_type='counts', z=z, W=n_z)
        SourceWindows.append(window)
    pars.SourceWindows = SourceWindows
    results = camb.get_results(pars)
    cls = results.get_source_cls_dict(raw_cl=True)
    
    lmax = len(cls['W1xW1'])
    
    Cl = np.zeros((N_Z_BINS, N_Z_BINS, lmax))

    for i in range(N_Z_BINS):
        cross_bin = 'W'+str(i+1)+'xW'+str(i+1)
        Cl[i,i] = cls[cross_bin]
        for j in range(N_Z_BINS):
            if(i!=j):
                Cl[i,j] = 0. * cls['W1xW1']
    
    for i in range(N_Z_BINS):
        Cl[i,i,:2] = 1e-15                
    
    toc = time.time()
    print("Time taken for power spectrum calculations: %2.3f seconds"%(toc - tic))
    return Cl

def get_sigma8(Omega_m, A_s, h=0.7, ns=0.97, Omega_b=0.046):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100 * h, ombh2=Omega_b * h * h, omch2=(Omega_m - Omega_b) * h * h)
    pars.InitPower.set_params(As=A_s, ns=ns)
    #Note non-linear corrections couples to smaller scales than you want
    pars.set_matter_power(redshifts=[0.], kmax=2.0)
    #Linear spectra
    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    sig8 = np.array(results.get_sigma8())
    return sig8

def sample_kappa(Cl_arr, N, N_Z_BINS):
    N_Y = N//2+1
    kappa_var = Cl_arr
    kappa_arr = np.zeros((N_Z_BINS, N, N_Y))
    for i in range(N):
        for j in range(N_Y):
            cov = kappa_var[:,:,i,j]
            kappa = np.random.multivariate_normal(np.zeros(N_Z_BINS), cov)
            kappa_arr[:,i,j]   = kappa           
    return kappa_arr

def get_mass_eigs(mass_matrix, N, N_bins=2):
    cov = mass_matrix
    eig_vals, eig_vecs = np.linalg.eig(cov.T)
    eig_vals_arr = eig_vals.T
    eig_vecs_arr = np.swapaxes(eig_vecs.T, 0, 1)
    print("Done getting the eigen-system of the mass-matrix...")
    return eig_vals_arr, eig_vecs_arr

def get_mass_inv(mass_matrix, N, N_bins=2):
    print("Getting the inverse of the mass-matrix...")
    print("N: "+str(N))
    M_inv_arr = np.linalg.inv(mass_matrix.T).T
    print("Done getting the inverse of the mass-matrix...")
    return M_inv_arr