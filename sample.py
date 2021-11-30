import h5py as h5
import sys
import time
import numpy as np
import jax.numpy as jnp

# Import functions from modules
from karmic_harmonies.map_objects import GaussianMap, LogNormalMap
from karmic_harmonies.samplers import HMCSampler, SliceSampler, MHSampler
from karmic_harmonies import get_lensing_spectra, config_map, config_io, config_sampling, config_cosmo, config_cosmo_pars, config_lognormal, IOHandler

restart_flag = sys.argv[1]

assert restart_flag == 'INIT' or restart_flag == 'RESUME', "The restart flag (1st command line argument) must be either INIT or RESUME"

configfile = sys.argv[2]

N_Z_BINS, n_z_file, N_grid, theta_max, probe_list = config_map(configfile)
datafile, savedir, N_SAVE, N_RESTART        = config_io(configfile)
lognormal, precalculated, shift, var_gauss  = config_lognormal(configfile)
cosmo_pars                             = config_cosmo_pars(configfile)
# n_bar, sigma_eps                       = config_mock(configfile)
sample_map, sample_cosmo, N_COSMO,\
            N_ADAPT, N_MCMC, dt, N_LEAPFROG,\
            precalculated_mass_matrix       = config_sampling(configfile)
cosmo_sampler, sigma_Om, sigma_A, rho, emulator_file = config_cosmo(configfile)

# Load n_z's from file.
with h5.File(n_z_file, 'r') as f:
    zs   = f['zs'][:]
    n_zs = f['n_zs'][:]
assert len(zs)==N_Z_BINS,"Size of the zs list must be equal to the number of redshift bins"
assert len(n_zs)==N_Z_BINS,"Size of the n_zs list must be equal to the number of redshift bins"

nz         = [zs, n_zs]

io_handler = IOHandler(savedir, N_SAVE, N_RESTART, sample_map, sample_cosmo, cosmo_pars[0])

if(lognormal):
    print("Using LogNormal maps....")
    combined_map = LogNormalMap(N_Z_BINS, N_grid, theta_max, nz, probe_list, cosmo_pars, shift, precalculated)
    if var_gauss is not None:
        combined_map.var_gauss = var_gauss
else:
    print("Using Gaussian maps....")
    combined_map = GaussianMap(N_Z_BINS, N_grid, theta_max, nz, probe_list, cosmo_pars)

# Emu = None
# if emulator_file is not None:
#     from karmic_harmonies.cosmology import Emulator
#     Emu = Emulator(N_Z_BINS,cosmo_pars[1], cosmo_pars[0], lognormal)
#     Emu.setup_from_file(emulator_file)

# Default initial parameter values     
cosmo_pars = np.array([0., 0.])        
fields     = combined_map.init_field(0.1)
cosmo_lnprob = 0.
        
combined_map.set_data(datafile)
combined_map.set_mass_matrix()

if(restart_flag == 'INIT'):
    N_START = 0    
                    
elif(restart_flag == 'RESUME'):
    print("Restarting run....")    
    with h5.File(savedir+'/restart.h5', 'r') as f_restart:
        N_START = f_restart['N_STEP'].value
        fields = f_restart['kappa_l'][:]
        #print("x.shape: "+str(x.shape))
#         if(sample_cosmo):
#             try:
#                 cosmo_pars = f_restart['cosmo_pars'][:]
#                 log_Om, log_A = cosmo_pars
#                 OmegaM = np.exp(log_Om) * Omega_m
#                 A      = np.exp(log_A)
#                 Cl     = get_lensing_spectra(N_Z_BINS, zs, n_zs, OmegaM, A * ShearMap.As_fid)
#                 if(lognormal):
#                     ShearMap.shift = Emu.get_shift_emu(log_Om, log_A)
#                     Cl_g  = ShearMap.cl2clg(Cl[0,0,:ShearMap.lmax])
#                     Cl = Cl_g
#                     ShearMap.var_gauss = ShearMap.get_var_gauss(Cl_g)
#                 ShearMap.set_Cl_arr(Cl)
#                 ShearMap.set_eigs(ShearMap.Cl_arr_real, ShearMap.Cl_arr_imag)
#             except:
#                 print("No cosmo pars in restart file. Setting to default value...")              

io_handler.N_START = N_START                           
x = combined_map.field2x(fields)        

VERBOSE = True
N_MODES = 2 * N_Z_BINS * np.sum(combined_map.map_tool.fourier_symm_mask)
N_pix   = N_Z_BINS * N_grid * N_grid
N_DIM   = x.shape
    
if(precalculated_mass_matrix):
    mass_matrix = np.load(savedir+'/mass_matrix.npy')    
    print("Using precomputed mass matrix...")
else:
    print("Mass matrix not found in save_dir. Using default mass matrix instead...")

# mass_matrix = np.tile(np.eye(N_Z_BINS)[:,:,np.newaxis],N_grid**2-1)
combined_map.mass_arr = mass_matrix

MapSampler = HMCSampler(N_DIM, combined_map.psi, combined_map.grad_psi, combined_map.mass_arr, N_grid, N_Z_BINS, verbose=VERBOSE)
# MapSampler = HMCSampler(N_DIM, ShearMap.log_prior, ShearMap.grad_prior, ShearMap.mass_arr, N_grid, N_Z_BINS, verbose=VERBOSE)
    
# if(sample_cosmo):
#     step_cov       = np.array([[sigma_Om**2, rho * sigma_A * sigma_Om],
#                                [rho * sigma_Om * sigma_A,  sigma_A**2]])
#     if(cosmo_sampler=='slice'):
#         print("Using Slice Sampler...")
#         CosmoSampler   = SliceSampler(2, lnprob=combined_map.log_prob_cosmo, verbose=VERBOSE)
#     elif(cosmo_sampler=='mh'):
#         print("Using MH Sampler...")
#         CosmoSampler   = MHSampler(lnprob=combined_map.log_prob_cosmo, verbose=VERBOSE)
#     CosmoSampler.set_cov(step_cov)  
#     cosmo_chain = []

for i in range(N_START, N_START + N_MCMC):
    print("MCMC step: %d"%(i))
    if(sample_map):    
        start_time = time.time()
        x, ln_prob, acc, KE = MapSampler.sample_one_step(x, dt, N_LEAPFROG)    
        end_time = time.time()        
        print("Time taken for HMC sampling: %2.3f"%(end_time - start_time))
        fields = combined_map.x2field(x)        
        ln_prior = float(combined_map.log_prior(x))
        ln_like  = float(combined_map.log_like(x))        
        io_handler.map_sample_output(acc, i, N_MODES, N_pix, ln_prior, ln_like)
        io_handler.write_map_output(i, fields, x, KE, ln_prob, ln_like, ln_prior)        
#     if(sample_cosmo):
#         for ii in range(N_COSMO):
#             cosmo_pars, cosmo_acc, cosmo_lnprob = CosmoSampler.sample_one_step(cosmo_pars, lnprob_kwargs={'x':x, 'Emu':Emu})
#         if(i < N_START + N_ADAPT):
#             print("Adapting covariance...")
#             CosmoSampler.update_cov(cosmo_pars, i+2-N_START) 
#         cosmo_chain.append(cosmo_pars)
#         np.save(savedir+'/cosmo_chain.npy',np.array(cosmo_chain))
#         ShearMap = io_handler.cosmo_sample_output(i, cosmo_pars, Emu, ShearMap, lognormal)        
#         io_handler.write_cosmo_output(i, cosmo_pars, cosmo_lnprob)
    fields = combined_map.x2field(x)
    io_handler.write_restart(i, fields, x, cosmo_pars, cosmo_lnprob)        
