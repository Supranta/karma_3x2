import h5py as h5
import sys
import time
import numpy as np
import jax.numpy as jnp

# Import functions from modules
from mbi_ia.map_objects import GaussianMap, LogNormalMap
from mbi_ia.samplers import HMCSampler, SliceSampler, MHSampler
from mbi_ia.io import *
from mbi_ia.utils import get_lensing_spectra

restart_flag = sys.argv[1]

assert restart_flag == 'INIT' or restart_flag == 'RESUME', "The restart flag (1st command line argument) must be either INIT or RESUME"

configfile = sys.argv[2]

N_Z_BINS, n_z_file, N_grid, theta_max       = config_map(configfile)
datafile, savedir, N_SAVE, N_RESTART        = config_io(configfile)
lognormal, precalculated, shift, var_gauss  = config_lognormal(configfile)
cosmo_ia_pars                             = config_cosmo_ia_pars(configfile)
# n_bar, sigma_eps                       = config_mock(configfile)
sample_map, sample_pars, sample_bias, sample_bta,\
            N_ADAPT, N_MCMC, dt, N_LEAPFROG,\
            precalculated_mass_matrix       = config_sampling(configfile)

# Load n_z's from file.
with h5.File(n_z_file, 'r') as f:
    zs   = f['zs'][:]
    n_zs = f['n_zs'][:]
assert len(zs)==N_Z_BINS,"Size of the zs list must be equal to the number of redshift bins"
assert len(n_zs)==N_Z_BINS,"Size of the n_zs list must be equal to the number of redshift bins"

nz         = [zs, n_zs]

io_handler = IOHandler(savedir, N_SAVE, N_RESTART, sample_map, sample_pars, cosmo_ia_pars[0])

if(lognormal):
    print("Using LogNormal maps....")
    combined_map = LogNormalMap(N_Z_BINS, N_grid, theta_max, nz, cosmo_ia_pars, shift, precalculated, sample_bias, sample_bta)
    if var_gauss is not None:
        combined_map.var_gauss = var_gauss
else:
    print("Using Gaussian maps....")
    combined_map = GaussianMap(N_Z_BINS, N_grid, theta_max, nz, cosmo_ia_pars, sample_bias, sample_bta)

# Default initial parameter values     
params       = np.array([cosmo_ia_pars[-1], 0., 1.])        
fields       = combined_map.init_field(0.1)
pars_lnprob = 0.
        
combined_map.set_data(datafile)
combined_map.set_mass_matrix()

if(restart_flag == 'INIT'):
    N_START = 0    
                    
elif(restart_flag == 'RESUME'):
    print("Restarting run....")    
    with h5.File(savedir+'/restart.h5', 'r') as f_restart:
        N_START = f_restart['N_STEP'][()]
        fields = f_restart['kappa_l'][:]

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

combined_map.mass_arr = mass_matrix

MapSampler = HMCSampler(N_DIM, combined_map.psi, combined_map.grad_psi, combined_map.mass_arr, N_grid, N_Z_BINS, verbose=VERBOSE)

select_params_indices = [True, sample_bta, sample_bias]
def get_params_dim():
    return 1 + np.sum(sample_bta) + np.sum(sample_bias)

def get_params_cov():
    params_std = np.array([1., 0.1, 0.1])
    return np.diag(params_std[select_params_indices]**2)

if(sample_pars):
    N_PARAMS_DIM = get_params_dim()
    step_cov     = get_params_cov()
    params = params[select_params_indices]
    ParamSampler   = SliceSampler(N_PARAMS_DIM, lnprob=combined_map.log_prob_ia, verbose=VERBOSE)
    ParamSampler.set_cov(step_cov)  
    param_chain = []

for i in range(N_START, N_START + N_MCMC):
    print("MCMC step: %d"%(i))
    if(sample_map):    
        start_time = time.time()
        x, ln_prob, acc, KE = MapSampler.sample_one_step(x, dt, N_LEAPFROG, psi_kwargs={'params': params}, grad_psi_kwargs={'params': params})    
        end_time = time.time()        
        print("Time taken for HMC sampling: %2.3f"%(end_time - start_time))
        fields = combined_map.x2field(x)        
        ln_prior = float(combined_map.log_prior(x))
        ln_like  = float(combined_map.log_like(x, params))        
        io_handler.map_sample_output(acc, i, N_MODES, N_pix, ln_prior, ln_like)
        io_handler.write_map_output(i, fields, x, KE, ln_prob, ln_like, ln_prior)        
    if(sample_pars):
        params, pars_acc, pars_lnprob = ParamSampler.sample_one_step(params, lnprob_kwargs={'x':x})
        print("IA bias pars: "+str(params))
        if(i < N_START + N_ADAPT):
            print("Adapting covariance...")
            ParamSampler.update_cov(params, i+2-N_START) 
        param_chain.append(params)
        np.save(savedir+'/parameter_chain.npy',np.array(param_chain))
    fields = combined_map.x2field(x)
    io_handler.write_restart(i, fields, x, params, pars_lnprob)        
