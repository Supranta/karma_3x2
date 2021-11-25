import h5py as h5
import sys
import time
import numpy as np
import jax.numpy as jnp

# Import functions from modules
from karmic_harmonies.map_objects import GaussianGalaxyMap
from karmic_harmonies.samplers import HMCSampler, SliceSampler, MHSampler
from karmic_harmonies import get_lensing_spectra, config_map, config_io, config_sampling, config_cosmo, config_cosmo_pars, config_lognormal, IOHandler

restart_flag = sys.argv[1]

assert restart_flag == 'INIT' or restart_flag == 'RESUME', "The restart flag (1st command line argument) must be either INIT or RESUME"

configfile = sys.argv[2]

N_Z_BINS, n_z_file, N_grid, theta_max       = config_map(configfile)            
datafile, savedir, N_SAVE, N_RESTART        = config_io(configfile)
sample_map, sample_cosmo, N_COSMO,\
            N_ADAPT, N_MCMC, dt, N_LEAPFROG,\
            precalculated_mass_matrix       = config_sampling(configfile)
cosmo_sampler, sigma_Om, sigma_A, rho, emulator_file = config_cosmo(configfile)
lognormal, precalculated, shift, var_gauss  = config_lognormal(configfile)
cosmo_pars                                  = config_cosmo_pars(configfile)

# Load n_z's from file.
with h5.File(n_z_file, 'r') as f:
    zs   = f['zs'][:]
    n_zs = f['n_zs'][:]
assert len(zs)==N_Z_BINS,"Size of the zs list must be equal to the number of redshift bins"
assert len(n_zs)==N_Z_BINS,"Size of the n_zs list must be equal to the number of redshift bins"

nz         = [zs, n_zs]

io_handler = IOHandler(savedir, N_SAVE, N_RESTART, sample_map, sample_cosmo, cosmo_pars[0])

GalaxyMap = GaussianGalaxyMap(N_Z_BINS, N_grid, theta_max, nz, cosmo_pars)

cosmo_pars = np.array([0., 0.])        
delta_l = GalaxyMap.init_delta_l(0.1)
cosmo_lnprob = 0.

with h5.File(datafile, 'r') as f:
    galaxy_counts = f['galaxy_counts'][:]
    n_bar         = f['n_bar'][:]
    delta_l_true  = f['delta_l_true'][:]
    
GalaxyMap.set_data(n_bar, galaxy_counts)

if(restart_flag == 'INIT'):
    N_START = 0    
    if not sample_map:
        delta_l = delta_l_true            

elif(restart_flag == 'RESUME'):
    print("Restarting run....")    
    with h5.File(savedir+'/restart.h5', 'r') as f_restart:
        N_START = f_restart['N_STEP'].value
        delta_l = f_restart['delta_l'][:]
        
io_handler.N_START = N_START                           
x = GalaxyMap.delta2x(delta_l)        
### ==========================================
for i in range(10):
    print("========================")
print(x.min(), x.max(), x.mean())
for i in range(10):
    print("========================")
### ==========================================

VERBOSE = True
N_MODES = 2 * N_Z_BINS * np.sum(GalaxyMap.map_tool.fourier_symm_mask)
N_pix   = N_Z_BINS * N_grid * N_grid
N_DIM   = x.shape

mass_matrix = np.tile(np.eye(N_Z_BINS)[:,:,np.newaxis],N_grid**2-1)
GalaxyMap.mass_arr = mass_matrix

MapSampler = HMCSampler(N_DIM, GalaxyMap.psi, GalaxyMap.grad_psi, GalaxyMap.mass_arr, N_grid, N_Z_BINS, verbose=VERBOSE)
# MapSampler = HMCSampler(N_DIM, GalaxyMap.log_prior, GalaxyMap.grad_prior, GalaxyMap.mass_arr, N_grid, N_Z_BINS, verbose=VERBOSE)

for i in range(N_START, N_START + N_MCMC):
    print("MCMC step: %d"%(i))
    start_time = time.time()
    x, ln_prob, acc, KE = MapSampler.sample_one_step(x, dt, N_LEAPFROG)    
    end_time = time.time()        
    print("Time taken for HMC sampling: %2.3f"%(end_time - start_time))
    delta_l = GalaxyMap.x2delta(x)        
    ln_prior = float(GalaxyMap.log_prior(x))
    ln_like  = float(GalaxyMap.log_like(x))        
    io_handler.map_sample_output(acc, i, N_MODES, N_pix, ln_prior, ln_like)
    io_handler.write_map_output(i, delta_l, x, KE, ln_prob, ln_like, ln_prior)            
    kappa_l = GalaxyMap.x2delta(x)
    io_handler.write_restart(i, delta_l, x, cosmo_pars, cosmo_lnprob)        
