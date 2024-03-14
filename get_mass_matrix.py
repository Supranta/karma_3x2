import sys
import h5py as h5
import numpy as np
import jax.numpy as jnp

# Import functions from modules
from karmic_harmonies.map_objects import GaussianMap, LogNormalMap
from karmic_harmonies.samplers import HMCSampler, SliceSampler, MHSampler
from karmic_harmonies import get_lensing_spectra, config_map, config_io, config_sampling, config_cosmo, config_cosmo_ia_pars, config_lognormal, config_mock, IOHandler

configfile = sys.argv[1]

N_Z_BINS, n_z_file, N_grid, theta_max = config_map(configfile)
datafile, savedir, _, _               = config_io(configfile)
lognormal, precalculated, shift, var_gauss = config_lognormal(configfile)
cosmo_ia_pars = config_cosmo_ia_pars(configfile)
n_bar, sigma_eps                       = config_mock(configfile)    
# Load n_z's from file.
with h5.File(n_z_file, 'r') as f:
    zs   = f['zs'][:]
    n_zs = f['n_zs'][:]
assert len(zs)==N_Z_BINS,"Size of the zs list must be equal to the number of redshift bins"
assert len(n_zs)==N_Z_BINS,"Size of the n_zs list must be equal to the number of redshift bins"

# Default cosmological parameters
Omega_m = 0.279
A_s     = 2.249e-9  # corresponds to sigma_8 \approx 0.82   

nz         = [zs, n_zs]

if(lognormal):
    print("Using LogNormal maps....")
    combined_map = LogNormalMap(N_Z_BINS, N_grid, theta_max, nz, cosmo_ia_pars, shift, precalculated)
    if var_gauss is not None:
        combined_map.var_gauss = var_gauss
else:
    print("Using Gaussian maps....")
    combined_map = GaussianMap(N_Z_BINS, N_grid, theta_max, nz, cosmo_ia_pars)

combined_map.set_data(datafile)

import jax.numpy as jnp
from jax import grad, jvp

EPS = 1e-15

def get_mass_matrix():
    field_true, n_gals, ellipticity = combined_map.create_synthetic_data(n_bar, sigma_eps)
    x = combined_map.field2x(field_true)
    mass_matrix = np.zeros((N_Z_BINS, N_Z_BINS, N_grid**2 - 1))
     
    for i in range(N_Z_BINS):
        combined_map.data[i]['eps'] = ellipticity[i]
        combined_map.data[i]['counts'] = n_gals[i]
    
        x_i_dir = np.zeros(x.shape)
        x_i_dir[i] = 1
        x_i_dir = jnp.array(x_i_dir)
        grad_like = grad(combined_map.log_like, 0)
        _, v_like = jvp(grad_like, (x,), (x_i_dir,))
        
        mass_matrix[i] = np.abs(v_like)
    
    for i in range(N_Z_BINS):
        mass_matrix[i,i] = mass_matrix[i,i] + 1.
    
    return mass_matrix

N_MOCKS = 10
mass_matrix_list = []

mean_mass_matrix = np.zeros(np.array(get_mass_matrix()).shape)

for i in range(N_MOCKS):
    print("i: %d"%(i))
    N = (i+1)
    mass_matrix = get_mass_matrix()
    mean_mass_matrix = (mean_mass_matrix * (N-1) + mass_matrix) / N
    np.save(savedir+'/mass_matrix.npy', np.array(mean_mass_matrix))
