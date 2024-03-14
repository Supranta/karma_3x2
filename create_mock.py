import sys
import h5py as h5

# Import functions from modules
from mbi_ia.map_objects import GaussianMap, LogNormalMap
from mbi_ia.io import *

configfile = sys.argv[1]

N_Z_BINS, n_z_file, N_grid, theta_max  = config_map(configfile)
datafile, _, _, _                      = config_io(configfile)
lognormal, precalculated, shift, var_gauss   = config_lognormal(configfile)
cosmo_ia_pars                          = config_cosmo_ia_pars(configfile)
n_bar, sigma_eps                       = config_mock(configfile)


# Load n_z's from file.
with h5.File(n_z_file, 'r') as f:
    zs   = f['zs'][:]
    n_zs = f['n_zs'][:]
    
nz         = [zs, n_zs]

if(lognormal):
    print("Creating lognormal mocks...")
    combined_map = LogNormalMap(N_Z_BINS, N_grid, theta_max, nz, cosmo_ia_pars, shift, precalculated)
    if var_gauss is not None:
        combined_map.var_gauss = var_gauss
else:
    print("Creating Gaussian mocks...")
    combined_map = GaussianMap(N_Z_BINS, N_grid, theta_max, nz, cosmo_ia_pars)

print("Creating mock data...")    
field_true, n_gals, ellipticity = combined_map.create_synthetic_data(n_bar, sigma_eps)    # Create a synthetic map  
print("Done creating mock data...")
for i in range(N_Z_BINS):
    delta_x_true = combined_map.map_tool.fourier2map(field_true[i])
    delta_x_min  = delta_x_true.min()
    print("delta_x_min: %2.4f"%(delta_x_min))

with h5.File(datafile, 'w') as f:    
    for i in range(N_Z_BINS):
        f.create_group('bin_%d'%(i))
        g = f['bin_%d'%(i)]
        g.create_group('data')
        f['bin_%d'%(i)]['data']['nbar']      = n_bar[i]
        f['bin_%d'%(i)]['data']['eps']       = ellipticity[i]
        f['bin_%d'%(i)]['data']['sigma_eps'] = sigma_eps[i]
        f['bin_%d'%(i)]['data']['counts']    = n_gals[i]
            
    f['true_field'] = field_true    
