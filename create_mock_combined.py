import sys
import h5py as h5

# Import functions from modules
from karmic_harmonies.map_objects import GaussianCombinedMap
from karmic_harmonies import config_map, config_io, config_mock, config_lognormal, config_cosmo_pars, IOHandler

configfile = sys.argv[1]

N_Z_BINS, n_z_file, N_grid, theta_max, probe_list = config_map(configfile)
datafile, _, _, _                      = config_io(configfile)
lognormal, precalculated, shift, var_gauss   = config_lognormal(configfile)
cosmo_pars                             = config_cosmo_pars(configfile)
n_bar, sigma_eps                       = config_mock(configfile)


# Load n_z's from file.
with h5.File(n_z_file, 'r') as f:
    zs   = f['zs'][:]
    n_zs = f['n_zs'][:]
    
nz         = [zs, n_zs]

combined_map = GaussianCombinedMap(N_Z_BINS, N_grid, theta_max, nz, probe_list, cosmo_pars)

print("Creating mock data...")    
data, field_true = combined_map.create_synthetic_data(n_bar, sigma_eps, probe_list)    # Create a synthetic map  
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
        probe = probe_list[i]
        f['bin_%d'%(i)]['probe'] = probe
        f['bin_%d'%(i)]['data']['nbar']  = n_bar[i]
        if(probe=='lensing'):
            f['bin_%d'%(i)]['data']['eps'] = data[i]
            f['bin_%d'%(i)]['data']['sigma_eps'] = sigma_eps[i]
        elif(probe=='galaxy'):
            f['bin_%d'%(i)]['data']['counts'] = data[i]
            
    f['true_field'] = field_true    
