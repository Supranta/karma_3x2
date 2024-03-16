import sys
import h5py as h5
import treecorr

# Import functions from modules
from mbi_ia.map_objects import GaussianMap, LogNormalMap
from mbi_ia.io import *

configfile = sys.argv[1]

N_Z_BINS, n_z_file, N_grid, theta_max  = config_map(configfile)
datafile, savedir, _, _                = config_io(configfile)
lognormal, precalculated, shift, var_gauss   = config_lognormal(configfile)
cosmo_ia_pars                          = config_cosmo_ia_pars(configfile)
n_bar, sigma_eps                       = config_mock(configfile)

def get_radec_grid(THETA_MAX, N_GRID):
    DELTA_THETA = THETA_MAX / N_GRID
    theta_arr = (0.5 + np.arange(N_GRID) - N_GRID/2) * DELTA_THETA
    ra_grid = np.zeros((N_GRID, N_GRID))
    dec_grid = np.zeros((N_GRID, N_GRID))

    for i in range(N_GRID):
        for j in range(N_GRID):
            ra_grid[i,j]  = theta_arr[i]
            dec_grid[i,j] = theta_arr[j]
    return ra_grid, dec_grid

def get_cats(n_gals, delta, e_field, T_field, Q_field):
    weights = n_gals / n_gals.mean()
    e_cat = treecorr.Catalog(ra=ra_grid.flatten(), dec=dec_grid.flatten(),
                         g1=e_field[0].flatten(), g2=e_field[1].flatten(), w=weights.flatten(),
                         ra_units='deg', dec_units='deg')
    T_cat = treecorr.Catalog(ra=ra_grid.flatten(), dec=dec_grid.flatten(),
                         g1=T_field[0].flatten(), g2=T_field[1].flatten(), w=weights.flatten(),
                         ra_units='deg', dec_units='deg')
    Q_cat = treecorr.Catalog(ra=ra_grid.flatten(), dec=dec_grid.flatten(),
                         g1=Q_field[0].flatten(), g2=Q_field[1].flatten(),
                         ra_units='deg', dec_units='deg')
    g_cat = treecorr.Catalog(ra=ra_grid.flatten(), dec=dec_grid.flatten(), w=weights.flatten(),
                         ra_units='deg', dec_units='deg')

    return [g_cat, e_cat, T_cat, Q_cat]

ra_grid, dec_grid = get_radec_grid(theta_max, N_grid)
    
THETA_MIN_CORR = theta_max / N_grid
THETA_MAX_CORR = 3.

BIN_SLOP = 0.

gg = treecorr.GGCorrelation(min_sep=THETA_MIN_CORR, max_sep=THETA_MAX_CORR, nbins=10, bin_slop=BIN_SLOP)
ng = treecorr.NGCorrelation(min_sep=THETA_MIN_CORR, max_sep=THETA_MAX_CORR, nbins=10, bin_slop=BIN_SLOP)


def get_ge_corr(g_cat, e_cat, ng=ng):
    ng.process(g_cat, e_cat)
    return ng.xi.copy()

def get_ee_corr(e_cat_1, e_cat_2, gg=gg):
    gg.process(e_cat_1, e_cat_2)
    return np.hstack([gg.xip.copy(), gg.xim.copy()])

def get_all_ge_corr(cats, ng=ng):
    g_cat, e_cat, T_cat, Q_cat = cats
    ge_corr = get_ge_corr(g_cat, e_cat, ng)
    gT_corr = get_ge_corr(g_cat, T_cat, ng)
    gQ_corr = get_ge_corr(g_cat, Q_cat, ng)
    return ge_corr, gT_corr, gQ_corr
    
def get_all_ee_corr(cats, gg=gg):
    g_cat, e_cat, T_cat, Q_cat = cats
    ee_corr = get_ee_corr(e_cat, e_cat, gg=gg)
    TT_corr = get_ee_corr(T_cat, T_cat, gg=gg)
    QQ_corr = get_ee_corr(Q_cat, Q_cat, gg=gg)
    TQ_corr = get_ee_corr(T_cat, Q_cat, gg=gg)    
    return ee_corr, TT_corr, QQ_corr, TQ_corr


# Load n_z's from file.
with h5.File(n_z_file, 'r') as f:
    zs   = f['zs'][:]
    n_zs = f['n_zs'][:]
    
nz         = [zs, n_zs]

def get_data_corr():
    with h5.File(datafile, 'r+') as f:
        for i in range(N_Z_BINS):
            e_field_true = f['bin_%d'%(i)]['data']['eps'][:]
            counts_true  = f['bin_%d'%(i)]['data']['counts'][:]
            weights = counts_true / counts_true.mean()
            e_cat = treecorr.Catalog(ra=ra_grid.flatten(), dec=dec_grid.flatten(),
                                     g1=e_field_true[0].flatten(), g2=e_field_true[1].flatten(), w=weights.flatten(),
                                     ra_units='deg', dec_units='deg')
            g_cat = treecorr.Catalog(ra=ra_grid.flatten(), dec=dec_grid.flatten(), w=weights.flatten(),
                                     ra_units='deg', dec_units='deg')
            xi_ee = get_ee_corr(e_cat, e_cat)
            xi_ge = get_ge_corr(g_cat, e_cat)
            for x in ['xi_ge', 'xi_ee']:
                if x in f['bin_%d'%(i)]:
                    del f['bin_%d'%(i)][x]
            f['bin_%d'%(i)]['xi_ge'] = xi_ge
            f['bin_%d'%(i)]['xi_ee'] = xi_ee

if(lognormal):
    print("Creating lognormal mocks...")
    combined_map = LogNormalMap(N_Z_BINS, N_grid, theta_max, nz, cosmo_ia_pars, shift, precalculated)
    if var_gauss is not None:
        combined_map.var_gauss = var_gauss
else:
    print("Creating Gaussian mocks...")
    combined_map = GaussianMap(N_Z_BINS, N_grid, theta_max, nz, cosmo_ia_pars)

get_data_corr()
from tqdm import trange

N_samples = 2000

for n in trange(N_samples):
    field_true, n_gals, e_field = combined_map.create_synthetic_data(n_bar, sigma_eps)    # Create a synthetic map  
    ps_filename = savedir + '/corr_cov_mean/corr_%d.h5'%(n)
    with h5.File(ps_filename, 'w') as f:
        pass

    for i in range(N_Z_BINS):
        delta_l = field_true[i]
        delta_x = combined_map.map_tool.fourier2map(delta_l)
        T1, T2 = combined_map.map_tool.do_fwd_KS(delta_l)
        Q1 = delta_x * T1
        Q2 = delta_x * T2

        T_field = np.array([T1, T2])
        Q_field = np.array([Q1, Q2])

        cats = get_cats(n_gals[i], delta_x, e_field[i], T_field, Q_field)
        ge_corr, gT_corr, gQ_corr          = get_all_ge_corr(cats, ng=ng)
        ee_corr, TT_corr, QQ_corr, TQ_corr = get_all_ee_corr(cats, gg=gg)
        with h5.File(ps_filename, 'r+') as f:
            g = f.create_group('bin_%d'%(i))
            g['ge_corr'] = ge_corr
            g['gT_corr'] = gT_corr
            g['gQ_corr'] = gQ_corr
            g['ee_corr'] = ee_corr
            g['TT_corr'] = TT_corr
            g['TQ_corr'] = TT_corr
            g['QQ_corr'] = QQ_corr


        
