import configparser
import h5py as h5
import numpy as np

def parse_string_array(string):
    split_string = string.split(',')
    return np.array(split_string).astype(float)

def config_ps(configfile):
    print('Entering config_ps....')
    
    config = configparser.ConfigParser()
    config.read(configfile)
    
    N_ELL_BINS = int(config['PS']['N_ELL_BINS'])
    N_MOCKS    = int(config['PS']['N_MOCKS'])
    
    print('Exiting config_ps....')
    
    return N_ELL_BINS, N_MOCKS

def config_mock(configfile):
    print('Entering config_mock....')
    
    config = configparser.ConfigParser()
    config.read(configfile)
    
    n_bar = parse_string_array(config['MAP']['n_bar'])
    sigma_eps = parse_string_array(config['MAP']['sigma_eps'])    
    
    print('Exiting config_mock....')
    
    return n_bar, sigma_eps
    
def config_map(configfile):
    print('Entering config_map....')
    
    config = configparser.ConfigParser()
    config.read(configfile)
    
    N_Z_BINS  = int(config['MAP']['N_Z_BINS'])
    n_z_file  = config['MAP']['n_z_file']    
        
    N_GRID    = int(config['MAP']['N_GRID'])
    theta_max = float(config['MAP']['theta_max'])    
    
    print('Exiting config_map....')
    
    return N_Z_BINS, n_z_file, N_GRID, theta_max

def config_io(configfile):
    print('Entering config_io....')
    
    config = configparser.ConfigParser()
    config.read(configfile)
    
    datafile  = config['IO']['datafile']
    savedir   = config['IO']['save_dir']
    N_SAVE    = int(config['IO']['N_SAVE'])
    N_RESTART = int(config['IO']['N_RESTART'])    
    
    print('Exiting config_io....')
    
    return datafile, savedir, N_SAVE, N_RESTART

def config_cosmo_pars(configfile):
    print('Entering config_cosmo_pars....')
    
    config = configparser.ConfigParser()
    config.read(configfile)
    
    try:
        Omega_m = float(config['COSMOLOGY']['Omega_m'])
        As      = float(config['COSMOLOGY']['As']) * 1e-9
        h       = float(config['COSMOLOGY']['h'])
        ns      = float(config['COSMOLOGY']['ns'])
        Omega_b = float(config['COSMOLOGY']['Omega_b'])
    except:
        Omega_m = 0.279
        As = 2.249e-9
        h = 0.7
        ns = 0.97
        Omega_b = 0.046
    print('Exiting config_cosmo_pars....')
    cosmo_pars = [Omega_m, As, h, ns, Omega_b]
    
    return cosmo_pars
    
def config_sampling(configfile):
    print('Entering config_sampling....')
    
    config = configparser.ConfigParser()
    config.read(configfile)
    
    sample_map    = bool(config['SAMPLING']['sample_map'].lower()=="true")
    sample_cosmo  = bool(config['SAMPLING']['sample_cosmo'].lower()=="true")
    N_COSMO       = int(config['SAMPLING']['N_COSMO'])
    N_MCMC        = int(config['SAMPLING']['N_MCMC'])
    N_ADAPT       = int(config['SAMPLING']['N_ADAPT'])
    dt            = float(config['SAMPLING']['dt'])
    N_LEAPFROG    = int(config['SAMPLING']['N_LEAPFROG'])
    try:
        precalculated_mass_matrix = bool(config['SAMPLING']['precalculated_mass_matrix'].lower()=="true")
    except:
        precalculated_mass_matrix = False

    print('Exiting config_sampling....')
    
    return sample_map, sample_cosmo, N_COSMO, N_ADAPT, N_MCMC, dt, N_LEAPFROG, precalculated_mass_matrix

def config_lognormal(configfile, base_dir='./'):
    print('Entering config_lognormal....')
    
    config = configparser.ConfigParser()
    config.read(configfile)
    
    try:
        lognormal = True
        precalculated_file = base_dir + config['LOGNORMAL']['precalculated_file']
        print("precalculated_file: "+precalculated_file)
        with h5.File(precalculated_file, 'r') as f:
            Pl_theta = f['Pl_theta'][:]
            x_i      = f['x_i'][:]   
            theta_i  = f['theta_i'][:]
            w_i      = f['w_i'][:]   
            ls       = f['ls'][:]
            lmax     = f['lmax'][()]
        precalculated = [Pl_theta, w_i, ls, lmax]
        shift = parse_string_array(config['LOGNORMAL']['shift'])
        try:
            var_gauss = parse_string_array(config['LOGNORMAL']['var_gauss'])
        except:
            var_gauss = None
    except:
        print("Precalculation file not found...Will run Gaussian prior...")
        lognormal     = False
        precalculated = None
        shift         = None
        var_gauss     = None

    print('Exiting config_lognormal....')
    
    return lognormal, precalculated, shift, var_gauss
    
def config_cosmo(configfile):
    print('Entering config_cosmo....')
    
    config = configparser.ConfigParser()
    config.read(configfile)
    
    sigma_Om = float(config['COSMO']['sigma_Om'])
    sigma_A  = float(config['COSMO']['sigma_A'])
    rho      = float(config['COSMO']['rho'])
    try:      
        emulator_file = config['COSMO']['emulator_file']
    except:     
        emulator_file = None
    try:
        cosmo_sampler = config['COSMO']['cosmo_sampler']
    except:
        cosmo_sampler = 'slice'
    
    print('Exiting config_cosmo....')
    
    return cosmo_sampler, sigma_Om, sigma_A, rho, emulator_file
        
class IOHandler:
    def __init__(self, savedir, N_SAVE, N_RESTART, sample_map, sample_cosmo, OmegaM):
        self.savedir = savedir     
        self.N_SAVE = N_SAVE
        self.N_RESTART = N_RESTART
        self.accepted_count       = 0
        self.cosmo_accepted_count = 0
        self.sample_map   = sample_map
        self.sample_cosmo = sample_cosmo
        self.OmegaM_fid   = OmegaM
        
    def write_synthetic_data(self, eps, sigma_eps, kappa_l):
        with h5.File(self.savedir+'/synthetic_data.h5', 'w') as f:
            f['eps'] = eps
            f['sigma_eps'] = sigma_eps
            f['kappa_l']   = kappa_l
        
    def map_sample_output(self, acc, i, N_MODES, N_pix, ln_prior, ln_like):
        if(acc):   
            self.accepted_count += 1                
        acceptance_rate = self.accepted_count / (i - self.N_START + 1)                    
        print("HMC acceptance rate: %2.3f"%(acceptance_rate))                        
        print("===================")
        print("Log-prior chi-squared per mode: "+str(2. * ln_prior / N_MODES))
        print("Chi-squared per pixel: "+str(ln_like / N_pix))
        print("===================")  
    
    def cosmo_sample_output(self, i, cosmo_pars, Emu, ShearMap, lognormal=False, sample_map=True):
        log_Om, log_A = cosmo_pars
        OmegaM = np.exp(log_Om) * self.OmegaM_fid
        A      = np.exp(log_A)
        if(sample_map):
            print("Setting the new Cl_arr") 
            if Emu is not None:                
                if(lognormal):
                    ShearMap.shifts    = Emu.get_shift_emu(log_Om, log_A)            
                    ShearMap.Cl_g      = Emu.get_Cl_g_emu(log_Om, log_A)            
                    ShearMap.var_gauss = Emu.get_var_gauss_emu(log_Om, log_A)            
                    Cl = ShearMap.Cl_g                    
                else:
                    Cl = Emu.get_lensing_spectrum_emu(log_Om, log_A)                
            ShearMap.set_Cl_arr(Cl)
            ShearMap.set_eigs(ShearMap.Cl_arr_real, ShearMap.Cl_arr_imag)             
        print("Omega_m: %2.6f"%(OmegaM))
        print("A: %2.6f"%(A))    
        return ShearMap
            
    def write_map_output(self, i, kappa_l, x, KE, ln_prob, ln_like, ln_prior):
        if(i%self.N_SAVE==0):
            print("Saving sample...")
            f = h5.File(self.savedir+'/mcmc_'+str(i//self.N_SAVE)+'.h5', 'w')                    
            
            f['kappa_l'] = kappa_l
            f['x']       = x
            f['KE']      = KE
            f['ln_prob'] = ln_prob            
            f['ln_like']  = ln_like                
            f['ln_prior'] = ln_prior
            
            f.close()
            
    def write_cosmo_output(self, i, cosmo_pars, cosmo_lnprob):
        if(i%self.N_SAVE==0):
            print("Saving sample...")
            try:
                f = h5.File(self.savedir+'/mcmc_'+str(i//self.N_SAVE)+'.h5', 'r+')
            except:
                f = h5.File(self.savedir+'/mcmc_'+str(i//self.N_SAVE)+'.h5', 'w')                                           
            f['cosmo_pars']   = cosmo_pars
            f['cosmo_lnprob'] = cosmo_lnprob
            f.close()
            
            
    def write_restart(self, i, kappa_l, x, cosmo_pars, cosmo_lnprob): 
        if(i%self.N_RESTART==0):
            print("Saving restart file...")
            f = h5.File(self.savedir+'/restart.h5', 'w')
            f['N_STEP'] = i
            f['kappa_l'] = kappa_l
            f['x']       = x
            f['cosmo_pars']   = cosmo_pars
            f['cosmo_lnprob'] = cosmo_lnprob
            f.close()               