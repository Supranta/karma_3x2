import numpy as np
import jax
import jax.numpy as jnp
import h5py as h5
from jax import grad
from scipy.interpolate import interp1d
from jax.config import config
config.update("jax_enable_x64", True)
from jax import jit
from functools import partial
from ..utils import sample_kappa, get_spectra
import time

from .map_tools import MapTools

EPS = 1e-20
J = np.complex(0., 1.)

class CombinedMap:
    def __init__(self, N_Z_BINS, N_grid, theta_max, n_z, cosmo_pars):
        """
        :N_Z_BINS:   Number of redshift bins
        :N_grid:     Number of pixels on each side. At the moment, we assume square geometry
        :theta_max:  Each side of the square (in degrees)
        :n_z:        n(z) for each redshift bin. Provided as a list of z and n(z) in each redshift bin
        :cosmo_pars: Cosmological parameters (Omega_m, A_s)
        """
        self.set_map_properties(N_Z_BINS, N_grid, theta_max)
        
        self.map_tool = MapTools(N_grid, theta_max)
        self.N_Y = self.map_tool.N_Y
        
        self.OmegaM_fid, self.As_fid, h, ns, Omega_b = cosmo_pars
        self.zs, self.n_zs           = n_z
        
        Cl = get_spectra(N_Z_BINS, self.zs, self.n_zs, self.OmegaM_fid, self.As_fid, 
                         h=h, ns=ns, Omega_b=Omega_b)

        self.Cl = Cl
        self.set_Cl_arr(Cl)
    
    def get_synthetic_lensing_data(self, kappa_l, sigma_eps, galaxy_number):
        sigma_eps_i = sigma_eps / np.sqrt(galaxy_number)

        gamma_1, gamma_2 = self.map_tool.do_fwd_KS(kappa_l)

        eps_1_i = gamma_1 + sigma_eps_i * np.random.normal(size=gamma_1.shape)
        eps_2_i = gamma_2 + sigma_eps_i * np.random.normal(size=gamma_2.shape)
        
        data = np.array([eps_1_i, eps_2_i])
        return data
    
    def get_synthetic_galaxy_data(self, delta_l, nbar):
        delta_x = self.map_tool.fourier2map(delta_l)
        lambda_i = (1. + delta_x) * nbar * self.PIXEL_AREA
        data = np.random.poisson(lambda_i)        
        return data
        
    def create_synthetic_data(self, nbars, sigma_eps):
        print("Creating synthetic map....")
        field = self.init_field(1.)
        n_gals_list = []       
        ellipticity_list = []       
        for i in range(self.N_Z_BINS):
            n_gals = self.get_synthetic_galaxy_data(field[i], nbars[i])
            ellipticity = self.get_synthetic_lensing_data(field[i], sigma_eps[i], n_gals)
            n_gals_list.append(n_gals)           
            ellipticity_list.append(ellipticity)            
        return field, n_gals_list, ellipticity_list                   
    
    @partial(jit, static_argnums=(0,))
    def log_prior(self, x):        
        ln_prior = 0.5 * jnp.sum(x**2)
        return ln_prior 
    
    @partial(jit, static_argnums=(0,))
    def grad_prior(self, x):
        return grad(self.log_prior, 0)(x)                    
    
    @partial(jit, static_argnums=(0,))
    def log_like_1bin_shear(self, kappa_l, data):
        eps       = data['eps']
        sigma_eps = data['sigma_eps']
        n_bar     = data['nbar']
        
        gamma_1, gamma_2 = self.map_tool.do_fwd_KS(kappa_l)

        eps_1, eps_2 = eps      
        
        residual_1 = (gamma_1 - eps_1) / sigma_eps
        residual_2 = (gamma_2 - eps_2) / sigma_eps
        
        return 0.5 * jnp.sum(residual_1**2) + 0.5 * jnp.sum(residual_2**2) 
    
    @partial(jit, static_argnums=(0,))
    def log_like_1bin_galaxy(self, delta_l, data):
        nbar   = data['nbar']
        counts = data['counts']
        delta_x = self.map_tool.fourier2map(delta_l)
        lambda_i = (1. + delta_x) * nbar * self.PIXEL_AREA      
        return jnp.sum(lambda_i - counts * jnp.log(lambda_i))
    
    def log_like(self, x):   
        ln_like = 0.        
        field = self.x2field(x)
        for i in range(self.N_Z_BINS):
            ln_like_bin_shape = self.log_like_1bin_shear(field[i], self.data[i])
            ln_like_bin_count = self.log_like_1bin_galaxy(field[i], self.data[i])
            ln_like = ln_like + ln_like_bin_shape + ln_like_bin_count
        return ln_like 
    
    def grad_like(self, x):
        grad_like = np.array(grad(self.log_like, 0)(x))
        return grad_like
    
    def grad_psi(self, x):
        return self.grad_prior(x) + self.grad_like(x)
    
    def psi(self, x):
        return self.log_prior(x) + self.log_like(x)

#=============== WRAP to Fourier object ====================
    @partial(jit, static_argnums=(0,))
    def wrap_fourier(self, x):
        x_fourier = jnp.zeros((self.N_Z_BINS, 2, self.N_grid, self.N_Y))
        for n in range(self.N_Z_BINS):
            y = self.map_tool.array2fourier_plane(x[n])
            x_fourier = x_fourier.at[n].set(y)
        return x_fourier
    
    @partial(jit, static_argnums=(0,))
    def reverse_wrap_fourier(self, F_x):
        x = jnp.zeros((self.N_Z_BINS, self.N_grid * self.N_grid - 1))
        for n in range(self.N_Z_BINS):
            y = self.map_tool.fourier_plane2array(F_x[n])
            x = x.at[n].set(y)
        return x
    
#=============== SET OBJECT PROPERTIES ====================
    def set_map_properties(self, N_Z_BINS, N_grid, theta_max):
        self.N_Z_BINS = N_Z_BINS
        self.N_grid   = N_grid
        self.theta_max = theta_max * np.pi / 180.
        self.Omega_s = self.theta_max**2
        self.PIXEL_AREA = (theta_max * 60. / N_grid)**2
        
        arcminute_in_radian    = (1./60) * np.pi / 180.
        self.arcminute_sq_in_radian = arcminute_in_radian**2

    def set_Cl_arr(self, Cl):    
        Cl_arr_real, Cl_arr_imag , inv_Cl_arr_real, inv_Cl_arr_imag = self.get_Cl_arr(Cl)
    
        self.Cl_arr_imag = Cl_arr_imag
        self.Cl_arr_real = Cl_arr_real
        
        self.inv_Cl_arr_real = inv_Cl_arr_real
        self.inv_Cl_arr_imag = inv_Cl_arr_imag             
    
    def get_data(self, g):                
        bin_data = {}
        bin_data['nbar']      = g['nbar'][()]
        bin_data['counts']    = g['counts'][:]
        sigma_eps             = g['sigma_eps'][()]
        sigma_eps_pixel       = sigma_eps / np.sqrt(bin_data['counts'])
        bin_data['eps']       = g['eps'][:]
        bin_data['sigma_eps'] = sigma_eps_pixel
        
        return bin_data

    def set_data(self, datafile):        
        data_list = []
        with h5.File(datafile, 'r') as f:
            for i in range(self.N_Z_BINS):
                bin_data = self.get_data(f['bin_%d'%(i)]['data'])
                data_list.append(bin_data)    
        self.data = data_list
            
    def set_mass_matrix(self):
        mass_arr_real    = self.inv_Cl_arr_real
        mass_arr_imag    = self.inv_Cl_arr_imag
        self.mass_arr    = np.concatenate([mass_arr_real[:,:,np.newaxis], 
                                           mass_arr_imag[:,:,np.newaxis]], axis=2)
        
        self.mass_arr *= 0.
        for i in range(self.N_Z_BINS):
            self.mass_arr[i,i] = 1.

#=============== HELPER ROUTINES ====================
    def get_Cl_arr(self, Cl):
        ls = np.arange(len(Cl[0,0]))
        lmax = ls[-1]
        self.ls = ls
        
        Cl_arr = np.zeros((self.N_Z_BINS, self.N_Z_BINS, self.N_grid, self.N_Y))
        
        for i in range(self.N_Z_BINS):
            for j in range(self.N_Z_BINS):
                Cl_arr[i,j]     = 0.5 * self.interp_arr(ls, Cl[i,j])

        Cl_arr_real = Cl_arr.copy()                
        Cl_arr_imag = Cl_arr.copy()
        
        Cl_arr_imag[:,:,0,0]          = 1e-20 * np.diag(np.ones(self.N_Z_BINS))
        Cl_arr_imag[:,:,0,-1]         = 1e-20 * np.diag(np.ones(self.N_Z_BINS))
        Cl_arr_imag[:,:,self.N_grid//2,0]  = 1e-20 * np.diag(np.ones(self.N_Z_BINS))
        Cl_arr_imag[:,:,self.N_grid//2,-1] = 1e-20 * np.diag(np.ones(self.N_Z_BINS))

        Cl_arr_real[:,:,0,0]          = 2. * Cl_arr_real[:,:,0,0] 
        Cl_arr_real[:,:,0,-1]         = 2. * Cl_arr_real[:,:,0,-1] 
        Cl_arr_real[:,:,self.N_grid//2,0]  = 2. * Cl_arr_real[:,:,self.N_grid//2,0] 
        Cl_arr_real[:,:,self.N_grid//2,-1] = 2. * Cl_arr_real[:,:,self.N_grid//2,-1] 
        
        Cl_arr_imag = Cl_arr_imag * self.Omega_s
        Cl_arr_real = Cl_arr_real * self.Omega_s
        
        inv_Cl_arr_real = np.linalg.inv(Cl_arr_real.T).T 
        inv_Cl_arr_imag = np.linalg.inv(Cl_arr_imag.T).T 
            
        return Cl_arr_real, Cl_arr_imag , inv_Cl_arr_real, inv_Cl_arr_imag
    
    def interp_arr(self, ls, y):
        interp_func = interp1d(ls, y)
        return interp_func(self.map_tool.ell)
    
    
    def enforce_symmetry(self, x):
        x_real, x_imag = x        
        
        x_imag[:,0,0]          = 0.
        x_imag[:,0,-1]         = 0.
        x_imag[:,self.N_grid//2,0]  = 0.
        x_imag[:,self.N_grid//2,-1] = 0.
        
        x_real[:,(self.N_Y):,0]  = x_real[:,1:self.N_Y-1,0][::-1]
        x_real[:,(self.N_Y):,-1] = x_real[:,1:self.N_Y-1,-1][::-1]
        
        x_imag[:,(self.N_Y):,-1] = -x_imag[:,1:self.N_Y-1,-1][::-1]
        x_imag[:,(self.N_Y):,0]  = -x_imag[:,1:self.N_Y-1,0][::-1]
        
        return x_real, x_imag
    
    @partial(jit, static_argnums=(0,))
    def matmul(self, A, x):
        y = jnp.zeros(x.shape)
        for i in range(self.N_Z_BINS):
            a = jnp.sum(A[i,:] * x, axis=0)
            y = y.at[i].add(a)
        return y
        
    def set_eigs(self, Cl_arr_real, Cl_arr_imag):
        self.eig_val_real, eig_vec_real = np.linalg.eig(Cl_arr_real.T)            
        self.eig_val_imag, eig_vec_imag = np.linalg.eig(Cl_arr_imag.T)        
        self.R_real = np.swapaxes(eig_vec_real.T, 0, 1)
        self.R_imag = np.swapaxes(eig_vec_imag.T, 0, 1)
        self.R_real = self.eig_vec_normalize(self.R_real)
        self.R_imag = self.eig_vec_normalize(self.R_imag)
        self.R_real_T = np.swapaxes(self.R_real, 0, 1)
        self.R_imag_T = np.swapaxes(self.R_imag, 0, 1)
    
    def eig_vec_normalize(self, x):
        sign_matrix = (x[0] + 1e-25) / np.abs(x[0] + 1e-25)
        sign_matrix = sign_matrix[np.newaxis]
        return sign_matrix * x
