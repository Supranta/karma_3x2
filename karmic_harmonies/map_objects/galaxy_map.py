import numpy as np
import jax
import jax.numpy as jnp
from jax import grad
from scipy.interpolate import interp1d
from jax.config import config
config.update("jax_enable_x64", True)
from jax import jit
from functools import partial
from ..utils import sample_kappa, get_galaxy_spectra
import time

from .map_tools import MapTools

EPS = 1e-20
J = np.complex(0., 1.)

class GalaxyMap:
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

        Cl = get_galaxy_spectra(N_Z_BINS, self.zs, self.n_zs, self.OmegaM_fid, self.As_fid, h=h, ns=ns, Omega_b=Omega_b)
        self.Cl = Cl
        self.set_Cl_arr(Cl)        
    
    def create_synthetic_data(self, nbars):
        print("Creating synthetic map....")
        delta_l = self.init_delta_l(1.)
        counts_list = []
        for i in range(self.N_Z_BINS):
            delta_x = self.map_tool.fourier2map(delta_l[i])
            lambda_i = (1. + delta_x) * nbars[i]
            counts = np.random.poisson(lambda_i)
            counts_list.append(counts)
        print("Done creating synthetic map....")
        return np.array(counts_list), delta_l                      
    
    def grad_psi(self, x):
        return self.grad_prior(x) + self.grad_like(x)
    
    def psi(self, x):
        return self.log_prior(x) + self.log_like(x)            
    
    @partial(jit, static_argnums=(0,))
    def grad_prior(self, x):
        return grad(self.log_prior, 0)(x)                    
    
    @partial(jit, static_argnums=(0,))
    def log_like_1bin(self, delta_l, counts, nbar):
        delta_x = self.map_tool.fourier2map(delta_l)
        lambda_i = (1. + delta_x) * nbar        
        return jnp.sum(lambda_i - counts * jnp.log(lambda_i))
        

#=============== WRAP to Fourier object ====================
    @partial(jit, static_argnums=(0,))
    def wrap_fourier(self, x):
        x_fourier = jnp.zeros((self.N_Z_BINS, 2, self.N_grid, self.N_Y))
        for n in range(self.N_Z_BINS):
            y = self.map_tool.array2fourier_plane(x[n])
            x_fourier = jax.ops.index_update(x_fourier, n, y)
        return x_fourier
    
    @partial(jit, static_argnums=(0,))
    def reverse_wrap_fourier(self, F_x):
        x = jnp.zeros((self.N_Z_BINS, self.N_grid * self.N_grid - 1))
        for n in range(self.N_Z_BINS):
            y = self.map_tool.fourier_plane2array(F_x[n])
            x = jax.ops.index_update(x, n, y)
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
        
    def set_data(self, nbars, counts):               
        self.nbars = nbars
        self.counts = counts

        mass_arr_real    = self.inv_Cl_arr_real
        mass_arr_imag    = self.inv_Cl_arr_imag
        self.mass_arr    = np.concatenate([mass_arr_real[:,:,np.newaxis], 
                                           mass_arr_imag[:,:,np.newaxis]], axis=2)
        
        self.mass_arr *= 0.
        for i in range(self.N_Z_BINS):
            self.mass_arr[i,i] = 1.
    
    def set_mass_matrix(self):
        pass

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