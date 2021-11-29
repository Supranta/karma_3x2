import numpy as np
import jax.numpy as jnp
from jax import grad, jit
from .combined_map import CombinedMap
from ..utils import sample_kappa, get_lensing_spectra
import jax
from functools import partial
from jax.config import config
config.update("jax_enable_x64", True)

class GaussianMap(CombinedMap):    
    def __init__(self, N_Z_BINS, N_grid, theta_max, n_z, probe_list, cosmo_pars):
        super().__init__(N_Z_BINS, N_grid, theta_max, n_z, probe_list, cosmo_pars)        
        self.set_eigs(self.Cl_arr_real, self.Cl_arr_imag)
        
    def init_field(self, scaling=1.):
        field_real = sample_kappa(self.Cl_arr_real, self.N_grid, self.N_Z_BINS)
        field_imag = sample_kappa(self.Cl_arr_imag, self.N_grid, self.N_Z_BINS)                       
            
        field_real, field_imag = self.enforce_symmetry([field_real, field_imag])        
        field_l = np.concatenate([field_real[:,np.newaxis], field_imag[:,np.newaxis]], axis=1)        
        
        return scaling * field_l              
            
    def x2field(self, x):
        x = self.wrap_fourier(x)        
        delta_real = self.matmul(self.R_real, x[:,0] * jnp.sqrt(self.eig_val_real.T))
        delta_imag = self.matmul(self.R_imag, x[:,1] * jnp.sqrt(self.eig_val_imag.T))
        return jnp.swapaxes(jnp.array([delta_real, delta_imag]), 0, 1)
    
    def field2x(self, delta):
        x_real = self.matmul(self.R_real_T, delta[:,0]) / np.sqrt(self.eig_val_real.T)
        x_imag = self.matmul(self.R_imag_T, delta[:,1]) / np.sqrt(self.eig_val_imag.T)
        x = np.swapaxes(np.array([x_real, x_imag]), 0, 1)
        x = self.reverse_wrap_fourier(x)
        x = jax.ops.index_update(x, jax.ops.index[:,-1], 0.)
        return x
    
    @partial(jit, static_argnums=(0,))
    def matmul(self, A, x):
        y = jnp.zeros(x.shape)
        for i in range(self.N_Z_BINS):
            a = jnp.sum(A[i,:] * x, axis=0)
            y = jax.ops.index_add(y, i, a)
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
