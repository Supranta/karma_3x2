import numpy as np
import jax.numpy as jnp
from jax import grad, jit
from .shear_map import ShearMap
from ..utils import sample_kappa, get_lensing_spectra
import jax
from functools import partial
from jax.config import config
config.update("jax_enable_x64", True)

class GaussianMap(ShearMap):    
    def __init__(self, N_Z_BINS, N_grid, theta_max, n_z, cosmo_pars):
        super().__init__(N_Z_BINS, N_grid, theta_max, n_z, cosmo_pars)        
        self.set_eigs(self.Cl_arr_real, self.Cl_arr_imag)
        
    def init_kappa_l(self, scaling=1.):
        kappa_real = sample_kappa(self.Cl_arr_real, self.N_grid, self.N_Z_BINS)
        kappa_imag = sample_kappa(self.Cl_arr_imag, self.N_grid, self.N_Z_BINS)                       
            
        kappa_real, kappa_imag = self.enforce_symmetry([kappa_real, kappa_imag])        
        kappa_l = np.concatenate([kappa_real[:,np.newaxis], kappa_imag[:,np.newaxis]], axis=1)        
        
        return scaling * kappa_l
    
    @partial(jit, static_argnums=(0,))
    def log_prior(self, x):        
        ln_prior = 0.5 * jnp.sum(x**2)
        return ln_prior
    
    def log_like(self, x):   
        ln_like = 0.
        kappa = self.x2kappa(x)
        for i in range(self.N_Z_BINS):
            ln_like += self.log_like_1bin(kappa[i], self.eps[i], self.sigma_eps[i])        
        return ln_like 
    
    def grad_like(self, x):
        grad_like = np.array(grad(self.log_like, 0)(x))
        return grad_like
    
    def log_prob_cosmo(self, cosmo_pars, x, Emu=None):
        log_Om, log_A = cosmo_pars
        OmegaM = np.exp(log_Om) * self.OmegaM_fid
        A      = np.exp(log_A)
        As     = A * self.As_fid
        if (OmegaM < 0.17) or (OmegaM > 0.5):
            return - np.inf
        if (As < 1.2e-9) or (As > 4e-9):
            return - np.inf       
        Cl = Emu.get_lensing_spectrum_emu(log_Om, log_A)            
        Cl_arr_real, Cl_arr_imag, inv_Cl_arr_real, inv_Cl_arr_imag = self.get_Cl_arr(Cl)
        self.set_eigs(Cl_arr_real, Cl_arr_imag)
        return -self.log_like(x) + log_A + log_Om
    
    def x2kappa(self, x):
#         x = self.filter_1d[jnp.newaxis] * x
        x = self.wrap_fourier(x)        
        kappa_real = self.matmul(self.R_real, x[:,0] * jnp.sqrt(self.eig_val_real.T))
        kappa_imag = self.matmul(self.R_imag, x[:,1] * jnp.sqrt(self.eig_val_imag.T))
        return jnp.swapaxes(jnp.array([kappa_real, kappa_imag]), 0, 1)
    
    def kappa2x(self, kappa):
        x_real = self.matmul(self.R_real_T, kappa[:,0]) / np.sqrt(self.eig_val_real.T)
        x_imag = self.matmul(self.R_imag_T, kappa[:,1]) / np.sqrt(self.eig_val_imag.T)
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
