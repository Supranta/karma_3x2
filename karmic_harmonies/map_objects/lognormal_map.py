from math import pi
import numpy as np
import jax.numpy as jnp
from jax import grad, jit
from .shear_map import ShearMap
from ..utils import sample_kappa, get_lensing_spectra
import jax
from functools import partial
from jax.config import config
config.update("jax_enable_x64", True)

class LogNormalMap(ShearMap):    
    def __init__(self, N_Z_BINS, N_grid, theta_max, n_z, cosmo_pars, shifts, precalculated):
        super().__init__(N_Z_BINS, N_grid, theta_max, n_z, cosmo_pars)  
        self.N_pix = N_grid * N_grid
        self.Pl_theta, self.w_i, self.ls, self.lmax = precalculated
        self.shifts = shifts
        self.pixel_window = self.pixel_window_func(self.ls)
        self.Cl    = self.Cl[:,:,:self.lmax]
        self.Cl_g  = self.cl2clg(self.Cl)
        self.set_Cl_arr(self.Cl_g)
        
        self.var_gauss = self.get_var_gauss(self.Cl_g)
        #=================================
        self.set_eigs(self.Cl_arr_real, self.Cl_arr_imag)
        
    def init_kappa_l_gauss(self, scaling):
        kappa_real = sample_kappa(self.Cl_arr_real, self.N_grid, self.N_Z_BINS)
        kappa_imag = sample_kappa(self.Cl_arr_imag, self.N_grid, self.N_Z_BINS)                       
            
        kappa_real, kappa_imag = self.enforce_symmetry([kappa_real, kappa_imag])        
        kappa_l = np.concatenate([kappa_real[:,np.newaxis], kappa_imag[:,np.newaxis]], axis=1)        
        
        for n in range(self.N_Z_BINS):
            kappa_l[n] = self.map_tool.symmetrize_fourier(kappa_l[n])            
        return scaling * kappa_l
    
    def init_kappa_l(self, scaling):
        kappa_l_g   = self.init_kappa_l_gauss(scaling)
        kappa_l     = np.zeros(kappa_l_g.shape)
        for i in range(self.N_Z_BINS):
            kappa_g_map = self.map_tool.fourier2map(kappa_l_g[i])
            kappa_map   = self.shifts[i] * (np.exp(kappa_g_map - 0.5 * self.var_gauss[i]) - 1.)
            kappa_l[i]     = self.map_tool.map2fourier(kappa_map) 
        return kappa_l
    
    @partial(jit, static_argnums=(0,))
    def log_prior(self, x):        
        ln_prior = 0.5 * jnp.sum(x**2)
        return ln_prior
    
    def log_like(self, x):   
        ln_like = 0.
        kappa_l = self.x2kappa(x)
        for n in range(self.N_Z_BINS):
            kappa_bin = self.map_tool.symmetrize_fourier(kappa_l[n])
            ln_like += self.log_like_1bin(kappa_bin, self.eps[n], self.sigma_eps[n])        
        return ln_like  

    @partial(jit, static_argnums=(0,))
    def log_like_kappa(self, kappa_l):   
        ln_like = 0.
        for n in range(self.N_Z_BINS):
            ln_like += self.log_like_1bin(kappa_l[n], self.eps[n], self.sigma_eps[n])        
        return ln_like 
    
    def grad_like(self, x):
        grad_like = grad(self.log_like, 0)(x)
        return grad_like

    def x2kappa(self, x):
        x              = self.wrap_fourier(x)
        kappaG_Fourier = self.x2kappaG_fourier(x)
        kappaG_map     = self.kappaG_fourier2map(kappaG_Fourier)
        kappa_l = self.kappaGmap_2_kappa_l(kappaG_map, self.shifts, self.var_gauss)
        return kappa_l
    
    @partial(jit, static_argnums=(0,))
    def kappaGmap_2_kappa_l(self, kappaG_map, shifts, var_gauss):
        kappa_l        = jnp.zeros((self.N_Z_BINS, 2, self.N_grid, self.N_Y))
        for i in range(self.N_Z_BINS):
            kappa_map  = shifts[i] * (jnp.exp(kappaG_map[i] - 0.5 * var_gauss[i]) - 1.)
            kappa_l    = jax.ops.index_add(kappa_l, i, self.map_tool.map2fourier(kappa_map))
        return kappa_l
    
    @partial(jit, static_argnums=(0,))
    def kappaG_fourier2map(self, kappaG_Fourier):                       
        kappaG_map = jnp.zeros((self.N_Z_BINS, 
                                self.map_tool.N_grid, 
                                self.map_tool.N_grid))
        for n in range(self.N_Z_BINS):
            kappaG_l   = self.map_tool.symmetrize_fourier(kappaG_Fourier[n])
            kappaG_map = jax.ops.index_add(kappaG_map, n, self.map_tool.fourier2map(kappaG_l))
        return kappaG_map
    
    def x2kappaG_fourier(self, x):
        kappa_real = self.matmul(self.R_real, x[:,0] * jnp.sqrt(self.eig_val_real.T))
        kappa_imag = self.matmul(self.R_imag, x[:,1] * jnp.sqrt(self.eig_val_imag.T))
        return jnp.swapaxes(jnp.array([kappa_real, kappa_imag]), 0, 1)
    
    def get_var_gauss(self, Cl_g):
        var_gauss_arr = np.zeros(self.N_Z_BINS)        
        for i in range(self.N_Z_BINS):
            var_gauss_arr[i]= self._var_gauss_single_bin(Cl_g[i,i])
        return var_gauss_arr
    
    def _var_gauss_single_bin(self, Cl_g):
        return np.sum(self.pixel_window * (1. + 2. * self.ls[:self.lmax]) * Cl_g)/ 4. / np.pi
    
    def log_prob_cosmo(self, cosmo_pars, x, Emu=None):
        log_Om, log_A = cosmo_pars
        OmegaM = np.exp(log_Om) * self.OmegaM_fid
        A      = np.exp(log_A)
        As     = A * self.As_fid
        if (OmegaM < 0.17) or (OmegaM > 0.5):
            return - np.inf
        if (As < 1.2e-9) or (As > 4e-9):
            return - np.inf                                       
        
        self.shifts    = Emu.get_shift_emu(log_Om, log_A)
        self.Cl_g      = Emu.get_Cl_g_emu(log_Om, log_A)
        self.var_gauss = Emu.get_var_gauss_emu(log_Om, log_A)
        self.set_Cl_arr(self.Cl_g)        
        
        self.set_eigs(self.Cl_arr_real, self.Cl_arr_imag)
        log_L = self.log_like(x)
        log_P = -log_L + log_A + log_Om
        return log_P
    
    def kappa2x(self, kappa_l):
        kappa_g_l = np.zeros(kappa_l.shape)
        for n in range(self.N_Z_BINS):
            kappa_map = np.array(self.map_tool.fourier2map(kappa_l[n]))
            kappa_g_map = 0.5 * self.var_gauss[n] + np.log(1. + kappa_map/self.shifts[n])
            kappa_g_l[n] = self.map_tool.map2fourier(kappa_g_map)        
        x_real = self.matmul(self.R_real_T, kappa_g_l[:,0]) / np.sqrt(self.eig_val_real.T)
        x_imag = self.matmul(self.R_imag_T, kappa_g_l[:,1]) / np.sqrt(self.eig_val_imag.T)
        x = jnp.swapaxes(jnp.array([x_real, x_imag]), 0, 1)   
        return self.reverse_wrap_fourier(x)
    
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
    
    def cl2clg(self, Cl):
        Cl_g = np.zeros((self.N_Z_BINS, self.N_Z_BINS, self.lmax))
        for i in range(self.N_Z_BINS):
            for j in range(i+1):
                Cl_g[i,j] = self._cl2clg_single_bin(Cl[i,j], self.shifts[i], self.shifts[j])
                if(i!=j):
                    Cl_g[j,i] = Cl_g[i,j]
        Cl_g[:,:,0] = 1e-15 * np.eye(self.N_Z_BINS)
        Cl_g[:,:,1] = 1e-15 * np.eye(self.N_Z_BINS)
        return Cl_g
    
    def _cl2clg_single_bin(self, Cl, shift1, shift2=None):
        chi = 1./(4. * pi) * np.sum(self.Pl_theta * Cl[:self.lmax] * (1 + 2. * self.ls[:self.lmax]), axis=1)
        if shift2 is None:
            shift2 = shift11
        chi_g = np.log(1. + chi / shift1 / shift2)
        Cl_g = (2. * pi) * np.sum(((self.w_i * chi_g).reshape(-1,1) * self.Pl_theta), axis=0)
        Cl_g[:2] = 1e-15
        return Cl_g
    
    def pixel_window_func(self, ls, anisotropic=False):
        ΔΘ = self.theta_max / self.N_grid
        π = np.pi
        filter_arr = np.sinc(0.5 * self.ls * ΔΘ / np.pi)
        return filter_arr    
