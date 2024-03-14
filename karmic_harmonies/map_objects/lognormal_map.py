from math import pi
import numpy as np
import jax.numpy as jnp
from jax import grad, jit
from .combined_map import CombinedMap
from ..utils import sample_kappa, get_lensing_spectra
import jax
from functools import partial
from jax.config import config
config.update("jax_enable_x64", True)

class LogNormalMap(CombinedMap): 
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
        
    def init_field_gauss(self, scaling):
        field_real = sample_kappa(self.Cl_arr_real, self.N_grid, self.N_Z_BINS)
        field_imag = sample_kappa(self.Cl_arr_imag, self.N_grid, self.N_Z_BINS)                       
            
        field_real, field_imag = self.enforce_symmetry([field_real, field_imag])        
        fields = np.concatenate([field_real[:,np.newaxis], field_imag[:,np.newaxis]], axis=1)        
        
        for n in range(self.N_Z_BINS):
            fields[n] = self.map_tool.symmetrize_fourier(fields[n])            
        return scaling * fields
    
    def init_field(self, scaling):
        field_g = self.init_field_gauss(scaling)
        fields  = np.zeros(field_g.shape)
        for i in range(self.N_Z_BINS):
            field_g_map = self.map_tool.fourier2map(field_g[i])
            field_map   = self.shifts[i] * (np.exp(field_g_map - 0.5 * self.var_gauss[i]) - 1.)
            fields[i]     = self.map_tool.map2fourier(field_map) 
        return fields    

    def x2field(self, x):
        x              = self.wrap_fourier(x)
        fieldG_Fourier = self.x2fieldG_fourier(x)
        fieldG_map     = self.fieldG_fourier2map(fieldG_Fourier)
        fields         = self.fieldGmap_2_fourier(fieldG_map, self.shifts, self.var_gauss)
        return fields
    
    @partial(jit, static_argnums=(0,))
    def fieldGmap_2_fourier(self, fieldG_map, shifts, var_gauss):
        fields  = jnp.zeros((self.N_Z_BINS, 2, self.N_grid, self.N_Y))
        for i in range(self.N_Z_BINS):
            field_map = shifts[i] * (jnp.exp(fieldG_map[i] - 0.5 * var_gauss[i]) - 1.)
            fields    = jax.ops.index_add(fields, i, self.map_tool.map2fourier(field_map))
        return fields
    
    @partial(jit, static_argnums=(0,))
    def fieldG_fourier2map(self, fieldG_Fourier):                       
        fieldG_map = jnp.zeros((self.N_Z_BINS, 
                                self.map_tool.N_grid, 
                                self.map_tool.N_grid))
        for n in range(self.N_Z_BINS):
            fieldG_l   = self.map_tool.symmetrize_fourier(fieldG_Fourier[n])
            fieldG_map = jax.ops.index_add(fieldG_map, n, self.map_tool.fourier2map(fieldG_l))
        return fieldG_map
    
    def x2fieldG_fourier(self, x):
        field_real = self.matmul(self.R_real, x[:,0] * jnp.sqrt(self.eig_val_real.T))
        field_imag = self.matmul(self.R_imag, x[:,1] * jnp.sqrt(self.eig_val_imag.T))
        return jnp.swapaxes(jnp.array([field_real, field_imag]), 0, 1)
    
    def get_var_gauss(self, Cl_g):
        var_gauss_arr = np.zeros(self.N_Z_BINS)        
        for i in range(self.N_Z_BINS):
            var_gauss_arr[i]= self._var_gauss_single_bin(Cl_g[i,i])
        return var_gauss_arr
    
    def _var_gauss_single_bin(self, Cl_g):
        return np.sum(self.pixel_window * (1. + 2. * self.ls[:self.lmax]) * Cl_g)/ 4. / np.pi
        
    def field2x(self, fields):
        fields_g_l = np.zeros(fields.shape)
        for n in range(self.N_Z_BINS):
            field_map = np.array(self.map_tool.fourier2map(fields[n]))
            field_g_map = 0.5 * self.var_gauss[n] + np.log(1. + field_map/self.shifts[n])
            fields_g_l[n] = self.map_tool.map2fourier(field_g_map)        
        x_real = self.matmul(self.R_real_T, fields_g_l[:,0]) / np.sqrt(self.eig_val_real.T)
        x_imag = self.matmul(self.R_imag_T, fields_g_l[:,1]) / np.sqrt(self.eig_val_imag.T)
        x = jnp.swapaxes(jnp.array([x_real, x_imag]), 0, 1)   
        return self.reverse_wrap_fourier(x)
        
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
