import h5py as h5
import numpy as np
from .pce_emulator import PCEEmulator
from ..utils import get_lensing_spectra

class Emulator:
    def __init__(self, N_Z_BINS, A_s, Om, lognormal):
        self.N_Z_BINS = N_Z_BINS
        self.A_s      = A_s
        self.Om       = Om
        self.N_DIM    = 2
        self.lognormal = lognormal
        
    def get_lensing_spectrum_emu(self, log_Om, log_A):
        Om = np.exp(log_Om) * self.Om
        As = np.exp(log_A) * self.A_s
        theta = np.array([Om, As])
        pca_coeff_pred = self.Cl_emu.predict(theta)
        log_Cl_pred    = self.Cl_emu.do_inverse_pca(pca_coeff_pred)
        Cl_pred        = np.exp(log_Cl_pred)
        Cl_pred        = Cl_pred.reshape((self.N_Z_BINS, self.N_Z_BINS, -1))
        Cl_pred[:,:,0] = 1e-15 * np.diag(np.ones(self.N_Z_BINS))
        Cl_pred[:,:,1] = 1e-15 * np.diag(np.ones(self.N_Z_BINS))
        return Cl_pred
    
    def get_Cl_g_emu(self, log_Om, log_A):
        Om = np.exp(log_Om) * self.Om
        As = np.exp(log_A) * self.A_s
        theta = np.array([Om, As])
        pca_coeff_pred = self.Cl_g_emu.predict(theta)
        log_Cl_pred    = self.Cl_g_emu.do_inverse_pca(pca_coeff_pred)
        Cl_pred        = np.exp(log_Cl_pred)
        Cl_pred        = Cl_pred.reshape((self.N_Z_BINS, self.N_Z_BINS, -1))
        Cl_pred[:,:,0] = 1e-15 * np.eye(self.N_Z_BINS)
        Cl_pred[:,:,1] = 1e-15 * np.eye(self.N_Z_BINS)
        return Cl_pred
    
    def get_theta(self, log_Om, log_A):
        Om = np.exp(log_Om) * self.Om
        As = np.exp(log_A) * self.A_s
        return np.array([Om, As])
    
    def get_shift_emu(self, log_Om, log_A):
        theta = self.get_theta(log_Om, log_A)
        shift_pca_pred = self.shift_emu.predict(theta)
        shift_pred = self.shift_emu.do_inverse_pca(shift_pca_pred)
        return shift_pred
    
    def get_var_gauss_emu(self, log_Om, log_A):
        theta = self.get_theta(log_Om, log_A)
        var_gauss_pca_pred = self.var_gauss_emu.predict(theta)
        var_gauss_pred = self.var_gauss_emu.do_inverse_pca(var_gauss_pca_pred)
        return var_gauss_pred
    
    def setup_from_file(self, filename):
        with h5.File(filename, 'r') as f:
            fit_Cls = f['Cl'][:]
            fit_theta = f['theta'][:]
            prior_lims = f['prior_lims'][:]
        N_SAMPLES = len(fit_theta)            
        
        self.Cl_emu = PCEEmulator(self.N_DIM, 8, prior_lims)       
        fit_Cls = fit_Cls.reshape((N_SAMPLES, -1))
        log_Cl  = np.log(fit_Cls.astype(np.float64) + 1e-50)
        pca_coeff = self.Cl_emu.do_pca(log_Cl)
        self.Cl_emu.train(fit_theta, pca_coeff, 8)
        
        if(self.lognormal):
            with h5.File(filename, 'r') as f:
                fit_Cl_g = f['Cl_g'][:]
                fit_shifts = f['shift'][:]
                fit_var_gauss = f['var_gauss'][:]

            self.shift_emu = PCEEmulator(self.N_DIM, min(self.N_Z_BINS, 3), prior_lims)
            shift_pca = self.shift_emu.do_pca(fit_shifts)
            self.shift_emu.train(fit_theta, shift_pca)
            
            self.var_gauss_emu = PCEEmulator(self.N_DIM, min(self.N_Z_BINS, 3), prior_lims)
            var_gauss_pca = self.var_gauss_emu.do_pca(fit_var_gauss)
            self.var_gauss_emu.train(fit_theta, var_gauss_pca)
            
            self.Cl_g_emu = PCEEmulator(self.N_DIM, 8, prior_lims)       
            fit_Cl_g = fit_Cl_g.reshape((N_SAMPLES, -1))
            log_Cl_g  = np.log(fit_Cl_g.astype(np.float64) + 1e-50)
            Cl_g_pca_coeff = self.Cl_g_emu.do_pca(log_Cl_g)
            self.Cl_g_emu.train(fit_theta, Cl_g_pca_coeff, 8)
                
        