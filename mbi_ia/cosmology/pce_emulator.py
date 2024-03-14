from pyDOE import lhs
import chaospy
import numpy as np
from sklearn.decomposition import PCA

def map2_unitinterval(samples, prior_lims):
    prior_upper = prior_lims[:,1]
    prior_lower = prior_lims[:,0]
    delta_prior = prior_upper - prior_lower
    return 2. * (samples - prior_lower) / delta_prior - 1.

class PCEEmulator:
    def __init__(self, N_DIM, N_PCA, prior_lims):
        self.N_DIM = N_DIM
        self.N_PCA = N_PCA
        self.prior_lims = prior_lims

    def get_lhs_samples(self, N_SAMPLES):
        lh_samples = lhs(self.N_DIM, N_SAMPLES)
        prior_upper = self.prior_lims[:,1]
        prior_lower = self.prior_lims[:,0]
        delta_prior = prior_upper - prior_lower
        return prior_lower + delta_prior * lh_samples
    
    def do_pca(self, data_vector):
        """
        Do PCA decomposition of the given set of data vectors. Returns the PCA coefficients.
        
        :param data_vector: Array of shape (N_SIMS, N_DATA_VECTOR)
        :returns pca_coeff: Array of shape (N_SIMS, N_PCA)
        """
        pca = PCA(self.N_PCA)
        pca.fit(data_vector)
        self.pca = pca
        pca_coeff = pca.transform(data_vector)
        return pca_coeff
    
    def do_inverse_pca(self, pca_coeff):
        """
        Do inverse pca transform for a given set of PCA coefficients.
        """
        return self.pca.inverse_transform(pca_coeff)
    
    def train(self, fit_theta, pca_coeff, N_ORDER=6):
        fit_x = map2_unitinterval(fit_theta, self.prior_lims)
        self.pca_mean = np.mean(pca_coeff, axis=0)
        self.pca_std  = np.std(pca_coeff, axis=0)
        pca_coeff_norm = (pca_coeff - self.pca_mean) / self.pca_std
        
        distribution = chaospy.J(chaospy.Uniform(-1., 1.), chaospy.Uniform(-1., 1.))
        for i in range(self.N_DIM - 2):
            distribution = chaospy.J(distribution, chaospy.Uniform(-1., 1.))        
        expansion = chaospy.generate_expansion(N_ORDER, distribution)
        solver_list = []
        for i in range(self.N_PCA):
            solver = chaospy.fit_regression(expansion, fit_x.T, pca_coeff_norm[:,i])
            solver_list.append(solver)
            del solver
        self.solver_list = solver_list
        self.trained = True
        
    def predict(self, theta_pred):
        assert self.trained, "The emulator needs to be trained first before predicting"
        
        pca_pred_list = []
        pred_x = map2_unitinterval(theta_pred, self.prior_lims)
        
        for i, solver in enumerate(self.solver_list):
            pca_pred_i = chaospy.call(solver, pred_x.T)
            pca_pred_i = self.pca_std[i] * pca_pred_i + self.pca_mean[i]
            pca_pred_list.append(pca_pred_i)
    
        return np.array(pca_pred_list).T
         