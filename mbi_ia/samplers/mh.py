import numpy as np
from . import mcmc_helper as helper

__all__ = ["sampler"]

class MHSampler(object):
    def __init__(self, lnprob, lnprob_args=[], verbose=False):
        self.lnprob        = lnprob
        self.lnprob_args   = lnprob_args
        self.lnprob_kwargs = None
        self.verbose       = verbose
        self.mu            = None
    
    def sample_one_step(self, x_old, lnprob_kwargs=None):
        if(self.mu is None):
            self.mu = x_old
        if(lnprob_kwargs is not None):
            self.lnprob_kwargs=lnprob_kwargs
        lnprob_old = self.get_lnprob(x_old)

        x_proposed = np.random.multivariate_normal(mean=x_old, cov=self.cov)
        
        lnprob_prop = self.get_lnprob(x_proposed)
        
        diff = lnprob_prop - lnprob_old
        if(self.verbose):
            print("diff: %2.3f"%(diff))
        
        lnprob0 = lnprob_old
        acc = True
        if(diff > 0.):
            x_new = x_proposed
            lnprob0 = lnprob_prop
        else:
            u = np.random.uniform(0.,1.)
            if(np.log(u) < diff):
                x_new = x_proposed
                lnprob0 = lnprob_prop
            else:
                acc = False
                x_new = x_old
        self.current_lnprob = lnprob0
        return x_new, acc, lnprob0
    
    def set_cov(self, step_cov):
        self.cov = step_cov
    
    def update_cov(self, x, n):
        self.mu, self.cov = helper.update_step_size(n, x, self.mu, self.cov)
        self.eig_vals, self.eig_vecs = np.linalg.eig(self.cov)
        self.step_size = np.sqrt(self.eig_vals)
        print("self.mu: "+str(self.mu))
        print("self.cov: "+str(self.cov))
        
    def get_lnprob(self, x):
        """Return lnprob at the given position."""
        if(self.lnprob_kwargs is not None):
            return self.lnprob(x, *self.lnprob_args, **self.lnprob_kwargs)
        return self.lnprob(x, *self.lnprob_args)