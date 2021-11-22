import numpy as np
from scipy.linalg import sqrtm

from . import mcmc_helper as helper
from ..utils import get_mass_eigs, get_mass_inv

__all__ = ["sampler"]

class HMCSampler(object):
    """
    A class for doing Hamiltonian Monte-Carlo
    :param ndim: Dimension of the parameter space being sampled
    :param psi: The negative of the log-posterior, a.k.a Potential
    :param grad_psi: The gradient of the Potential
    :param Hamiltonian_mass: array of masses for the hamil
    :param psi_args: (optional) extra positional arguments for the function psi
    :param grad_psi_args: (optional) extra positional arguments for the function grad_psi
    """

    def __init__(self, ndim, psi, grad_psi, mass_matrix, N, N_Z_BINS, verbose=False,
                 psi_args=[], grad_psi_args=[]):
        self.ndim             = ndim
        self.psi              = psi
        self.grad_psi         = grad_psi

        self.N                = N
        self.N_Y              = N//2 + 1
        self.N_Z_BINS         = N_Z_BINS
        
        self.mass_matrix   = mass_matrix
        
        eig_vals, eig_vecs = get_mass_eigs(mass_matrix, self.N)

        self.eig_vals      = np.real(eig_vals)
        self.eig_vecs      = np.real(eig_vecs)
        
        self.M_inv         = get_mass_inv(mass_matrix, self.N)
                
        self.psi_args         = psi_args
        self.grad_psi_args    = grad_psi_args
        self.psi_kwargs       = None
        self.grad_psi_kwargs  = None
        self.verbose          = verbose
            
    def rotate_p(self, p):
        p_prime = np.zeros(p.shape)
        for i in range(self.N_Z_BINS):
            for j in range(self.N_Z_BINS):
                p_prime[i] += self.eig_vecs[i,j,:]*p[j]
        return p_prime
    
    def sample_one_step(self, x_old, time_step, N_LEAPFROG, psi_kwargs=None, grad_psi_kwargs=None):
        """
        A function to run one step of HMC
        """
        if(psi_kwargs is not None):
            self.psi_kwargs=psi_kwargs
        if(grad_psi_kwargs is not None):
            self.grad_psi_kwargs=grad_psi_kwargs

        p_prime = np.random.normal(0., np.sqrt(self.eig_vals))
        
        p = self.rotate_p(p_prime)
        
        H_old, _, _ = self.Hamiltonian(x_old, p)

        dt = np.random.uniform(0, time_step)
        N = np.random.randint(1,N_LEAPFROG+1)

        if(self.verbose):
            print("dt: %2.3f, N:%d"%(dt, N))
        
        x_proposed, p_proposed = self.leapfrog(x_old, p, dt, N)

        H_proposed, KE, _ = self.Hamiltonian(x_proposed, p_proposed)

        diff = H_proposed-H_old

        if(self.verbose):
            print("diff: %2.3f"%(diff))
        
        accepted = False
        if(diff < 0.0):
            x_old = x_proposed
            accepted = True
        else:
            rand = np.random.uniform(0.0, 1.0)
            log_rand = np.log(rand)

            if(-log_rand > diff):
                x_old = x_proposed
                accepted = True

        return x_old, -self.get_psi(x_old), accepted, KE
    
    def leapfrog(self, x, p, dt, N):
        """
        Returns the new position and momenta evolved from the position {x,p} in phase space, with `N_LEAPFROG` leapfrog iterations.
        :param x: The current position in the phase space
        :param p: The current momenta
        :param dt: The time step to which it is evolved
        :param N_LEAPFROG: Number of leapfrog iterations
        """
        x_old       = x
        p_old       = p
        
        for i in range(N):
            psi_grad = self.get_grad_psi(x_old)        
                    
            p_new = p_old-(dt/2)*psi_grad
            x_new = x_old.copy()
            ##=============================================
            x_new = x_new + dt * self.matrix_multiply(self.M_inv, p_new)
            ##=============================================
            psi_grad = self.get_grad_psi(x_new)
            p_new = p_new-(dt/2)*psi_grad
            
            x_old, p_old = x_new, p_new

        return x_new, p_new

    def Hamiltonian(self, x, p):
        """
        Returns the hamiltonian for a given position and momenta
        :param x: The position in the parameter space
        :param p: The set of momenta
        """        
        KE = 0.5 * np.sum(p * self.matrix_multiply(self.M_inv, p))
        
        PE = self.get_psi(x)
        if(self.verbose):
            print("KE: %2.4f"%(KE))
            print("PE: %2.4f"%(PE))
        H = KE + PE 
        return H, KE, PE
    
    def matrix_multiply(self, A, x):
        N_b = A.shape[0]
        assert A.shape[1]==N_b,"Unexpected shape for A"
        assert x.shape[0]==N_b,"Unexpected shape for x"
        y = np.zeros(x.shape)
        for i in range(N_b):
            for j in range(N_b):
                y[i] += A[i,j]*x[j]
        return y
    
    def get_psi(self, x):
        """Return psi at the given position."""
        if(self.psi_kwargs is not None):
            return self.psi(x, *self.psi_args, **self.psi_kwargs)
        return self.psi(x, *self.psi_args)

    def get_grad_psi(self, x):
        """Return grad_psi at the given position."""
        if(self.grad_psi_kwargs is not None):
            return self.grad_psi(x, *self.grad_psi_args, **self.grad_psi_kwargs)
        return self.grad_psi(x, *self.grad_psi_args)