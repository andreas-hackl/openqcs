import openqcs.tools as tools
import numpy as np
from numpy import sqrt, sin, cos 
from scipy.linalg import logm, expm


# Look up table for variety of parameter of diagonal pseudo Kraus operator

dKraus = [
    [
        np.matrix([[0, -1j], [1j, 0]]),
        np.matrix([[1, 0], [0, 1]])
    ]


]

# Output message
class Message:
    def __init__(self):
        self.ier = False
        self.err = 0.0
        self.variant = 0 # variant of param for diagonal pseudo Kraus operator
        


def get_kraus_(U, init_env=0, sys_bits=[0], env_bits=[], nbits=2):
    total_bits=np.array(range(nbits))
    if env_bits==[]:
        env_bits=[bit for bit in total_bits if bit not in sys_bits]
    n_env_bits=len(env_bits)
    n_sys_bits=len(sys_bits)
    n_env_st=2**(n_env_bits)
    n_sys_st=2**(n_sys_bits)

    Ks = []
    for i in range(n_env_st):
        K = U[init_env*n_sys_st:(init_env+1)*n_sys_st, i*n_sys_st:(i+1)*n_sys_st]
        Ks.append(K)

    return Ks

def channel_(rho0, Ks):
    ch = np.matrix(np.zeros_like(rho0), dtype=np.complex)
    for K in Ks:
        ch+=K*rho0*K.H

    return ch



def get_iterative_kraus_op(rho_in, rho_out, variant=0):
    # check if rho_in and rho_out are 2x2 density matrices
    if rho_in.shape[0]!=2 or rho_out.shape[0]!=2:
        raise ValueError("wrong dimension")
        
    if not tools.is_density_matrix(rho_in, pseudo=False) or not tools.is_density_matrix(rho_out, pseudo=False):
        raise ValueError("rho_in and rho_out have to be density matrices")
        
    # Define return values
    msg = Message()
    
        
    D0, U0 = tools.diagonalize(rho_in)
    D1, U1 = tools.diagonalize(rho_out)
    
    D0 = D0.real
    D1 = D1.real
    
    s0 = D0[0,0]
    s1 = D1[0,0]
    
    if 1-s0 <= s1 <= s0:
        msg.ier=True
        p = (s0 - s1)/(2*s0 - 1)
        
        S0 = dKraus[variant][0]
        S1 = dKraus[variant][1]
        
        E0 = U1@S0@U0.H
        E1 = U1@S1@U0.H
        
        rho_out_proj = p*E0@rho_in@E0.H + (1-p)*E1@rho_in@E1.H
        msg.err = np.linalg.norm(rho_out_proj - rho_out)
        
        return p, E0, E1, msg
    
    else:
        p = 0
        E0 = np.matrix(np.zeros((2,2)))
        E1 = np.matrix(np.zeros((2,2)))
        return p, E0, E1, msg