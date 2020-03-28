import openqcs.tools as tools
import numpy as np
from numpy import sqrt, sin, cos 
from scipy.linalg import logm, expm


# Look up table for variety of parameter of diagonal pseudo Kraus operator

dKraus = [
    [
        np.matrix([[0, -1], [1, 0]]),
        np.matrix([[1, 0], [0, 1]])
    ],
    [
        np.matrix([[0,1j],[1j,0]]),
        np.matrix([[1j,0],[0,-1j]])
    ]


]

# Output message
class Message:
    def __init__(self):
        self.ier = False
        self.err = 0.0
        self.variant = 0 # variant of param for diagonal pseudo Kraus operator


def get_iterative_kraus_op(rho_in, rho_out, variant=0):
    # check if rho_in and rho_out are 2x2 density matrices
    if rho_in.shape[0]!=2 or rho_out.shape[0]!=2:
        raise ValueError("wrong dimension")
        
    if not tools.is_density_matrix(rho_in, pseudo=False) or not tools.is_density_matrix(rho_out, pseudo=False):
        raise ValueError("rho_in and rho_out have to be density matrices")
        
    # Define return values
    msg = Message()
    

    phase = False

    D0, U0 = tools.diagonalize(rho_in)
    D1, U1 = tools.diagonalize(rho_out)
    
    D0 = D0.real
    D1 = D1.real

    if D0[0,0] > D0[1,1] and D1[0,0] < D1[1,1]:
        phase = True
    
    s0 = D0[0,0]
    s1 = D1[0,0]
    
    if 1-s0 <= s1 <= s0:
        msg.ier=True

        
        
        p = (s0 - s1)/(2*s0 - 1)
        
        
        S0 = dKraus[variant][0]
        S1 = dKraus[variant][1]

        if phase==True:
            p = 1-p
            S0 = np.matrix([[1j, 0], [0, -1j]])
            S1 = np.matrix([[0, -1],[1, 0]], dtype=np.complex)


        
        E0 = U1@S0@U0.H
        E1 = U1@S1@U0.H
        
        rho_out_proj = p*E0@rho_in@E0.H + (1-p)*E1@rho_in@E1.H
        msg.err = np.linalg.norm(rho_out_proj - rho_out)
        
        return p, E0, E1, msg

    elif s0 <= s1 <= 1-s0:
        msg.ier=True
        p = (s0-s1)/(2*s0-1)


        S0 = dKraus[variant][0]
        S1 = dKraus[variant][1]

        E0 = U1@S0@U0.H
        E1 = U1@S1@U0.H
        
        rho_out_proj = (1-p)*E1@rho_in@E1.H + p*E0@rho_in@E0.H
        msg.err = np.linalg.norm(rho_out_proj - rho_out)
        
        return p, E0, E1, msg
    
    else:
        p = 0
        E0 = np.matrix(np.zeros((2,2)))
        E1 = np.matrix(np.zeros((2,2)))
        return p, E0, E1, msg