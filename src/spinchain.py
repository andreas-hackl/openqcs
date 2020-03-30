import openqcs.tools as tools
import numpy as np 
from scipy.linalg import expm

def multi_gate_(pidx, bit=0, nbits=2):   
    # pidx = Pauli index
    # 0 -> identity
    # 1 -> X
    # 2 -> Y
    # 3 -> Z
    
    if bit>=nbits:
        raise ValueError("bit not assigned")
    
    tmp = 1
    for i in reversed(range(nbits)):
        if i == bit:
            tmp=np.kron(tmp, tools.Pauli[pidx])
        else:
            tmp=np.kron(tmp, tools.Pauli[0])
    return tmp


def H_(a, b, N=2, boundary="o"):
    valid_bound_expr=["o", "p"]
    if not boundary in valid_bound_expr:
        raise ValueError("cannot interpret the boundary condition\n valid expressions are:\n\"o\" for open \n\"p\" for periodic")
    H = np.matrix(np.zeros((2**N, 2**N), dtype=np.complex))
    
    # Z part
    for i in range(N-1):
        H-=a*multi_gate_(3,bit=i,nbits=N)@multi_gate_(3,bit=i+1,nbits=N)
    
    if boundary=="p":
        H-=a*multi_gate_(3,bit=N-1,nbits=N)@multi_gate_(3,bit=0,nbits=N)
        
    # X part
    for i in range(N):
        H+=b*multi_gate_(1,bit=i,nbits=N)
        
    H = H.real
    return H

def T_(H, dt=0.1):
    T = expm(-1j*H*dt)
    return np.matrix(T)
