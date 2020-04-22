import numpy as np

def partical_op(nbits=2):
    N = np.zeros((2**nbits, 2**nbits), dtype=np.double)
    for i in range(2**nbits):
        N[i,i]=i
        
    return N

def H0_(w,nbits=2):
    return w*(1/2*np.matrix(np.eye(2**nbits))+partical_op(nbits=nbits))

def a(nbits=2):
    A = np.matrix(np.zeros((2**nbits, 2**nbits)))
    for i in range(1,2**nbits):
        A[i-1,i] = np.sqrt(i)
    return A

def a_dag(nbits=2):
    return a(nbits=nbits).H


def x_op(m,w,nbits=2):
    return 1/np.sqrt(2*m*w)*(a(nbits=nbits) + a_dag(nbits=nbits))

def p_op(m,w,nbits=2):
    return 1j* np.sqrt(m*w/2) * (a_dag(nbits=nbits)-a(nbits=nbits))

def Hint_(a,nbits=2):
    return a*(x_op(1.0, 1.0, nbits=nbits))**4

def H_(w,a,nbits=2):
    return H0_(w,nbits=nbits) + Hint_(a,nbits=nbits)
