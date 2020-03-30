import numpy as np
from numpy import exp, sin, cos, sqrt
from scipy.linalg import expm, logm
import warnings
import matplotlib.pyplot as plt



Pauli = [
    np.matrix(np.eye(2),        dtype=np.complex),   # identity
    np.matrix([[0,  1],[1, 0]], dtype=np.complex),   # X
    np.matrix([[0,-1j],[1j,0]], dtype=np.complex),   # Y
    np.matrix([[1,  0],[0,-1]], dtype=np.complex)    # Z
]




# Useful function to get a quick overview over the entries of a large matrix

def color_matrix(M, bound_opt=True, cmap="Blues", low_=-2.0, high_=2.0):
    if isinstance(M,np.matrix):
        M = np.array(M)

    if len(M.shape)!=2:
        raise IndexError()

    if bound_opt:
        max_ = np.amax(M)
        if type(max_)==np.complex128 or type(max_)==np.complex64:
            max_ = max(max_.real, max_.imag)
        min_ = np.amin(M)
        if type(min_)==np.complex128 or type(min_)==np.complex64:
            min_ = min(min_.real, min_.imag)
        range_ = max_-min_
        delta = 0.1*range_

        low_ = min_ - delta
        high_ = max_ + delta
    
    if M.dtype == np.complex:
        fig, axs = plt.subplots(1,2,figsize=(10,3), constrained_layout=True)
        psm0 = axs[0].pcolormesh(M.real, cmap=cmap, rasterized=True, vmin=low_, vmax=high_)
        axs[0].set_title("real")
        axs[0].set_ylim(M.shape[1],0)
        psm1 = axs[1].pcolormesh(M.imag, rasterized=True, cmap=cmap, vmin=low_, vmax=high_)
        axs[1].set_title("imag")
        axs[1].set_ylim(M.shape[1],0)
        fig.colorbar(psm1, ax=axs[1])
        plt.show()
    else:
        fig, ax = plt.subplots(1,1,figsize=(5,3), constrained_layout=True)
        psm = ax.pcolormesh(M, cmap=cmap, rasterized=True,  vmin=low_, vmax=high_)
        ax.set_ylim(M.shape[1],0)
        fig.colorbar(psm, ax=ax)
        plt.show()




def is_hermitian(M):
    
    if (len(M.shape)!=2):
        raise LinAlgError("M not a matrix")
    
    if (M.shape[0] != M.shape[1]):
        raise LinAlgError("M has to be a square matrix")
        
    M = np.matrix(M)
    dist = np.linalg.norm(M-M.H)
    return np.isclose(dist, 0.0)


def is_unitary(M):

    if (len(M.shape)!=2):
        raise LinAlgError("M is not a matrix")

    M = np.matrix(M)

    dist = np.linalg.norm(M*M.H - np.matrix(np.eye(M.shape[0])))
    return np.isclose(dist, 0.0)



# The keyword pseudo indicates if one wants to check the positivity of the matrix


def is_density_matrix(rho, pseudo=True):
    # Check if Hermitean
    if not is_hermitian(rho):
        return False
    # Check if trace is 1
    if not np.isclose(np.trace(rho), 1):
        print(rho)
        print("trace: ", np.trace(rho))
        return False
    if not pseudo:
        #Check positivity
        W, v = np.linalg.eig(rho)
        for w in W:
            if not np.isclose(w, 0.0):
                if w.real < 0:
                    return False
    return True



#
#   Following are two diagonalization algorithm
#   
#   The first is for 2x2 matrices, which orders the eigenvalues from highest to lowest
#   The second is for nxn matrices, which doesn't order the eigenvalues
#  


def diag(rho):
    v, w = np.linalg.eigh(rho)
    
    if v[0] > v[1]:
        raise ValueError()
        
    S = np.matrix(np.diag(v))
    U = np.matrix(w, dtype=np.complex)
    
    if not np.isclose(np.linalg.norm(U.H@U - Pauli[0]), 0.0):
        raise ValueError()
    
    
    dist = np.linalg.norm(rho-U@S@U.H)
    if not np.isclose(dist, 0.0):
        raise ValueError()
        
    return S, U

def diagonalize_2(rho):
    if rho.shape[0] != 2:
        raise ValueError("wrong dimension")
    w, v = np.linalg.eig(rho)
    v = np.array(v)
    U = np.matrix([v[i,:]/np.linalg.norm(v[i,:]) for i in range(len(w))])
    if np.linalg.norm(U.H@U - np.matrix(np.eye(U.shape[0]))) > 1e-6:
        raise ValueError()
    S = np.matrix(np.diag(w))
    
    if S[0,0]<S[1,1]:
        X = np.matrix([[0,1],[1,0]])
        S = X@S@X
        U = U@X
    
    return S, U



def diagonalize(rho):
    #if rho.shape[0] == 2:
    #    return diagonalize_2(rho)

    D, eigenvecs = np.linalg.eig(rho)
    
    norms = [np.linalg.norm(v) for v in eigenvecs]
    U = [v / np.linalg.norm(v) for v in eigenvecs]
    U = np.matrix(np.array(U))
    
    S = np.matrix(np.diag(D))
    
    
    if not np.isclose(np.linalg.norm(rho - U@S@U.H), 0.0):
        raise ValueError("something went wrong, rho not diagonizable")
        
    return S, U

#
#   The following function are for generate SU(2) and U(2) matrices
#
#

def su2(param, generator=False):

    if not isinstance(param, list):
        if len(param.shape)!=1:
            raise ValueError("array not one dimensional!")
    if (len(param)!=3):
        raise ValueError("wrong number of SU(2) angles")


    if generator:
        #
        # param[0] = a_x
        # param[1] = a_y
        # param[2] = a_z
        #
        #
        # U = exp(-i \sum_{i\in\{x,y,z\}} a_i \sigma_i)
        #

        H = sum([param[i]*Pauli[i+1] for i in range(3)])
        return expm(-1j*H)

    else:

        #
        #   param[0] = \theta
        #   param[1] = \phi_1
        #   param[2] = \phi_2
        #
        #      The SU(2) matrix is
        #     
        #           / e^(i \phi_1) \cos(\theta)          e^(i \phi_2) \sin(\theta)   \
        #       U = |                                                                |
        #           \ -e^(-i\phi_2) \sin(\theta)         e^(-i \phi_1) \cos(\theta)  /
        
        theta, phi1, phi2 = param
        
        return np.matrix([[exp(1j*phi1)*cos(theta), exp(1j*phi2)*sin(theta)]
                        ,[-exp(-1j*phi2)*sin(theta), exp(-1j*phi1)*cos(theta)]])

def get_su2_param(U, generator=True):
    detU = np.linalg.det(U)
    if not np.isclose(detU.real, 1, rtol=1e-3):
        raise ValueError()
    if generator:
        ###################################
        # U = exp(-i \sum_k a_k \sigma_k) #
        ###################################
        
        LogU = 1j * logm(U)
        a0 = LogU[0,1].real
        a1 = -LogU[0,1].imag
        a2 = LogU[0,0].real
        param = np.array([a0, a1, a2])
    else:
        #######################################################
        #      / e^{i t1} cos(t0)      e^{i t2} sin(t0)  \    #
        # U = |                                           |   #
        #      \  -e^{-i t2} sin(t0)   e^{-i t1} cos(t0) /    #
        #######################################################
        
        a = np.abs(U[0,0])
        t1 = np.angle(U[0,0])
        t2 = np.angle(U[0,1])
        t0 = np.arccos(a)
        
        param = np.arra

    return param


def u2(param):
    #
    #    param[0] = global angle of U(2) \phi
    #    param[1:] = SU(2) angle
    #
    if not isinstance(param, list):
        if len(param.shape)!=1:
            raise ValueError("array not one dimensional!")

    if (len(param)!=4):
        raise ValueError("wrong number of U(2) angles")
        
    phi = param[0]
    su2_matrix = su2(param[1:])
    return exp(1j*phi/2)*su2_matrix


def pseudo_random_density_matrix(n=2):
    #
    # This function generates a random hermitian matrix with trace=1,
    # but it ignors the semi-positivity of the matrix
    #

    rho = np.zeros((n, n), dtype=np.complex)
    residue = 1.0
    for i in range(n-1):
        val = np.random.rand()*residue
        rho[i,i] = val
        residue -= val
        
    rho[-1, -1] = residue
        
    
    for i in range(n):
        for j in range(i+1,n):
            val = np.random.rand() + 1j*np.random.rand()
            rho[i,j] = val
            rho[j,i] = np.conjugate(val)
            
    return np.matrix(rho)



def random_density_matrix(n=2, warn=True):
    if n != 2:
        if warn:
            warnings.warn("In general, the output is not a positive semi-define matrix for n!=2") 
        return pseudo_random_density_matrix(n=n)

    a = np.random.rand()*0.5 + 0.5
    D = np.matrix([[a,0],[0,1-a]])

    U = su2(np.random.rand(3))

    return U@D@U.H


def test_checks(M):
    print(is_density_matrix(M, pseudo=False), is_density_matrix(M, pseudo=True), is_hermitian(M), is_unitary(M))

if __name__ == "__main__":
    rho0 = random_density_matrix()
    color_matrix(rho0)

    test_checks(rho0)

    S, U = diagonalize(rho0)
    color_matrix(S)
    color_matrix(U)


    rho1 = random_density_matrix(n=4)
    color_matrix(rho1)

    test_checks(rho1)

    S, U = diagonalize(rho1)
    color_matrix(S)
    color_matrix(U)

    U0 = u2([1.2, 4.1, 5.2, 1.9])
    color_matrix(U0)

    test_checks(U0)

    U1 = su2([1.2, 3.1, 2.1])
    color_matrix(U1)
    get_su2_param(U1)

    U2 = su2([1.2, 1.2, 1.5], generator=True)
    color_matrix(U2)

    test_checks(U1)

