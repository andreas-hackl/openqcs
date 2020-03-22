import numpy as np 


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

def split_number(k, sysbit, nbits):
    s = -sysbit-1
    binary = "{0:b}".format(k)
    while len(binary) < nbits:
        binary = "0"+binary
        
    env_binary_arr = [binary[i] for i in range(len(binary)) if i!=len(binary)+s]
    sys_value = int(binary[len(binary)+s])

    env_binary = ""
    for v in env_binary_arr:
        env_binary+=v
    
    env_value = int(env_binary,2)
    
    return sys_value, env_value


def partial_trace(M, sysbit=0):
    if M.shape[0]!=M.shape[1]:
        raise ValueError()
    nbits = int(np.log2(M.shape[0]))
    print("nbits = ", nbits)
    
    
    envbits = [i for i in range(nbits) if i!=sysbit]
    
    
    X = np.matrix(np.zeros((2,2)), dtype=np.complex)
    
    partial_matrices = [np.zeros((2,2), dtype=np.complex) for i in range(2**(nbits-1))]
    for k in range(2**nbits):
        for l in range(2**nbits):
            k_sys, k_env = split_number(k, sysbit, nbits)
            l_sys, l_env = split_number(l, sysbit, nbits)
            
            if k_env == l_env:
                partial_matrices[k_env][k_sys, l_sys] = M[k,l]
            
    for tmp in partial_matrices:
        X += np.matrix(tmp)
            
    
    return X