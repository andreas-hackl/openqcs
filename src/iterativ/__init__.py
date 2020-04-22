import openqcs.tools as tools
import openqcs as op
from openqcs.tools import get_su2_param
import numpy as np
from numpy import sqrt, sin, cos 
from scipy.linalg import logm, expm

import matplotlib.pyplot as plt 


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

Pauli = [
    np.matrix(np.eye(2),        dtype=np.complex),   # identity
    np.matrix([[0,  1],[1, 0]], dtype=np.complex),   # X
    np.matrix([[0,-1j],[1j,0]], dtype=np.complex),   # Y
    np.matrix([[1,  0],[0,-1]], dtype=np.complex)    # Z
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


def it_kraus(rho_in, rho_out, phase_check=True):
    msg = {"ier": False, "err": 0.0, "phase":False}
    
    D0, U0 = tools.diag(rho_in)
    D1, U1 = tools.diag(rho_out)
    
    if phase_check:
        if rho_in[0,0]>1/2 and rho_out[0,0]<1/2:
            msg["phase"]=True
            msg["ier"]=True
            p = 1
            E0 = np.matrix(np.eye(2))
            E1 = np.matrix(np.eye(2))
            return p, E0, E1, msg
    
    s0 = D0[1,1]
    s1 = D1[1,1]
    
    if 1-s0 <= s1 <= s0:
        msg["ier"]=True
        
        p = (s0 - s1)/(2*s0 - 1)
        
        S0 = 1j*Pauli[2]
        S1 = Pauli[0]
        
        E0 = U1@S0@U0.H
        E1 = U1@S1@U0.H
        
        uni0 = np.linalg.norm(E0.H@E0 - Pauli[0])
        uni1 = np.linalg.norm(E1.H@E1 - Pauli[0])
        
        if not np.isclose(uni0, 0.0):
            raise ValueError()
            
        if not np.isclose(uni1, 0.0):
            raise ValueError()
        
        rho_out_proj = p*E0@rho_in@E0.H + (1-p)*E1@rho_in@E1.H
        
        msg["err"] = np.linalg.norm(rho_out_proj - rho_out)
        
        return p, E0, E1, msg
    
    elif s0 <= s1 <= 1-s0:
        msg["ier"]=True
        
        p = (s0 - s1)/(2*s0 - 1)
        
        S0 = 1j*Pauli[2]
        S1 = Pauli[0]
        
        E0 = U1@S0@U0.H
        E1 = U1@S1@U0.H
        
        uni0 = np.linalg.norm(E0.H@E0 - Pauli[0])
        uni1 = np.linalg.norm(E1.H@E1 - Pauli[0])
        
        if not np.isclose(uni0, 0.0):
            raise ValueError()
            
        if not np.isclose(uni1, 0.0):
            raise ValueError()
        
        rho_out_proj = p*E0@rho_in@E0.H + (1-p)*E1@rho_in@E1.H
        
        msg["err"] = np.linalg.norm(rho_out_proj - rho_out)
        
        return p, E0, E1, msg
    
    else:
        p = 0
        E0 = np.matrix(np.eye(2))
        E1 = np.matrix(np.eye(2))
        return p, E0, E1, msg


def iterative_evolution(T, sysbit=0):
    nbits = np.log2(T.shape[0])
    if nbits!=2:
        raise ValueError()
        

    valid = True
    n_t = 1
    n_kraus = 2**(nbits-1)
    
    data = []
    err = []
    
    rho_0 = np.matrix([[1,0],[0,0]], dtype=np.complex)
    rho_in = rho_0
    
    repair = [] 
    # indices, at which one need to do interpolation, e.g.
    # at the critical time t_crit where rho(t_crit)_{1,1} = 1/2
    
    
    while valid:
        print("#", end="", flush=True)
        Ks = op.get_kraus_(T**n_t)
        rho_out = op.channel_(rho_0, Ks)
        
        
        p, S0, S1, msg = it_kraus(rho_in, rho_out)
        
        
    
        valid = msg["ier"]
        
        if msg["phase"]:
            repair.append(n_t-1)
        
        
        if valid:
            try:
                p0 = get_su2_param(S0)
                p1 = get_su2_param(S1)
                data.append([p] + list(p0) + list(p1))
                err.append(msg["err"])
            except ValueError:

                #print(np.linalg.det(S0), np.linalg.det(S1))


                repair.append(n_t-1)
                data.append([0]*7)
                err.append(msg["err"])
            n_t += 1
            
            
        rho_in = rho_out
        
    print("\nvalid time steps: ", n_t-1)
    data = np.array(data)
    err = np.array(err)
    print(repair)
    # repairing
    for idx in repair:
        if idx != 0:
            for j in range(7):
                data[idx,j] = 1/2 * (data[idx-1,j] + data[idx+1,j])
        else:
            for j in range(7):
                data[idx,j] = data[idx+1,j]
    
    return data, err, n_t-1

def plot_param(data, err, n_t, dt):
    tdata = np.arange(n_t)*dt

    labeling = [r"$\phi_0$", r"$\phi_1$", r"$\phi_2$"]
    color = ["tab:blue", "tab:red", "tab:green"]

    fig, axes = plt.subplots(2,2, figsize=(20,20))

    axes[0,0].plot(tdata, data[:,0], color="black")
    axes[0,0].set_xlabel("t")
    axes[0,0].set_ylabel("p")

    axes[0,1].plot(tdata, err, color="black")
    axes[0,1].set_ylabel("error")
    axes[0,1].set_xlabel("t")

    for i in range(3):
        axes[1,0].plot(tdata, data[:,1+i], color=color[i], label=labeling[i])
        axes[1,1].plot(tdata, data[:,4+i], color=color[i], label=labeling[i])

    for l in range(2):
        axes[1,l].set_ylabel("U(2) parameter of Matrix {}".format(l))
        axes[1,l].set_xlabel("t")
        axes[1,l].legend()

    plt.show()
