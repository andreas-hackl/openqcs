import numpy as np
import openqcs as o
import openqcs.iterativ as it
import openqcs.spinchain as sp
import openqcs.tools as t

import pyqcs
import openqcs.qcirc as qu
import openqcs.statistic as st
import time



def get_data(a, b, dt, plotting=False):
    H = sp.H_(a,b)
    T = sp.T_(H,dt)
    
    data, err, n_t = it.iterative_evolution(T)
    
    if plotting:
        it.plot_param(data, err, n_t, dt)
    
    return data, n_t

def QuantumEvolution(data, n_sample, iSt=pyqcs.State.new_zero_state(1), n_measure=1000, log=False):

    probs = data[:,0]

    circs = np.ndarray((probs.shape[0],2), dtype=pyqcs.AnonymousCompoundGateCircuit)

    for i, d in enumerate(data):
        c1 = qu.su2_to_circuit(0, d[1:4])
        c2 = qu.su2_to_circuit(0, d[4:])
        circs[i,:] = [c1, c2]
        
        
    output = np.zeros((n_sample, 2), dtype=float)
    
    paths = []
    
    for i in range(n_sample):
        
        evolute_gates = np.ndarray(probs.shape[0], dtype=pyqcs.AnonymousCompoundGateCircuit)
        path=""
        for k, p in enumerate(probs):
            
            q = np.random.rand()
            if q < p:
                evolute_gates[k] = circs[k,0]
                path+="0"
            else:
                evolute_gates[k] = circs[k,1]
                path+="1"
        
        U_gate = pyqcs.list_to_circuit(evolute_gates)
        
        psi = U_gate*iSt
        
        res = pyqcs.sample(psi, 1, n_measure)
        paths.append(path)
        
        for key, val in res.items():
            output[i,key] = float(val)/n_measure
    
    paths = np.array(paths)
    
    means = np.zeros(10, dtype=np.double)
    for i in range(10):
        tmp = np.random.choice(output[:,0], size=data.shape[0], replace=True)
        means[i] = np.mean(tmp)
    mean0, err0 = st.bootstrap(output[:,0])
    mean1, err1 = st.bootstrap(output[:,1])
    if log:
        return mean0, err0, mean1, err1, paths
    else:
        return mean0, err0, mean1, err1

def ClassicalEvolution(idx, T, rho0):
    
    rho = o.rho_sys_(T**idx, rho0)
    
    return rho[0,0], rho[1,1]
        
    
def main():
    a = 0.5
    b = 0.5
    dt = 0.05

    n_samples = np.logspace(2, 3, 5, dtype=int)


    data, n_t = get_data(a,b,dt)


    H = sp.H_(a,b)
    T = sp.T_(H,dt)

    iSt = pyqcs.State.new_zero_state(1)
    rho0 = np.matrix([[1,0],[0,0]])

    
    f = open("qc_res.txt", "w")
    g = open("times.txt", "w")

    for i, n in enumerate(n_samples):
        print("n_sample = {}".format(n), flush=True)
        t_begin = time.time()
        m0, e0, m1, e1, paths = QuantumEvolution(data, n, iSt=iSt, log=True)
        f.write("{}\t{}\t{}\t{}\t{}\n".format(n, m0, e0, m1, e1))

        t_end = time.time()
        g.write("{}\t{}\n".format(n,t_end-t_begin))

        print("exicution time: ", t_end - t_begin)


    

    r0, r1 = ClassicalEvolution(i, T, rho0)
    print(r0, r1)    

    

main()
        
