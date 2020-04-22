import pyqcs
from pyqcs import R, X, H
import numpy as np


def expIphiZ(act, phi):
    gates = R(act, -phi) | X(act) | R(act, phi) | X(act)
    return gates

def phase(act, phi):
    return (X(act) | R(act, phi) | X(act) | R(act, phi))    

def su2_to_circuit(act,param):
    phi0, phi1, phi2 = param
    th0 = phi0
    th1 = 1/2*(phi1 + phi2)
    th2 = 1/2*(phi1 - phi2)
    S = R(act, np.pi/2)
    Sd = R(act, -np.pi/2)
    gates = []
    gates.append(expIphiZ(act,th2))
    gates.append( Sd | H(act) | expIphiZ(act,th0) | H(act) | S )
    gates.append(expIphiZ(act,th1))
    
    return pyqcs.list_to_circuit(gates)


def u2_to_circuit(act, param):
    phi = param[0]

    gates = []
    gates.append(phase(act, phi/2))

    su2gates = su2_to_circuit(act, param[1:])

    gates.append(su2gates)

    return pyqcs.list_to_circuit(gates)
