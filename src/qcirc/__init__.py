import pyqcs
from pyqcs import R, X, H
import numpy as np


def expIphiZ(act, phi):
    gates = pyqcs.R(act, -phi)|pyqcs.X(act)|pyqcs.R(act, phi)|pyqcs.X(act)
    return gates
    

def su2_to_circuit(act,param):
    phi0, phi1, phi2 = param
    th0 = phi0
    th1 = 1/2*(phi1 + phi2)
    th2 = 1/2*(phi1 - phi2)
    S = pyqcs.R(act, np.pi/2)
    Sd = pyqcs.R(act, -np.pi/2)
    gates = []
    gates.append(expIphiZ(act,th2))
    gates.append(Sd|pyqcs.H(act)|expIphiZ(act,th0)|pyqcs.H(act)|S)
    gates.append(expIphiZ(act,th1))
    
    return pyqcs.list_to_circuit(gates)
