# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 17:35:16 2023

@author: Charlie
"""

import numpy as np
from qutip import *
from matplotlib import pyplot as plt 

XX_YY = 0.5*0.5*(tensor(sigmax(),sigmax())+tensor(sigmay(),sigmay()))

times = np.linspace(0.0, 5, 1000)

Num_qubits = 4
H= tensor(XX_YY,identity(2),identity(2))+tensor(identity(2), XX_YY, identity(2))\
+tensor(identity(2),identity(2),XX_YY)


psi_1001 = tensor(basis(2,1),basis(2,0),basis(2,0),basis(2,1))
results = sesolve(H, psi_1001, times)


fidelities = []
entropy = []
for i in range(1000):
    fidelities.append(fidelity(results.states[0], results.states[i]))
    entropy.append(entropy_vn(results.states[i].ptrace([0,1])))
    

    
    
fig, ax = plt.subplots()
plt.title("1001 qutip")
ax.plot(results.times, fidelities,label='fidelity')
ax.plot(results.times, entropy, label='entropy')
plt.legend()