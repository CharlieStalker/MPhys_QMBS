# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 10:22:21 2022

@author: Charl
"""
import numpy as np
from numpy import linalg as LA 
from scipy.linalg import expm, sinm, cosm, logm, svd
from matplotlib import pyplot as plt
import scipy.sparse as sparse
from scipy.sparse.linalg import expm as expms
import qiskit

def genBasis(N):
    """
    Generates the set of half filled basis states for an N atom system

    Parameters
    ----------
    N : INT
        Dimensions

    Returns
    -------
    states : List
        List of half filled Basis States

    """
    fullbasisstates =[]
    halffilledstates = []
    for i in range(2**N):
        fullbasisstates.append(format(i,'0{n}b'.format(n=N)))
        x= format(i,'0{n}b'.format(n=N))
        total = sum(int(a) for a in x)
        if total == N/2:           
            halffilledstates.append(format(i,'0{n}b'.format(n=N)))
    return fullbasisstates, halffilledstates

def GenHamiltonian2(States, Two_qubit_hamiltonian, J_a, J_e):
    row=[]
    column=[]
    data=[]
    
    #hamiltonian = np.zeros((len(States),len(States)),dtype='complex')
    #hamiltonian2 = np.zeros((len(States),len(States)),dtype='complex')
    A_basis = ['00','01','10','11']
    for basis in States:
        a = States.index(basis)
        for i in range(len(basis)-1):
            dimer = basis[i:i+2]
            compbasis_of_dimer = A_basis.index(dimer)
            inputvector = np.zeros(len(A_basis))
            inputvector[compbasis_of_dimer] = 1
            if i%2 == 0:
                outputstate = J_a * inputvector@Two_qubit_hamiltonian
            elif i%2 ==1:
                outputstate = J_e * inputvector@Two_qubit_hamiltonian
                
            for index, element in enumerate(outputstate):
                if element != 0:
                    outputstatestring = basis[:i]+str(A_basis[index])+basis[i+2:]
                    outputstate_hamiltonian_index = States.index(outputstatestring)
                    #hamiltonian2[a,outputstate_hamiltonian_index] += element
                    row.append(a)
                    column.append(outputstate_hamiltonian_index)
                    data.append(element)
    hamiltonian = sparse.csc_matrix((data,(row,column)))
    return hamiltonian
            
        
        
def Fidelity(initial,current):    
    fidelity = np.vdot(initial,current)*np.conj(np.vdot(initial,current))
    return fidelity

def PsiT(hamiltonian,psi_0,no_times,start,end):
    psi_T=[psi_0]
    times = np.linspace(start, end, no_times)
    timestep = times[1]
    exp = expms(-1j*timestep*hamiltonian)
    psi_t=exp@psi_0
    for time in times:
        psi_t = exp@psi_t
        psi_T.append(psi_t)
    return psi_T

def PartialTracefull(DensMatrix,num_qubits,num_qubits_a):
    A_basis = genBasis(num_qubits_a)
    partialtrace = np.zeros((2**(num_qubits-num_qubits_a),2**(num_qubits-num_qubits_a)),dtype = complex)
    for i in range(len(A_basis)):
        tempmatrix = np.zeros(len(A_basis))
        tempmatrix[i] = 1
        psii_identity = np.kron(tempmatrix, np.identity(2**(num_qubits-num_qubits_a)))
        partialtrace += psii_identity@DensMatrix@psii_identity.T
    
    return partialtrace

def PauliZ_i(L,i):
    S_z = 1/2*np.array([[1, 0],[0, -1]])
    if i ==0:
        return np.kron(S_z, np.eye(2**(L-1)))
    else:
        temp_mat = np.kron( np.identity(2**(i)) , S_z)
        return np.kron(temp_mat,np.identity(2**(L-i-1)))
    
def imbalance(psi0, psit, N):
    imbalance=0
    for i in range(int(N)):
        pauliz_i = PauliZ_i(int(N), int(i))
        S0 = psi0@pauliz_i@psi0.T
        St = psit@pauliz_i@psit.T
        imbalance += St*S0
    return 1/N * imbalance         

def VonNeumannEntropy(rho):
    if np.any(rho):
        
        R=rho@(logm(rho)) #/logm(np.matrix([[2]])))
        S=-np.trace(R)
    else:
        S=0
    return S

def svdpartialtrace(S, U, N):
    rho = np.zeros((2**N,2**N),dtype=complex)
    for i in range(2**N):
        alpha = S[i]
        u_i = U[:2**N,i]
        rho += np.abs(alpha**2) * np.outer(u_i, u_i)
    return rho
        
def svdentropy(S):
    entropy=0
    for i in range(len(S)):
        alpha = S[i]
        entropy +=-1* abs(alpha)**2 * np.log(abs(alpha)**2)
    return entropy

def EnergyOfState(hamiltonian,psi):
    return (psi.T.conj())@hamiltonian@psi
        
def norm(state):
    return np.vdot(state,state)
                
#Hamiltonian = GenHamiltonian(Basis)
S_x = 1/2*np.array([[0,1],[1,0]])
S_y = 1/2*np.array([[0, -1j],[1j, 0]])
S_z = 1/2*np.array([[1, 0],[0, -1]])

XX_YY = (np.kron(S_x, S_x) + np.kron(S_y, S_y))

def StatetoFullBasis(statevector, fullbasis,partialbasis):
    state = np.zeros(len(fullbasis),dtype= complex)
    for i in range(len(statevector)):
        fullindex = fullbasis.index(partialbasis[i])
        state[fullindex]= statevector[i]
    return state

def qiskitpartialtrace(statevector, fullbasis,partialbasis,qubits_A):
    state = np.zeros(len(fullbasis),dtype= complex)
    for i in range(len(statevector)):
        fullindex = fullbasis.index(partialbasis[i])
        state[fullindex]= statevector[i]
    rho_A = qiskit.quantum_info.partial_trace(state,qubits_A)
    return rho_A 


if __name__ == '__main__':
    N=10
    fullBasis, Basis = genBasis(N)
    Hamiltonian = GenHamiltonian2(Basis, XX_YY,-5,-2)
    pi_index=Basis.index('1010101010')#('10'* int(N/2)))
    for i in [pi_index]:
        entropies = []
        entropy_qiskit = []
        psi_0 = np.zeros(len(Basis), dtype = complex)
        #psi_0[i] = 1
        for i in range(252):
            psi_0[i]= np.random.rand() + 1j*np.random.rand()
            
        normalisation = np.abs(np.sum(psi_0))
        psi_0=psi_0/normalisation
        fidelities=[]
        imbalancelist = []
        mag = []
        energy=[]
        Y = []
        S_T = []
        Timeevo = PsiT(Hamiltonian, psi_0, 100,0,10)
        psi_0full = StatetoFullBasis(psi_0, fullBasis, Basis)
        for Psi_T in Timeevo:
            fidelities.append(Fidelity(psi_0, Psi_T))
            #mattest = np.reshape(Psi_T, (4,231))
            #U,S,Vh = svd(mattest)
            #S_T.append(S[0])
            #rho = (svdpartialtrace(S,U,2))
            #rho = PartialTracefull(densMat,10,2)
            rho_qiskit =qiskitpartialtrace(Psi_T, fullBasis, Basis, [0,1,2,3])
            entropy_q = VonNeumannEntropy(rho_qiskit)
            #entropy=svdentropy(S)
            entropy_qiskit.append(entropy_q)
            #entropies.append(entropy/2)
            Psi_tfull = StatetoFullBasis(Psi_T, fullBasis, Basis)
            imbalance_t = imbalance(psi_0full, Psi_tfull, N)
            imbalancelist.append(imbalance_t)
            #mag.append(norm(Psi_T))
            #energy.append(EnergyOfState(Hamiltonian, Psi_T))

            
            
        plt.figure()
        plt.plot(fidelities, label = 'fidelity')
        plt.plot(entropies,label = 'entropy')
        plt.plot(entropy_qiskit, label = 'qiskit entropy')
        plt.plot(imbalance,label = 'imbalance')

        plt.legend()
        plt.grid(True)
        plt.title(str(Basis[i]))
        plt.show()

