import numpy as np
import qutip

# generate test matrix (using qutip for convenience)
dm = qutip.rand_dm_hs(8, dims=[[2, 4]] * 2).full()
# reshape to do the partial trace easily using np.einsum
reshaped_dm = dm.reshape([2, 4, 2, 4])
# partial trace the second space
reduced_dm = np.einsum('jiki->jk', reshaped_dm)
# check results with qutip
qutip_dm = qutip.Qobj(dm, dims=[[2, 4]] * 2)
reduced_dm_via_qutip = qutip_dm.ptrace([0]).full()
# check consistency of results
np.allclose(reduced_dm, reduced_dm_via_qutip)

sy = qutip.sigmay()
sx = qutip.sigmax()
H =1/2

