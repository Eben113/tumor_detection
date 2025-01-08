import numpy as np
import scipy.linalg as lin
class FDM():
    def __init__(self, Nx, Ny):
        self.Nx = Nx
        self.Ny = Ny

    def build_lhs(self):
        mainMatrix_diag = np.diag([4]*(self.Nx-1)) + np.diag([-1]*(self.Nx-2), 1) + np.diag([-1]*(self.Nx-2), -1)
        lhs = lin.block_diag(*[mainMatrix_diag]*3)

        lhs += np.diag([-1]*6, 3) + np.diag([-1]*6, -3)
        return lhs

    def build_rhs(self):
        a = [75,0,50]
        rhs = np.array([75,0,50]*3)
        rhs += np.array([0]*6 + [100]*3)
        return rhs

    def solve(self):
        return np.linalg.solve(self.build_lhs(), self.build_rhs())
fdm.solve()