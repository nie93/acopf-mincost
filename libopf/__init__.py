from pdb import *
from .solvers import ScipyOptimizer
from .solvers import IpoptOptimizer

def runcopf(c, flat_start):

    opfsolver = IpoptOptimizer(c)
    # opfsolver = ScipyOptimizer(c)

    res = opfsolver.solve(flat_start)
    return res
