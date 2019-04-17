from pdb import *
from .solvers import ScipyOptimizer


def runcopf(c, flat_start):
    opfsolver = ScipyOptimizer(c)
    res = opfsolver.solve(flat_start)
    return res