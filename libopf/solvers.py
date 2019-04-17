from .case import *
from .arithmetic import *
import numpy as np
from scipy.optimize import minimize

try:
    import ipopt
except ImportError:
    print('***  Module "cyipopt" not found. Can only solve by "scipy.optimize"')


class OpfResult(object):

    def __init__(self):
        self.solved = False
        self.success = None
        self.x = None
        self.status = None
        self.objval = None
        self.message = None
        self.nit = None


class IpoptOptimizer(object):

    def __init__(self, c):
        self.case = c
        self.x0 = None
        self.xmin = None
        self.xmax = None
        self.opf_mdl = None
        self.cl = None
        self.cu = None
        self.result = OpfResult()

    def build_optmdl(self, flat_start):

        self.x0 = build_varinit(self.case, flat_start)
        self.xmin, self.xmax = build_varbounds(self.case)

        self.opf_mdl = IpoptModel(self.case)

        nb = self.case.bus.shape[0]
        nbr = self.case.branch.shape[0]
        self.cl = np.concatenate((np.zeros(2 * nb), np.zeros(2 * nbr)))
        self.cu = np.concatenate((np.zeros(2 * nb), np.inf * np.ones(2 * nbr)))


    def solve(self, flat_start):
        self.build_optmdl(flat_start)
        nlp = ipopt.problem(n=len(self.x0), m=len(self.cl), lb=self.xmin, ub=self.xmax, \
            cl=self.cl, cu=self.cu, problem_obj=self.opf_mdl)

        res_x, res_info = nlp.solve(self.x0)
        self.result.solved = True
        self.result.x = res_x
        self.result.status = res_info['status']
        self.result.objval = res_info['obj_val']
        self.result.message = res_info['status_msg']
        self.result.nit = -1 # Not callable from cyipopt
        return self.result


class IpoptModel(object):
    
    def __init__(self, c):
        self.case = c

    def objective(self, x):
        return costfcn(x, self.case)
        
    def gradient(self, x):
        return costfcn_jac(x, self.case)

    def constraints(self, x):
        return np.concatenate((acpf_consfcn(x, self.case), linerating_consfcn(x, self.case)))

    def jacobian(self, x):
        return np.concatenate((acpf_consfcn_jac(x, self.case), linerating_consfcn_jac(x, self.case)))

    def intermediate(
        self,
        alg_mod,
        iter_count,
        obj_value,
        inf_pr,
        inf_du,
        mu,
        d_norm,
        regularization_size,
        alpha_du,
        alpha_pr,
        ls_trials
        ):
        
        #
        # Example for the use of the intermediate callback.
        #
        print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))


class ScipyOptimizer(object):

    def __init__(self, c):
        self.case = c
        self.x0 = None
        self.xmin = None
        self.xmax = None
        self.fun = None
        self.jac = None
        self.hess = None
        self.bounds = None
        self.constraints = None
        self.result = OpfResult()

    def build_optmdl(self, flat_start):

        self.fun = lambda x: costfcn(x, self.case)
        self.jac = lambda x: costfcn_jac(x, self.case)
        self.hess = lambda x: costfcn_hess(x, self.case)
        self.x0 = build_varinit(self.case, flat_start)
        self.xmin, self.xmax = build_varbounds(self.case)

        self.bounds = ()
        for vi in range(len(self.xmin)):
            self.bounds += ((self.xmin[vi], self.xmax[vi]),)
        
        eqcons   = {'type': 'eq',
                    'fun' : lambda x: acpf_consfcn(x, self.case),
                    'jac' : lambda x: acpf_consfcn_jac(x, self.case)}
        ineqcons = {'type': 'ineq',
                    'fun' : lambda x: linerating_consfcn(x, self.case),
                    'jac' : lambda x: linerating_consfcn_jac(x, self.case)}

        self.constraints = (eqcons, ineqcons)


    def solve(self, flat_start):
        self.build_optmdl(flat_start)
        res = minimize(self.fun, self.x0, jac=self.jac, hess=self.hess, bounds=self.bounds, \
            constraints=self.constraints, options={'disp': False})

        
        self.result.solved = True
        self.result.x = res.x
        self.result.status = res.status
        self.result.objval = res.fun
        self.result.message = res.message
        self.result.nit = res.nit
        return self.result


class MipsOptimizer(object):

    def __init__(self, c):
        self.case = c
        self.x0 = None
        self.fun = None
        self.jac = None
        self.hess = None
        self.bounds = None
        self.constraints = None
        self.result = OpfResult()

    def build_optmdl(self, flat_start):
        return 0

    def solve(self, flat_start):
        self.build_optmdl(flat_start)
        return self.result


