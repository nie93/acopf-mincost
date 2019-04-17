from .case import *
from .arithmetic import *
import numpy as np
import ipopt
from scipy.optimize import minimize
from pdb import set_trace


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
        const = Const()
        
        nb     = self.case.bus.shape[0]
        ng     = self.case.gen.shape[0]
        nbr    = self.case.branch.shape[0]
        neq    = 2 * nb
        niq    = 2 * ng + nb + nbr
        neqnln = 2 * nb
        niqnln = nbr

        ii = get_var_idx(self.case)

        if flat_start:
            self.x0 = np.concatenate((deg2rad(self.case.bus.take(const.VA, axis=1)), \
                self.case.bus[:,[const.VMIN, const.VMAX]].mean(axis=1), \
                self.case.gen[:,[const.PMAX, const.PMIN]].mean(axis=1) / self.case.mva_base, \
                self.case.gen[:,[const.QMAX, const.QMIN]].mean(axis=1) / self.case.mva_base), axis=0)
            # self.x0 = np.concatenate((np.zeros(nb), \
            #     self.case.bus[:,[const.VMIN, const.VMAX]].mean(axis=1), \
            #     self.case.gen[:,[const.PMAX, const.PMIN]].mean(axis=1) / self.case.mva_base, \
            #     self.case.gen[:,[const.QMAX, const.QMIN]].mean(axis=1) / self.case.mva_base), axis=0)
        else:
            self.x0 = np.genfromtxt(os.path.join(self.case.path, "x0.csv"), delimiter=',')


        self.xmin = np.concatenate((-np.inf * np.ones(nb), \
                            self.case.bus[:, const.VMIN], \
                            self.case.gen[:, const.PMIN] / self.case.mva_base, \
                            self.case.gen[:, const.QMIN] / self.case.mva_base), axis=0)
        self.xmax = np.concatenate((np.inf * np.ones(nb), \
                            self.case.bus[:, const.VMAX], \
                            self.case.gen[:, const.PMAX] / self.case.mva_base, \
                            self.case.gen[:, const.QMAX] / self.case.mva_base), axis=0)

        self.xmin[(self.case.bus[:, const.BUS_TYPE] == 3).nonzero()] = 0
        self.xmax[(self.case.bus[:, const.BUS_TYPE] == 3).nonzero()] = 0

        self.opf_mdl = IpoptModel(self.case)

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
        self.fun = None
        self.jac = None
        self.hess = None
        self.bounds = None
        self.constraints = None
        self.result = OpfResult()

    def build_optmdl(self, flat_start):
        
        const = Const()
        
        nb     = self.case.bus.shape[0]
        ng     = self.case.gen.shape[0]
        nbr    = self.case.branch.shape[0]
        neq    = 2 * nb
        niq    = 2 * ng + nb + nbr
        neqnln = 2 * nb
        niqnln = nbr

        ii = get_var_idx(self.case)

        if flat_start:
            self.x0 = np.concatenate((deg2rad(self.case.bus.take(const.VA, axis=1)), \
                self.case.bus[:,[const.VMIN, const.VMAX]].mean(axis=1), \
                self.case.gen[:,[const.PMAX, const.PMIN]].mean(axis=1) / self.case.mva_base, \
                self.case.gen[:,[const.QMAX, const.QMIN]].mean(axis=1) / self.case.mva_base), axis=0)
            # self.x0 = np.concatenate((np.zeros(nb), \
            #     self.case.bus[:,[const.VMIN, const.VMAX]].mean(axis=1), \
            #     self.case.gen[:,[const.PMAX, const.PMIN]].mean(axis=1) / self.case.mva_base, \
            #     self.case.gen[:,[const.QMAX, const.QMIN]].mean(axis=1) / self.case.mva_base), axis=0)
        else:
            self.x0 = np.genfromtxt(os.path.join(self.case.path, "x0.csv"), delimiter=',')

        self.fun = lambda x: costfcn(x, self.case)
        self.jac = lambda x: costfcn_jac(x, self.case)
        self.hess = lambda x: costfcn_hess(x, self.case)


        xmin = np.concatenate((-np.inf * np.ones(nb), \
                            self.case.bus[:, const.VMIN], \
                            self.case.gen[:, const.PMIN] / self.case.mva_base, \
                            self.case.gen[:, const.QMIN] / self.case.mva_base), axis=0)
        xmax = np.concatenate((np.inf * np.ones(nb), \
                            self.case.bus[:, const.VMAX], \
                            self.case.gen[:, const.PMAX] / self.case.mva_base, \
                            self.case.gen[:, const.QMAX] / self.case.mva_base), axis=0)

        xmin[(self.case.bus[:, const.BUS_TYPE] == 3).nonzero()] = 0
        xmax[(self.case.bus[:, const.BUS_TYPE] == 3).nonzero()] = 0
        
        self.bounds = ()
        for vi in range(len(xmin)):
            self.bounds += ((xmin[vi], xmax[vi]),)
        
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
