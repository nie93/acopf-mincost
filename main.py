from libopf import case, runcopf
import numpy as np
import time
from pdb import set_trace
from pprint import pprint


def import_opf_case(casepath):
    const = case.Const()
    c = case.Case()
    c.import_case(casepath)
    return c

def set_starting_output(c, pg):
    const = case.Const()
    c.set_gen_prop(const.PMAX, [1,2,3], pg)
    return c

def execute_opf(c, number):
    for exec_count in range(0, number):

        start_time = time.time()
        res = runcopf(c, flat_start=True)
        end_time = time.time()

        ii = case.get_var_idx(c)

        float_fmtr = {'float_kind': lambda x: "%7.3f" % x}
        # pprint(res['x'])
        case.write_results(None, c, res)
        print("Optimal Outputs: %s" % str(res.x[ii['i1']['pg']:ii['iN']['pg']] * c.mva_base))
        print('Optimization execution time: %.8f' % (end_time - start_time))
    
    return res

if __name__ == "__main__":
    const = case.Const()
    c = import_opf_case(r'./data/case30')
    res = execute_opf(c, 10)
    case.write_results(r'./var/log/opfout_%s.log' % c.casename, c, res)
    