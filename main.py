from arithmetic import *
from case import *
import numpy as np
import opf
import time
from pdb import set_trace
from pprint import pprint


def import_opf_case(casepath):
    const = Const()
    c = Case()
    c.import_case(casepath)
    return c

def set_starting_output(c, pg):
    const = Const()
    c.set_gen_prop(const.PMAX, [1,2,3], pg)
    return c

def execute_opf(c, number):
    for exec_count in range(0, number):
        start_time = time.time()
        res = opf.runcopf(c, flat_start=True)
        end_time = time.time()
        # print("Optimal Outputs: %s" % str(res['PG']))
        print('Optimization execution time: %.8f' % (end_time - start_time))
    
    return res

def main():
    const = Const()
    c = import_opf_case(r'./data/case14/')
    res = execute_opf(c, 1)
    write_results(r'./var/log/opfout_%s.log' % c.casename, c, res)
    # set_trace()

if __name__ == "__main__":
    main()