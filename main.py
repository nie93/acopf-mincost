from case import Case, Const
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
    for i in range(0, number):
        start_time = time.time()
        res = opf.runcopf(c, flat_start=True)
        end_time = time.time()
        print("Optimal Outputs: %s" % str(res['PG']))
        print('Optimization execution time: %.8f' % (end_time - start_time))
    
    return res

def main():
    const = Const()
    c = import_opf_case('./data/case30/')
    # c = set_starting_output(c, [40.05814367749094, 70.12544088354686, 78.47923733719254])
    # c.set_branch_prop('RATE', [14], [34.999])
    # c.scale_branch_prop([const.BR_R, const.BR_X], multi=1.0)    

    res = execute_opf(c, 1)
    set_trace()

if __name__ == "__main__":
    main()