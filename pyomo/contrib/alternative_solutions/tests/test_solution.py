#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as pe
from pyomo.common.collections import ComponentSet
import pyomo.common.unittest as unittest
import pyomo.contrib.alternative_solutions.aos_utils as au
import pyomo.contrib.alternative_solutions.solution as sol

class TestSolutionUnit(unittest.TestCase):
    def get_model(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(domain=pe.NonNegativeReals)
        m.y = pe.Var(domain=pe.Binary)
        m.z = pe.Var(domain=pe.NonNegativeIntegers)
        m.f = pe.Var(domain=pe.Reals)
        
        m.f.fix(1)
        m.obj = pe.Objective(expr=m.x + m.y + m.z + m.f, sense=pe.maximize)
        
        m.con_x = pe.Constraint(expr=m.x <= 1.5)
        m.con_y = pe.Constraint(expr=m.y <= 1)
        m.con_z = pe.Constraint(expr=m.z <= 3)
        return m
        
    def test_multiple_objectives(self):
        model = self.get_model()
        opt = pe.SolverFactory('cplex')
        opt.solve(model)
        all_vars = au.get_model_variables(model, include_fixed=True)
 
        solution = sol.Solution(model, all_vars, include_fixed=False)
        solution.pprint()
        
        solution = sol.Solution(model, all_vars)
        solution.pprint(round_discrete=True)
 
        sol_val = solution.get_variable_name_values(include_fixed=True, 
                                                    round_discrete=True)
        self.assertEqual(set(sol_val.keys()), {'x','y','z','f'})
        self.assertEqual(set(solution.get_fixed_variable_names()), {'f'})

if __name__ == '__main__':
    unittest.main()