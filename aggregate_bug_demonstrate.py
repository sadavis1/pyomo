from pyomo.environ import *
import sys

def get_nonlinear_model():

    m = ConcreteModel()

    m.x = Var(within=Reals, bounds=(0,10), initialize=1)
    m.obj = Objective(sense=minimize, expr=m.x)
    m.e = Constraint(expr= m.x**1.5 >= 3)

    return m

def get_model():
    model = get_nonlinear_model()
    TransformationFactory('contrib.piecewise.nonlinear_to_pwl').apply_to(
        model,
        n=5,
        method='simple_uniform_point_grid',
        additively_decompose=True,
        allow_quadratic_cons=True,
        allow_quadratic_objs=True,
    )
    xf = TransformationFactory('contrib.piecewise.nested_inner_repn_gdp')
    xf.CONFIG.identify_variables = True
    xf.apply_to(model)
    xf.CONFIG.identify_variables = False
    TransformationFactory('gdp.hull').apply_to(model)

    return model

sys.path.append('/home/sadavi/repos/cimor-collab/pwl_transformation_comparisons')
from PWLTransformation import NonlinearToPWL

# try 1
solver = SolverFactory('gurobi')
solver.options['nonconvex'] = 2

m1 = get_model()
print("---- solving without disaggregate ----")
print("pprint(m1):")
#m1.pprint()
print('------------------------------------')
solver.solve(m1, tee=True)
print()
print("---- solving with disaggregate ----")
m2 = get_model()
TransformationFactory('contrib.aggregate_vars').apply_to(m2)
print("pprint(m2):")
#m2.pprint()
print('------------------------------------')
solver.solve(m2, tee=True)