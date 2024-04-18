from pyomo.environ import *
from pyomo.gdp import *
import sys
from pyomo.contrib.piecewise import PiecewiseLinearFunction
from pyomo.environ import ConcreteModel, Constraint, log, Objective, Var

def make_log_x_model():
    m = ConcreteModel()
    m.x = Var(bounds=(1, 10))
    m.pw_log = PiecewiseLinearFunction(points=[1, 3, 5, 7, 10], function=log)

    m.log_expr = m.pw_log(m.x)
    m.obj = Objective(expr=m.log_expr)
    m.cons = Constraint(expr=m.x >= 2)
    return m

def get_pre_nested_model():
    m = ConcreteModel()
    m.x = Var(bounds=(1, 10))
    m.cons = Constraint(expr=m.x >= 2)

    # DIY the equivalent of this
    #pw_log = PiecewiseLinearFunction(points=[1, 3, 5, 7, 10], function=log)
    #log_expr = pw_log(m.x)
    #m.obj = Objective(expr=m.log_expr)

    m.substitute_var = Var(bounds=(0, 4.94))
    @m.Disjunct()
    def d_l(b):
        @b.Disjunct()
        def d_l(b2):
            b2.lambda0 = Var(bounds=(0, 1))
            b2.lambda1 = Var(bounds=(0, 1))
            b2.convex_combo = Constraint(expr=b2.lambda0 + b2.lambda1 == 1)
            b2.linear_combo = Constraint(expr=b2.lambda0 + 3 * b2.lambda1 == m.x)
            b2.set_substitute = Constraint(expr=0.55 * m.x - 0.55 == m.substitute_var)
        @b.Disjunct()
        def d_r(b2):
            b2.lambda0 = Var(bounds=(0, 1))
            b2.lambda1 = Var(bounds=(0, 1))
            b2.convex_combo = Constraint(expr=b2.lambda0 + b2.lambda1 == 1)
            b2.linear_combo = Constraint(expr=3 * b2.lambda0 + 5 * b2.lambda1 == m.x)
            b2.set_substitute = Constraint(expr=0.25 * m.x + 0.33 == m.substitute_var)
        b.inner_disjunction_l = Disjunction(expr=[b.d_l, b.d_r])
    @m.Disjunct()
    def d_r(b):
        @b.Disjunct()
        def d_l(b2):
            b2.lambda0 = Var(bounds=(0, 1))
            b2.lambda1 = Var(bounds=(0, 1))
            b2.convex_combo = Constraint(expr=b2.lambda0 + b2.lambda1 == 1)
            b2.linear_combo = Constraint(expr=5 * b2.lambda0 + 7 * b2.lambda1 == m.x)
            b2.set_substitute = Constraint(expr=0.17 * m.x + 0.77 == m.substitute_var)
        @b.Disjunct()
        def d_r(b2):
            b2.lambda0 = Var(bounds=(0, 1))
            b2.lambda1 = Var(bounds=(0, 1))
            b2.convex_combo = Constraint(expr=b2.lambda0 + b2.lambda1 == 1)
            b2.linear_combo = Constraint(expr=7 * b2.lambda0 + 10 * b2.lambda1 == m.x)
            b2.set_substitute = Constraint(expr=0.12 * m.x + 1.11 == m.substitute_var)
        b.inner_disjunction_r = Disjunction(expr=[b.d_l, b.d_r])
    m.disj = Disjunction(expr=[m.d_l, m.d_r])

    m.var_id_l = Constraint(expr=m.d_r.d_l.binary_indicator_var == m.d_l.d_l.binary_indicator_var)
    m.var_id_r = Constraint(expr=m.d_r.d_r.binary_indicator_var == m.d_l.d_r.binary_indicator_var)

    m.obj = Objective(expr=m.substitute_var)
    return m

def get_model_example():
    model = make_log_x_model()
    xf = TransformationFactory('contrib.piecewise.nested_inner_repn_gdp')
    xf.CONFIG.identify_variables = True
    xf.apply_to(model)
    xf.CONFIG.identify_variables = False
    print('PRINTING THE THING====================')
    #model.pprint()
    print('DONE PRINTING THE THING====================')
    TransformationFactory('gdp.hull').apply_to(model)
    return model

def get_model_no_pw():
    model = get_pre_nested_model()
    xf = TransformationFactory('contrib.piecewise.nested_inner_repn_gdp')
    print("got model, applying hull")
    TransformationFactory('gdp.hull').apply_to(model)
    print("applied hull")
    return model


# try 1
solver = SolverFactory('gurobi')
solver.options['nonconvex'] = 2
get_model = get_model_no_pw

m1 = get_model()
print("---- solving without disaggregate ----")
print("pprint(m1):")
#m1.pprint()
print('------------------------------------')
solver.solve(m1, tee=True)
print()
print("---- solving with disaggregate ----")
m2 = get_model()
TransformationFactory('contrib.aggregate_vars').apply_to(m2, detect_fixed_vars=False)
print("pprint(m2):")
#m2.pprint()
print('------------------------------------')
solver.solve(m2, tee=True)
