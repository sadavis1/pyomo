# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import logging
import re
import sys
import random
from io import StringIO

import pyomo.common.unittest as unittest
import unittest.mock as mock

from pyomo.common.dependencies import dill_available, numpy_available
from pyomo.common.log import LoggingIntercept
from pyomo.common.fileutils import this_file_dir

from pyomo.environ import (
    TransformationFactory,
    Block,
    Set,
    Constraint,
    Var,
    RealSet,
    ComponentMap,
    value,
    log,
    ConcreteModel,
    Any,
    Suffix,
    SolverFactory,
    RangeSet,
    Param,
    Objective,
    TerminationCondition,
    NonNegativeReals,
)
from pyomo.core.expr.compare import (
    assertExpressionsEqual,
    assertExpressionsStructurallyEqual,
)
import pyomo.core.expr as EXPR
from pyomo.core.base import constraint
from pyomo.repn import generate_standard_repn
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.repn.quadratic import QuadraticRepnVisitor
from pyomo.repn.util import OrderedVarRecorder
from pyomo.core.base import SortComponents

from pyomo.gdp import Disjunct, Disjunction, GDP_Error
import pyomo.gdp.plugins.hull as hull_module
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct


# todo remove
import pyomo.gdp.plugins.reverse_polar_enumeration_cuts as rpec_module

class TestReversePolarEnumerationCuts(unittest.TestCase):
    def test_example(self):
        m = ConcreteModel()
        m.x1 = Var(bounds=(0, 20))
        m.x2 = Var(bounds=(0, 20))
        m.x3 = Var(bounds=(0, 20))
        m.x4 = Var(bounds=(0, 20))

        m.d1 = Disjunct()
        m.d1.c = Constraint(expr=5 * m.x1 - 3 * m.x2 + m.x4 >= 1)
        m.d2 = Disjunct()
        m.d2.c = Constraint(expr=3 * m.x1 - m.x2 + 2 * m.x3 - 3 * m.x4 >= 1)
        m.d3 = Disjunct()
        m.d3.c = Constraint(expr=4 * m.x1 - 6 * m.x2 + 4 * m.x3 - 2 * m.x4 >= 1)
        m.d4 = Disjunct()
        m.d4.c = Constraint(expr=2 * m.x1 - 2 * m.x2 - 2 * m.x3 >= 1)
        m.d = Disjunction(expr=[m.d1, m.d2, m.d3, m.d4])

        a = TransformationFactory('gdp.reverse_polar_enumeration_cuts').apply_to(m)
        breakpoint()
