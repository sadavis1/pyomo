#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.base import Transformation, TransformationFactory
import pyomo.gdp.plugins.hull
from pyomo.contrib.piecewise.transform.piecewise_linear_transformation_base import (
    PiecewiseLinearTransformationBase,
)


@TransformationFactory.register(
    'contrib.piecewise.logarithmic_gdp',
    doc="""
    Represent a piecewise using a nested linear GDP as in 
    contrib.piecewise.nested_inner_repn, but perform a variable identification 
    pass to reduce the number of Boolean variables from linearly to
    logarithmically many.""",
)
class LogarithmicGDPTransformation(Transformation):
    """
    Represent a piecewise using a nested linear GDP as in
    contrib.piecewise.nested_inner_repn, but perform a variable identification
    pass to reduce the number of Boolean variables from linearly to
    logarithmically many. After this identification step and then a hull
    transformation, the formulation should match the disaggregated logarithmic
    formulation of [1].

    References
    ----------
    [1] J.P. Vielma, S. Ahmed, and G. Nemhauser, "Mixed-integer models
        for nonseparable piecewise-linear optimization: unifying framework
        and extensions," Operations Research, vol. 58, no. 2, pp. 305-315,
        2010.
    """

    CONFIG = PiecewiseLinearTransformationBase.CONFIG()
    _transformation_name = "pw_linear_logarithmic_gdp"

    # Apply nested repn with variable identification, then disaggregate newly
    # identified variables
    def _apply_to(self, instance, **kwds):
        xf = TransformationFactory('contrib.piecewise.nested_inner_repn_gdp')
        xf.CONFIG.identify_variables = True
        # issue: ephemerally set values are not passed all the way through to
        # _transform_pw_linear_expr, so I need to persistently set it and then
        # undo (because it's static)
        # kwds['identify_variables'] = True
        xf.apply_to(instance, **kwds)
        xf.CONFIG.identify_variables = False
        TransformationFactory('contrib.aggregate_vars').apply_to(instance)
