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
    doc='TODO'
)
class LogarithmicGDPTransformation(Transformation):
    """
    """

    CONFIG = PiecewiseLinearTransformationBase.CONFIG()
    _transformation_name = "pw_linear_logarithmic_gdp"

    # Apply nested repn with variable identification, then disaggregate said
    # variables
    def _apply_to(self, instance, **kwds):
        xf = TransformationFactory('contrib.piecewise.nested_inner_repn_gdp')
        xf.CONFIG.identify_variables = True
        xf.apply_to(instance, **kwds)
        #TransformationFactory('contrib.piecewise.nested_inner_repn_gdp').apply_to(instance, options={'identify_variables': True}, **kwds)
        TransformationFactory('contrib.aggregate_vars').apply_to(instance, **kwds)
