#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging

from pyomo.common.collections import ComponentSet
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.core import (
    TransformationFactory,
    Transformation,
    Block,
    VarList,
    Set,
    SortComponents,
    Objective,
    Constraint,
)
from pyomo.core.util import target_list
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import get_gdp_tree
from pyomo.repn import generate_standard_repn

logger = logging.getLogger('pyomo.gdp')


@TransformationFactory.register(
    'gdp.apply_recursively',
    doc="Given a nested GDP, repeatedly apply the given transformation to the"
    "bottommost disjunctions until the model is no longer disjunctive.",
)
class NestedApplyRecursively(Transformation):
    CONFIG = ConfigDict('gdp.apply_recursively')
    CONFIG.declare(
        'targets',
        ConfigValue(
            default=None,
            domain=target_list,
            description="target or list of targets to transform",
            doc="""
            This specifies the list of Disjunctions or Blocks to be (partially)
            transformed. If None (default), the entire model is transformed.
            Note that if the transformation is done out of place, the list of
            targets should be attached to the model before it is cloned, and
            the list will specify the targets on the cloned instance.
            """,
        ),
    )
    CONFIG.declare(
        'transformation',
        ConfigValue(
            default=None,
            domain=str,
            description="transformation to apply recursively",
            doc="""
            This should be a string that refers to a valid key for TransformationFactory;
            additionally that transformation should remove disjunctive structure when
            called using apply_to, such that applying it recursively will actually
            terminate.
            """,
        ),
    )
    CONFIG.declare(
        'transformation_options',
        ConfigValue(
            default=None,
            # domain=dict,
            description="options dict for the transformation to apply recursively",
            doc="""
            This dict is passed unchanged into the `options` parameter of the call to
            apply_to on the instantiated transformation.
            """,
        ),
    )
    transformation_name = 'apply_recursively'

    def __init__(self):
        super().__init__()

    def _apply_to(self, instance, **kwds):
        self._config = self.CONFIG(kwds.pop('options', {}))
        self._config.set_value(kwds)

        targets = self._config.targets
        if targets is None:
            targets = (instance,)
        transformation = self._config.transformation
        if transformation is None:
            raise ValueError("need a transformation to apply")
        transformation_options = self._config.transformation_options

        while True:
            to_transform = ComponentSet()
            tree = get_gdp_tree(targets, instance)
            for t in tree.topological_sort():
                if tree.is_leaf(t):
                    component = tree.parent_disjunct(t)
                    if component is not None:
                        to_transform.add(component)

            if not to_transform:
                break

            for t in to_transform:
                TransformationFactory(transformation).apply_to(
                    t, options=transformation_options
                )
