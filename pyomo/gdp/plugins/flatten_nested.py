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

from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.modeling import unique_component_name
from pyomo.core import (
    Any,
    Block,
    Constraint,
    NonNegativeIntegers,
    SortComponents,
    value,
    Var,
    Reference,
)
from pyomo.core.base import Transformation, TransformationFactory, LogicalConstraint
from pyomo.core.expr import identify_variables, exactly, atleast
from pyomo.core.util import target_list
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import is_child_of, get_gdp_tree
from pyomo.repn.standard_repn import generate_standard_repn
import logging

logger = logging.getLogger(__name__)


@TransformationFactory.register('gdp.flatten_nested', doc="")
class FlattenNested(Transformation):
    """ """

    CONFIG = ConfigDict('gdp.flatten_nested')
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
    transformation_name = 'flatten_nested'

    def __init__(self):
        super().__init__()
        self.logger = logger

    def _apply_to(self, instance, **kwds):
        if instance.ctype not in (Block, Disjunct):
            raise GDP_Error(
                "Transformation called on %s of type %s. 'instance'"
                " must be a ConcreteModel, Block, or Disjunct"
                % (instance.name, instance.ctype)
            )

        self._config = self.CONFIG(kwds.pop('options', {}))
        self._config.set_value(kwds)

        targets = self._config.targets
        if targets is None:
            targets = (instance,)

        # TODO: when is a dict not enough?
        transformation_blocks = ComponentMap()
        for t in targets:
            b = Block(Any)
            name = unique_component_name(t, "flatten_nested_transform")
            t.add_component(name, b)
            transformation_blocks[t] = b

        whole_tree = get_gdp_tree(targets, instance)
        # TODO: does this need to be sorted?
        for t in whole_tree.topological_sort():
            if t.ctype is Disjunction and whole_tree.in_degree(t) == 0:
                target_block = t.parent_block()
                while target_block not in targets:
                    target_block = target_block.parent_block()
                transformation_block_indexed = transformation_blocks[target_block]
                transformation_block_data = transformation_block_indexed[
                    t.getname(fully_qualified=True)
                ]
                self._transform_disjunction(t, instance, transformation_block_data)

    # Basic unit of transformation: transform one root Disjunction.
    def _transform_disjunction(self, disjunction, instance, transform_block):
        tree = get_gdp_tree((disjunction,), instance)
        leaves = []
        logical_constraints = transform_block.derived_logical_constraints = (
            LogicalConstraint(Any)
        )
        had_or = False
        for node in tree.topological_sort():
            if not tree.is_leaf(node):
                # Non-leaf Disjuncts: Deactivate but otherwise do
                # nothing. We will reference all the Constraints and
                # Vars from these later, and the other components will
                # be ignored.
                #
                # Non-leaf Disjunctions: Set up a logical constraint,
                # then deactivate. Goal is that we make all the non-root
                # Disjuncts unreachable from Disjunctions, but reachable
                # again from some References we will set up.
                if node.ctype == Disjunction:
                    if node.xor:
                        logical_constraints[len(logical_constraints)] = exactly(
                            1, *(disj.indicator_var for disj in node.disjuncts)
                        )
                    else:
                        had_or = True
                        logical_constraints[len(logical_constraints)] = atleast(
                            1, *(disj.indicator_var for disj in node.disjuncts)
                        )
                    node.deactivate()
                elif node.ctype == Disjunct:
                    node._deactivate_without_fixing_indicator()
            else:
                # Leaves: we had better be a Disjunct, not an empty
                # Disjunction. Add refs to all Constraints and Vars from
                # parent Disjuncts. The Vars are only necessary to
                # satisfy writers that make too many assumptions.
                if node.ctype == Disjunct:
                    leaves.append(node)
                    refs_block = Block(Any)
                    node.add_component(
                        unique_component_name(node, 'parent_disjunct_refs'), refs_block
                    )
                    parent_disjunct = tree.parent_disjunct(node)
                    while parent_disjunct is not None:
                        print(f"{parent_disjunct.getname(fully_qualified=True)=}")
                        working_block = refs_block[
                            parent_disjunct.getname(fully_qualified=True)
                        ]
                        for component in parent_disjunct.component_objects(
                            descend_into=False
                        ):
                            if component.ctype in {Constraint, Var}:
                                working_block.add_component(
                                    component.name, Reference(component)
                                )

                        parent_disjunct = tree.parent_disjunct(parent_disjunct)
        # Set up the new Disjunction. It's almost possible to skip the
        # logical_constraints if had_or == False, but it might allow the
        # model to cheat at user-defined LogicalConstraints so we can't.
        transform_block.transformed_disjunction = Disjunction(
            expr=leaves, xor=not had_or
        )
