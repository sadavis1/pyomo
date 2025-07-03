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
from pyomo.core.base import (
    Transformation,
    TransformationFactory,
    LogicalConstraint,
    BooleanVar,
)
from pyomo.core.expr import identify_variables, exactly, atleast, implies, equivalent
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
        # TODO: you should be able to call this on a Disjunction
        
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
        transformed_disjuncts = transform_block.transformed_disjuncts = Disjunct(Any)
        had_or = False

        to_deactivate = ComponentSet()

        for node in tree.topological_sort():
            if not tree.is_leaf(node):
                # Non-leaf Disjuncts: Deactivate but otherwise do
                # nothing. We will copy the constraints and reference the
                # Vars from these later, and the other components will
                # be ignored.
                #
                # Non-leaf Disjunctions: Set up a logical constraint,
                # then deactivate. Never fix the indicator vars because
                # we are replacing the disjunctive structure with logic
                # on the indicator vars.
                if node.ctype == Disjunction:
                    parent_indicator = True
                    if tree.parent_disjunct(node) is not None:
                        parent_indicator = tree.parent_disjunct(node).indicator_var
                    if node.xor:
                        logical_constraints[len(logical_constraints)] = implies(
                            parent_indicator,
                            exactly(
                                1, *(disj.indicator_var for disj in node.disjuncts)
                            ),
                        )
                    else:
                        had_or = True
                        logical_constraints[len(logical_constraints)] = implies(
                            parent_indicator,
                            atleast(
                                1, *(disj.indicator_var for disj in node.disjuncts)
                            ),
                        )
                    to_deactivate.add(node)
                elif node.ctype == Disjunct:
                    to_deactivate.add(node)
            else:
                # Leaves: we had better be a Disjunct, not an empty
                # Disjunction. Add refs to all Constraints and Vars from
                # parent Disjuncts. The Vars are only necessary to
                # satisfy writers that make too many assumptions.
                if node.ctype == Disjunct:
                    to_deactivate.add(node)

                    leaf_proxy = transformed_disjuncts[
                        node.getname(fully_qualified=True)
                    ]
                    leaves.append(leaf_proxy)

                    # Here is a case where it would be nice to choose
                    # the indicator var manually
                    logical_constraints[len(logical_constraints)] = equivalent(
                        node.indicator_var, leaf_proxy.indicator_var
                    )

                    refs_block = Block(Any)
                    leaf_proxy.add_component(
                        unique_component_name(node, 'disjunct_refs'), refs_block
                    )
                    visiting_disjunct = node
                    # Quit once we make it past the disjunction we are transforming
                    while (
                        visiting_disjunct is not None
                        and disjunction not in tree.children(visiting_disjunct)
                    ):
                        # print(f"{parent_disjunct.getname(fully_qualified=True)=}")
                        working_block = refs_block[
                            visiting_disjunct.getname(fully_qualified=True)
                        ]
                        for component in visiting_disjunct.component_data_objects(
                            descend_into=Block, active=True
                        ):
                            if component.ctype in {Var, BooleanVar}:
                                working_block.add_component(
                                    component.name, Reference(component)
                                )
                            if component.ctype == Constraint:
                                working_block.add_component(
                                    component.name, Constraint(expr=component.expr)
                                )
                                to_deactivate.add(component)
                            if component.ctype == LogicalConstraint:
                                working_block.add_component(
                                    component.name,
                                    LogicalConstraint(expr=component.expr),
                                    # Using a Reference is not possible
                                    # here. The logical-to-linear
                                    # transformation will deactivate
                                    # after transforming the first one,
                                    # and transform no others as they
                                    # are not active.
                                    #
                                    # component.name,
                                    # Reference(component),
                                )
                                to_deactivate.add(component)
                            # if component.ctype == Block:
                            # (descending into)

                        visiting_disjunct = tree.parent_disjunct(visiting_disjunct)

        for component in to_deactivate:
            if component.ctype == Disjunct:
                component._deactivate_without_fixing_indicator()
            else:
                component.deactivate()

        # Set up the new Disjunction. It's almost possible to skip the
        # logical_constraints if had_or == False, but it might allow the
        # model to cheat at user-defined LogicalConstraints so we can't.
        transform_block.transformed_disjunction = Disjunction(
            expr=leaves, xor=not had_or
        )
