# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import logging
from pyomo.core.base import Transformation, TransformationFactory
from pyomo.core.base.component import ActiveComponent
from pyomo.core.base.block import SubclassOf
from pyomo.core.util import target_list
from pyomo.core.base.enums import SortComponents
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.core import Block, Constraint
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import get_gdp_tree
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.repn.util import OrderedVarRecorder


logger = logging.getLogger(__name__)


@TransformationFactory.register(
    'gdp.reverse_polar_enumeration_cuts',
    doc="Add cuts to a GDP with 'simple disjunctions', according to the reverse polar "
    "vertex enumeration algorithm of [TODO REF]. A simple disjunction is one in which "
    "each disjunct contains only exactly one linear inequality on nonnegative "
    "variables.",
)
class ReversePolarEnumerationCuts(Transformation):
    """
    Add cuts to a GDP with 'simple disjunctions', according to the reverse polar
    vertex enumeration algorithm of [TODO REF]. A simple disjunction is one in which
    each disjunct contains only exactly one linear inequality on nonnegative variables.
    """

    transformation_name = 'reverse_polar_enumeration_cuts'  # necessary?
    CONFIG = ConfigDict('gdp.reverse_polar_enumeration_cuts')
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
        'num_cuts',
        ConfigValue(
            default=1,
            # domain=int,
            description="number of cuts to generate",
            doc="""
            Maximum number of cuts to generate. If None is passed, keep going until there
            are no more cuts available, at which point the cuts that have been added will
            fully determine the closed convex hull of the feasible region. This is
            applied separately on each target, so if you need to make a different
            number of cuts to each of multiple targets you should call this
            transformation multiple times.
            """,
        ),
    )
    CONFIG.declare(
        'num_skip',
        ConfigValue(
            default=0,
            domain=int,
            description="Skip the first N cuts generated.",
            doc="""
            Skip the first N cuts generated. Applied separately to each target like
            num_cuts.
            """,
        ),
    )

    def __init__(self):
        super().__init__()
        self.logger = logger

    def _apply_to(self, instance, **kwds):
        if instance.ctype not in (Block, Disjunct):
            raise GDP_Error(
                "Transformation called on %s of type %s. 'instance'"
                " must be a ConcreteModel, Block, or Disjunct (in "
                "the case of nested disjunctions)." % (instance.name, instance.ctype)
            )

        self._config = self.CONFIG(kwds.pop('options', {}))
        self._config.set_value(kwds)

        targets = self._config.targets
        if targets is None:
            targets = (instance,)

        tree = get_gdp_tree(targets, instance)
        for t in tree.reverse_topological_sort():
            if t.ctype is Disjunction:
                self._validate_disjunction(t, tree)
                for cut in self._generate_cuts(t, tree):
                    pass
                    # TODO add cut to model in a nice way

    def _validate_disjunction(self, disj, tree):
        if tree.root_disjunct(disj) is not None:
            raise GDP_Error("we don't support nested for now")
        for b in tree.children(disj):
            found = False
            for c in b.component_data_objects(SubclassOf(ActiveComponent)):
                # no active components except exactly one constraint are permitted.
                # Non-active things like params and vars are fine.
                if found or c.ctype is not Constraint:
                    raise GDP_Error(
                        "No active components except exactly one constraint "
                        "are permitted on disjuncts of a disjunction "
                        f"transformed by {self.transformation_name}."
                    )
                found = True
            if not found:
                # probably an error? It's trivial in any case.
                raise GDP_Error(
                    "Empty disjunct on disjunction transformed by "
                    f"{self.transformation_name}."
                )

    def _generate_cuts(self, disj, tree):
        num_cuts = self._config.num_cuts
        given_cuts = 0

        # Bijectively label the vars as I find them since I need a dummy variable zero. (
        # x_0 is always treated as 1). This can probably be eliminated later
        idx_to_var = {0: None}
        var_to_idx = ComponentMap()
        coef = {}  # coef[(k, j)] = d_k^j
        Jm = {0}  # {k | \forall t d_k^t < 0} \cup {0}
        Jp = set()  # {k | \exists t d_k^t > 0}
        disjunct_idx = 1

        # Preprocess
        visitor = LinearRepnVisitor(
            {}, var_recorder=OrderedVarRecorder({}, {}, SortComponents.deterministic)
        )
        for d in tree.children(disj):
            con = next(d.component_data_objects(Constraint))
            repn = visitor.walk_expression(con.body)
            if repn.nonlinear:  # TODO move this?
                raise GDP_Error(
                    f"Disjunction transformed by {self.transformation_name} "
                    "must not have a nonlinear constraint."
                )
            # standardize form to dx >= d0, d0 = 1
            # TODO: presently we are assuming the RHS is all > 0 (for >= constraints);
            # this will need to be eliminated later (see doc from connor)
            # note: repn.multiplier is always 1 when obtained from LinearRepnVisitor
            multiplier = 1
            if con.ub is not None:
                if con.lb is not None:
                    raise GDP_Error(
                        "Equality constraint is not permitted in "
                        f"{self.transformation_name} transformation."
                    )
                else:
                    lb = repn.constant - con.ub
                    multiplier *= -1
            else:
                lb = con.lb - repn.constant
            if lb <= 0:
                raise GDP_Error(
                    "TODO: we will need to do something painful to handle this"
                )
            multiplier /= lb

            for v, c in repn.linear.items():
                if v not in var_to_idx:
                    idx = len(idx_to_var)
                    idx_to_var[idx] = v
                    var_to_idx[v] = idx
                else:
                    idx = var_to_idx[v]

                if c * multiplier > 0:
                    Jp.add(idx)
                    Jm.discard(idx)
                elif c * multiplier < 0:  # can this be zero?
                    if idx not in Jp:
                        Jm.add(idx)

                coef[(idx, disjunct_idx)] = c * multiplier

            disjunct_idx += 1

        # Additional preprocessing: mixed-sign lemma and sparse positive intersections
        # lemma (from Connor)

        # Mixed-sign lemma: turn certain negative numbers to zero without changing the
        # convex hull of disjunction


        # Sparse positive intersetions: eliminate all-negatives disjunctions (they would
        # be empty), then 
        

        # First: NEEC cut

        # For now, and probably for later, we will attempt to avoid taking the logarithm;
        # i.e., we are currently working on the set S_0^# instead of D^#
        breakpoint()
        alpha = {}
        for j in Jp:
            alpha[j] = coef[j, j]
        for k in Jm:
            alpha[k] = -1 * min([abs(coef[k, j]) for j in Jp])

        # is this what it should be?
        assert alpha[0] == -1

        while given_cuts < num_cuts:
            # Convert alpha to a cut and return it

            # Use the auxiliary graph algorithm to proceed from alpha to alpha_next
            return alpha
