# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import logging
import itertools
from pyomo.core.base import Transformation, TransformationFactory, NonNegativeIntegers
from pyomo.core.base.component import ActiveComponent
from pyomo.core.base.block import SubclassOf
from pyomo.core.util import target_list
from pyomo.core.base.enums import SortComponents
from pyomo.common.autoslots import AutoSlots
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.modeling import unique_component_name
from pyomo.common.enums import Enum
from pyomo.core import Block, Constraint
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import get_gdp_tree
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.repn.util import OrderedVarRecorder
from math import log, exp

from pyomo.common.dependencies import networkx as nx, networkx_available


logger = logging.getLogger(__name__)
EPS = 1e-6


class _ReversePolarEnumerationCutsData(AutoSlots.Mixin):
    __slots__ = "disjunction_constraints_map"

    def __init__(self):
        self.disjunction_constraints_map = ComponentMap()


Block.register_private_data_initializer(_ReversePolarEnumerationCutsData)


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

    transformation_name = 'reverse_polar_enumeration_cuts'
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
            default=None,
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
        if not networkx_available:
            raise GDP_Error("Networkx is required for this transformation.")
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

        xf_block = Block()
        instance.add_component(
            unique_component_name(instance, "_reverse_polar_enumeration_cuts"), xf_block
        )
        tree = get_gdp_tree(targets, instance)
        for t in tree.reverse_topological_sort():
            if t.ctype is Disjunction:
                self._validate_disjunction(t, tree)
                self._generate_cuts(instance, xf_block, t, tree)

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
                    f"{self.transformation_name} - no cuts are possible."
                )

    def _add_cut(self, instance, xf_block, delta, disj, idx_to_var, var_map, Jp, Jm):
        expr = 0
        for k, v in delta.items():
            if k != 0:
                var = var_map[idx_to_var[k]]
                if k in Jp:
                    alpha = exp(delta[k])
                elif k in Jm:
                    alpha = -exp(delta[k])
                expr += alpha * var
        if disj in instance.private_data().disjunction_constraints_map:
            con = instance.private_data().disjunction_constraints_map[disj]
        else:
            con = Constraint(NonNegativeIntegers)
            xf_block.add_component(unique_component_name(xf_block, disj.name), con)
            instance.private_data().disjunction_constraints_map[disj] = con
        con[len(con)] = expr >= 1
        # print(f"Added a cut: {str(expr >= 1)}\n=========================")

    def _near_match(self, d1, d2):
        for k, v in d1.items():
            if not abs(v - d2[k]) < EPS:
                return False
        return True

    def _generate_cuts(self, instance, xf_block, disj, tree):
        num_cuts = self._config.num_cuts
        if num_cuts == 0:
            return
        num_skip = self._config.num_skip

        # Bijectively label the vars as I find them since I need a dummy
        # variable zero. (x_0 is always treated as 1). This can
        # probably be eliminated later
        idx_to_var = {0: None}
        var_to_idx = ComponentMap()
        coef = {}  # coef[(k, t)] = d_k^t
        # These should have fast lookup, but they also need to have
        # stable iteration order for testing and consistency, so I will
        # use a dict to None instead of a set or list
        Jm = {0: None}  # {k | \forall t d_k^t < 0} \cup {0}
        Jp = {}  # {k | \exists t d_k^t > 0}
        disjunct_idx = 1

        # Preprocess
        visitor = LinearRepnVisitor(
            {}, var_recorder=OrderedVarRecorder({}, {}, SortComponents.deterministic)
        )
        for d in tree.children(disj):
            con = next(d.component_data_objects(Constraint))
            repn = visitor.walk_expression(con.body)
            if repn.nonlinear:
                raise GDP_Error(
                    f"Disjunction transformed by {self.transformation_name} "
                    "must not have a nonlinear constraint."
                )
            # standardize form to dx >= d0, d0 = 1
            # NOTE: We are assuming the RHS is all > 0 (for >= constraints);
            # this will need to be handled before passing to this transformation.

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
                    "Nonpositive RHS is not valid for reverse polar cut generator."
                )
            multiplier /= lb

            for v, c in repn.linear.items():
                # here v is the var id
                c = c * multiplier
                if v not in var_to_idx:
                    idx = len(idx_to_var)
                    idx_to_var[idx] = v
                    var_to_idx[v] = idx
                else:
                    idx = var_to_idx[v]

                if c > 0:
                    Jp[idx] = None
                    Jm.pop(idx, None)
                # NOTE: a variable can be neither Jp nor Jm at this
                # stage, but this will put such vars in Jm since we
                # aren't catching zero coefficients. We handle this
                # below
                elif c < 0:
                    if idx not in Jp:
                        Jm[idx] = None

                coef[(idx, disjunct_idx)] = c

            disjunct_idx += 1

        # Keep these sorted. Only Jp could fail to be here (since items
        # can be added late if they were initially in Jm).
        Jp = dict(sorted(Jp.items()))

        # Fill in default entries. Eliminate this later to save effort when sparse
        for t in range(1, disjunct_idx):
            coef[0, t] = 1
            for j in range(1, len(idx_to_var)):
                if (j, t) not in coef:
                    coef[j, t] = 0
                    if j in Jm:
                        # In this case, we effectively delete this
                        # variable completely from the disjunction. It
                        # is never necessary to include it on a
                        # generated cut.

                        # TODO: This _is_ the only way zero coefficients
                        # can arise (ie, they never show up in the
                        # repn), right?
                        Jm.pop(j, None)
        # Preprocessing (sparse positive intersections lemma from
        # Connor): Recreate the disjunction to have one disjunct for
        # each Jp variable, performing various alterations to the
        # coefficients. In the end d_k^t has a block form consisting of
        # a square diagonal matrix of size |Jp|x|Jp| with positive
        # diagonal values, and below that a block of all negative values
        # corresponding to variables in Jm.
        coef_new = {}
        for j in Jp:
            for k in Jp:
                if j == k:
                    coef_new[j, j] = max([coef[j, t] for t in range(1, disjunct_idx)])
                else:
                    coef_new[k, j] = 0
            for k in Jm:
                coef_new[k, j] = (
                    -min(
                        [
                            abs(coef[k, t]) / coef[j, t]
                            for t in range(1, disjunct_idx)
                            if coef[j, t] > 0
                        ]
                    )
                    * coef_new[j, j]
                )
        coef = coef_new
        # debug_vars(Jp, Jm, idx_to_var, visitor.var_map)

        # First: NEEC cut
        # TODO: remove use of logarithms throughout
        delta = {}
        for j in Jp:
            delta[j] = log(coef[j, j])
        for k in Jm:
            delta[k] = log(min([abs(coef[k, j]) for j in Jp]))

        if not num_skip:
            self._add_cut(
                instance, xf_block, delta, disj, idx_to_var, visitor.var_map, Jp, Jm
            )
        # breakpoint()
        # print("=====================")
        added_cuts = 1
        if num_cuts == 1:
            return

        cost = {}
        for j in Jp:
            for k in Jm:
                # print(f"here coef[{k}, {j}]={coef[k,j]}, coef[{j}, {j}]={coef[j,j]}")
                cost[j, k] = log(abs(coef[k, j]) / coef[j, j])
                # print(f"cost[{j}, {k}]={cost[j, k]}")

        # State machine: perform breadth-first search on D^# by using
        # the properties of the auxiliary graph G_dstar at each vertex
        # dstar in D^# to find vertices adjacent to dstar, checking each
        # against used_list in case they are not new.
        vertex_queue = [delta]
        used_list = [delta]
        # indexes into Jm
        k0 = 0
        # tuples of lists: (X, Xbar)
        cuts_stack = []

        class Targets(Enum):
            get_vertex = 0
            start_graph_cut = 1
            graph_cuts_inner = 2
            mip_cut = 3

        jump_target = Targets.get_vertex

        while True:
            match jump_target:
                case Targets.get_vertex:
                    if not vertex_queue:
                        return  # all cuts generated
                    dstar = vertex_queue.pop(0)
                    G_dstar = nx.Graph()
                    G_dstar.add_nodes_from(Jp)
                    G_dstar.add_nodes_from(Jm)
                    for j in Jp:
                        for k in Jm:
                            if abs(dstar[k] - dstar[j] - cost[j, k]) < EPS:
                                G_dstar.add_edge(k, j)

                    jump_target = Targets.start_graph_cut
                    continue

                case Targets.start_graph_cut:
                    if k0 == len(Jm) - 1:
                        k0 = 0
                        jump_target = Targets.get_vertex
                        continue
                    k0 = k0 + 1  # skip 0
                    Xbar = []
                    it = iter(Jm)
                    for i in range(k0):
                        Xbar.append(next(it))
                    X = [next(it)]
                    cuts_stack.append((X, Xbar))
                    jump_target = Targets.graph_cuts_inner
                    continue

                case Targets.graph_cuts_inner:
                    X, Xbar = cuts_stack.pop(-1)
                    # Forcing rules that necessarily put certain nodes in X
                    for k in Jm:
                        if k in X:
                            for j in G_dstar.neighbors(k):
                                # these are in Jp only
                                X.append(j)
                                # print("did forcing rule 1")
                    # Going forward we often need access to G_dstar[N_0 \ X]
                    G_working = G_dstar.copy()
                    G_working.remove_nodes_from(X)
                    for k in Jm:
                        if k not in X and k not in Xbar:
                            for j in G_dstar.neighbors(k):
                                if j in X and not nx.has_path(G_working, k, 0):
                                    X.append(k)
                                    if k in G_working.nodes:
                                        G_working.remove_node(k)
                                    # print("did forcing rule 2")
                                    break
                    for j in Jp:
                        if j in X:
                            done = False
                            for k in G_dstar.neighbors(j):
                                # these are in Jm only
                                if k not in X and k not in Xbar:
                                    if nx.has_path(G_working, k, 0):
                                        # print(
                                        #     f"had path to 0; double branch for {k=}"
                                        # )
                                        cuts_stack.append((X, Xbar + [k]))
                                        cuts_stack.append((X + [k], Xbar))
                                    else:
                                        # print(
                                        #     f"no path to 0; single branch for {k=}"
                                        # )
                                        cuts_stack.append((X + [k], Xbar))
                                    done = True
                                    break
                            if done:
                                # we will see the other neighbors on
                                # subsequent iterations
                                continue  # jump_target is still graph_cuts_inner
                    # from here on any remaining elements of Jp and Jm
                    # are treated as part of Xbar
                    if self._validate_cut(X, G_dstar, G_working, Jp, Jm):
                        jump_target = Targets.mip_cut
                        continue
                    if not cuts_stack:
                        jump_target = Targets.start_graph_cut
                    # otherwise return to graph_cuts_inner
                    continue

                # This could just be inlined to underneath
                # `if self._validate_cut(...)` but it's conceptually
                # distinct so let's maintain some semblance of order
                # by moving it here.
                case Targets.mip_cut:
                    # occurs regardless of how we exit this
                    jump_target = (
                        Targets.graph_cuts_inner
                        if cuts_stack
                        else Targets.start_graph_cut
                    )

                    lstar = min(
                        [
                            cost[j, k] - dstar[k] + dstar[j]
                            for j in Jp
                            for k in Jm
                            if k in X and j not in X
                        ]
                    )
                    # print(f"Calculated lambda*={lstar}")
                    d_candidate = {
                        k: (v + lstar if k in X else v) for k, v in dstar.items()
                    }
                    # floating point...
                    done = False
                    for d in used_list:
                        if self._near_match(d_candidate, d):
                            # near match to a vertex already used: do not add cut
                            # print("was near match")
                            done = True
                            break
                    if done:
                        continue
                    vertex_queue.append(d_candidate)
                    used_list.append(d_candidate)
                    added_cuts += 1
                    if added_cuts > num_skip:
                        self._add_cut(
                            instance,
                            xf_block,
                            d_candidate,
                            disj,
                            idx_to_var,
                            visitor.var_map,
                            Jp,
                            Jm,
                        )
                    if added_cuts == num_cuts:
                        # early termination
                        return
                    continue

    def _validate_cut(self, cut, G_dstar, G_Xbar, Jp, Jm):
        # print(f"validating cut {cut}")
        G_X = G_dstar.copy()
        G_X.remove_nodes_from(G_Xbar.nodes)
        # (1) and (3) are known to be able to fail
        # (3) X intersects Jm and Xbar intersects Jp
        if set(Jm).isdisjoint(set(cut)) or set(Jp).isdisjoint(set(G_Xbar.nodes)):
            # print("failed: X disjoint from Jm or Xbar disjoint from Jp")
            return False
        # (1) X and Xbar induce connected subgraphs of G_dstar
        if not nx.is_connected(G_Xbar) or not nx.is_connected(G_X):
            # print("failed: G[X] or G[Xbar] not connected")
            return False

        # (2) No directed edges run from X to Xbar
        # This is probably not a possible failure case, but let's check just in case.
        for src, dst in G_dstar.edges:
            if src in cut and src in Jm and dst not in cut and dst in Jp:
                # print("failed: there was an edge of G going from X to Xbar")
                return False
        return True


def debug_vars(Jp, Jm, idx_to_var, var_map):
    print()
    for j in Jp:
        print(f"Index {j} (Jp) corresponds to {var_map[idx_to_var[j]].name}")
    for k in Jm:
        if k == 0:
            print("Index 0 (Jm) is the dummy variable")
        else:
            print(f"Index {k} (Jm) corresponds to {var_map[idx_to_var[k]].name}")


def get_constraint(transformed_block, disjunction):
    if disjunction in transformed_block.private_data().disjunction_constraints_map:
        return transformed_block.private_data().disjunction_constraints_map[disjunction]
    else:
        raise ValueError(
            f"Disjunction {disjunction} was not used for cut generation by "
            f"a call to gdp.reverse_polar_enumeration_cuts on model {transformed_block}"
        )
