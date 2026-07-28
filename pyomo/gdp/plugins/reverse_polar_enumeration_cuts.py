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
        # b = disj.parent_block()
        # if not hasattr(b, '_reverse_polar_enumeration_cuts'):
        #     b._reverse_polar_enumeration_cuts = Constraint(NonNegativeIntegers)
        # b._reverse_polar_enumeration_cuts[len(b._reverse_polar_enumeration_cuts)] = (
        #     expr >= 1
        # )
        if disj in instance.private_data().disjunction_constraints_map:
            con = instance.private_data().disjunction_constraints_map[disj]
        else:
            con = Constraint(NonNegativeIntegers)
            xf_block.add_component(unique_component_name(xf_block, disj.name), con)
            instance.private_data().disjunction_constraints_map[disj] = con
        con[len(con)] = expr >= 1
        print(f"Added a cut: {str(expr >= 1)}")

    def _near_match(self, d1, d2):
        for k, v in d1.items():
            if not abs(v - d2[k]) < EPS:
                return False
        return True

    def _generate_cuts(self, instance, xf_block, disj, tree):
        num_cuts = self._config.num_cuts

        # Bijectively label the vars as I find them since I need a dummy variable zero. (
        # x_0 is always treated as 1). This can probably be eliminated later
        idx_to_var = {0: None}
        var_to_idx = ComponentMap()
        coef = {}  # coef[(k, t)] = d_k^t
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
                    Jp.add(idx)
                    Jm.discard(idx)
                # NOTE: a variable can be neither Jp nor Jm at this
                # stage, but this will put such vars in Jm since we
                # aren't catching zero coefficients. We handle this
                # below
                elif c < 0:
                    if idx not in Jp:
                        Jm.add(idx)

                coef[(idx, disjunct_idx)] = c

            disjunct_idx += 1

        # Fill in dummy/default entries. Eliminate this later to save effort when sparse
        for t in range(1, disjunct_idx):
            coef[0, t] = 1  # or -1?
            for j in range(1, len(idx_to_var)):
                if (j, t) not in coef:
                    coef[j, t] = 0
                    if j in Jm:
                        # TODO verify this is actually valid. In this
                        # case, we effectively delete this variable
                        # completely from the disjunction. I believe
                        # this should be correct in light of Connor's
                        # mixed-sign variables lemma. In any case the
                        # variable certainly cannot go into Jm.
                        # TODO: This _is_ the only way zero coefficients
                        # can arise (ie, they don't show up in the
                        # repn), right?
                        Jm.discard(j)
        # Additional preprocessing (sparse positive intersections lemma
        # from Connor): Eliminate all-negatives disjuncts (they would be
        # empty), and perform various alterations to the
        # coefficients. In the end d[k, t] has a block form consisting
        # of a square diagonal matrix (possibly taller than the
        # original, since the index of t is replaced with Jp), and below
        # that a block of all negative values corresponding to variables
        # in Jm.
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
        debug_vars(Jp, Jm, idx_to_var, visitor.var_map)
        # for k in Jm:
        #     for j in Jp:
        #         print(f"coef[{k},{j}]={coef[k,j]} (supposed to be <0; this is {("true" if coef[k, j]<0 else "false")})")
        # First: NEEC cut

        # For later, we will attempt to avoid taking the logarithm;
        # i.e., we will try to work on the set S_0^# instead of D^#
        alpha = {}
        for j in Jp:
            alpha[j] = coef[j, j]
        for k in Jm:
            alpha[k] = -1 * min([abs(coef[k, j]) for j in Jp])

        # this is correct, right?
        assert alpha[0] == -1

        cost = {}
        for j in Jp:
            for k in Jm:
                print(f"here coef[{k}, {j}]={coef[k,j]}, coef[{j}, {j}]={coef[j,j]}")
                cost[j, k] = log(abs(coef[k, j]) / coef[j, j])
                print(f"cost[{j}, {k}]={cost[j, k]}")
        delta = {k: log(abs(v)) for k, v in alpha.items()}
        # vertices already added
        used_list = [delta]
        # vertices to construct G_delta from
        vertex_queue = [delta]
        self._add_cut(
            instance, xf_block, delta, disj, idx_to_var, visitor.var_map, Jp, Jm
        )
        # breakpoint()
        print("=====================")
        added_cuts = 1

        while vertex_queue and (not num_cuts or added_cuts <= num_cuts):
            print(f"before popping, {len(vertex_queue)=}")
            dstar = vertex_queue.pop(0)
            print(f"working form vertex {dstar=}")
            print("calling _enumerate_graph_cuts")
            # for cut in self._enumerate_graph_cuts_exhaustive_debug(dstar, Jp, Jm, cost):
            for cut in self._enumerate_graph_cuts(dstar, Jp, Jm, cost):
                print(f"Using cut: {cut}")
                l = min(
                    [
                        cost[j, k] - dstar[k] + dstar[j]
                        for j in Jp
                        for k in Jm
                        if k in cut and j not in cut
                    ]
                )
                print(f"Calculated lambda*={l}")
                d_candidate = {k: (v + l if k in cut else v) for k, v in dstar.items()}
                # floating point...
                found_near_match = False
                for d in used_list:
                    if self._near_match(d_candidate, d):
                        print("was near match")
                        found_near_match = True
                        break
                if found_near_match:
                    continue
                vertex_queue.append(d_candidate)
                print(f"after appending, {len(vertex_queue)=}")
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
                used_list.append(d_candidate)
                added_cuts += 1
                # breakpoint()
                print("=====================")
            print("finished call to _enumerate_graph_cuts")
            print(f"after enumerate_graph_cuts, {len(vertex_queue)=}")
            # breakpoint()
            print("=====================")

    # generator yielding graph cuts (we yield the X sets) in G_dstar
    def _enumerate_graph_cuts(self, dstar, Jp, Jm, cost):
        G_dstar = nx.DiGraph()
        G_dstar.add_nodes_from(Jp)
        G_dstar.add_nodes_from(Jm)
        for j in Jp:
            for k in Jm:
                if abs(dstar[k] - dstar[j] - cost[j, k]) < EPS:
                    G_dstar.add_edge(k, j)

        # Ternary Booleans in honor of George Boole. True means in X,
        # False means in X_bar, None means not yet assigned
        for k0 in Jm:
            if k0 == 0:
                continue
            print(f"iterating {k0=}")
            label = {}
            for i in Jm:
                if i < k0:
                    label[i] = False
                elif i == k0:
                    label[i] = True
                else:
                    label[i] = None
            for i in Jp:
                label[i] = None
            yield from self._branch(G_dstar, label, Jp, Jm)
        return

    def _branch(self, G_dstar, label, Jp, Jm):
        print("calling branch()")
        print(f"here {G_dstar.edges=}")
        print(f"here initially {label=}")
        # Here we will mostly deal with G_dstar[N_0 \ X]. Also we only
        # check for undirected paths.
        G_working = G_dstar.to_undirected(as_view=False)
        for k, v in label.items():
            if v:
                G_working.remove_node(k)
        # Forcing rules that necessarily put certain nodes in X
        for k in Jm:
            if label[k]:
                for j in G_dstar.successors(k):
                    print("did forcing rule 1")
                    label[j] = True
                    if j in G_working.nodes:
                        G_working.remove_node(j)
        for k in Jm:
            if label[k] is None:
                for j in G_dstar.successors(k):
                    if label[j]:
                        if not nx.has_path(G_working, k, 0):
                            print("did forcing rule 2")
                            label[k] = True
                            if k in G_working.nodes:
                                G_working.remove_node(k)
                        break

        for j in Jp:
            if label[j]:
                # these are in Jm only
                for k in G_dstar.predecessors(j):
                    if label[k] is None:
                        if nx.has_path(G_working, k, 0):
                            # I will trust that this is never exponential time
                            print(f"had path to 0, doing a double branch for {k=}")
                            l1 = label.copy()
                            l1[k] = True
                            l2 = label.copy()
                            l2[k] = False
                            yield from self._branch(G_dstar, l1, Jp, Jm)
                            yield from self._branch(G_dstar, l2, Jp, Jm)
                            return
                        else:
                            l1 = label.copy()
                            l1[k] = True
                            yield from self._branch(G_dstar, l1, Jp, Jm)
                            return
        # no unassigned neighbors to any j in X intersect Jp
        for k in label.keys():
            if label[k] is None:
                label[k] = False
        cut = set()
        for k in label.keys():
            if label[k]:
                cut.add(k)
        if self._validate_cut(cut, G_dstar, Jp, Jm):
            print(f"Yielding valid cut {cut}")
            yield cut
            return
        print(f"Cut {cut} failed validation")
        return

    def _validate_cut(self, cut, G_dstar, Jp, Jm):
        cut_complement = set()
        G_X = G_dstar.to_undirected(as_view=False)
        G_Xbar = G_dstar.to_undirected(as_view=False)
        for j in itertools.chain(Jp, Jm):
            if j in cut:
                G_Xbar.remove_node(j)
            else:
                G_X.remove_node(j)
                cut_complement.add(j)
        # (1) and (3) are known to be able to fail
        # (3) X intersects Jm and Xbar intersects Jp
        if Jm.isdisjoint(cut) or Jp.isdisjoint(cut_complement):
            print("failed: X disjoint from Jm or Xbar disjoint from Jp")
            return False
        # (1) X and Xbar induce connected subgraphs of G_dstar
        if not nx.is_connected(G_Xbar) or not nx.is_connected(G_X):
            print("failed: G[X] or G[Xbar] not connected")
            return False

        # (2) No directed edges run from X to Xbar
        # This is probably not a possible failure case, but let's check just in case.
        for src, dst in G_dstar.edges:
            if src in cut and dst in cut_complement:
                print("failed: there was an edge of G going from X to Xbar")
                return False
        return True

    def _enumerate_graph_cuts_exhaustive_debug(self, dstar, Jp, Jm, cost):
        G_dstar = nx.DiGraph()
        G_dstar.add_nodes_from(Jp)
        G_dstar.add_nodes_from(Jm)
        for j in Jp:
            for k in Jm:
                if abs(dstar[k] - dstar[j] - cost[j, k]) < EPS:
                    G_dstar.add_edge(k, j)
        nodes = list(Jp.union(Jm))
        # power set
        for cut in itertools.chain.from_iterable(
            itertools.combinations(nodes, r) for r in range(len(nodes) + 1)
        ):
            if 0 in cut:
                continue
            if self._validate_cut(cut, G_dstar, Jp, Jm):
                yield cut
        return


def debug_vars(Jp, Jm, idx_to_var, var_map):
    print()
    for j in Jp:
        print(f"Index {j} (Jp) corresponds to {var_map[idx_to_var[j]].name}")
    for k in Jm:
        if k != 0:
            print(f"Index {k} (Jm) corresponds to {var_map[idx_to_var[k]].name}")
        else:
            print("Index 0 (Jm) is the dummy variable")


def get_constraint(transformed_block, disjunction):
    if disjunction in transformed_block.private_data().disjunction_constraints_map:
        return transformed_block.private_data().disjunction_constraints_map[disjunction]
    else:
        raise ValueError(
            f"Disjunction {disjunction} was not used for cut generation by "
            f"a call to gdp.reverse_polar_enumeration_cuts on model {transformed_block}"
        )
