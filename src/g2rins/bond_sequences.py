# (C) 2026 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""User-facing attachment-site sequences derived from finalized unit topology."""

from functools import lru_cache


HYDROGEN_TERMINATION_POLICY = "implicit/v1"


@lru_cache(maxsize=4096)
def _is_plain_hydrogen_cap(psmiles):
    """Recognize only a neutral, non-isotopic H bonded to one mapped port."""
    if not isinstance(psmiles, str) or not psmiles:
        return False
    from rdkit import Chem

    parameters = Chem.SmilesParserParams()
    parameters.removeHs = False
    mol = Chem.MolFromSmiles(psmiles, parameters)
    if mol is None or mol.GetNumAtoms() != 2 or mol.GetNumBonds() != 1:
        return False
    atoms = list(mol.GetAtoms())
    return (
        sorted(atom.GetAtomicNum() for atom in atoms) == [0, 1]
        and all(atom.GetIsotope() == 0 and atom.GetFormalCharge() == 0
                and atom.GetNumRadicalElectrons() == 0 for atom in atoms)
        and any(atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() > 0 for atom in atoms)
        and mol.GetBondWithIdx(0).GetBondType() == Chem.BondType.SINGLE
    )


def normalize_hydrogen_termini(topology, unit_info):
    """Project explicit ordinary H termini to implicit valence hydrogen.

    Only singly bonded, degree-one T units with canonical H-only chemistry
    are omitted. Isotopic/charged hydrogen, initiators, repeat units, isolated
    hydrogen and H-H molecules remain explicit. This never changes sampling,
    atom graphs, unit/contact counts or molecular weights. Surviving occurrence
    IDs are compacted without changing growth order; existing roots survive.
    """
    units = dict(topology["occurrences"])
    hydrogen_units = {
        unit for unit in set(units.values())
        if _is_plain_hydrogen_cap(unit_info.get(unit, {}).get("psmiles"))
    }
    candidates = {oid for oid, unit in units.items()
                  if unit.startswith("T") and unit in hydrogen_units}
    if not candidates:
        return topology
    adjacency = {oid: [] for oid in candidates}
    for edge in topology["edges"]:
        left, _ls, right, _rs, _bond, _aromatic = edge
        if left in adjacency:
            adjacency[left].append((right, edge))
        if right in adjacency:
            adjacency[right].append((left, edge))
    removed = {
        oid for oid, neighbors in adjacency.items()
        if len(neighbors) == 1
        and units[neighbors[0][0]] not in hydrogen_units
        and neighbors[0][1][4:] == ["1", False]
    }
    if not removed:
        return topology
    surviving = [(oid, unit) for oid, unit in topology["occurrences"] if oid not in removed]
    remap = {oid: index for index, (oid, _unit) in enumerate(surviving)}
    edges = [
        [remap[left], ls, remap[right], rs, bond, aromatic]
        for left, ls, right, rs, bond, aromatic in topology["edges"]
        if left not in removed and right not in removed
    ]
    components = []
    for component in topology["components"]:
        members = [oid for oid in component["occurrence_ids"] if oid not in removed]
        if members:
            root = component["root"]
            if root in removed:
                root = min(members)
            components.append({"root": remap[root],
                               "occurrence_ids": sorted(remap[oid] for oid in members)})
    components.sort(key=lambda component: component["root"])
    return {
        "schema": topology["schema"],
        "occurrences": [[index, unit] for index, (_oid, unit) in enumerate(surviving)],
        "edges": edges,
        "components": components,
    }


def _validate_token_budget(max_tokens):
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")


def _rooted_bond_sequence_forest(topology):
    """Orient finalized edges, rejecting cycles before any sequence rendering."""
    if not isinstance(topology, dict) or topology.get("schema") != "bond-sequence-topology/v1":
        raise ValueError("Expected a bond-sequence-topology/v1 record")
    units = {}
    for occurrence_id, unit_id in topology["occurrences"]:
        if occurrence_id in units or not isinstance(unit_id, str) or not unit_id:
            raise ValueError("Invalid bond-sequence occurrences")
        units[occurrence_id] = unit_id
    adjacency = {occurrence_id: [] for occurrence_id in units}
    for edge_id, (left, left_site, right, right_site, _bond, _aromatic) in enumerate(topology["edges"]):
        adjacency[left].append((right, left_site, right_site, edge_id))
        adjacency[right].append((left, right_site, left_site, edge_id))
    for neighbors in adjacency.values():
        neighbors.sort(key=lambda edge: (edge[1], edge[0], edge[2], edge[3]))

    # Build an oriented spanning forest first, rejecting closure edges before
    # expanding any paths. Use iterative traversals for long polymer chains.
    children = {}
    visited = set()
    roots = []
    for component in topology["components"]:
        root = component["root"]
        if root not in units or root in visited:
            raise ValueError("Invalid bond-sequence component root")
        roots.append(root)
        visited.add(root)
        members = {root}
        stack = [(root, None)]
        while stack:
            current, parent_edge = stack.pop()
            children[current] = []
            for neighbor, outgoing, incoming, edge_id in adjacency[current]:
                if edge_id == parent_edge:
                    continue
                if neighbor in visited:
                    raise ValueError("Cyclic bond sequences require bond_sequence_format='compact'")
                visited.add(neighbor)
                members.add(neighbor)
                children[current].append((neighbor, outgoing, incoming))
                stack.append((neighbor, edge_id))
        if members != set(component["occurrence_ids"]):
            raise ValueError("Bond-sequence components do not match connectivity")
    if visited != set(units):
        raise ValueError("Bond-sequence components must cover every occurrence")
    return units, children, roots


def render_bond_sequence_paths(topology, *, max_tokens=1_000_000):
    """Render a tree/forest as root-to-leaf lists of ``incoming.unit.outgoing``.

    Sites are the map labels of the canonical unit pSMILES, not atom indices.
    ``null`` denotes a missing path endpoint (not an unreacted chemical site).
    Shared trunks repeat across paths. This is a view, not an exact graph key:
    occurrence identity and bond order/aromaticity are deliberately not encoded.
    Cycles, self-links and parallel edges require the ``compact`` format.
    ``max_tokens`` bounds expanded output, including repeated shared trunks.
    """
    _validate_token_budget(max_tokens)
    units, children, roots = _rooted_bond_sequence_forest(topology)

    paths = []
    token_count = 0
    for root in roots:
        path = []
        # Each frame retains the incoming site and the next child to visit.
        stack = [(root, None, 0)]
        while stack:
            current, incoming, child_index = stack[-1]
            branches = children[current]
            if not branches:
                token_count += len(path) + 1
                if token_count > max_tokens:
                    raise ValueError("Bond-sequence path expansion exceeds max_tokens; use compact format")
                paths.append(path + [f"{incoming or 'null'}.{units[current]}.null"])
                stack.pop()
                if path:
                    path.pop()
            elif child_index < len(branches):
                neighbor, outgoing, child_incoming = branches[child_index]
                stack[-1] = (current, incoming, child_index + 1)
                path.append(f"{incoming or 'null'}.{units[current]}.{outgoing}")
                stack.append((neighbor, child_incoming, 0))
            else:
                stack.pop()
                if path:
                    path.pop()
    return paths


def render_bond_sequence_branches(topology, *, max_tokens=1_000_000):
    """Return ``(segments, parents)`` without repeating root-to-junction trunks.

    Each component starts with a singleton root, as in legacy ``sequences``.
    Every other segment starts with a *reference* to an existing junction,
    followed by newly introduced occurrences. ``parents[i] = [segment, position]``
    identifies that reference (zero-based); root singletons have parent ``None``.
    Thus identical unit labels never imply identity by common-prefix matching.

    At each junction the earliest surviving occurrence ID is continued first;
    other children start separate segments. This growth-order convention is
    deterministic, not a chemical backbone-identification heuristic. Junctions
    inside a segment keep their continuation site; the branch's first token
    supplies the outgoing site for that branch. ``null`` means no preceding or
    following unit *in this segment*, not a chemical termination/unreacted site.

    Together the lists preserve tree connectivity and occupied endpoint sites,
    but not bond order/aromaticity or original occurrence numbering. Cycles
    require compact format. Both traversals are iterative and storage is O(V+E).
    """
    _validate_token_budget(max_tokens)
    units, children, roots = _rooted_bond_sequence_forest(topology)
    children = {
        occurrence: sorted(edges, key=lambda edge: (edge[0], edge[1], edge[2]))
        for occurrence, edges in children.items()
    }
    segments, parents = [], []
    locations = {}
    token_count = 0

    def append_token(segment, incoming, occurrence, outgoing):
        nonlocal token_count
        token_count += 1
        if token_count > max_tokens:
            raise ValueError("Bond-sequence branch rendering exceeds max_tokens; use compact format")
        segment.append(f"{incoming or 'null'}.{units[occurrence]}.{outgoing or 'null'}")

    for root in roots:
        locations[root] = [len(segments), 0]
        singleton = []
        append_token(singleton, None, root, None)
        segments.append(singleton)
        parents.append(None)
        pending = [(root, None, edge) for edge in reversed(children[root])]
        while pending:
            junction, junction_incoming, (current, outgoing, incoming) = pending.pop()
            segment_index = len(segments)
            segment = []
            parents.append(list(locations[junction]))
            segments.append(segment)
            append_token(segment, junction_incoming, junction, outgoing)
            while True:
                locations[current] = [segment_index, len(segment)]
                branches = children[current]
                if not branches:
                    append_token(segment, incoming, current, None)
                    break
                next_occurrence, next_outgoing, next_incoming = branches[0]
                append_token(segment, incoming, current, next_outgoing)
                pending.extend(
                    (current, incoming, edge) for edge in reversed(branches[1:])
                )
                current, incoming = next_occurrence, next_incoming
    return segments, parents


def bond_sequence_branch_metadata(metadata):
    """Describe explicit junction references without changing coverage semantics."""
    return {
        **metadata,
        "schema": "bond-sequence-branches/v1",
        "format": "branches",
        "token_format": "incoming.unit.outgoing; null denotes a segment endpoint",
        "parent_references": "bond_sequence_parents[chain][segment]: null for a root, otherwise [parent_segment, junction_position] (zero-based); first token references that junction",
        "continuation_policy": "earliest-surviving-occurrence-first/v1",
        "connectivity_preserved": True,
        "topology_preserved": False,
        "omitted_edge_attributes": ["bond_type", "aromatic"],
        "shared_trunks": "stored-once; only junction references repeat",
    }


def bond_sequence_path_metadata(metadata):
    """Describe the path view without changing population/coverage semantics."""
    return {
        **metadata,
        "schema": "bond-sequence-paths/v1",
        "format": "paths",
        "token_format": "incoming.unit.outgoing; null denotes a path endpoint",
        "topology_preserved": False,
        "shared_trunks": "repeated-per-root-to-leaf-path",
    }