# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

import threading
import warnings
from collections import deque

# RDKit's SMILES writer recurses per atom in C++; the default stack overflows
# (hard 0xC00000FD on Windows) near 3k atoms. Empirically a 2000-atom linear
# chain is safe on the default stack and 4000 crashes, so gate with margin.
_BIG_STACK_ATOM_THRESHOLD = 2000
_BIG_STACK_SIZE = 0x0FFFF000  # just under CPython's 256 MiB Windows cap
_RANDOM_SMILES_ATTEMPTS = 16
_FAST_RING_ATOM_THRESHOLD = 10000
_FAST_RING_EXCESS_THRESHOLD = 1000
_PREFER_DIRECT_SMILES_PROPERTY = "_g2rinsPreferDirectSmiles"

# threading.stack_size is process-global: serialize set/start/restore.
_STACK_SIZE_LOCK = threading.Lock()
# RDKit's random generator is also process-global. Keep explicit reseeding and
# randomized serialization atomic so concurrent callers cannot perturb it.
_RANDOM_SMILES_LOCK = threading.Lock()


def _apply_atom_chirality_tokens(mol, mol_graph, graph_idx_to_mol_idx, chem):
    """Apply parsed bracket-atom chirality markers to RDKit atom tags.

    Only tetrahedral forms are mapped here. More exotic symbols stay unset
    until explicitly implemented.
    """
    chiral_tag_by_token = {
        "@": chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        "@@": chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        "@TH1": chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        "@TH2": chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    }
    unsupported_tokens = set()

    for graph_idx, data in mol_graph.nodes(data=True):
        token = data.get("atom_chiral_token")
        if token is None:
            continue
        if data.get("atomic_num", 0) <= 0:
            continue

        chiral_tag = chiral_tag_by_token.get(token)
        if chiral_tag is None:
            unsupported_tokens.add(token)
            continue

        atom = mol.GetAtomWithIdx(graph_idx_to_mol_idx[graph_idx])
        atom.SetChiralTag(chiral_tag)

    if unsupported_tokens:
        warnings.warn(
            "Unsupported atom chirality tokens were ignored: "
            + ", ".join(sorted(unsupported_tokens)),
            RuntimeWarning,
            stacklevel=3,
        )


def _apply_directional_bond_tokens(
    mol,
    mol_graph,
    graph_idx_to_mol_idx,
    chem,
    suppressed_bond_keys=None,
):
    """Map slash/backslash bond tokens onto RDKit single-bond directions."""
    bond_dir_by_token = {
        "/": chem.rdchem.BondDir.ENDUPRIGHT,
        "\\": chem.rdchem.BondDir.ENDDOWNRIGHT,
    }
    if suppressed_bond_keys is None:
        suppressed_bond_keys = set()

    for u_idx, v_idx, attr in mol_graph.edges(data=True):
        if attr.get("bond_type", 1) != 1:
            continue
        if frozenset((u_idx, v_idx)) in suppressed_bond_keys:
            continue
        bond_symbol_raw = attr.get("bond_symbol_raw")
        if bond_symbol_raw not in bond_dir_by_token:
            continue

        bond = mol.GetBondBetweenAtoms(
            graph_idx_to_mol_idx[u_idx],
            graph_idx_to_mol_idx[v_idx],
        )
        if bond is None:
            continue
        bond.SetBondDir(bond_dir_by_token[bond_symbol_raw])


def _iter_incident_edge_data(mol_graph, atom_idx):
    """Yield incident edges for an atom across directed and undirected graph types."""
    if mol_graph.is_multigraph():
        if hasattr(mol_graph, "out_edges"):
            for _u_idx, v_idx, _key, edge_data in mol_graph.out_edges(atom_idx, keys=True, data=True):
                yield v_idx, edge_data
            for u_idx, _v_idx, _key, edge_data in mol_graph.in_edges(atom_idx, keys=True, data=True):
                yield u_idx, edge_data
            return

        for neighbor in mol_graph.neighbors(atom_idx):
            edge_bundle = mol_graph.get_edge_data(atom_idx, neighbor)
            if edge_bundle is None:
                continue
            for edge_data in edge_bundle.values():
                yield neighbor, edge_data
        return

    seen_neighbors = set()
    if hasattr(mol_graph, "adj"):
        for neighbor, edge_data in mol_graph.adj.get(atom_idx, {}).items():
            if neighbor in seen_neighbors:
                continue
            seen_neighbors.add(neighbor)
            yield neighbor, edge_data
    if hasattr(mol_graph, "pred"):
        for neighbor, edge_data in mol_graph.pred.get(atom_idx, {}).items():
            if neighbor in seen_neighbors:
                continue
            seen_neighbors.add(neighbor)
            yield neighbor, edge_data
    if not hasattr(mol_graph, "adj") and not hasattr(mol_graph, "pred"):
        for neighbor in mol_graph.neighbors(atom_idx):
            edge_data = mol_graph.get_edge_data(atom_idx, neighbor)
            if edge_data is None:
                continue
            if neighbor in seen_neighbors:
                continue
            seen_neighbors.add(neighbor)
            yield neighbor, edge_data


def _warn_on_ambiguous_directional_markers(mol_graph):
    """Warn on unresolved directional markers and return single-bond markers to ignore.

    Ambiguous or conflicting marker patterns are discarded (Option A) so E/Z
    stereochemistry cannot be inferred from contradictory input.
    """

    def directional_token_sets(atom_idx, partner_idx):
        tokens_by_bond = {}

        def add_token(other_idx, edge_data):
            if other_idx == partner_idx:
                return
            if edge_data.get("bond_type", 1) != 1:
                return
            token = edge_data.get("bond_symbol_raw")
            if token not in {"/", "\\"}:
                return
            bond_key = frozenset((atom_idx, other_idx))
            tokens_by_bond.setdefault(bond_key, set()).add(token)

        for other_idx, edge_data in _iter_incident_edge_data(mol_graph, atom_idx):
            add_token(other_idx, edge_data)

        return tokens_by_bond

    warned = set()
    provenance_key = "double_bond_stereo_defined"
    has_source_stereo_provenance = any(
        edge_data.get("bond_type", 1) == 2 and provenance_key in edge_data
        for _u_idx, _v_idx, edge_data in mol_graph.edges(data=True)
    )
    suppressed_bond_keys = set()
    for u_idx, v_idx, edge_data in mol_graph.edges(data=True):
        if edge_data.get("bond_type", 1) != 2:
            continue
        if has_source_stereo_provenance and not edge_data.get(provenance_key, False):
            # Provenance-aware mode: only source-defined stereogenic double bonds
            # are eligible for "stereo information lost" warnings.
            continue

        left_tokens_by_bond = directional_token_sets(u_idx, v_idx)
        right_tokens_by_bond = directional_token_sets(v_idx, u_idx)
        left_token_sets = list(left_tokens_by_bond.values())
        right_token_sets = list(right_tokens_by_bond.values())

        if not left_token_sets and not right_token_sets:
            continue

        bond_key = frozenset((u_idx, v_idx))
        if bond_key in warned:
            continue

        if not left_token_sets or not right_token_sets:
            warnings.warn(
                (
                    "Incomplete double-bond directional markers near atoms "
                    f"{u_idx}-{v_idx}; E/Z stereochemistry is left unspecified."
                ),
                RuntimeWarning,
                stacklevel=3,
            )
            warned.add(bond_key)
            continue

        if len(left_token_sets) > 1 or len(right_token_sets) > 1:
            suppressed_bond_keys.update(left_tokens_by_bond)
            suppressed_bond_keys.update(right_tokens_by_bond)
            warnings.warn(
                (
                    "Ambiguous double-bond directional markers near atoms "
                    f"{u_idx}-{v_idx}; directional stereoinformation was discarded "
                    "and E/Z stereochemistry is left unspecified."
                ),
                RuntimeWarning,
                stacklevel=3,
            )
            warned.add(bond_key)
            continue

        if any(len(token_set) > 1 for token_set in left_token_sets + right_token_sets):
            suppressed_bond_keys.update(left_tokens_by_bond)
            suppressed_bond_keys.update(right_tokens_by_bond)
            warnings.warn(
                (
                    "Conflicting double-bond directional markers near atoms "
                    f"{u_idx}-{v_idx}; directional stereoinformation was discarded "
                    "and E/Z stereochemistry is left unspecified."
                ),
                RuntimeWarning,
                stacklevel=3,
            )
            warned.add(bond_key)

    return suppressed_bond_keys


def _warn_and_collect_unresolved_directional_markers(
    mol_graph,
    strip_unresolved_directional_markers=False,
):
    """Warn on unresolved E/Z markers and return single-bond markers to ignore.

    When ``strip_unresolved_directional_markers`` is true, single-bond
    directional markers that belong to unresolved E/Z assignments are removed
    from the exported molecule so SMILES output cannot retain dangling ``/`` or
    ``\\`` tokens around unspecified double bonds.
    """

    suppressed_bond_keys = _warn_on_ambiguous_directional_markers(mol_graph)
    if not strip_unresolved_directional_markers:
        return suppressed_bond_keys

    def neighboring_directional_bond_keys(atom_idx, partner_idx):
        bond_keys = set()

        def add_key(other_idx, edge_data):
            if other_idx == partner_idx:
                return
            if edge_data.get("bond_type", 1) != 1:
                return
            token = edge_data.get("bond_symbol_raw")
            if token not in {"/", "\\"}:
                return
            bond_keys.add(frozenset((atom_idx, other_idx)))

        for other_idx, edge_data in _iter_incident_edge_data(mol_graph, atom_idx):
            add_key(other_idx, edge_data)

        return bond_keys

    for u_idx, v_idx, edge_data in mol_graph.edges(data=True):
        if edge_data.get("bond_type", 1) != 2:
            continue
        left_keys = neighboring_directional_bond_keys(u_idx, v_idx)
        right_keys = neighboring_directional_bond_keys(v_idx, u_idx)
        if not left_keys and not right_keys:
            continue
        if left_keys and right_keys:
            # Fully directional context exists; keep these markers.
            continue
        suppressed_bond_keys.update(left_keys)
        suppressed_bond_keys.update(right_keys)

    return suppressed_bond_keys


def _assign_stereochemistry(mol, chem):
    """Finalize atom and double-bond stereochemistry from current tags/directions."""
    set_bond_stereo = getattr(chem, "SetBondStereoFromDirections", None)
    if callable(set_bond_stereo):
        set_bond_stereo(mol)
    chem.AssignStereochemistry(mol, cleanIt=True, force=True)


def _run_with_big_stack(fn, *args):
    """Run fn(*args) on a daemon thread with a ~256 MiB stack.

    A KeyboardInterrupt in the caller abandons the thread; a long RDKit call
    also holds the GIL, so the interrupt lands only once that call returns.
    """
    result, error = [], []

    def runner():
        try:
            result.append(fn(*args))
        except BaseException as exc:
            error.append(exc)

    try:
        with _STACK_SIZE_LOCK:
            old_size = threading.stack_size(_BIG_STACK_SIZE)
            try:
                thread = threading.Thread(target=runner, daemon=True)
                thread.start()  # stack size is read at start(), not at Thread()
            finally:
                threading.stack_size(old_size)
    except (ValueError, RuntimeError):
        warnings.warn(
            "Could not start a big-stack thread; running RDKit conversion inline. "
            "Molecules over ~3000 atoms may crash the process (stack overflow).",
            RuntimeWarning,
            stacklevel=3,
        )
        thread = None
    if thread is None:
        return fn(*args)  # outside the except handler: clean tracebacks
    while thread.is_alive():
        thread.join(timeout=0.5)  # interruptible between calls, not during one
    if error:
        raise error[0]
    if not result:
        raise RuntimeError("big-stack conversion thread finished without a result")
    return result[0]


def mol_graph_to_rdkit_mol(
    mol_graph,
    kekulize=True,
    native_stage_callback=None,
    _on_big_stack=False,
    strip_unresolved_directional_markers=False,
):
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError("RDKit is  an optional dependency, but to generate RDKit molecules it is required. Please install RDKit for example with `pip install rdkit`.") from exc

    if (
        not _on_big_stack
        and mol_graph.number_of_nodes() >= _BIG_STACK_ATOM_THRESHOLD
    ):
        return _run_with_big_stack(
            mol_graph_to_rdkit_mol,
            mol_graph,
            kekulize,
            native_stage_callback,
            True,
            strip_unresolved_directional_markers,
        )

    def convert_bond_type(bond_attr):
        if bond_attr["aromatic"]:
            return Chem.BondType.AROMATIC
        if bond_attr["bond_type"] == 1:
            return Chem.BondType.SINGLE
        if bond_attr["bond_type"] == 2:
            return Chem.BondType.DOUBLE
        if bond_attr["bond_type"] == 3:
            return Chem.BondType.TRIPLE
        if bond_attr["bond_type"] == 4:
            return Chem.BondType.QUADRUPLE

    if native_stage_callback is not None:
        native_stage_callback("build")
    mol = Chem.RWMol()
    graph_idx_to_mol_idx = {}
    for graph_idx, data in mol_graph.nodes(data=True):
        atom = Chem.Atom(data["atomic_num"])
        atom.SetIsAromatic(data["aromatic"])
        atom.SetFormalCharge(data["charge"])
        # Preserve the written H count of aromatic bracket atoms that specify one
        # (e.g. [nH]); a negative value (or a caller-supplied None) leaves RDKit to
        # infer implicit H by valence. This is a public API taking a caller-built
        # graph, so tolerate a missing/None attribute rather than raising.
        # Never on dummy atoms (atomic_num 0): connection placeholders copy every
        # attribute of the neighboring real atom, whose H count must not render
        # as a phantom hydrogen on the [*:n] stub.
        num_explicit_h = data.get("num_explicit_h", -1)
        if num_explicit_h is not None and num_explicit_h >= 0 and data["atomic_num"] > 0:
            atom.SetNumExplicitHs(int(num_explicit_h))
            atom.SetNoImplicit(True)
        if "connection" in data:
            atom.SetAtomMapNum(data["connection"] + 1)

        graph_idx_to_mol_idx[graph_idx] = mol.AddAtom(atom)

    for u, v, attr in mol_graph.edges(data=True):
        # bond_type 0 = association edge (e.g, ion pair): no covalent bond, the
        # counterion renders as a separate "." fragment.
        if attr["bond_type"] == 0:
            continue
        u_mol_idx = graph_idx_to_mol_idx[u]
        v_mol_idx = graph_idx_to_mol_idx[v]
        if mol.GetBondBetweenAtoms(u_mol_idx, v_mol_idx) is not None:
            continue
        mol.AddBond(u_mol_idx, v_mol_idx, convert_bond_type(attr))

    _apply_atom_chirality_tokens(mol, mol_graph, graph_idx_to_mol_idx, Chem)
    suppressed_bond_keys = _warn_and_collect_unresolved_directional_markers(
        mol_graph,
        strip_unresolved_directional_markers=strip_unresolved_directional_markers,
    )
    _apply_directional_bond_tokens(
        mol,
        mol_graph,
        graph_idx_to_mol_idx,
        Chem,
        suppressed_bond_keys=suppressed_bond_keys,
    )
    if kekulize:
        has_stereochemistry = any(
            data.get("atom_chiral_token")
            for _node, data in mol_graph.nodes(data=True)
        ) or any(
            data.get("bond_symbol_raw") in {"/", "\\"}
            for _u, _v, data in mol_graph.edges(data=True)
        )
        use_fast_ring_path = (
            mol.GetNumAtoms() >= _FAST_RING_ATOM_THRESHOLD
            and mol.GetNumBonds() - mol.GetNumAtoms()
            >= _FAST_RING_EXCESS_THRESHOLD
            and not has_stereochemistry
        )
        if use_fast_ring_path:
            # Symmetric SSSR perception can take minutes for polymers made of
            # thousands of repeated fused-ring units. The generated graph
            # already carries aromatic atom/bond assignments, so initialize
            # valence properties and a non-symmetric ring basis instead. This
            # path is deliberately limited to huge, non-stereochemical graphs;
            # ordinary and stereo-bearing molecules retain full sanitization.
            if native_stage_callback is not None:
                native_stage_callback("property-cache-fast-rings")
            mol.UpdatePropertyCache(strict=True)
            Chem.FastFindRings(mol)
            mol.SetBoolProp(_PREFER_DIRECT_SMILES_PROPERTY, True)
        else:
            if native_stage_callback is not None:
                native_stage_callback("sanitize")
            Chem.SanitizeMol(mol)
            if native_stage_callback is not None:
                native_stage_callback("property-cache")
            mol.UpdatePropertyCache()
            _assign_stereochemistry(mol, Chem)
    else:
        # Fragment mode (per-unit bookkeeping): a unit is a static-connected piece,
        # so an aromatic ring atom that bears an inter-unit (non-static) bond has a
        # dangling valence here and cannot be kekulized in isolation, even though
        # the assembled molecule kekulizes fine. Skip only kekulization; the
        # dangling (under-valent) bond does not trip the valence check.
        if native_stage_callback is not None:
            native_stage_callback("sanitize-fragment")
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        if native_stage_callback is not None:
            native_stage_callback("property-cache-fragment")
        mol.UpdatePropertyCache(strict=False)
        _assign_stereochemistry(mol, Chem)
    return mol


def rdkit_mol_to_smiles(
    mol,
    native_stage_callback=None,
    smiles_policy="auto",
):
    """Serialize ``mol`` to SMILES, with large-molecule RDKit safeguards.

    ``smiles_policy="auto"`` first requests canonical output and recovers from
    ring-label overflow. ``"canonical"`` disables recovery, while ``"fast"``
    first uses the deterministic non-canonical extended-label writer.

    Canonical traversal can exhaust RDKit's finite set of simultaneously open
    ring labels for large, ring-rich polymers even though the molecule is
    valid. In that specific case, retry with alternate roots, deterministic
    breadth-first atom orderings, independently seeded randomized traversals,
    and a stereo-aware writer using extended ring labels; the resulting SMILES
    is non-canonical but represents the same molecule.
    """
    from rdkit import Chem, rdBase

    supported_policies = {"auto", "canonical", "fast"}
    if smiles_policy not in supported_policies:
        raise ValueError(
            f"Unsupported SMILES policy: {smiles_policy!r}. "
            f"Choose from {sorted(supported_policies)}."
        )

    if native_stage_callback is not None:
        native_stage_callback("smiles")

    def _is_ring_label_overflow(exc):
        # RDKit can surface this condition as ValueError or RuntimeError
        # depending on version/build bindings; only match the known message.
        if not isinstance(exc, (ValueError, RuntimeError)):
            return False
        return "rings open at once" in str(exc).lower()

    def _direct_smiles_with_extended_ring_labels():
        """Write a stereo-aware molecule without RDKit's finite open-ring pool."""
        prefer_direct = (
            mol.HasProp(_PREFER_DIRECT_SMILES_PROPERTY)
            and mol.GetBoolProp(_PREFER_DIRECT_SMILES_PROPERTY)
        )
        tetrahedral_tags = {
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        }
        if any(
            atom.GetChiralTag() not in tetrahedral_tags
            for atom in mol.GetAtoms()
        ):
            return None
        supported_bond_stereo = {
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS,
        }
        supported_bond_directions = {
            Chem.rdchem.BondDir.NONE,
            Chem.rdchem.BondDir.ENDUPRIGHT,
            Chem.rdchem.BondDir.ENDDOWNRIGHT,
        }
        if any(
            bond.GetStereo() not in supported_bond_stereo
            or bond.GetBondDir() not in supported_bond_directions
            for bond in mol.GetBonds()
        ):
            return None

        supported_bond_symbols = {
            Chem.rdchem.BondType.DOUBLE: "=",
            Chem.rdchem.BondType.TRIPLE: "#",
            Chem.rdchem.BondType.AROMATIC: "",
            Chem.rdchem.BondType.QUADRUPLE: "$",
        }

        def bond_symbol(bond, from_atom_idx=None):
            direction = bond.GetBondDir()
            if direction != Chem.rdchem.BondDir.NONE:
                symbol = "/" if direction == Chem.rdchem.BondDir.ENDUPRIGHT else "\\"
                if from_atom_idx is not None and bond.GetBeginAtomIdx() != from_atom_idx:
                    symbol = "\\" if symbol == "/" else "/"
                return symbol
            bond_type = bond.GetBondType()
            if bond_type == Chem.rdchem.BondType.SINGLE:
                # An omitted bond between aromatic atoms would become aromatic
                # on reparse (for example, the central bond in biphenyl).
                if (
                    bond.GetBeginAtom().GetIsAromatic()
                    and bond.GetEndAtom().GetIsAromatic()
                ):
                    return "-"
                return ""
            return supported_bond_symbols.get(bond_type)

        if any(bond_symbol(bond) is None for bond in mol.GetBonds()):
            return None

        n_atoms = mol.GetNumAtoms()
        adjacency = [
            [neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors()]
            for atom_idx in range(n_atoms)
        ]
        seen = set()
        parent = {}
        children = [[] for _ in range(n_atoms)]
        traversal = []
        component_roots = []
        for component_root in range(n_atoms):
            if component_root in seen:
                continue
            component_roots.append(component_root)
            seen.add(component_root)
            traversal.append(component_root)
            stack = [(component_root, iter(adjacency[component_root]))]
            while stack:
                atom_idx, neighbors = stack[-1]
                try:
                    neighbor_idx = next(neighbors)
                except StopIteration:
                    stack.pop()
                    continue
                if neighbor_idx in seen:
                    continue
                seen.add(neighbor_idx)
                parent[neighbor_idx] = atom_idx
                children[atom_idx].append(neighbor_idx)
                traversal.append(neighbor_idx)
                stack.append((neighbor_idx, iter(adjacency[neighbor_idx])))

        tree_edges = {
            (min(atom_idx, parent_idx), max(atom_idx, parent_idx))
            for atom_idx, parent_idx in parent.items()
        }
        traversal_position = {
            atom_idx: position for position, atom_idx in enumerate(traversal)
        }
        ring_annotations = [[] for _ in range(n_atoms)]
        ring_neighbors = [[] for _ in range(n_atoms)]

        def ring_label(number):
            if number < 10:
                return str(number)
            if number < 100:
                return f"%{number}"
            return f"%({number})"

        ring_number = 0
        for bond in mol.GetBonds():
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            edge = (min(begin, end), max(begin, end))
            if edge in tree_edges:
                continue
            ring_number += 1
            if traversal_position[begin] < traversal_position[end]:
                first, second = begin, end
            else:
                first, second = end, begin
            label = ring_label(ring_number)
            ring_annotations[first].append(bond_symbol(bond, first) + label)
            ring_annotations[second].append(label)
            ring_neighbors[first].append(second)
            ring_neighbors[second].append(first)

        atom_token_cache = {}

        def atom_token(atom_idx):
            atom = mol.GetAtomWithIdx(atom_idx)
            chiral_tag = atom.GetChiralTag()
            if chiral_tag != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
                old_neighbors = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
                emitted_neighbors = []
                if atom_idx in parent:
                    emitted_neighbors.append(parent[atom_idx])
                emitted_neighbors.extend(ring_neighbors[atom_idx])
                emitted_neighbors.extend(children[atom_idx])
                if set(old_neighbors) != set(emitted_neighbors):
                    return None
                old_positions = {
                    neighbor_idx: position
                    for position, neighbor_idx in enumerate(old_neighbors)
                }
                permutation = [old_positions[neighbor_idx] for neighbor_idx in emitted_neighbors]
                inversions = sum(
                    left > right
                    for index, left in enumerate(permutation)
                    for right in permutation[index + 1 :]
                )
                # For a component-root center with a bracket hydrogen, SMILES
                # treats that virtual neighbor as preceding all written bonds;
                # RDKit's stored tag orders it after the explicit neighbors.
                if (
                    atom_idx not in parent
                    and atom.GetDegree() == 3
                    and atom.GetNumExplicitHs() + atom.GetNumImplicitHs() == 1
                ):
                    inversions += 1
                marker = (
                    "@@"
                    if chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW
                    else "@"
                )
                if inversions % 2:
                    marker = "@" if marker == "@@" else "@@"
                token = atom.GetSmarts()
                original_marker = (
                    "@@"
                    if chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW
                    else "@"
                )
                return token.replace(original_marker, marker, 1)
            key = (
                atom.GetAtomicNum(),
                atom.GetIsAromatic(),
                atom.GetIsotope(),
                atom.GetFormalCharge(),
                atom.GetNumExplicitHs(),
                atom.GetNumImplicitHs(),
                atom.GetNoImplicit(),
                atom.GetNumRadicalElectrons(),
                atom.GetAtomMapNum(),
            )
            token = atom_token_cache.get(key)
            if token is None:
                # On the huge-ring fast path, MolFragmentToSmiles can invoke
                # the same expensive ring machinery that this writer avoids.
                # GetSmarts emits the same atom token for the supported atom
                # features (including aromatic [nH], charge, isotope, and map).
                if prefer_direct:
                    token = atom.GetSmarts()
                else:
                    token = Chem.MolFragmentToSmiles(
                        mol,
                        atomsToUse=[atom_idx],
                        canonical=False,
                    )
                atom_token_cache[key] = token
            return token

        output = []
        for component_index, component_root in enumerate(component_roots):
            if component_index:
                output.append(".")
            events = [("node", component_root)]
            while events:
                event, value = events.pop()
                if event == "text":
                    output.append(value)
                    continue
                atom_idx = value
                output.append(atom_token(atom_idx))
                output.extend(ring_annotations[atom_idx])
                atom_children = children[atom_idx]
                if not atom_children:
                    continue
                continuation = atom_children[-1]
                events.append(("node", continuation))
                events.append(
                    (
                        "text",
                        bond_symbol(
                            mol.GetBondBetweenAtoms(atom_idx, continuation),
                            atom_idx,
                        ),
                    )
                )
                for branch in reversed(atom_children[:-1]):
                    events.append(("text", ")"))
                    events.append(("node", branch))
                    events.append(
                        (
                            "text",
                            bond_symbol(
                                mol.GetBondBetweenAtoms(atom_idx, branch),
                                atom_idx,
                            ),
                        )
                    )
                    events.append(("text", "("))
        return "".join(output)

    def serialize():
        if smiles_policy == "canonical":
            return Chem.MolToSmiles(mol)
        if (
            smiles_policy == "fast"
            or (
                mol.HasProp(_PREFER_DIRECT_SMILES_PROPERTY)
                and mol.GetBoolProp(_PREFER_DIRECT_SMILES_PROPERTY)
            )
        ):
            direct = _direct_smiles_with_extended_ring_labels()
            if direct is not None:
                return direct
        try:
            if smiles_policy == "fast":
                return Chem.MolToSmiles(mol, canonical=False)
            return Chem.MolToSmiles(mol)
        except Exception as exc:
            if not _is_ring_label_overflow(exc):
                raise

            # Once canonical traversal has exhausted RDKit's finite ring-label
            # pool, prefer the deterministic extended-label writer. It avoids
            # up to sixteen additional native traversals before the same
            # fallback, while retaining the slower RDKit repairs for features
            # the direct writer does not support.
            direct = _direct_smiles_with_extended_ring_labels()
            if direct is not None:
                return direct

            n_atoms = mol.GetNumAtoms()
            roots = [0, 1, 2, 10]
            if n_atoms > 1:
                roots.extend(
                    [
                        n_atoms // 8,
                        n_atoms // 4,
                        n_atoms // 2,
                        (3 * n_atoms) // 4,
                        (7 * n_atoms) // 8,
                        n_atoms - 1,
                    ]
                )
            roots = [root for root in dict.fromkeys(root for root in roots if 0 <= root < n_atoms)]

            # Root zero is the cheapest compatibility fallback for ordinary
            # molecules. On large molecules, repair the pathological atom
            # ordering first: another original-order traversal can be very slow
            # before exhausting the same ring-label pool.
            initial_roots = roots[:1] if n_atoms < _BIG_STACK_ATOM_THRESHOLD else []
            for root in initial_roots:
                try:
                    return Chem.MolToSmiles(mol, canonical=False, rootedAtAtom=root)
                except Exception as root_exc:
                    if not _is_ring_label_overflow(root_exc):
                        raise

            def seeded_randomized_smiles():
                # RDKit's randomized traversal can avoid pathological spanning
                # trees that no fixed atom ordering repairs. Passing a positive
                # ``randomSeed`` to MolToRandomSmilesVect does not reset RDKit's
                # process-global generator after its first construction. Reset
                # it explicitly so every attempt is an independent, reproducible
                # traversal rather than merely the next state of one stream.
                with _RANDOM_SMILES_LOCK:
                    for seed in range(1, _RANDOM_SMILES_ATTEMPTS + 1):
                        rdBase.SeedRandomNumberGenerator(seed)
                        try:
                            return Chem.MolToRandomSmilesVect(
                                mol,
                                1,
                                randomSeed=0,
                            )[0]
                        except Exception as random_exc:
                            if not _is_ring_label_overflow(random_exc):
                                raise
                return None

            # Failed breadth-first traversals are disproportionately costly on
            # these large molecules. For unsupported direct-writer features,
            # prefer the bounded randomized repair; deterministic BFS remains
            # available if every randomized traversal still overflows.
            if n_atoms >= _BIG_STACK_ATOM_THRESHOLD:
                randomized = seeded_randomized_smiles()
                if randomized is not None:
                    return randomized

            # ``canonical=False`` still follows the molecule's atom ordering.
            # Incrementally assembled cyclic polymers can have an ordering that
            # leaves more than RDKit's finite ring-label pool open at once for
            # every root. Renumber atoms breadth-first so neighboring rings are
            # emitted close together. Reverse-neighbor BFS is first because it
            # keeps newly added cyclic repeats locally contiguous in the
            # reported long benzimidazole chain.
            for reverse_neighbors in (True, False):
                order = []
                seen = set()
                for component_root in range(n_atoms):
                    if component_root in seen:
                        continue
                    seen.add(component_root)
                    queue = deque([component_root])
                    while queue:
                        atom_idx = queue.popleft()
                        order.append(atom_idx)
                        neighbors = sorted(
                            (
                                neighbor.GetIdx()
                                for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
                            ),
                            reverse=reverse_neighbors,
                        )
                        for neighbor_idx in neighbors:
                            if neighbor_idx not in seen:
                                seen.add(neighbor_idx)
                                queue.append(neighbor_idx)

                renumbered = Chem.RenumberAtoms(mol, order)
                try:
                    return Chem.MolToSmiles(
                        renumbered,
                        canonical=False,
                        rootedAtAtom=0,
                    )
                except Exception as reordered_exc:
                    if not _is_ring_label_overflow(reordered_exc):
                        raise

            # Retain the broader root search for unusual molecules where the
            # original ordering is useful but the preceding deterministic
            # traversals still exceed the ring-label pool.
            for root in roots[len(initial_roots):]:
                try:
                    return Chem.MolToSmiles(
                        mol,
                        canonical=False,
                        rootedAtAtom=root,
                    )
                except Exception as root_exc:
                    if not _is_ring_label_overflow(root_exc):
                        raise

            return Chem.MolToSmiles(mol, canonical=False)

    if mol.GetNumAtoms() < _BIG_STACK_ATOM_THRESHOLD:
        return serialize()
    return _run_with_big_stack(serialize)


def rdkit_mol_weight(mol, native_stage_callback=None):
    """Compute molecular weight with large-stack protection and stage reporting."""
    from rdkit.Chem import Descriptors

    if native_stage_callback is not None:
        native_stage_callback("descriptor-molwt")
    if mol.GetNumAtoms() < _BIG_STACK_ATOM_THRESHOLD:
        return Descriptors.MolWt(mol)
    return _run_with_big_stack(Descriptors.MolWt, mol)


def mol_graph_to_smiles(
    mol_graph,
    kekulize=True,
    native_stage_callback=None,
    strip_unresolved_directional_markers=False,
    smiles_policy="auto",
):
    """Convert a mol graph to SMILES; safe for very large graphs.

    ``smiles_policy="auto"`` requests canonical output and falls back safely
    after ring-label overflow. ``"canonical"`` disables fallbacks, while
    ``"fast"`` prefers deterministic non-canonical output immediately.
    """
    return rdkit_mol_to_smiles(
        mol_graph_to_rdkit_mol(
            mol_graph,
            kekulize=kekulize,
            native_stage_callback=native_stage_callback,
            strip_unresolved_directional_markers=strip_unresolved_directional_markers,
        ),
        native_stage_callback=native_stage_callback,
        smiles_policy=smiles_policy,
    )
