# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

import threading
import warnings

# RDKit's SMILES writer recurses per atom in C++; the default stack overflows
# (hard 0xC00000FD on Windows) near 3k atoms. Empirically a 2000-atom linear
# chain is safe on the default stack and 4000 crashes, so gate with margin.
_BIG_STACK_ATOM_THRESHOLD = 2000
_BIG_STACK_SIZE = 0x0FFFF000  # just under CPython's 256 MiB Windows cap

# threading.stack_size is process-global: serialize set/start/restore.
_STACK_SIZE_LOCK = threading.Lock()


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

        if hasattr(mol_graph, "out_edges"):
            for _u_idx, v_idx, _key, edge_data in mol_graph.out_edges(atom_idx, keys=True, data=True):
                add_token(v_idx, edge_data)
            for u_idx, _v_idx, _key, edge_data in mol_graph.in_edges(atom_idx, keys=True, data=True):
                add_token(u_idx, edge_data)
        else:
            for neighbor in mol_graph.neighbors(atom_idx):
                edge_bundle = mol_graph.get_edge_data(atom_idx, neighbor)
                if edge_bundle is None:
                    continue
                if mol_graph.is_multigraph():
                    for edge_data in edge_bundle.values():
                        add_token(neighbor, edge_data)
                else:
                    add_token(neighbor, edge_bundle)

        return tokens_by_bond

    warned = set()
    suppressed_bond_keys = set()
    for u_idx, v_idx, edge_data in mol_graph.edges(data=True):
        if edge_data.get("bond_type", 1) != 2:
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
    suppressed_bond_keys = _warn_on_ambiguous_directional_markers(mol_graph)
    _apply_directional_bond_tokens(
        mol,
        mol_graph,
        graph_idx_to_mol_idx,
        Chem,
        suppressed_bond_keys=suppressed_bond_keys,
    )
    if kekulize:
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


def rdkit_mol_to_smiles(mol, native_stage_callback=None):
    """Serialize ``mol`` to SMILES, with large-molecule RDKit safeguards.

    Canonical traversal can exhaust RDKit's finite set of simultaneously open
    ring labels for large, ring-rich polymers even though the molecule is
    valid. In that specific case, retry in atom order; the resulting SMILES is
    non-canonical but represents the same molecule.
    """
    from rdkit import Chem

    if native_stage_callback is not None:
        native_stage_callback("smiles")

    def _is_ring_label_overflow(exc):
        # RDKit can surface this condition as ValueError or RuntimeError
        # depending on version/build bindings; only match the known message.
        if not isinstance(exc, (ValueError, RuntimeError)):
            return False
        return "rings open at once" in str(exc).lower()

    def serialize():
        try:
            return Chem.MolToSmiles(mol)
        except Exception as exc:
            if not _is_ring_label_overflow(exc):
                raise

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

            for root in roots:
                try:
                    return Chem.MolToSmiles(mol, canonical=False, rootedAtAtom=root)
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


def mol_graph_to_smiles(mol_graph, kekulize=True, native_stage_callback=None):
    """Convert a mol graph to a canonical SMILES string; safe for very large graphs."""
    return rdkit_mol_to_smiles(
        mol_graph_to_rdkit_mol(
            mol_graph,
            kekulize=kekulize,
            native_stage_callback=native_stage_callback,
        ),
        native_stage_callback=native_stage_callback,
    )
