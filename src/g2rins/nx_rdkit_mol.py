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
        mol.AddBond(graph_idx_to_mol_idx[u], graph_idx_to_mol_idx[v], convert_bond_type(attr))
    if kekulize:
        if native_stage_callback is not None:
            native_stage_callback("sanitize")
        Chem.SanitizeMol(mol)
        if native_stage_callback is not None:
            native_stage_callback("property-cache")
        mol.UpdatePropertyCache()
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

    def serialize():
        try:
            return Chem.MolToSmiles(mol)
        except ValueError as exc:
            if "Too many rings open at once" not in str(exc):
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
