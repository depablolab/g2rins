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


def mol_graph_to_rdkit_mol(mol_graph, kekulize=True):
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError("RDKit is  an optional dependency, but to generate RDKit molecules it is required. Please install RDKit for example with `pip install rdkit`.") from exc

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
        Chem.SanitizeMol(mol)
        mol.UpdatePropertyCache()
    else:
        # Fragment mode (per-unit bookkeeping): a unit is a static-connected piece,
        # so an aromatic ring atom that bears an inter-unit (non-static) bond has a
        # dangling valence here and cannot be kekulized in isolation, even though
        # the assembled molecule kekulizes fine. Skip only kekulization; the
        # dangling (under-valent) bond does not trip the valence check.
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        mol.UpdatePropertyCache(strict=False)
    return mol


def rdkit_mol_to_smiles(mol):
    """Chem.MolToSmiles guarded against stack overflow on very large molecules."""
    from rdkit import Chem

    if mol.GetNumAtoms() < _BIG_STACK_ATOM_THRESHOLD:
        return Chem.MolToSmiles(mol)
    return _run_with_big_stack(Chem.MolToSmiles, mol)


def mol_graph_to_smiles(mol_graph, kekulize=True):
    """Convert a mol graph to a canonical SMILES string; safe for very large graphs."""
    return rdkit_mol_to_smiles(mol_graph_to_rdkit_mol(mol_graph, kekulize=kekulize))
