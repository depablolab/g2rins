# SPDX-License-Identifier: GPL-3.0-only

"""Assemble finite motif fragments using canonical pSMILES attachment sites."""

import re
from functools import lru_cache
from itertools import combinations

from rdkit import Chem


@lru_cache(maxsize=4096)
def _unit_mol(psmiles):
    parameters = Chem.SmilesParserParams()
    parameters.removeHs = False
    mol = Chem.MolFromSmiles(psmiles, parameters)
    if mol is None:
        raise ValueError(f"Invalid motif unit pSMILES: {psmiles!r}")
    return mol


@lru_cache(maxsize=4096)
def _source_descriptors(psmiles, bigsmiles):
    """Recover connector identity by connectivity, not source traversal order.

    Native sites follow generative-graph node order, which need not be source
    descriptor order (notably when several connectors share an atom). Native
    pSMILES remains authoritative for stereochemistry: this lookup maps ports
    only, and must not reinterpret stereo from a second SMILES parser.
    """
    if not bigsmiles:
        return None
    # Amounts/probabilities describe generation, not the finite fragment.
    source = re.sub(r"\|[^|]*\|", "", bigsmiles)
    pattern = r"\[[<>$][^\]]*\]"
    descriptors = re.findall(pattern, source)
    canonical_unit = _unit_mol(psmiles)
    sites = sorted(atom.GetAtomMapNum() for atom in canonical_unit.GetAtoms()
                   if atom.GetAtomicNum() == 0)
    target = Chem.RemoveHs(canonical_unit)
    expected = Chem.MolToSmiles(target, isomericSmiles=True)
    target_sites = {atom.GetIdx(): atom.GetAtomMapNum() for atom in target.GetAtoms()
                    if atom.GetAtomicNum() == 0}
    for atom in target.GetAtoms():
        atom.SetAtomMapNum(0)
    matches = set()
    ordered_matches = set()
    # Context can deactivate a connector (e.g. a nested foreign descriptor).
    # Match full connectivity, retaining every symmetry-related port assignment.
    # Accept only if all assignments imply the same descriptors for native sites.
    for active in combinations(range(len(descriptors)), len(sites)):
        site_by_position = dict(zip(active, sites))
        descriptor_by_map = dict(zip(sites, (descriptors[index] for index in active)))
        position = iter(range(len(descriptors)))

        def replace(match):
            site = site_by_position.get(next(position))
            return f"[*:{site}]" if site is not None else "[H]"

        candidate = re.sub(pattern, replace, source)
        candidate_mol = Chem.RemoveHs(_unit_mol(candidate))
        # Preserve the established source-order convention when its mapped
        # chemistry verifies exactly, including symmetric directional repeaters.
        if Chem.MolToSmiles(candidate_mol, isomericSmiles=True) == expected:
            ordered_matches.add(tuple(descriptors[index] for index in active))
        if (candidate_mol.GetNumAtoms() != target.GetNumAtoms()
                or candidate_mol.GetNumBonds() != target.GetNumBonds()):
            continue
        candidate_descriptors = {
            atom.GetIdx(): descriptor_by_map[atom.GetAtomMapNum()]
            for atom in candidate_mol.GetAtoms() if atom.GetAtomicNum() == 0
        }
        for atom in candidate_mol.GetAtoms():
            atom.SetAtomMapNum(0)
        for assignment in target.GetSubstructMatches(
                candidate_mol, uniquify=False, useChirality=False, maxMatches=0):
            mapped = {target_sites[assignment[index]]: descriptor
                      for index, descriptor in candidate_descriptors.items()}
            matches.add(tuple(mapped[site] for site in sites))
    if len(ordered_matches) == 1:
        return dict(zip(sites, ordered_matches.pop()))
    if len(matches) > 1 or len(ordered_matches) > 1:
        raise ValueError(
            "Source BigSMILES connectors do not match canonical pSMILES sites: "
            "ambiguous connector identity"
        )
    if len(matches) != 1:
        raise ValueError(
            "Source BigSMILES connectors do not match canonical pSMILES sites: "
            f"psmiles={psmiles!r}, bigsmiles={bigsmiles!r}"
        )
    return dict(zip(sites, matches.pop()))


def native_source_descriptors(parsed_graph, native_graph):
    """Read exact active-site identities before parser connectors are discarded.

    Both graphs must come from the same GraphCreator. Native site labels refer
    to those original node IDs, including the separate placeholders introduced
    for multiple connectors on one atom. No molecular isomorphism is involved.
    """
    from .bond import BondConnector
    from .generative_graph import derive_unit_labels

    labels = derive_unit_labels(native_graph)
    result = {unit: {} for unit in labels.unit_nodes}
    for node, site in labels.bond_id.items():
        if node not in parsed_graph:
            raise ValueError("Native motif attachment site has no parser provenance")
        neighbors = set(parsed_graph.successors(node)) | set(parsed_graph.predecessors(node))
        descriptors = {
            re.sub(r"\|[^|]*\|", "", str(parsed_graph.nodes[neighbor]["obj"]))
            for neighbor in neighbors
            if isinstance(parsed_graph.nodes[neighbor].get("obj"), BondConnector)
        }
        if len(descriptors) != 1:
            raise ValueError("Native motif attachment site has ambiguous parser provenance")
        result[labels.unit_id[node]][site] = descriptors.pop()
    return result


def motif_chemical_strings(topology, units, *, unit_descriptors=None):
    """Return pSMILES and a BigSMILES *fragment*, not a stochastic object.

    Occupied cut-edge endpoints get local [*:n] maps and their source BigSMILES
    descriptors. Unoccupied template ports are absent from the sampled graph;
    they must not survive as wildcard atoms in the finite motif.
    molzip joins matched dummy atoms and preserves neighboring stereochemistry.
    """
    fragments = []
    ports = {}
    source_descriptors = {}
    for occurrence, unit in enumerate(topology["units"]):
        mol = Chem.Mol(_unit_mol(units[unit]["psmiles"]))
        if unit_descriptors is None:
            # Compatibility for stored units and source-less synthetic callers.
            source_descriptors[occurrence] = _source_descriptors(
                units[unit]["psmiles"], units[unit].get("bigsmiles"),
            )
        else:
            descriptors = unit_descriptors.get(unit)
            sites = {atom.GetAtomMapNum() for atom in mol.GetAtoms()
                     if atom.GetAtomicNum() == 0}
            if (not isinstance(descriptors, dict) or set(descriptors) != sites
                    or any(not isinstance(value, str)
                           or re.fullmatch(r"\[[<>$]\d*\]", value) is None
                           for value in descriptors.values())):
                raise ValueError("Native motif connector provenance does not match pSMILES sites")
            source_descriptors[occurrence] = descriptors
        for atom in mol.GetAtoms():
            site = atom.GetAtomMapNum()
            if atom.GetAtomicNum() == 0 and site:
                if atom.GetDegree() != 1 or (occurrence, site) in ports:
                    raise ValueError("Motif pSMILES ports must be unique degree-one dummy atoms")
                ports[occurrence, site] = atom
            atom.SetAtomMapNum(0)
        fragments.append(mol)

    bond_types = {"1": Chem.BondType.SINGLE, "2": Chem.BondType.DOUBLE,
                  "3": Chem.BondType.TRIPLE, "4": Chem.BondType.AROMATIC}
    used_ports = set()

    def set_port_bond(occurrence, site, bond_type, aromatic):
        key = (occurrence, site)
        if key in used_ports:
            raise ValueError("A motif attachment site was used more than once")
        used_ports.add(key)
        atom = ports[key]
        bond = fragments[occurrence].GetBondWithIdx(atom.GetBonds()[0].GetIdx())
        bond.SetBondType(bond_types[bond_type])
        bond.SetIsAromatic(aromatic)
        return atom

    for label, (left, ls, right, rs, bond, aromatic) in enumerate(topology["edges"], 1):
        set_port_bond(left, ls, bond, aromatic).SetAtomMapNum(label)
        set_port_bond(right, rs, bond, aromatic).SetAtomMapNum(label)
    boundary_descriptors = {}
    for label, (occurrence, site, bond, aromatic) in enumerate(topology["boundary"], 1):
        set_port_bond(occurrence, site, bond, aromatic).SetIntProp("_motif_boundary", label)
        descriptors = source_descriptors[occurrence]
        boundary_descriptors[label] = descriptors[site] if descriptors is not None else f"[${label}]"
    # Replacing an absent port with H before removal lets RDKit adjust chiral
    # parity correctly. These are implicit valence hydrogens, not added units.
    for key, atom in ports.items():
        if key not in used_ports:
            atom.SetAtomicNum(1)
            atom.SetNoImplicit(True)
            atom.GetBonds()[0].SetBondType(Chem.BondType.SINGLE)
    combined = fragments[0]
    for fragment in fragments[1:]:
        combined = Chem.CombineMols(combined, fragment)
    assembled = Chem.molzip(combined)
    for atom in assembled.GetAtoms():
        atom.SetAtomMapNum(atom.GetIntProp("_motif_boundary") if atom.HasProp("_motif_boundary") else 0)
    Chem.SanitizeMol(assembled)
    assembled = Chem.RemoveHs(assembled)
    psmiles = Chem.MolToSmiles(assembled, canonical=True, isomericSmiles=True)
    bigsmiles = re.sub(r"\[\*:(\d+)\]", lambda match: boundary_descriptors[int(match[1])], psmiles)
    return {"psmiles": psmiles, "bigsmiles": bigsmiles}
