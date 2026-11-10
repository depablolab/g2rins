"""Focused contracts for the bounded ensemble cleanup (no pool required)."""

import json
import warnings
from collections import Counter
from dataclasses import FrozenInstanceError

import networkx as nx
import numpy as np
import pytest

from g2rins import _metadata_equality
from g2rins import ensemble_creator as module
from g2rins._ensemble_export import _fixed_ensemble_json_data
from g2rins._ensemble_options import _ChainConversionOptions, _ConversionMode


@pytest.mark.parametrize("value, expected", [
    (False, _ConversionMode.IMMEDIATE), (None, _ConversionMode.IMMEDIATE),
    (0, _ConversionMode.IMMEDIATE), (True, _ConversionMode.DEFERRED),
    (1, _ConversionMode.DEFERRED), ("legacy-truthy", _ConversionMode.DEFERRED),
    ("materialize", _ConversionMode.MATERIALIZE),
    (_ConversionMode.IMMEDIATE, _ConversionMode.IMMEDIATE),
    (_ConversionMode.MATERIALIZE, _ConversionMode.MATERIALIZE),
])
def test_named_conversion_modes_preserve_legacy_values(value, expected):
    assert _ConversionMode.from_legacy(value) is expected


def test_conversion_options_are_immutable_and_not_transport_payloads():
    options = _ChainConversionOptions("smiles", False, True, None, _ConversionMode.IMMEDIATE, False, "fast")
    with pytest.raises(FrozenInstanceError):
        options.smiles_policy = "canonical"
    assert _ConversionMode.IMMEDIATE.value is False
    assert _ConversionMode.DEFERRED.value is True
    assert _ConversionMode.MATERIALIZE.value == "materialize"


@pytest.mark.parametrize("worker", [False, True])
def test_conversion_configuration_is_created_once_per_batch(monkeypatch, worker):
    creator = module.EnsembleCreator.__new__(module.EnsembleCreator)
    constructed = []
    observed = []

    def configure(*args):
        options = _ChainConversionOptions(*args)
        constructed.append(options)
        return options

    def convert(sample, options, *args, **kwargs):
        observed.append(options)
        return sample

    monkeypatch.setattr(module, "_ChainConversionOptions", configure)
    monkeypatch.setattr(module, "_convert_sample_with_options", convert)
    monkeypatch.setattr(module, "_attempt_chain", lambda *args: ("accepted", (), None, []))
    if worker:
        records = module._sample_chain_batch(
            creator, list(enumerate(np.random.SeedSequence(9).spawn(3))), "smiles", False, 1, None,
        )
    else:
        records = list(creator._iter_chain_records(3, "smiles", False, 1, None, False, None, 9))
    assert len(constructed) == 1
    assert len(records) == len(observed) == 3
    assert all(options is constructed[0] for options in observed)
    assert all("conversion" not in record for record in records)


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("materialize_units", [False, True])
def test_record_assembly_preserves_legacy_and_compact_metadata(monkeypatch, legacy, materialize_units):
    creator = module.EnsembleCreator.__new__(module.EnsembleCreator)
    contact = (("R0.1", "left"), ("T0.1", "right"))
    calls = []
    monkeypatch.setattr(creator, "_materialize_unit_counts", lambda counts: {key: {"count": value, "materialized": True} for key, value in counts.items()})
    if legacy:
        unit = nx.Graph()
        unit.add_node(0, origin_idx="origin")
        creator._unit_id_by_origin = {"origin": "R0"}
        creator._origin_endpoint = object()
        record = {
            "molecule": "C", "molecular_weight": 16.0, "molecule_units": {unit: 2},
            "bonds": {contact: 1}, "sequences": [["C"]],
            "mol_weights": {0: [16.0]}, "distributions": {0: "target"},
        }
        def bonds(counts, endpoint, origins):
            assert endpoint is creator._origin_endpoint
            calls.append(dict(counts))
            return ["legacy"]
        monkeypatch.setattr(module, "_bond_records", bonds)
    else:
        # Deliberately no legacy lookup attributes: compact assembly must not
        # access them eagerly just to pass dependencies to the new helper.
        record = module._ChainRecord("C", {"R0": 2}, None, {contact: 1}, [["C"]], {0: [16.0]}, {0: "target"}, 16.0, sequence_unit_ids=[["R0"]])
        def bonds(counts, origins):
            calls.append(dict(counts))
            return ["compact"]
        monkeypatch.setattr(module, "_bond_records_from_labeled", bonds)
    result = creator._records_to_ensemble_data([record, record], materialize_units=materialize_units)
    assert result.chains == ["C", "C"]
    assert result.molecular_weights == [16.0, 16.0]
    assert result.units == {"R0": {"count": 4, **({"materialized": True} if materialize_units else {})}}
    assert result.bonds == (["legacy"] if legacy else ["compact"])
    assert calls == [{contact: 2}]
    assert result.sequences == [[["C"]], [["C"]]]
    assert result.sequence_unit_ids == (None if legacy else [[["R0"]], [["R0"]]])
    assert result.mol_weights == {0: [16.0, 16.0]}
    assert result.distributions == {0: "target"}


def _caught(message, lineno):
    return warnings.WarningMessage(UserWarning(message), UserWarning, "attempt.py", lineno)


def _cause():
    try:
        try:
            raise ValueError("inner")
        except ValueError as inner:
            raise module.DeadSamplingPath("first") from inner
    except module.DeadSamplingPath as error:
        return error


@pytest.mark.parametrize("succeeds", [False, True])
def test_retry_stream_counts_first_cause_and_warning_order(monkeypatch, succeeds):
    rng = np.random.default_rng(19)
    expected_rng = np.random.default_rng(19)
    first = _cause()
    later = module.DeadSamplingPath("later")
    accepted = object()
    draws = []
    seen_rngs = []
    attempts = []

    def attempt(atom_graph, collect_info, termination_flag, actual_rng, repeat):
        assert (atom_graph, collect_info, termination_flag, repeat) == ("graph", False, 8, True)
        seen_rngs.append(actual_rng)
        draws.append(actual_rng.random())
        index = len(attempts)
        attempts.append(index)
        sample = accepted if succeeds and index == 2 else None
        return sample, ("earlier", "later") if index == 0 else ("later",), first if index == 0 else later, [_caught(str(index), index + 1)]

    monkeypatch.setattr(module, "_attempt_chain", attempt)
    sample, discards, reasons, cause, caught = module._retry_chain(
        "graph", False, 8, rng, True, 3,
    )
    assert sample is (accepted if succeeds else None)
    assert seen_rngs == [rng, rng, rng]
    assert draws == [expected_rng.random() for _ in range(3)]
    assert discards == (2 if succeeds else 3)
    assert tuple(reasons.items()) == (("earlier", 1), ("later", discards))
    assert cause is first
    assert cause.__traceback__ is None
    assert cause.__cause__.__traceback__ is None
    assert cause.__context__.__traceback__ is None
    assert [str(item.message) for item in caught] == ["0", "1", "2"]


@pytest.mark.parametrize("budget", [0, -1, 1])
def test_retry_budget_still_attempts_once(monkeypatch, budget):
    calls = []

    def attempt(*args):
        calls.append(args)
        return None, (), None, []

    monkeypatch.setattr(module, "_attempt_chain", attempt)
    result = module._retry_chain(object(), False, None, None, False, budget)
    assert len(calls) == 1
    assert result == (None, 1, Counter(), None, [])


def test_retry_does_not_catch_fatal_errors(monkeypatch):
    error = ValueError("fatal model error")

    def attempt(*args):
        raise error

    monkeypatch.setattr(module, "_attempt_chain", attempt)
    with pytest.raises(ValueError) as caught:
        module._retry_chain(object(), False, None, None, False, 10)
    assert caught.value is error


@pytest.mark.parametrize("worker", [False, True])
@pytest.mark.parametrize("mode", [False, True, "materialize"])
def test_callers_construct_one_rng_and_keep_conversion_policy(monkeypatch, worker, mode):
    creator = module.EnsembleCreator.__new__(module.EnsembleCreator)
    real_default_rng = np.random.default_rng
    rngs = []
    attempts = []
    conversions = []
    accepted = object()

    def default_rng(seed):
        rng = real_default_rng(seed)
        rngs.append(rng)
        return rng

    def attempt(graph, info, termination, rng, repeat):
        attempts.append(rng)
        return (accepted if len(attempts) == 2 else None), (), None, [_caught("attempt", len(attempts))]

    def convert(*args, **kwargs):
        conversions.append((args, kwargs))
        warnings.warn("conversion", UserWarning)
        return {"molecule": "accepted"}

    monkeypatch.setattr(module.np.random, "default_rng", default_rng)
    monkeypatch.setattr(module, "_attempt_chain", attempt)
    monkeypatch.setattr(module, "_convert_chain", convert)
    monkeypatch.setattr(module, "_defer_chain_conversion", convert)
    if worker:
        records = module._sample_chain_batch(
            creator, [(5, np.random.SeedSequence(23))], "smiles", False, 3, None,
            defer_conversion=mode,
        )
    else:
        records = list(creator._iter_chain_records(
            1, "smiles", False, 3, None, False, None, 23,
            start_index=5, defer_conversion=mode,
        ))
    assert len(rngs) == 1
    assert attempts == [rngs[0], rngs[0]]
    assert len(conversions) == 1
    args, kwargs = conversions[0]
    assert args[0] is accepted
    if worker and mode:
        assert kwargs["materialize_output"] is (mode == "materialize")
        assert kwargs["molecule_format"] == "smiles"
    else:
        assert "materialize_output" not in kwargs
        assert "molecule_format" not in kwargs
    assert list(records[0]) == [
        "chain_index", "record", "discards", "reasons", "first_cause", "warnings",
    ]
    assert records[0]["chain_index"] == 5
    assert records[0]["discards"] == 1
    assert [str(item[0]) for item in records[0]["warnings"]] == [
        "attempt", "attempt", "conversion",
    ]


def test_serial_failure_stops_but_worker_keeps_later_survivors(monkeypatch):
    creator = module.EnsembleCreator.__new__(module.EnsembleCreator)

    def outcomes():
        return iter([(None, (), None, []), ("later", (), None, [])])

    iterator = outcomes()
    monkeypatch.setattr(module, "_attempt_chain", lambda *args: next(iterator))
    monkeypatch.setattr(module, "_convert_chain", lambda sample, *args, **kwargs: sample)
    serial = list(creator._iter_chain_records(2, "smiles", False, 1, None, False, None, 7))
    assert len(serial) == 1
    assert serial[0]["record"] is None
    iterator = outcomes()
    jobs = list(enumerate(np.random.SeedSequence(7).spawn(2)))
    worker = module._sample_chain_batch(creator, jobs, "smiles", False, 1, None)
    assert [record["record"] for record in worker] == [None, "later"]
    assert all("building_block_statistics" not in record for record in worker)


def test_warning_routing_preserves_objects_provenance_and_summary_order(monkeypatch):
    counts = Counter()
    examples = {}
    replayed = []
    ordinary = UserWarning("ordinary")
    prefix = module._DIRECTIONAL_STEREO_WARNING_PREFIXES
    messages = [
        (RuntimeWarning(prefix["conflicting"] + "9"), RuntimeWarning, "c.py", 9),
        (ordinary, UserWarning, "original.py", 17),
        (RuntimeWarning(prefix["incomplete"] + "1"), RuntimeWarning, "a.py", 1),
        (RuntimeWarning(prefix["incomplete"] + "2"), RuntimeWarning, "a.py", 2),
        # Matching text with the wrong category must still be replayed.
        (prefix["ambiguous"] + "3", UserWarning, "b.py", 3),
    ]
    monkeypatch.setattr(module.warnings, "warn_explicit", lambda *args: replayed.append(args))
    module._replay_chain_warnings(iter(messages), counts, examples)
    assert replayed == [messages[1], messages[4]]
    assert replayed[0][0] is ordinary
    assert counts == {"conflicting": 1, "incomplete": 2}
    assert examples["incomplete"] == prefix["incomplete"] + "1"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module._emit_directional_stereo_warning_summary(counts, examples, True)
    assert len(caught) == 1
    text = str(caught[0].message)
    assert "3 warning event(s): incomplete=2, conflicting=1" in text
    assert text.index(examples["incomplete"]) < text.index(examples["conflicting"])
    assert "were stripped" in text


def _ensemble(chains=None, sequences=None):
    return module.EnsembleData(
        ["C", "CC", "CCC"] if chains is None else chains,
        {}, [],
        [[["C"]], [["CC"]], [["CCC"]]] if sequences is None else sequences,
        {0: [np.float64(16.0)]}, {},
    )


@pytest.mark.parametrize("limit", [None, 0, 1, -1])
def test_export_key_order_slicing_normalization_and_disabled_optionals(limit):
    graph = nx.Graph(g2rins_string=np.str_("C"))
    data = _ensemble()
    # Disabled fields must neither be read nor normalized.
    data.chain_motif_index = object()
    data.motif_population_statistics = object()
    data.building_block_statistics = object()
    paths = []

    def normalize(value, path="$"):
        paths.append(path)
        return module._json_safe(value, path)

    def no_sequence_serialization(unit):
        pytest.fail("SMILES sequences must pass through unchanged")

    result = _fixed_ensemble_json_data(
        graph, data, "smiles", limit, {}, no_sequence_serialization,
        normalize, lambda graph: {"format": {}, "graph": {}},
    )
    assert paths == ["$['string']", "$['ensemble']", "$"]
    assert list(result) == ["string", "format", "graph", "ensemble"]
    assert list(result["ensemble"]) == [
        "units", "chains", "bonds", "mol_weights", "distributions", "sequences", "sequence_unit_ids",
    ]
    assert result["ensemble"]["chains"] == (data.chains if limit is None else data.chains[:limit])
    assert result["ensemble"]["sequences"] == data.sequences
    assert result["ensemble"]["sequence_unit_ids"] is None
    assert result["ensemble"]["mol_weights"] == {0: [16.0]}
    json.dumps(result, allow_nan=False)


def test_export_graph_projection_and_optional_field_order():
    chain = nx.Graph()
    chain.add_node(0, extra="drop", origin_idx="raw", charge=np.int64(0), atomic_num=6)
    chain.add_node(1, atomic_num=8)
    chain.add_edge(0, 1, extra="drop", bond_type=1, aromatic=False)
    data = _ensemble([chain], [[[chain]]])
    data.chain_motif_index = {"rows": []}
    data.motif_population_statistics = {"accepted": np.int64(1)}
    data.building_block_statistics = {"units": []}
    calls = []

    def sequence(unit):
        calls.append(unit)
        return "CO"

    result = _fixed_ensemble_json_data(
        nx.Graph(), data, "mol_graph", None, {"raw": 42}, sequence,
        module._json_safe, lambda graph: {"format": {}},
        retain_motif_membership=True, building_block_statistics=True,
    )["ensemble"]
    assert calls == [chain]
    assert list(result)[-3:] == [
        "chain_motif_index", "motif_population_statistics", "building_block_statistics",
    ]
    node = result["chains"][0]["nodes"][0]
    assert list(node) == ["id", "atomic_num", "charge", "origin_idx"]
    assert node == {"id": 0, "atomic_num": 6, "charge": 0, "origin_idx": 42}
    edge = result["chains"][0]["edges"][0]
    assert list(edge) == ["source", "target", "bond_type", "aromatic"]
    assert result["sequences"] == [[["CO"]]]
    assert chain.nodes[0]["extra"] == "drop"


@pytest.mark.parametrize("failure_stage", ["string", "graph", "ensemble", "final", "sequence"])
def test_export_errors_precede_file_open(monkeypatch, tmp_path, failure_stage):
    creator = module.EnsembleCreator.__new__(module.EnsembleCreator)
    creator._generative_graph = nx.Graph(g2rins_string="C")
    data = _ensemble([nx.Graph()], [[[nx.Graph()]]])
    monkeypatch.setattr(creator, "_iter_chain_records", lambda **kwargs: iter([
        {"record": {"molecule": data.chains[0]}, "warnings": [], "discards": 0, "reasons": ()},
    ]))
    monkeypatch.setattr(creator, "_records_to_ensemble_data", lambda records: data)
    error = ValueError("projection failed")
    json_safe = module._json_safe

    def normalize(value, path="$"):
        if failure_stage == {"$['string']": "string", "$['ensemble']": "ensemble", "$": "final"}[path]:
            raise error
        return json_safe(value, path)

    def graph_json(graph):
        if failure_stage == "graph":
            raise error
        return {"format": {}}

    def sequence(unit, cache, **options):
        if failure_stage == "sequence":
            raise error
        return "C"

    monkeypatch.setattr(module, "_json_safe", normalize)
    monkeypatch.setattr(module, "generative_graph_json_data", graph_json)
    monkeypatch.setattr(module, "_sequence_unit_to_smiles", sequence)
    target = tmp_path / "ensemble.json"
    target.write_text("keep existing contents")
    with pytest.raises(ValueError) as caught:
        creator.create_ensemble(1, json_file=str(target))
    assert caught.value is error
    assert target.read_text() == "keep existing contents"


def test_no_export_work_without_json_file(monkeypatch):
    creator = module.EnsembleCreator.__new__(module.EnsembleCreator)
    data = _ensemble()
    monkeypatch.setattr(creator, "_iter_chain_records", lambda **kwargs: iter([
        {"record": {"molecule": "C"}, "warnings": [], "discards": 0, "reasons": ()},
    ]))
    monkeypatch.setattr(creator, "_records_to_ensemble_data", lambda records: data)

    def unexpected(*args, **kwargs):
        pytest.fail("JSON-disabled sampling must not enter export helpers")

    monkeypatch.setattr(module, "_fixed_ensemble_json_data", unexpected)
    monkeypatch.setattr(module, "_json_safe", unexpected)
    monkeypatch.setattr(module, "generative_graph_json_data", unexpected)
    monkeypatch.setattr(module, "_sequence_unit_to_smiles", unexpected)
    assert creator.create_ensemble(1, ensemble_info=True) is data


def test_equality_private_reexports_and_pickle_class_locations():
    for name in (
        "_edge_ids", "_graphs_equal", "_array_equal", "_masked_equal",
        "_comparison_result", "_is_collection", "_metadata_equal", "_RECURSIVE_METADATA_TYPES",
    ):
        assert getattr(module, name) is getattr(_metadata_equality, name)
    for name in ("EnsembleData", "ConvergedEnsembleData", "ConvergenceCheckpoint", "_ChainRecord", "_DeferredChainRecord", "_UnitOccurrence"):
        assert getattr(module, name).__module__ == "g2rins.ensemble_creator"
