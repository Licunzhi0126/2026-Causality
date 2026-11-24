"""Independent toy-graph checks for the five added experiment conditions."""
from itertools import permutations
import json

import numpy as np
import pandas as pd
import pytest

from mignet_ce.downstream.perturbation.condition import PAIR_IDS
from mignet_ce.downstream.perturbation.config import PerturbationConfig
from mignet_ce.downstream.perturbation.grn_io import GRNGraph
from mignet_ce.downstream.perturbation.mirror_perturbation import apply_mirrored_weight_perturbation
from mignet_ce.downstream.perturbation.motif_index import HIGHER_ORDER_TEMPLATES, build_motif_index, motif_edge_participation
from mignet_ce.downstream.perturbation.selection import discover_region, selection_from_region
from mignet_ce.downstream.perturbation.topology_metrics import TARGET_METRIC, compute_motif_metrics
from mignet_ce.downstream.perturbation import runner
from mignet_ce.downstream.perturbation import __main__ as cli
from mignet_ce.downstream.perturbation.symmetry import evaluate_mirror_pair


def toy(template, weight=2.0):
    n = max(max(edge) for edge in template) + 1
    adjacency = np.zeros((n, n))
    for edge in template:
        adjacency[edge] = weight
    return GRNGraph(tuple(map(str, range(n))), adjacency)


@pytest.mark.parametrize('pair,template', HIGHER_ORDER_TEMPLATES.items())
def test_exact_template_metrics_mirror_and_nested_selection(pair, template):
    graph = toy(template)
    index = build_motif_index(graph)
    assert len(index.occurrences(pair)) == 1
    assert motif_edge_participation(index, pair) == {edge: 1.0 for edge in template}
    config = PerturbationConfig(region_node_fraction=1.0, region_edge_fraction=1.0)
    region = discover_region(graph, pair_id=pair, communities={node: 0 for node in graph.nodes}, seed=12, config=config, motif_index=index)
    previous = set()
    for strength in config.strengths:
        selection = selection_from_region(region, strength=strength)
        selected = set(selection.selected_edges)
        assert previous <= selected
        assert selected <= set(template)
        previous = selected
        positive, negative, _ = apply_mirrored_weight_perturbation(graph, selected_edges=selection.selected_edges, mirror_fraction=1.0)
        assert np.array_equal(positive.adjacency + negative.adjacency, 2 * graph.adjacency)
        for edge in template:
            assert positive.adjacency[edge] == (4.0 if edge in selected else 2.0)
            assert negative.adjacency[edge] == (0.0 if edge in selected else 2.0)
        metric = TARGET_METRIC[pair]
        assert compute_motif_metrics(graph, baseline_scale=2.0, index=index)[metric] == 1.0
        assert compute_motif_metrics(positive, baseline_scale=2.0, index=index)[metric] == pytest.approx(2 ** (len(selected) / len(template)))
        assert compute_motif_metrics(negative, baseline_scale=2.0, index=index)[metric] == 0.0


@pytest.mark.parametrize('pair,template', HIGHER_ORDER_TEMPLATES.items())
def test_sparse_matches_against_directed_edge_set_oracle(pair, template):
    # Extra edges intentionally permit overlapping, non-induced instances.
    graph = GRNGraph(tuple('abcde'), np.array([
        [0, 1, 1, 0, 0], [1, 0, 1, 1, 0], [1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1], [0, 1, 1, 0, 0]], dtype=float))
    size = max(max(edge) for edge in template) + 1
    expected = {frozenset((nodes[a], nodes[b]) for a, b in template)
                for nodes in permutations(range(5), size)
                if all(graph.adjacency[nodes[a], nodes[b]] > 0 for a, b in template)}
    index = build_motif_index(graph)
    actual = [frozenset((nodes[a], nodes[b]) for a, b in template) for nodes in index.occurrences(pair)]
    assert len(actual) == len(set(actual))
    assert set(actual) == expected
    counts = {edge: sum(edge in occurrence for occurrence in expected) for occurrence in expected for edge in occurrence}
    assert motif_edge_participation(index, pair) == counts


def test_defaults_and_all_conditions_reporting(tmp_path):
    assert len(PAIR_IDS) == 13
    assert PerturbationConfig().strengths == tuple(i / 10 for i in range(1, 10))
    edges = tmp_path / 'grn.csv'
    edges.write_text('regulator,target,weight\na,b,2\nb,c,2\nc,d,2\nd,a,2\na,c,2\n', encoding='utf-8')
    records = runner.run_experiment(time_pair='toy', source_sample='s', target_sample='t', baseline_inputs={'grn_t': edges}, output_root=tmp_path / 'out', evaluate=False)
    assert len(records) == 234
    for pair in HIGHER_ORDER_TEMPLATES:
        rows = [row for row in records if row['pair_id'] == pair]
        assert len(rows) == 18
        assert all(row['baseline_motif_count'] > 0 and row['overall_pass'] for row in rows)
        assert all(row['topology_delta'] > 0 if row['direction'] == 'positive' else row['topology_delta'] < 0 for row in rows)
    csv = pd.read_csv(tmp_path / 'out/toy/perturbation_results.csv')
    assert len(csv) == 234
    payload = json.loads((tmp_path / 'out/toy/P9_four_node_feedback_cycle/strength_0.1/positive/topology/topology_metrics.json').read_text())
    assert payload['four_node_feedback_cycle'] > 0


@pytest.mark.parametrize('pair', HIGHER_ORDER_TEMPLATES)
def test_absent_motif_reports_zero_without_random_edges(tmp_path, pair):
    edges = tmp_path / 'grn.csv'
    edges.write_text('regulator,target,weight\na,b,1\n', encoding='utf-8')
    records = runner.run_experiment(time_pair='empty', source_sample='s', target_sample='t', baseline_inputs={'grn_t': edges}, output_root=tmp_path / 'out', pair_ids=(pair,), strengths=(0.1,), evaluate=False)
    assert all(row['selection_status'] == 'NO_BASELINE_MOTIF' and row['baseline_motif_count'] == 0 and row['selected_edge_count'] == 0 and row['actual_topology'] == 0 for row in records)


@pytest.mark.parametrize('pair,template', HIGHER_ORDER_TEMPLATES.items())
def test_unequal_weights_use_only_template_edges(pair, template):
    graph = toy(template)
    for i, edge in enumerate(template):
        graph.adjacency[edge] = float(2 ** (i + 1))
    index = build_motif_index(graph)
    expected = 2 ** ((len(template) + 1) / 2) / 3
    assert compute_motif_metrics(graph, baseline_scale=3, index=index)[TARGET_METRIC[pair]] == pytest.approx(expected)


def test_cli_topology_only_new_pair_and_strength_override(tmp_path):
    # Factory discovery requires the complete path layout even in topology-only mode.
    for time in ('1', '2'):
        stem = f'spot_heart_{time}'
        for relative in (f'spot/heart/{stem}.h5ad', f'cci/spot/{stem}_CCI_total.npz',
                         f'cci/spot/{stem}_index.tsv', f'grn/spot/{stem}/grn_edges.csv'):
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    _, _, inputs = cli._factory_inputs(data_root=tmp_path, organ='heart', layer='spot', source_time='1', target_time='2')
    inputs['grn_t'].write_text('regulator,target,weight\na,b,1\nb,c,1\nc,d,1\nd,a,1\n', encoding='utf-8')
    cli.main(['--data-root', str(tmp_path), '--organ', 'heart', '--source-time', '1', '--target-time', '2', '--output-root', str(tmp_path / 'out'), '--pair', 'P9_four_node_feedback_cycle', '--strength', '0.1', '--topology-only'])
    records = pd.read_csv(tmp_path / 'out/heart_spot_1_to_2/perturbation_results.csv')
    assert len(records) == 2
    assert set(records['strength']) == {0.1}
    assert set(records['training_status']) == {'NOT_REQUESTED'}


@pytest.mark.parametrize('invalid', [-1.0, float('nan'), float('inf')])
def test_existing_qc_rejects_invalid_motif_weights(invalid):
    pair = 'P9_four_node_feedback_cycle'
    graph = toy(HIGHER_ORDER_TEMPLATES[pair])
    region = discover_region(graph, pair_id=pair, communities={node: 0 for node in graph.nodes}, seed=1, config=PerturbationConfig())
    selection = selection_from_region(region, strength=0.1)
    positive, negative, _ = apply_mirrored_weight_perturbation(graph, selected_edges=selection.selected_edges, mirror_fraction=1.0)
    positive.adjacency[selection.selected_edges[0]] = invalid
    assert not evaluate_mirror_pair(baseline=graph, positive=positive, negative=negative, selection=selection, mirror_tolerance=1e-12).overall_pass
