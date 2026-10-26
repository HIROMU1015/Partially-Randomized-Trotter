"""Contract path binding regression; real evidence is byte/metadata-only."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from test_research_c_c1_h4_strong_ca_resources import artificial as strong_artificial, m
from test_research_c_c1_h4_ca_resources import artificial as legacy_artificial


def test_distinct_equal_byte_contract_paths_and_fixed_artificial_execution(tmp_path, strong_artificial):
    path, _ = strong_artificial()
    c = m.io.read(path)
    canonical = 'docs/canonical_strong_contract.json'
    target = tmp_path / canonical; target.parent.mkdir()
    target.write_bytes((tmp_path / c['records']['strong_contract']).read_bytes())
    b3path = tmp_path / c['records']['B3_contract']
    old = m.io.read(b3path); old['previous_contract'] = canonical
    b3path.write_text(json.dumps(old))
    c['saved_inputs'][c['records']['B3_contract']] = m.io.sha(b3path)
    # Historical schema: equal bytes alone do not satisfy the literal binding.
    with pytest.raises(ValueError, match='SAVED_REFERENCE_SHELL_GROUND_BINDING'):
        m.load_bundle(tmp_path, c)
    c['records']['strong_contract'] = canonical
    c['saved_inputs'][canonical] = m.io.sha(target)
    path.write_text(json.dumps(c))
    out = tmp_path / m.FAMILY / '2026-10-26/run01'
    with patch.object(m.prior, 'source_commit', return_value='synthetic-binding-regression'):
        result = m.run(tmp_path, path, out)
    assert result['scientific_units'] == 86
    assert all(v['status'] == 'NUMERICAL_PILOT_CA_PASS' for v in result['candidates'].values())
    assert not m.io.read(out / 'STOP.json')['failed']


def test_real_contract_binding_passive_only():
    root = Path(__file__).resolve().parents[2]
    prefix = root / 'docs/research'
    old = m.io.read(prefix / 'research_c_c1_h4_strong_baseline_ca_resources_contract_draft_20261026.json')
    path = prefix / 'research_c_c1_h4_strong_baseline_ca_resources_contract_binding_v2_20261026.json'
    new = m.preflight(root, path)
    previous = m.io.read(root / old['records']['B3_contract'])['previous_contract']
    assert old['records']['strong_contract'] != previous
    assert m.io.sha(root / old['records']['strong_contract']) == m.io.sha(root / previous)
    assert new['records']['strong_contract'] == previous
    assert new['source_manifest_sha256'] == old['source_manifest_sha256']
    assert new['conditions'] == old['conditions'] and new['call_limits'] == old['call_limits']
    assert not (root / new['proposed_fresh_output']).exists()
