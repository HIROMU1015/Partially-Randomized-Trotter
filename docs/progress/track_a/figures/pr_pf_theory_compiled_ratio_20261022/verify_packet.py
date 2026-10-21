"""Technical packet checks only. Does not render or invoke scientific code."""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote
import xml.etree.ElementTree as ET

from build_figures import HERE, check, coverage_rows, data, exported_rows


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_matches(name, expected):
    actual = list(csv.DictReader((HERE / name).open()))
    normalize = lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else '' if v is None else str(v)
    assert actual == [{k: normalize(v) for k, v in r.items()} for r in expected], name


def verify():
    d = data()
    result = check(d)
    csv_matches('figure_data.csv', exported_rows(d))
    csv_matches('coverage_matrix.csv', coverage_rows(d))
    csv_matches('coverage_methods.csv', d['method_coverage'])
    csv_matches('historical_coverage.csv', d['historical_S2_coverage'])
    inputs = json.loads((HERE / 'input_sources.json').read_text())
    for entry in inputs:
        fixed = subprocess.check_output(['git', 'show', entry['commit'] + ':' + entry['source_path']], cwd=HERE)
        assert (HERE / entry['local_path']).read_bytes() == fixed
    # Check model and source identities against the actual saved pair records.
    pairs = json.loads((HERE / 'sources/stage3_pf_pr_comparisons.json').read_text())
    totals = json.loads((HERE / 'sources/stage3_shot_weighted_totals.json').read_text())
    shots = json.loads((HERE / 'sources/stage3_preserved_shots.json').read_text())
    for pair in d['references']:
        original = pairs[int(pair['source']['locator'][1:])]
        assert pair['rho_compiled'] == original['compiled_PR_PF']
        assert Fraction(pair['rho_theory_lower']) == Fraction(original['theory_PR_PF_lower'])
        assert Fraction(pair['rho_theory_upper']) == Fraction(original['theory_PR_PF_upper'])
        for role in ['PR', 'PF']:
            saved = pair[role]
            row = totals[int(saved['source']['locator'][1:])]
            shot = shots[int(saved['shot_source']['locator'][1:])]
            assert saved['candidate_id'] == row['candidate_id'] == shot['candidate_id']
            assert saved['G_compiled'] == row['compiled_G_saved']
            assert saved['G_theory_lower'] == row['theory_G_lower']
            assert saved['G_theory_upper'] == row['theory_G_upper']
            assert saved['shots'] == shot['shots']
    links = []
    for path in HERE.glob('*.md'):
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', path.read_text()):
            target = target.strip().strip('<>')
            if target.startswith(('http:', 'https:', '#', 'mailto:')):
                continue
            local = unquote(target.split('#', 1)[0])
            if local:
                assert (path.parent / local).exists(), (path.name, target)
                links.append([path.name, target])
    svg_text = []
    for stem in d['figures']:
        png = (HERE / (stem + '.png')).read_bytes()
        assert png.startswith(b'\x89PNG\r\n\x1a\n')
        assert (HERE / (stem + '.pdf')).read_bytes().startswith(b'%PDF-')
        tree = ET.parse(HERE / (stem + '.svg'))
        texts = [''.join(x.itertext()) for x in tree.iter() if x.tag.endswith('}text')]
        assert not any(re.search('[\u3040-\u30ff\u4e00-\u9fff]', t) for t in texts)
        assert any('H4' in t for t in texts)
        svg_text.append({'figure': stem, 'text_elements': len(texts), 'Japanese_text': False})
    prov = json.loads((HERE / 'provenance.json').read_text())
    for entry in prov['outputs']:
        path = HERE / entry['path']
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], entry['path']
    stop = json.loads((HERE / 'STOP.json').read_text())
    assert stop['provenance_sha256'] == sha(HERE / 'provenance.json')
    assert stop['push_executed'] is False and stop['new_scientific_calls'] == 0
    result.update(packet_outputs_verified=len(prov['outputs']), local_links=len(links),
                  SVG_structure=svg_text, source_bytes_equal_immutable_Git_blobs=True,
                  all_CSV_values_equal_JSON=True, STOP_provenance_binding=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(verify(), ensure_ascii=False))
