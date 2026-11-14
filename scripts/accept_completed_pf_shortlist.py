"""Metadata overlay on A's completed intake and C's alignment; no risk calls."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
A = '1adb7e65a9e7c3adc32491d853f31d124393ead7'
C = '0b4b2d941cd669aaa45231e1236bc2d3158a9d12'
ADIR = 'docs/research/pf_shortlist_mean_completion_20261115/'
CDIR = 'docs/research/h4_five_method_candidate_alignment_20261115/'
OUTPUT = ROOT/'docs/research/pauli_q3_three_quarter_20261115/PF_PACKET_ACCEPTANCE.json'


def selected(source):
    value = json.loads(subprocess.check_output(
        ['git', 'show', source['commit']+':'+source['path']], cwd=ROOT))
    for key in source.get('selector', ()):
        value = value[key]
    return value


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def check(item, evidence, config, packet, candidate, rows):
    """Reuse A's binding PASS; compare unchanged metadata to pinned originals."""
    cid = item['configuration_id']
    require(cid == config['id'] == packet['configuration_id'] == candidate['configuration_id']
            == evidence['configuration_id'], 'configuration mismatch')
    require(item['total_view_id'] == evidence['total_view_configuration_id'], 'total view mismatch')
    require(evidence['intake'] == item['B_intake'] == 'PASS' and evidence['science_calls'] == 0
            and evidence['mean_values_preserved'], 'A intake evidence insufficient')
    require(evidence['configuration_source'] == item['configuration']
            and evidence['packet_source'] == item['signal_packet'], 'SOURCE locator mismatch')
    require(packet['source'] == item['unified_numerical_SOURCE'], 'numerical SOURCE mismatch')
    require([r['m'] for r in config['rounds']] == [r['m'] for r in packet['means']]
            == item['all_rounds'] == list(range(13)), 'whole 13-round packet required')
    require(len(rows) == 26 and {(r['m'], r['axis']) for r in rows}
            == {(m, a) for m in range(13) for a in ('X', 'Y')}, 'C row coverage mismatch')
    require(config['spec']['law'] == candidate['law'], 'C law mismatch')
    require(config['spec']['binding'] == candidate['binding'], 'C physical binding mismatch')
    require(config['spec']['profile'] == candidate['compiler_profile'], 'C profile mismatch')
    for round_, mean in zip(config['rounds'], packet['means']):
        require(mean['identity'] == round_['identity'] and mean['law_binding'] == round_['law_binding'],
                'round identity/law mismatch')
        require(mean['source'] == packet['source'] and mean['target'] == packet['target'],
                'round SOURCE/target mismatch')
        require(mean['qualification'] == 'UNQUALIFIED', 'do not promote qualification')
        for row in (r for r in rows if r['m'] == round_['m']):
            require(row['physical_candidate'] == mean['physical_candidate'], 'C physical candidate mismatch')
            require(row['original_canonical_id'] == round_['candidate']['id'], 'C canonical round mismatch')
            require(row['structure']['canonical'] == round_['structure'], 'C signed schedule mismatch')
    # Raw/reference/normalization/scalar are never transformed or recomputed.
    return dict(candidate=item['candidate'], configuration_id=cid, total_view_id=item['total_view_id'],
        configuration=deepcopy(item['configuration']), signal_packet=deepcopy(item['signal_packet']),
        numerical_SOURCE=deepcopy(packet['source']), A_intake='REUSED_PASS',
        metadata_check='PASS', original_qualification='UNQUALIFIED', MSE=None, N=None,
        sufficient_N=None, G=None, rounds=[dict(m=r['m'], physical_id=mean['physical_candidate']['id'],
            original_round_id=r['candidate']['id'],
            signal_selector=['means', i],
            original_producer_SOURCE=deepcopy(mean['provider_binding']['input_source']),
            C_rows=[dict(m=x['m'], axis=x['axis']) for x in rows if x['m'] == r['m']])
            for i, (r, mean) in enumerate(zip(config['rounds'], packet['means']))])


def build():
    index_source = dict(commit=A, path=ADIR+'PACKET_INDEX.json')
    evidence_source = dict(commit=A, path=ADIR+'B_INTERFACE_CHECK.json')
    alignment_source = dict(commit=C, path=CDIR+'ALIGNMENT.json')
    index, evidence, alignment = map(selected, (index_source, evidence_source, alignment_source))
    require(index['B_intake_pass'] == 6 and evidence['all_intake_pass'], 'six prior intakes required')
    out = []
    for item in index['packets']:
        cid = item['configuration_id']
        prior = next(r for r in evidence['rows'] if r['configuration_id'] == cid)
        candidate = next(r for r in alignment['candidates'] if r['configuration_id'] == cid)
        rows = [r for r in alignment['rows'] if r['configuration_id'] == cid]
        out.append(check(item, prior, selected(item['configuration']), selected(item['signal_packet']),
                         candidate, rows))
    require(len(out) == 6 and len({r['configuration_id'] for r in out}) == 6, 'six distinct candidates required')
    return dict(status='COMPLETED_SIGNALS_METADATA_ACCEPTED', A_index=index_source,
        reused_B_intake=evidence_source, C_alignment=alignment_source,
        semantics='OVERLAY_ONLY_PRESERVE_HISTORICAL_C_STATUS_AND_PRICE_EVIDENCE', candidates=out,
        scientific_calls=0, numerical_reanalysis=False, risk_calls=0,
        sufficient_N=None, global_RMSE_bound=None, G=None, G_qualified=None, mandatory_stop=True)


if __name__ == '__main__':
    output = build()
    with OUTPUT.open('x') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print('Six completed PF packets: A intake reused; B/C metadata PASS. No scientific calls.')
