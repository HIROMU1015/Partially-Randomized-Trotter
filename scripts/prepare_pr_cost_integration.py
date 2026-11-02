"""Metadata-only fixed Git evidence receipt; never calls a scientific runner."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/trotterlib'))
import pr_cost_integration as core

FIXED = dict(H4='2588bb1ff8a03120050fdaa67b9cd30739131bec',
             H4_followup='97d25cc431516d720ce91c7cb4f148e061d8ea34',
             H6='648184eb2a6f1c2406df17c1796eb5cc195908a8',
             H8='839c403004cf9343730c43adb346024aea41999b',
             H8_D='bc9cceac6088d798dc0c6dca92490066258860e5',
             plan='41ba315b2ad369825577ad671c79b69c77e1f3cc',
             index='2c22c8edad6a497240f57f8654da3778c0df1fdd')
PREFIX = 'artifacts/resource_applicability/'
SERIES = dict(H4=('track_a_h4_native_validation', 12, 'legacy_fixed_rank'),
              H6=('track_a_h6_dba', 19, 'tol_only_1e-8'),
              H8=('track_a_h8_ab', 28, 'tol_only_1e-8'))


def git(*args):
    return subprocess.check_output(['git', '--no-optional-locks', '-c', 'maintenance.auto=false',
                                    '-c', 'gc.auto=0', '-C', str(ROOT), *args])


def blob(commit, path, receipts):
    raw = git('show', commit+':'+path)
    receipts.append(dict(commit=commit, path=path, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                         blob=git('rev-parse', commit+':'+path).decode().strip(),
                         url=f'https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/{commit}/{path}'))
    return raw


def load(commit, path, receipts):
    raw = blob(commit, path, receipts)
    return json.loads(gzip.decompress(raw) if path.endswith('.gz') else raw)


def stage_receipt(obj, expected_source=None):
    core.require(obj['native_exit']['exit_code'] == 0 and obj['native_exit']['same_parent_reaped'] is True
                 and obj['native_exit']['wait4_complete'] is True, 'SAVED_NATIVE_EXIT')
    core.require(obj['terminal']['status'] == 'COMPLETE_MANDATORY_STOP'
                 and obj['terminal']['exit_code'] == 0, 'SAVED_TERMINAL')
    if expected_source:
        core.require(obj['SOURCE'] == expected_source, 'SAVED_SOURCE')


def build():
    receipts, systems = [], {}
    reports = dict(H4='track_a_h4_native_validation_20261018.md',
                   H4_followup='track_a_h4_followup_20261018.md', H6='track_a_h6_dba_20261018.md',
                   H8='track_a_h8_ab_20261018.md', H8_D='track_a_h8_dba_20261018.md',
                   plan='track_a_targeted_parallel_validation_plan_20261018.md',
                   index='track_a_research_handoff_20261018.md')
    for key, name in reports.items():
        blob(FIXED[key], 'docs/research/'+name, receipts)
    for system, (name, rank, policy) in SERIES.items():
        commit = FIXED[system]; base = PREFIX+name+'/2026-10-18/'
        result_raw = blob(commit, base+'A_result.json.gz', receipts)
        result_bytes = gzip.decompress(result_raw); result = json.loads(result_bytes)
        core.require(result['stage'] == 'A' and result['status'] == 'COMPLETE_MANDATORY_STOP'
                     and len(result['signals']) == 16, 'COMPLETED_SIXTEEN_A_SIGNALS')
        inp = load(commit, base+('input_binding.json' if system == 'H4' else 'input_binding_v1.json'), receipts)
        receipt = load(commit, base+dict(H4='A_completion_receipt.json', H6='A_execution_receipt.json',
                                       H8='A_completion_receipt_v1.json')[system], receipts)
        if system == 'H6':
            terminal = load(commit, base+'completion/A_terminal_result.json', receipts)
            stage_receipt(terminal)
            source = terminal['SOURCE']
            # Public terminal carries the exact uncompressed scalar result.
            core.require(core.digest(terminal['result']) == core.digest(result), 'H6_SAVED_RESULT_CLOSURE')
        else:
            stage_receipt(receipt)
            core.require(receipt['final_gate']['status'] == 'PASS', 'A_FINAL_GATE')
            compressed = receipt.get('compressed_sha256', receipt.get('result_gzip_sha256'))
            uncompressed = receipt.get('uncompressed_sha256', receipt.get('result_sha256'))
            core.require(hashlib.sha256(result_raw).hexdigest() == compressed and
                         hashlib.sha256(result_bytes).hexdigest() == uncompressed, 'A_BYTE_RECEIPT')
            source = receipt['SOURCE']
        cells = [x['cell'] for x in result['signals']]
        acquisition = core.acquisition_candidates(result['signals'], cells)
        binding = dict(inp['binding']); profile = inp.get('profile', inp.get('profile_reference'))
        if profile is None:
            profile = dict(sha256='622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12')
        binding = dict(system=system, geometry='linear_1Angstrom', basis='STO-3G', rank=rank,
                       rank_policy=policy, T=.8, hamiltonian=binding['hamiltonian'], state=binding['state'],
                       snapshot=binding.get('snapshot', binding.get('input')), compiler=binding['compiler'],
                       profile=profile['sha256'], control='symmetric_directional')
        # Existing endpoint flags are copied, no bias/shot/normalization is recalculated.
        rows = [dict(cell=x['cell'], record_sha256=core.digest(x),
                     normalization_sha256=core.digest(dict(b=x['b'], log_B=x['log_B'])),
                     accounting_sha256=core.digest({k:v for k,v in x.items() if 'precision_rows' in k}))
                for x in result['signals']]
        n = 8  # independent proposal per system, matching that system's exploration cohort
        selected = acquisition['selected']
        systems[system] = dict(binding=binding, A_SIGNAL_SOURCE=source, A_result_commit=commit,
            A_result_path=base+'A_result.json.gz', records=rows, cells=cells,
            acquisition=acquisition, n_proposed=n, budget_is_execution_authorization=False,
            proposed_calls=dict(trajectory=n*len(selected), native_evolution=n*len(selected),
                wrapper_compile=2*n*len(selected), occurrence=n*sum(c['q'] for c in selected),
                RTE_event=n*sum(c['R'] for c in selected), control_probe=4*len(selected),
                independent_event_oracle=len(selected), signal=0, fit=0, quantum_shots=0),
            provider_SOURCE=inp['provider_SOURCE'], input_receipts=inp.get('inputs',inp.get('input_files')),
            source_receipt=receipt.get('SOURCE', source), profile_reference=profile,
            requires_before_scientific_execution=['explicit scoped user approval', 'sealed new SOURCE and runtime',
                'private input path bytes/SHA manifest', 'exact preparation view/export identities',
                'old master/derived seed exclusion receipt', 'fresh resources and exclusive output/one-shot lock'])
    # Inspect D/B schemas and native terminal evidence, not new cost arithmetic.
    evidence = []
    acceptance={}
    for role,key,path in [
        ('H4_D','H4','track_a_h4_native_validation/2026-10-18/D_completion_receipt.json'),
        ('H4_B','H4_followup','track_a_h4_followup/2026-10-18/B_completion_receipt.json'),
        ('H6','H6','track_a_h6_dba/2026-10-18/completion/final_acceptance_receipt.json'),
        ('H8_B','H8','track_a_h8_ab/2026-10-18/completion/final_acceptance_v1.json'),
        ('H8_D','H8_D','track_a_h8_dba/2026-10-18/completion/final_acceptance_v1.json')]:
        a=load(FIXED[key],PREFIX+path,receipts);acceptance[role]=a
        if role.startswith('H4'):
            stage_receipt(a);core.require(a['final_gate']['status']=='PASS','B_D_FINAL_GATE')
        elif role=='H6':
            for item in a['stages'].values():
                core.require(item['native_exit_code']==0 and item['same_parent_wait4_reaped'] is True
                             and item['final_gate']['status']=='PASS','H6_FINAL_GATE')
        else:
            for item in a.get('stages',a.get('cases')).values():
                stage_receipt(item);core.require(item['final_gate']['status']=='PASS','H8_FINAL_GATE')
    cases = [('H4', FIXED['H4'], PREFIX+'track_a_h4_native_validation/2026-10-18/D_result.json.gz'),
             ('H4_B', FIXED['H4_followup'], PREFIX+'track_a_h4_followup/2026-10-18/B_result.json.gz'),
             ('H6_D', FIXED['H6'], PREFIX+'track_a_h6_dba/2026-10-18/completion/D_terminal_result.json'),
             ('H6_B', FIXED['H6'], PREFIX+'track_a_h6_dba/2026-10-18/completion/B_terminal_result.json'),
             ('H8_B', FIXED['H8'], PREFIX+'track_a_h8_ab/2026-10-18/completion/B_terminal_result.json')]
    for role, commit, path in cases:
        value = load(commit, path, receipts)
        if role in ('H4','H4_B'):
            proof=acceptance['H4_D' if role=='H4' else 'H4_B']
            expected=proof.get('compressed_sha256',proof.get('result_gzip_sha256'))
            core.require(receipts[-1]['sha256']==expected,'H4_B_D_RESULT_BYTE_RECEIPT')
        elif role in ('H6_D','H6_B'):
            proof=acceptance['H6']['stages'][role[-1]]
            core.require(receipts[-1]['sha256']==proof['terminal_bundle_sha256'],'H6_TERMINAL_BYTE_RECEIPT')
        elif role=='H8_B':
            core.require(receipts[-1]['sha256']==acceptance['H8_B']['stages']['B']['published_result_sha256'],
                         'H8_B_TERMINAL_BYTE_RECEIPT')
        if 'terminal' in value:
            stage_receipt(value); body = value['result']
        else:
            body = value
        core.require(body['status'] == 'COMPLETE_MANDATORY_STOP', 'COMPLETED_B_D_RESULT')
        evidence.append(dict(role=role, commit=commit, path=path, top_keys=sorted(body),
            groups=body.get('groups'), cell_id=body.get('cell_id'),
            conditional_costs_are_not_unconditional=role.endswith('_B'),
            original_law_probability_receipts='requires private original_distribution/log_stratum_mass receipt'
                if role.endswith('_B') else None))
    for cid in ('B0_p15_q8_2nd', 'B0I_p16_q4_4th'):
        path=PREFIX+'track_a_h8_dba/2026-10-18/completion/'+cid+'_result.json'
        value=load(FIXED['H8_D'],path,receipts)
        core.require(receipts[-1]['sha256']==acceptance['H8_D']['cases'][cid]['published_result_sha256'],
                     'H8_D_RESULT_BYTE_RECEIPT')
        core.require(value['status']=='COMPLETE_MANDATORY_STOP' and len(value['rows'])==2, 'H8_D_TWO_AXES')
        evidence.append(dict(role='H8_D', cell_id=cid, commit=FIXED['H8_D'], path=path,
                             rows=value['rows']))
    return dict(schema='track_a_pr_cost_readiness_evidence_v1', fixed_commits=FIXED,
        systems=systems, B_D_evidence=evidence, files=receipts,
        scientific_values_reaccounted=False, ranking_performed=False, new_scientific_calls=0,
        H4_rank10='FP panel is a separate input/state series, excluded from legacy A_COST and B/D joins',
        H6_B_missing='K4 s2_multi unmeasured; K2 and K4 are different original laws',
        review_kind='author metadata checks; no independent AI reviewer')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args(); args.output.parent.mkdir(parents=True, exist_ok=True)
    value=build()
    with args.output.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False); f.write('\n')
    print(json.dumps(dict(status='METADATA_ONLY_PASS', files=len(value['files']),
                         A_candidates={k:len(v['cells']) for k,v in value['systems'].items()},
                         scientific_calls=0)))


if __name__=='__main__':
    main()
