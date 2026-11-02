"""Metadata-only H8 eight-case feasibility contract, never a full-map grant."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

from .ax2a_preparation import digest
from .ax2b_h4_contract_v5 import file_hash
from .ax2b_h6_contract import primitive_time_schedule
from .h6_matched_contract_v1 import plan as old_plan
from .h48_preflight_contract_v1 import profile, observation_policy, source_paths as old_sources

RUNNER = 'scripts/resource_applicability/run_track_a_h8_signal_preflight_v1.py'
PREPARER = 'scripts/resource_applicability/prepare_track_a_h8_signal_preflight_v1.py'
INPUT_SHA = 'd4c4346127e97c241439ae1c4ae010962b0456b19f461cfd3d9b06e7d4a50955'
INPUT_BYTES = 1299688
INPUT_SOURCE = '2ff3fc9f6e1091cbd9915507acfa74112c7d3055'
NAMESPACE = 'H8_EIGHT_CASE_FEASIBILITY_V1_20261013'
AUTHORIZATION = 'H8のsignal/reference・16-mode回路/control検証と小規模8ケース・20 compileを行って'


def cells():
    rows = []
    for q in (1, 8):
        for formula in ('2nd', '4th'):
            rows.append(dict(id=f'B1_p28_q{q}_{formula}', method='B1', prefix=28,
                             q=q, formula=formula, r=None, R=None, K=None))
    for q, r, K in ((1, 4, 2), (2, 4, 2), (4, 4, 4), (8, 2, 2)):
        rows.append(dict(id=f'B2_p7_q{q}_2nd_r{r}_K{K}', method='B2', prefix=7,
                         q=q, formula='2nd', r=r, R=q*r, K=K))
    return rows


def plan():
    p = deepcopy(old_plan())
    p.update(schema='h8_eight_case_preflight_plan_v1', kind='H8_TECHNICAL_PREFLIGHT',
             actual_rank=28, cells=cells(),
             target=dict(model='linear_H8', geometry_angstrom=1., basis='sto-3g',
                         n_qubits=16, nelec_alpha=4, nelec_beta=4, sector_dimension=4900),
             input_policy='EXISTING_H8_SNAPSHOT_UNCHANGED',
             H8_full_campaign_authorized=False, full_comparison_contract_adopted=False)
    p['matrix_free']['num_threads'] = 1
    p['cost_rule'].update(seed_namespace=NAMESPACE, exploratory_random_trajectories=1,
                         confirmation_random_trajectories=0, acquire='all eight feasibility cases regardless of epsilon')
    p['sensitivity_cells'] = ['B1_p28_q1_4th', 'B2_p7_q2_2nd_r4_K2']
    p['b3_diagnostic'].update(q=[], r=[], K=[])
    pairs = set(); schedules = []
    for c in p['cells']:
        s = primitive_time_schedule(c, T=p['T'])
        s['registered_validation_times_v2'] = s['unique_primitive_times']
        schedules.append(dict(cell_id=c['id'], schedule=s))
        pairs.update(('one' if i == 0 else str(i-1), t) for i, t in s['unique_primitive_times'])
    p['coverage'] = dict(cells=schedules, physical_times=sorted(pairs),
                         primitive_actions=3*len(pairs), primitive_probe_count=3)
    p['caps_proposed'].update(compile=20, trajectory=4, occurrence=15, control_probe=40,
        primitive=3*len(pairs), reference_matvec=2000000, reference_matvec_per_action=100000,
        output_bytes=2*2**30, progress_records=5000, diagnostics=5000)
    p['numerical'].update(scope='binary64 independent occupation CSR/expm action, split-time/reference residual, spin-sector spectral/native and Horner/forward agreement; empirical, not certified')
    p['reference_policy'] = dict(method='independent occupation CSR with expm_multiply',
        check='single versus two half steps plus saved eigenstate residual; production Numba action on three probes',
        dense_4900_matrix=False, full_16_qubit_unitary=False, certified=False)
    return p


def seed(source, cell):
    return int.from_bytes(hashlib.sha256((NAMESPACE+'|'+source+'|'+INPUT_SHA+'|'+cell+'|0').encode()).digest()[:8], 'big')


def registrations(source, scope, signals=None):
    if scope == 'SIGNAL':
        return [dict(index=0, task=dict(kind='H8_SIGNAL'))]
    if scope not in ('HEAVY', 'REMAINING'):
        raise ValueError('H8_PREFLIGHT_SCOPE')
    by = {x['cell']['id']: x for x in signals['signals']}
    rows = []
    for index, c in enumerate(cells()):
        heavy = c['id'] == 'B1_p28_q8_4th'
        if heavy != (scope == 'HEAVY'):
            continue
        rows.append(dict(index=index, task=dict(cell=c, cell_id=c['id'],
            phase='eight_case_feasibility', replica=0, seed=seed(source, c['id']),
            ordinary=c['id'] in plan()['sensitivity_cells'], validate=True,
            expected_signal=by[c['id']]['signals']['corrected'], kind='H8_COST')))
    return rows


def resources(scope, workers=1):
    if scope in ('SIGNAL', 'HEAVY') and workers != 1:
        raise ValueError('H8_SINGLE_HEAVY_FIRST')
    if scope not in ('SIGNAL', 'HEAVY', 'REMAINING') or type(workers) is not int or not 1 <= workers <= 3:
        raise ValueError('H8_PREFLIGHT_WORKERS')
    cpus = [30] if scope != 'REMAINING' else [20, 21, 24][:workers]
    co = [31] if scope != 'REMAINING' else [22, 23]
    return dict(assigned_cpus=sorted(cpus+co), coordinator_cpu_set=co,
        cost_cpu_sets=[[c] for c in cpus], cost_workers=workers, cost_threads=1,
        cost_address_space_bytes=32*2**30, worker_rss_stop_bytes=24*2**30,
        coordinator_address_space_bytes=8*2**30, coordinator_hard_address_space_bytes=32*2**30,
        aggregate_rss_bytes=(24*workers+4)*2**30,
        job_ram_budget_bytes=(32 if workers == 1 else 96)*2**30,
        coordinator_rss_reserve_bytes=2*2**30, aggregate_margin_bytes=2*2**30,
        memory_is_reservation=False, gpu=False)


def observation(scope, workers=1):
    p = observation_policy()
    ram = resources(scope, workers)['job_ram_budget_bytes']
    p.update(fresh_host_available_bytes=int(152.25*2**30)+ram, fresh_effective_available_bytes=ram)
    return p


def source_paths(root):
    return sorted(set(old_sources(root)) | {RUNNER, PREPARER})


def input_identity(directory):
    d = Path(directory).resolve()
    names = ('h8_input_snapshot.npz', 'snapshot_receipt.json', 'df_receipt.json',
             'projection_receipt.json', 'worker_terminal.json')
    if not d.is_relative_to('/home/AbeHiromu') or any(p.is_symlink() for p in (d, *d.parents)):
        raise ValueError('H8_PRIVATE_INPUT_PATH')
    hashes = {n: file_hash(d/n) for n in names}
    r = json.loads((d/'snapshot_receipt.json').read_text())
    t = json.loads((d/'worker_terminal.json').read_text())
    if (hashes[names[0]] != INPUT_SHA or (d/names[0]).stat().st_size != INPUT_BYTES
        or r['sha256'] != INPUT_SHA or r['bytes'] != INPUT_BYTES
        or t['status'] != 'COMPLETE' or t['reason'] is not None or t['source_commit'] != INPUT_SOURCE):
        raise ValueError('H8_FIXED_INPUT_RECEIPT')
    return dict(directory=str(d), file_hashes=hashes, snapshot_path=str(d/names[0]),
        snapshot_receipt_path=str(d/names[1]), df_receipt_path=str(d/names[2]),
        input_source=INPUT_SOURCE, hamiltonian_hash=r['metadata']['hamiltonian_hash'],
        state_hash=r['metadata']['state_hash'])


def verify_binding(root, m, g, *, worker=False):
    root = Path(root)
    if (g.get('approved_by_user') is not True or g.get('one_shot') is not True
        or g.get('authorization_basis') != AUTHORIZATION or g.get('retry') is not False
        or g.get('manifest_sha256') != digest(m) or g.get('output') != m['output']
        or g.get('source_commit') != m['source_commit']):
        raise ValueError('H8_NEW_SCOPE_AUTH_BINDING')
    if m.get('H8_full_campaign_authorized') is not False or g.get('H8_full_campaign_authorized') is not False:
        raise ValueError('H8_FULL_CAMPAIGN_FORBIDDEN')
    if digest(m['plan']) != digest(plan()) or m['profile'] != profile():
        raise ValueError('H8_PLAN_OR_PROFILE_CHANGED')
    if m['input_identity'] != input_identity(m['input_identity']['directory']):
        raise ValueError('H8_INPUT_CHANGED')
    scope = m['scope']; count = m['proposed_resources']['cost_workers']
    if m['proposed_resources'] != resources(scope, count) or m['observation_policy'] != observation(scope, count):
        raise ValueError('H8_RESOURCE_BINDING')
    if m['base_manifest'] != {'plan': {'caps_proposed': {'output_bytes': 2*2**30, 'log_bytes': 2*2**20}}}:
        raise ValueError('H8_OUTPUT_CAP_BINDING')
    out = Path(m['output'])
    if not out.is_absolute() or not out.is_relative_to('/home/AbeHiromu') or str(out.resolve()) != str(out) or out.is_symlink():
        raise ValueError('H8_OUTPUT_PATH')
    signals = None
    if scope != 'SIGNAL':
        s = m['signal_binding']; path = Path(s['path'])
        if file_hash(path) != s['sha256']:
            raise ValueError('H8_SIGNAL_CHANGED')
        signals = json.loads(path.read_text())
        t = json.loads((path.parent/'worker_terminal.json').read_text())
        if t['status'] != 'COMPLETE' or t['source_commit'] != m['source_commit'] or t['scope'] != 'SIGNAL':
            raise ValueError('H8_SIGNAL_TERMINAL')
    if digest(m['registrations']) != digest(registrations(m['source_commit'], scope, signals)):
        raise ValueError('H8_TASK_SEED_BINDING')
    if scope == 'REMAINING':
        h = m['heavy_binding']; p = Path(h['path'])
        if file_hash(p) != h['sha256']:
            raise ValueError('H8_HEAVY_CHANGED')
        t = json.loads(p.read_text())
        if t['first_stop_reason'] is not None or not t['all_dispatched_reaped'] or t['completed_indices'] != [3]:
            raise ValueError('H8_HEAVY_NOT_COMPLETE')
    if set(m['source_hashes']) != set(source_paths(root)):
        raise ValueError('H8_SOURCE_CLOSURE')
    if not worker:
        subprocess.run(['git', 'merge-base', '--is-ancestor', m['source_commit'], 'HEAD'], cwd=root, check=True)
    for p, sha in m['source_hashes'].items():
        if file_hash(root/p) != sha:
            raise ValueError('H8_SOURCE_CHANGED:'+p)
        if not worker and hashlib.sha256(subprocess.check_output(['git', 'show', m['source_commit']+':'+p], cwd=root)).hexdigest() != sha:
            raise ValueError('H8_SOURCE_COMMIT_CHANGED:'+p)
    return True
