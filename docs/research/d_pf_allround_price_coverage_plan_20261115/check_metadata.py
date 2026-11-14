"""Fixed-source coverage delta; stdlib/Git JSON and SOURCE text only.

Run from this worktree: python3 -B docs/research/d_pf_allround_price_coverage_plan_20261115/check_metadata.py
No project imports, circuit/QPY/NPZ load, price calculation, fit or prediction.
Outputs a compact selector overlay, not a second 156-row price registry.
"""
import ast
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
ALIGN = ('0b4b2d941cd669aaa45231e1236bc2d3158a9d12', 'docs/research/h4_five_method_candidate_alignment_20261115/ALIGNMENT.json')
CELL = ('43c15bfa34a22f313e55b824223b200332b7f23f', 'docs/research/truncated_p7_admission_pr_df_evidence_20261114/CELL_PLAN_V2.json')
STAGE1 = ('b190bd815c886446291fac8b281c3c7df7343372', 'docs/research/d_pf_stage1_calibration_20261115/C_HANDOFF_INDEX.json')
STAGE2 = ('a55813373b84b9128cbd10c6812c0d26143a4280', 'docs/research/d_pf_stage2_holdout_20261115/run01/RESULT.json')
FROZEN = ('12372fa6b53c878dc5dd124e036572ba0cd82061', 'docs/research/pf_stage1_model_freeze_20261115/run01/FROZEN_MODELS.json')
CARRIER = '155529e20a8fac39e00786c40b57830b71203c4f'
LONG = 'artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/'
COMMON = '0b1886f53f7c3a5e99e0c13cb5f7090581c5e38c'
CPATH = 'artifacts/resource_applicability/track_a_pf_common_H4_results/2026-10-22/'
PILOT = '3eca46148070ba4d4b0f5815ef45943407cc959e'
PPATH = 'artifacts/resource_applicability/track_a_h46_m5_tol_pilot_run01/2026-10-19/H4_TOL_PILOT_DET/H4_TOL_FULL_68d5c67e330e9c5f__0/result.json'
M5 = '5764aacb0714c6be42cb6dc7c7d8fdf1c8d98bdb'
MPATH = 'artifacts/resource_applicability/track_a_h46_m5_det_cost_run01_results/2026-10-20/'
MRAW = Path('/home/AbeHiromu/projects/h46-m5-cost-sampling-readiness/20261020/private/authorization_DET_run01/H4_DET/run01/output/M5_MIN_e847f177d7169dd3__0/result.json')
MCFG = MRAW.parents[3] / 'config_APPROVED_PRIVATE.json'
BASE = 'af284d853890155bacd19e61850c196bd1d5c660'
COMPILER = '1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
PREPS = {'FULL': 'bb736dc3158aa52f205ae0c23e4be374ae4e5edfa614d3cbceda04dcd5a28cfd',
         'B0I': 'fc83f26f1612a41d1d34caae79e78d3b5ce8e961d0e442ca4a9e92168006cf31'}
FIELDS = ('system', 'method', 'formula', 'prefix', 'T', 'q', 'delta', 'R_outer', 'K')
sources, checks = [], []


def blob(c, p):
    return subprocess.check_output(['git', 'show', c + ':' + p], cwd=ROOT)


def load(c, p):
    raw = blob(c, p)
    sources.append(dict(commit=c, path=p, sha256=hashlib.sha256(raw).hexdigest()))
    return json.loads(gzip.decompress(raw) if p.endswith('.gz') else raw)


def require(value, name):
    if not value:
        raise ValueError(name)
    checks.append(name)


def physical(c):
    return tuple(c.get(f) for f in FIELDS)


def canonical_sha(record):
    return hashlib.sha256((json.dumps(record, sort_keys=True, indent=2) + '\n').encode()).hexdigest()


def function(c, p, name):
    return ast.dump(next(n for n in ast.parse(blob(c, p)).body if isinstance(n, ast.FunctionDef) and n.name == name))


def main():
    alignment, cells, stage1, stage2 = (load(*s) for s in (ALIGN, CELL, STAGE1, STAGE2))
    frozen = load(*FROZEN)
    require(sources[-1]['sha256'] == '845c705082565a2b44d198bddb3034497620d991a00604454394ae5acda9651a', 'FROZEN_BYTES_UNCHANGED')
    ids = {c['configuration_id'] for c in cells}
    rows = [(i, r) for i, r in enumerate(alignment['rows']) if r['configuration_id'] in ids]
    require(len(ids) == 6 and len(rows) == 156, 'SIX_CONFIGURATIONS_156_AXES')
    require([i for i, _ in rows] == list(range(156)), 'COMPACT_ALIGNMENT_ROW_RANGE')
    keys = {(r['configuration_id'], r['m'], r['axis']) for _, r in rows}
    require(len(keys) == 156 and all(sum(x['configuration_id'] == c for _, x in rows) == 26 for c in ids), 'NO_MISSING_OR_DUPLICATE_AXIS')
    lookup = {(r['configuration_id'], r['m'], r['axis']): r for _, r in rows}
    require(len(cells) == 36 and all(lookup[(c['configuration_id'], c['m'], c['axis'])]['physical_candidate'] == c['physical_candidate'] for c in cells), 'CELL_PLAN_ALL36_EXACT_PHYSICAL_JOIN')
    require({tuple(c['model_group']) for c in cells} == {tuple(m['model_group']) for m in frozen['models'].values()}, 'TWELVE_FROZEN_GROUPS')
    require(stage2['counts'] == dict(PASS=12, FAIL=0, UNKNOWN=0) and all(r['m'] == 6 for r in (lookup[(r['configuration_id'], 6, r['axis'])] for r in stage2['rows'])), 'STAGE2_EXISTING_VERDICT_ONLY_NO_REJUDGEMENT')
    widths = {c['configuration_id']: c['physical_candidate']['delta'] for c in cells}
    require(all(r['physical_candidate']['delta'] != widths[r['configuration_id']] for _, r in rows if r['m'] < 4), 'ALL_SHORT_WIDTHS_OUTSIDE_FROZEN_GROUP')
    require(all(r['physical_candidate']['delta'] == widths[r['configuration_id']] for _, r in rows if r['m'] >= 4), 'LONG_WIDTHS_MATCH_BUT_Q_VALIDATION_REQUIRED')
    target = {physical(r['physical_candidate']): r for _, r in rows}
    scans = []
    for filename in ('exploration_costs.json.gz', 'extension_parallel_costs.json.gz', 'confirmation_costs.json.gz', 'fixed_delta_exploration_costs.json.gz', 'reused_compiled_prices.json.gz'):
        records = load(CARRIER, LONG + filename)
        matches = [dict(key=k, m=target[physical(v['candidate'])]['m']) for k, v in records.items() if physical(v['candidate']) in target]
        scans.append(dict(path=LONG + filename, entries=len(records), matching_cells=matches))
        require(all(m['m'] in (3, 4) for m in matches), 'NO_SHORT0_2_OR_LONG7_12_MATCH:' + filename)
    reused = load(CARRIER, LONG + 'reused_compiled_prices.json.gz')
    catalog = load('e7dca5f7301c8633b17a59b6a7161331303aef2d', 'docs/research/d_full_pf_full_h_v2_final_receipts_20261113/COVERAGE_V2.json')
    cat_m3 = [r for r in catalog['rows'] if physical(r['physical_candidate']) in target and target[physical(r['physical_candidate'])]['m'] == 3]
    require(len(cat_m3) == 3 and all(r['X'] == r['Y'] == 'EXACT_METADATA_MATCH' for r in cat_m3) and catalog['receipt_source_checks']['status'] == 'EXACT_METADATA_MATCH', 'REUSE_EXISTING_FULL_M3_CATALOG_AND_SOURCE_PROOF')
    native = load(COMMON, CPATH + 'native_results.json.gz')
    identity = load(COMMON, CPATH + 'source_input_identity.json')
    science = load(CARRIER, LONG + 'scientific_provenance.json')
    require(identity['binding'] == science['inputs']['H4']['binding'], 'PF_COMMON_SCIENTIFIC_BINDING_EXACT')
    by_sha = {canonical_sha(r): (i, r) for i, r in enumerate(native)}
    pilot = load(PILOT, PPATH)
    completion = load(M5, MPATH + 'stage_completion_receipts.json')
    raw = MRAW.read_bytes()
    m5 = json.loads(raw)
    m5_sha = hashlib.sha256(raw).hexdigest()
    require(any(x['result_sha256'] == m5_sha for s in completion for x in s['completion']['results']), 'M5_PRIMARY_TRIAL_HASH_IN_PUBLISHED_COMPLETION')
    require(m5['input_binding'] == identity['binding'], 'M5_PRIMARY_BINDING_EXACT')
    inventory = load(M5, MPATH + 'private_raw_inventory.json')
    cfg_bytes = MCFG.read_bytes()
    require(next(r['sha256'] for r in inventory if r['path'] == 'H4_DET/config_APPROVED_PRIVATE.json') == hashlib.sha256(cfg_bytes).hexdigest(), 'ORIGINAL_M5_CONFIG_HASH')
    cfg = json.loads(cfg_bytes)
    providers = []
    for path in ('src/trottertracks/resource_applicability/ax2b_native_df_v5.py', 'src/trotterlib/rte_compiled_cost.py'):
        seal = next(r for r in cfg['sealed_files'] if r['path'] == cfg['provider_root'] + '/' + path)
        require(seal['sha256'] == hashlib.sha256(blob(BASE, path)).hexdigest() and blob(BASE, path) == blob('c31ef424175c02379eafebe8fc6c7548c1d9aabe', path), 'M5_PROVIDER_NATIVE_COMPILER_BYTES:' + path)
        providers.append(dict(path=path, sha256=seal['sha256'], provider_SOURCE='c31ef424175c02379eafebe8fc6c7548c1d9aabe', current_native_SOURCE=BASE))
    evidence = []
    for i, row in rows:
        if row['m'] != 3 or row['axis'] != 'X':
            continue
        c = row['physical_candidate']; old = reused[c['id']]
        require(physical(old['candidate']) == physical(c), 'M3_ACTUAL_SCHEDULE:' + c['id'])
        if old.get('result_sha256'):
            index, rec = by_sha[old['result_sha256']]
            axes = rec['prices']
            locator = dict(commit=COMMON, path=CPATH + 'native_results.json.gz', selector=[index], result_sha256=old['result_sha256'])
            source, scalar, prep = identity['SOURCE'], rec['scalar'], rec['preparation_hash']
            require(all(v['PASS'] for v in rec['checks']), 'M3_SAVED_ACTION_GATES:' + c['id'])
            require(all(rec['candidate'][k] == c[k] for k in ('system', 'method', 'formula', 'prefix', 'q')), 'M3_ORIGIN_CANDIDATE:' + c['id'])
        elif c['formula'] == '4th':
            rec = pilot
            axes = {v['axis']: v for v in rec['records']}
            locator = dict(commit=PILOT, path=PPATH, sha256=hashlib.sha256(blob(PILOT, PPATH)).hexdigest())
            source, scalar, prep = rec['SOURCE'], rec['scalar'], rec['preparation_hash']
            require(rec['binding'] == identity['binding'], 'PILOT_BINDING_EXACT')
            require(rec['status'] == 'PASS' and {'hadamard_cosine', 'hadamard_sine', 'compiled_vector_cosine', 'compiled_vector_sine'} <= {v['check'] for v in rec['checks']}, 'PILOT_SAVED_ACCEPTANCE_AND_AXIS_GATES')
        else:
            rec = m5
            axes = {v['axis']: v for v in rec['axis_costs']}
            locator = dict(path=str(MRAW), sha256=m5_sha, public_commit=M5, public_path=MPATH + 'direct_measured_axes_92.csv', legacy_id=rec['task']['candidate']['id'])
            source, scalar, prep = rec['SOURCE'], rec['scalar'], rec['preparation_hash']
            require(all(v['PASS'] for v in rec['checks']), 'M5_SAVED_ACTION_GATES')
        require(prep == PREPS[c['method']], 'M3_PREPARATION_HASH:' + c['id'])
        require(scalar == (2.29310124732 if c['method'] == 'FULL' else 2.293146549384713), 'M3_SAVED_TOTAL_SCALAR:' + c['id'])
        per_axis = {}
        for a, label in (('X', 'cosine'), ('Y', 'sine')):
            v = axes[label]
            require(v.get('compiler_settings_hash', v.get('compiler')) == COMPILER and v['metrics']['rz_count'] == old[label] and bool(v['fingerprint']), 'M3_SEPARATE_AXIS_RECEIPT:' + c['id'] + ':' + a)
            per_axis[a] = dict(rz_count=v['metrics']['rz_count'], fingerprint=v['fingerprint'])
        evidence.append(dict(alignment_rows=[i, i + 1], physical_id=c['id'], origin=locator, generation_SOURCE=source, preparation_hash=prep, saved_total_scalar=scalar, axes=per_axis))
    # Existing source proof is reused, plus the older M5 direct generation SOURCE.
    origins = ('e2179f68f6979f3ad26392bff877fb422ddfb744', 'c7d53f04c3be131078ee91e8bef3c78ca434ed3e', '635ba563766875d688ad6912499b7f90397d0cc5')
    require(all(function(c, 'scripts/h46_m5_native.py', 'compose') == function(BASE, 'scripts/h46_m5_native.py', 'compose') for c in origins), 'THREE_LEGACY_COMPOSE_AST_IDENTICAL')
    require(all(blob(c, 'src/trotterlib/h46_m5_plan.py') == blob(BASE, 'src/trotterlib/h46_m5_plan.py') for c in origins), 'SIGNED_FORMULA_SCHEDULE_SOURCE_BYTES_IDENTICAL')
    missing_paths = []
    private = Path('/home/AbeHiromu/projects/h46-long-time-pf-pr/20261025/cost')
    for r in target.values():
        if r['m'] in (3, 4, 5, 6):
            continue
        for cohort in ('EXPLORATION', 'CONFIRMATION'):
            p = private / cohort / r['physical_id'] / '0/result.json'
            if p.is_file():
                missing_paths.append(str(p))
    require(not missing_paths, 'NO_EXTRA_TRIAL_AT_KNOWN_EXACT_PATHS')
    counts = Counter('A' if 3 <= r['m'] <= 6 else 'C' for _, r in rows)
    require(counts == dict(A=48, C=108), 'COMPACT_RULES_PARTITION156')
    return dict(status='METADATA_ONLY_PASS', baseline_commit=STAGE2[0], sources=sources,
        selected_alignment_row_indices=[dict(start=0, stop_exclusive=156)],
        coverage_rules=[dict(rounds=[3, 4, 5, 6], category='A', axes=48), dict(rounds=[0, 1, 2], category='C', reason='ACTUAL_WIDTH_OUTSIDE_FROZEN_GROUP', axes=36), dict(rounds=[7, 8, 9, 10, 11, 12], category='C', reason='LONG_Q_NOT_VALIDATED', axes=72)],
        counts=dict(A=48, B=0, C=108, D=0), saved_m3=evidence, carrier_scan=scans, m5_native_provider=providers,
        scientific_calls=0, predictions=0, scope='PINNED_CARRIERS_CATALOG_PRIMARY_TRIALS_AND_KNOWN_EXACT_PATHS_ONLY',
        m3_consumer_binding='PENDING_METADATA_ONLY_NOT_A_NEW_GUARD_ACCEPTANCE',
        unlisted_archives='UNKNOWN_RECHECK_BEFORE_SCIENCE', passed_checks=len(checks), checks=checks,
        stage1_source=STAGE1, frozen_source=FROZEN, stage2_source=STAGE2, mandatory_STOP=True)


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False, indent=2))
