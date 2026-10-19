"""Read-only metadata audit of disabled cost contracts; no scientific imports."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import h46_m5_cost_contract as c

ARTIFACT = 'artifacts/resource_applicability/track_a_h46_m5_cost_sampling_readiness/2026-10-20'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--private', required=True)
    p.add_argument('--output', help='Fresh directory for repeatable metadata audit outputs')
    a = p.parse_args()
    private = Path(a.private).resolve()
    bundle = c.ROOT / ARTIFACT
    out = Path(a.output).resolve() if a.output else bundle
    if a.output:
        out.mkdir(parents=True, exist_ok=False)
    checks = []

    def check(name, passed, **details):
        c.require(passed, name)
        checks.append(dict(check=name, PASS=True, **details))

    m = c.manifest()
    original = c.read(c.ROOT / 'artifacts/resource_applicability/track_a_h46_m5_signal_run02_results/2026-10-20/signal_records.json')['records']
    domain = c.design.selection(original)
    check('SAVED_LABEL_JOIN_ONLY_EXACT88', domain == m['candidate_table'] and len(domain) == 88)
    check('NO_UNKNOWN_LABELS_IMPUTED', all(not r['unresolved_epsilons'] for r in domain))
    check('ALL24_PR_RETAINED', sum(r['candidate']['method'] == 'PR' and r['proposal'] == 'COST_TARGET' for r in domain) == 24)
    check('DET48_ELIGIBLE_16_PRESERVED_NOT_CURRENT_TARGET',
          sum(r['candidate']['method'] != 'PR' and r['proposal'] == 'COST_TARGET' for r in domain) == 48 and
          sum(r['candidate']['method'] != 'PR' and r['proposal'] == 'NO_COMMON_ELIGIBLE_ENDPOINT' for r in domain) == 16)
    for name, n in [('candidate_cost_targets_88.csv', 88), ('candidate_endpoint_join_440.csv', 440), ('reuse_cost_axes_12.csv', 12)]:
        with (bundle / name).open(newline='') as f:
            rows = list(csv.DictReader(f))
        check('CSV_ROWS_' + name, len(rows) == n)
    check('SAVED12_COST_AXES_ACTIVE4', len(m['reuse_axes']) == 12 and sum(r['eligible_in_current_five_eps'] for r in m['reuse_axes']) == 4)
    check('ALL_PR_N_AND_TAIL_COSTS_NULL', len(m['sampling_proposal']) == 24 and all(
        r['n_E'] is None and r['confirmation_n_C'] is None and r['unseen_rare_cost'] is None and r['unseen_rare_variance'] is None for r in m['sampling_proposal']))
    check('SHOT_AD03_AND_EXECUTION_UNAPPROVED', m['shot_policy_adopted'] is False and m['AD03'] == 'DEFERRED' and m['valid_grants'] == m['scientific_dispatch'] == 0)

    stages = []
    for stage in c.STAGES:
        folder = private / ('proposal_' + stage)
        cfg = c.read(folder / 'config_DRAFT_PRIVATE.json')
        c.verify(cfg)
        c.require(str((c.ROOT/c.SIGNAL_RECORDS).resolve()) in {r['path'] for r in cfg['sealed_files']},'NORMALIZATION_RECORDS_SEALED')
        check(stage + '_SOURCE_AND_ALL_INPUT_SEALS', True, SOURCE=cfg['SOURCE'], seals=len(cfg['sealed_files']))
        check(stage + '_INVALID_UNCONSUMED', cfg['authorized'] is False and cfg['grant']['valid'] is False and
              not Path(cfg['run_root']).exists() and not any(Path(v).exists() for v in cfg['supervisor'].values()))
        for mode in ('--execute', '--supervise'):
            run = subprocess.run([sys.executable, '-B', str(c.ROOT / 'scripts/run_h46_m5_cost.py'), mode,
                                  '--config', str(folder / 'config_DRAFT_PRIVATE.json')], capture_output=True, text=True)
            reply = json.loads(run.stdout)
            check(stage + '_' + mode + '_REFUSED_BEFORE_SCIENCE', run.returncode == 2 and
                  reply['reason'] == 'EXPLICIT_UNIT_COST_EXECUTION_APPROVAL_REQUIRED' and not Path(cfg['run_root']).exists())
        if stage.endswith('DET'):
            expected = c.expected_tasks(stage, {})
            check(stage + '_EXACT_CALL_SCOPE', expected == cfg['tasks'] and
                  c.design.call_scope(expected, 'DET') == cfg['call_scope'] and
                  len(expected) == {'H4_DET': 19, 'H6_DET': 27}[stage])
        else:
            check(stage + '_NO_INVENTED_N_REGISTRY_OR_BUDGET', cfg['tasks'] is None and cfg['call_scope'] is None and
                  cfg['campaign_id'] is None and cfg['registry_path'] is None and len(cfg['decisions']['n_by_candidate']) == 12 and
                  all(x is None for x in cfg['decisions']['n_by_candidate'].values()))
        # These are disabled in-memory copies, never an approval or valid grant.
        changes = [
            ('SOURCE', '0' * 40), ('compiler_settings_hash', '0' * 64),
            ('compile_tolerance', 1e-6), ('other_tolerance', 1e-7),
            ('origin_signal_config', '/missing/origin.json'), ('campaign_id', 'unbound-campaign'),
            ('registry_path', '/missing/registry.json'), ('excluded_seeds_path', '/missing/exclusions.json'),
            ('approval_receipt_sha256', '0' * 64), ('runtime', {}),
            ('decisions', {}), ('resources', {})]
        for key, value in changes:
            modified = copy.deepcopy(cfg)
            modified[key] = value
            reason = None
            try:
                c.verify(modified)
            except Exception as exc:
                reason = str(exc)
            check(stage + '_REJECT_MUTATION_' + key, bool(reason), reason=reason)
        projection = {k: cfg['plan'][k] for k in ('status', 'SOURCE', 'stage', 'closure_digest', 'decisions',
                      'resources', 'tasks', 'call_scope', 'pipeline_identity', 'approval_receipt_sha256')}
        projection.update(private_plan_sha256=cfg['plan_sha256'], excluded_known_seed_count=4603,
                          registry_generated=False, authorized=False, valid_grant=False, retry=False,
                          seed_campaign=None, next_phase_authorized=False, mandatory_STOP=True)
        projection.update(compiler=cfg['compiler'], compiler_settings_hash=cfg['compiler_settings_hash'],
                          native_policy=c.design.NATIVE_POLICY, input_binding=cfg['systems'][stage[:2]]['binding'],
                          compile_tolerance=cfg['compile_tolerance'], other_tolerance=cfg['other_tolerance'],
                          old_main_profile_equivalence_certified=False, AD03='DEFERRED')
        c.save(out / (stage + '_contract_DRAFT.json'), projection)
        c.save(out / (stage + '_grant_INVALID.json'), dict(valid=False, SOURCE=cfg['SOURCE'], stage=stage,
               bound_private_plan_sha256=cfg['plan_sha256'], approval_reference=None, retry=False, next_phase_authorized=False))
        stages.append(dict(stage=stage, SOURCE=cfg['SOURCE'], private_plan_sha256=cfg['plan_sha256'],
                           seals=len(cfg['sealed_files']), tasks=None if cfg['tasks'] is None else len(cfg['tasks']), call_scope=cfg['call_scope']))

    exclusions = c.read(private / 'known_excluded_seeds_PRIVATE.json')
    check('KNOWN4603_SEEDS_ONLY_NO_FRESH_SEED_OR_SAMPLING', len(exclusions['values']) == 4603 == len(set(exclusions['values'])) and
          exclusions['future_fresh_registry_recheck_required'])
    old = c.read(private / 'preservation_before_PRIVATE.json')
    bad = [path for path, digest in old.items() if not Path(path).is_file() or c.sha(path) != digest]
    check('SCOPED3137_PROTECTED_FILES_UNCHANGED', len(old) == 3137 and not bad)
    catalog = c.read(bundle / 'fixed_evidence_catalog.json')
    for item in catalog:
        data = subprocess.check_output(['git', 'show', item['commit'] + ':' + item['path']], cwd=c.ROOT)
        check('GIT_EVIDENCE_' + item['commit'][:8], len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256'])
    c.save(out / 'metadata_audit.json', dict(status='PASS_METADATA_ONLY', checks=checks, stages=stages,
           new_molecular_signal=0, new_molecular_compile=0, new_molecular_sampling=0,
           new_shot_calculation=0, numerical_reanalysis=0, cost_ranking=0, valid_grants=0,
           author_self_checks=True, independent_review=False, mandatory_STOP=True))
    c.save(out / 'preservation_receipt.json', dict(status='UNCHANGED', files=len(old), changed=bad,
           private_snapshot_sha256=c.sha(private / 'preservation_before_PRIVATE.json'),
           scope='EXACT_KNOWN_USED_SOURCE_RUNTIME_RAW_GRANT_LEDGER_MARKER_STOP_PATHS_ONLY',
           unrelated_worktrees_frozen=False))
    print(json.dumps(dict(status='PASS', checks=len(checks), protected_files=len(old), stages=stages)))


if __name__ == '__main__':
    main()
