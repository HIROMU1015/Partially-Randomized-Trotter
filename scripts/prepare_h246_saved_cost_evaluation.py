"""Metadata-only sealing of the preauthorized H246 saved-value analysis."""
import argparse
import csv
import sys
from pathlib import Path
import subprocess
import run_h246_saved_cost_evaluation as c


def prepare(context_path, dest, source):
    context = c.read(context_path)
    dest = Path(dest).resolve()
    c.require(not dest.exists(), 'NEW_AUTHORIZATION_PACK_ONLY')
    dest.mkdir(parents=True)
    seals = {}
    def seal(p, role):
        p = Path(p).resolve()
        item = dict(path=str(p), bytes=p.stat().st_size, sha256=c.sha(p), role=role)
        if str(p) in seals:
            c.require(seals[str(p)]['sha256'] == item['sha256'], 'SAME_INPUT_BYTES')
        seals[str(p)] = item
        return item
    seal(context_path, 'input_specifications')
    groups, metadata = [], []
    records_path = c.ROOT/'artifacts/resource_applicability/track_a_h46_m5_signal_run02_results/2026-10-20/signal_records.json'
    costs_path = c.ROOT/'artifacts/resource_applicability/track_a_h46_higher_pf_dataset_v1/2026-10-20/measured_gate_costs_96.csv'
    conditions_path = costs_path.parent/'conditions.json'
    for p in (records_path, costs_path, conditions_path):
        c.require(p.read_bytes() == subprocess.check_output(['git', 'show',
            '605ca07a653241db49348fbd7f4e091f9f18ced7:'+str(p.relative_to(c.ROOT))], cwd=c.ROOT),
            'FIXED_PUBLIC_DATASET')
        seal(p, 'fixed_public_saved_data')
    h46_records = c.read(records_path)['records']
    conditions = c.read(conditions_path)
    prices = list(csv.DictReader(costs_path.open()))
    for spec in context['systems']:
        cfg = c.read(spec['config'])
        seal(spec['config'], 'original_acquisition_config')
        for item in cfg['sealed_files']:
            p = Path(item['path'])
            c.require(p.stat().st_size == item['bytes'] and c.sha(p) == item['sha256'], 'ORIGINAL_SEAL')
            seals[str(p)] = item
        root = Path(cfg['run_root'])
        for p in (Path(cfg['plan_path']), root/'end_audit.json', root/'completion_witness.json',
                  Path(cfg['supervisor']['exit']), Path(cfg['marker'])):
            seal(p, 'prior_acquisition_witness')
        audit = c.read(root/'end_audit.json')
        c.require(audit['status'] == 'ACQUISITION_AND_END_AUDIT_PASS_MANDATORY_STOP' and
                  audit['completed_trajectories'] == spec['trajectories'] and
                  audit['accepted_compile_axes'] == spec['axes'], 'FIXED_COMPLETE_ACQUISITION')
        accepted = {r['id']:r['result_sha256'] for r in audit['results']}
        if spec['system'] == 'H2':
            seal(cfg['signal_path'], 'H2_saved_outer32_signal')
            signal_bundle = c.read(cfg['signal_path'])
            records = [dict(r, preparation_hash=signal_bundle['preparation_hash'])
                       for r in signal_bundle['records']]
            binding = cfg['binding']
        else:
            records = [r for r in h46_records if r['candidate']['system'] == spec['system']]
            binding = cfg['systems'][spec['system']]['binding']
            c.require(binding == conditions['systems'][spec['system']]['binding'], 'DET_INPUT_BINDING')
        for record in records:
            candidate = record['candidate']
            entry = dict(record=record, binding=binding)
            if candidate['method'] == 'PR':
                tasks = [t for t in cfg['tasks'] if t['candidate']['id'] == candidate['id']]
                c.require(bool(tasks) and all(t['candidate'] == candidate for t in tasks), 'REGISTERED_PR_GROUP')
                indexes = []
                for task in tasks:
                    p = Path(cfg['output'])/task['id']/'result.json'
                    c.require(c.sha(p) == accepted[task['id']], 'PR_RESULT_SHA')
                    seal(p, 'accepted_exploration_result')
                    indexes.append(dict(task=task, result_path=str(p), sha256=accepted[task['id']]))
                entry.update(tasks=indexes, acquisition_SOURCE=cfg['SOURCE'],
                    acquisition_plan_sha256=cfg['plan_sha256'], direct_prices=None)
            else:
                rows = [r for r in prices if r['candidate_id'] == candidate['id']]
                c.require(len(rows) in (0, 2), 'ZERO_OR_TWO_DETERMINISTIC_AXES')
                rows.sort(key=lambda r: ['cosine', 'sine'].index(r['axis']))
                if rows:
                    c.require([r['axis'] for r in rows] == ['cosine', 'sine'] and all(
                        r['system'] == candidate['system'] and r['method'] == candidate['method'] and
                        r['formula'] == candidate['formula'] and int(r['prefix']) == candidate['prefix'] and
                        int(r['q']) == candidate['q'] and r['compiler_settings_hash'] == c.science.COMPILER and
                        r['preparation_hash'] == record['preparation_hash'] for r in rows), 'DET_IDENTITY_JOIN')
                entry.update(tasks=[], direct_prices=[int(r['rz_count']) for r in rows] if rows else None,
                    cost_provenance=rows,det_comparison_scope='SAME_NATIVE_M5_SUBSET_ONLY_NOT_ALL_S2_S4_CONTROLS')
            groups.append(entry)
        metadata.append(dict(system=spec['system'], public_commit=spec['public_commit'],
            SOURCE=cfg['SOURCE'], plan_sha256=cfg['plan_sha256'],
            end_audit_sha256=c.sha(root/'end_audit.json'), binding=binding))
    for rel in c.SOURCE_PATHS:
        c.require((c.ROOT/rel).read_bytes() == subprocess.check_output(
            ['git', 'show', source+':'+rel], cwd=c.ROOT), 'SEALED_COMMITTED_SOURCE')
        seal(c.ROOT/rel, 'analysis_source')
    for module in ('statistics', 'decimal', 'fractions', 'math'):
        imported = __import__(module)
        seal(imported.__file__, 'analysis_runtime_'+module)
    seal(sys.executable, 'analysis_python_binary')
    proposal = c.ROOT/'artifacts/resource_applicability/track_a_h246_saved_evaluation_preflight/2026-10-21/policy_adoption_required.json'
    seal(proposal, 'unchanged_prior_proposal')
    adoption = dict(adopted=True, proposal_id=c.read(proposal)['id'], applicable_series=c.read(proposal)['applicable_series'],
        adoption_evidence='User continuation instruction following explicit preflight policy proposal: 続きを行って',
        initial_numeric_one_shot_instruction_already_received=True,
        epsilon_grid=list(c.science.EPS), modes=list(c.science.MODES), alpha_axis=.025,
        allowance_factor=1, empirical_reference_conditional=True, strict_bias_certificate=None,
        frozen_shot_SOURCE_sha256=c.sha(c.ROOT/'scripts/d0_frozen_shot_rule_v1.py'),
        modifies_old_contracts=False, AD03='DEFERRED')
    c.save(dest/'policy_adoption_PRIVATE.json', adoption)
    seal(dest/'policy_adoption_PRIVATE.json', 'current_policy_adoption')
    plan = dict(SOURCE=source, stage='H246_SAVED_VALUE_COST_EVALUATION', scope=c.SCOPE,
        python_executable=sys.executable, python_version=sys.version, adoption=adoption,
        run_root=str(dest/'run01'), groups=groups, input_metadata=metadata,
        comparison_scope='FINITE_SAMPLE_PR_THREE_FORMULAS_AND_IDENTITY_BOUND_M5_DET_SUBSET',
        old_S2_S4_PF_controls_preserved=True, old_PF_complete_set_optimum_claimed=False,
        numerical_crosscheck=dict(moments='EXACT_RATIONAL_AND_STDLIB_CENTERED_FLOAT',
            shots='FROZEN_BINARY64_AND_DECIMAL_ROUND_EACH_PRIMITIVE',
            libm_shared=True, verification_relative_tolerance='256*binary64_epsilon',
            integer_shots_and_rational_identities_must_equal_exactly=True),
        sealed_files=list(seals.values()), retry=False, mandatory_STOP=True)
    pp = dest/'plan_APPROVED_PRIVATE.json'
    c.save(pp, plan)
    cfg = dict(authorized=True, SOURCE=source, plan=plan, plan_path=str(pp),
        plan_sha256=c.sha(pp), run_root=plan['run_root'])
    cfg['grant'] = dict(valid=True, SOURCE=source, plan_sha256=cfg['plan_sha256'],
        run_root=cfg['run_root'], scope=c.SCOPE, retry=False,
        grant_id='H246_SAVED_ANALYSIS_'+c.digest(plan))
    c.save(dest/'grant_APPROVED_PRIVATE.json', cfg['grant'])
    c.save(dest/'config_APPROVED_PRIVATE.json', cfg)
    c.verify(cfg)
    return dict(SOURCE=source, plan_sha256=cfg['plan_sha256'], groups=len(groups),
                sealed_files=len(seals), status='READY_NO_SCIENCE_STARTED')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--context', required=True)
    p.add_argument('--destination', required=True)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    print(prepare(a.context, a.destination, a.source))


if __name__ == '__main__':
    main()
