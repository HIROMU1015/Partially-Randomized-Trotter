"""Static receipt intake; never imports or evaluates scientific SOURCE.

Only hashes, metadata identities, coverage and source AST/text are checked.
No signal, bias, allowance, normalization, shot, cost or ranking is computed.
The sender's private-file and numerical checks are not rerun or claimed here.
"""
import ast
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RECEIVED = HERE / 'received'
AUDIT = ROOT / 'artifacts/resource_applicability/track_a_pf_order_reversal_existing_evidence_audit/2026-10-23'
PREP = ROOT / 'artifacts/resource_applicability/track_a_pf_order_reversal_h2_common_policy_preparation/2026-10-23'
BASE = '366b2fd24f7e86aa82b5b518b6e3774520f65596'
EXPORT = 'e3531956af6a6d359e2f1fb320e9a5179f41eb98'
ACQ = '4ec26921c6872ed6e8667f167074e1e405cf1a3f'
NEW_PROVIDER = 'c31ef424175c02379eafebe8fc6c7548c1d9aabe'
OLD_PROVIDER = '217c971dd537b1cc7fbc13742bc0f6c263ff7f45'
IDS = ('H2O32_c7ec590a32a859bc', 'H2O32_8a2ecdd0294ffb74')
POLICY = 'H2_QISKIT130_SYMBOLIC_ZERO_SCALAR_GUARD_V1'
PLAN = '4dda5e68815f8359b3890dc7a9e301956a631d6cea500e8a90703224db719373'
END_AUDIT = '38ad4d594a3be9289bdf1697d9a37a6d677384899a3320666c75761a304f7723'


def read(path):
    # Decimal parsing is used only for unchanged-value equality joins.
    return json.loads(path.read_bytes(), parse_float=Decimal)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def git_batch(requests):
    """Read only the explicit immutable blob list, without importing code."""
    raw = subprocess.check_output(['git', 'cat-file', '--batch'], cwd=ROOT,
                                  input=('\n'.join(requests)+'\n').encode())
    results = []; position = 0
    for request in requests:
        end = raw.index(b'\n', position)
        header = raw[position:end].split()
        if header[-1] == b'missing':
            results.append(None); position = end + 1; continue
        assert len(header) == 3 and header[1] == b'blob', request
        size = int(header[2]); results.append(raw[end+1:end+1+size])
        position = end+1+size+1
    assert position == len(raw)
    return results


def main():
    checks = []
    def check(name, condition):
        checks.append(dict(check=name, PASS=bool(condition)))
        if not condition:
            raise AssertionError(name)

    check('base_HEAD_unchanged', git('rev-parse','HEAD').decode().strip() == BASE)
    check('all_preexisting_tracked_files_unchanged', git('diff','--name-only',BASE) == b'')
    check('git_index_unchanged', git('diff','--cached','--name-only') == b'')
    old_inventory = read(PREP/'file_inventory.json')
    for r in old_inventory['files']:
        check('previous_preparation_unchanged:'+r['path'], sha((PREP/r['path']).read_bytes()) == r['sha256'])
    for r in read(PREP/'inherited_audit_files.json')['files']:
        check('fixed_audit_unchanged:'+Path(r['path']).name, sha((ROOT/r['path']).read_bytes()) == r['sha256'])

    intake = read(HERE/'acquisition_manifest.json')
    projection_inventory = {r['path']:r for r in read(RECEIVED/'file_inventory.json')['files']}
    check('remote_commit_and_branch_verified', intake['commit'] == intake['branch_SHA_verified'] == EXPORT)
    for r in intake['files']:
        raw = (RECEIVED/r['path']).read_bytes()
        blob = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        check('remote_git_blob:'+r['path'], blob == r['expected_git_blob_sha1'])
        if r['path'] in projection_inventory:
            expected = projection_inventory[r['path']]
            check('sender_projection_inventory:'+r['path'], sha(raw)==expected['projection_sha256'] and len(raw)==expected['bytes'])
    check('intake_150_files', len(intake['files']) == 150)

    origins = read(RECEIVED/'origin_inventory.json')['originals']
    original_by_id = {r['id']:r for r in origins}
    check('origin_ids_unique', len(original_by_id) == len(origins))
    for r in intake['files']:
        if not r['path'].endswith('.json'):
            continue
        def walk(v):
            if isinstance(v,dict):
                if {'origin','pointer','saved'} <= set(v):
                    check('origin_reference:'+r['path']+':'+v['origin']+':'+v['pointer'],
                          v['origin'] in original_by_id and isinstance(v['pointer'],str))
                for value in v.values(): walk(value)
            elif isinstance(v,list):
                for value in v: walk(value)
        walk(read(RECEIVED/r['path']))

    context = read(RECEIVED/'acquisition_context.json')
    stage = read(RECEIVED/'stage_end_metadata.json')
    binding = context['binding']['saved']
    end = stage['end_audit']['saved']
    check('H2_SOURCE_stage_plan_exact', context['SOURCE']['saved']==ACQ and
          context['stage']['saved']=='H2_PR_EXPLORATION' and context['plan_sha256']['saved']==PLAN)
    check('end_audit_origin_hash', original_by_id[stage['end_audit']['origin']]['original_sha256']==END_AUDIT)
    check('stage_and_guard_audit_join', end['SOURCE']==ACQ and end['plan_sha256']==PLAN and
          stage['compiler_policy_end_audit']['saved']['end_audit_sha256']==END_AUDIT)
    check('original_guard_stage_summary_PASS', stage['compiler_policy_end_audit']['saved']['status']=='PASS' and
          stage['compiler_policy_end_audit']['saved']['policy']==POLICY)

    prep = read(RECEIVED/'preparation_metadata.json')
    bundle = read(RECEIVED/'task_acceptance_67.json')
    tasks = bundle['tasks']; counts = Counter(t['task']['candidate']['id'] for t in tasks)
    check('target_67_tasks_unique', len(tasks)==len({t['task']['id'] for t in tasks})==67)
    check('target_cohort_counts_33_34', counts==Counter({IDS[0]:33,IDS[1]:34}))
    check('exploration_only_no_pooling', bundle['cohort']=='EXPLORATION' and bundle['other_cohorts_pooled'] is False)
    accepted = {r['id']:r for r in end['results']}
    guard_files = set(); check_names = None
    for t in tasks:
        task = t['task']; task_id = task['id']; result=t['result']['saved']
        check('task_config_identity:'+task_id, task==result['task'] and result['stage']=='H2_PR_EXPLORATION')
        check('task_accepted_original_hash:'+task_id,
              accepted[task_id]==t['acceptance']['saved'] and
              accepted[task_id]['result_sha256']==original_by_id[t['result']['origin']]['original_sha256'])
        check('task_binding_SOURCE_preparation:'+task_id,
              result['SOURCE']==ACQ and result['plan_sha256']==PLAN and result['input_binding']==binding and
              result['preparation_hash']==prep['cut1']['saved']==context['preparation_hash']['saved'])
        check('native_byte_acceptance_saved_flags:'+task_id,
              result['byte_restoration_PASS'] is True and result['native_scientific_PASS'] is True and
              result['paired_trajectory'] is True and result['cohort_pooling'] is False and
              result['conditional_sampling'] is False)
        check('task_two_ordered_axes:'+task_id, [x['axis'] for x in t['axes']]==['cosine','sine'])
        check('saved_correctness_records_identity:'+task_id,
              [x['saved'] for x in t['saved_correctness']]==result['checks'])
        names = {x['check'] for x in result['checks']}
        check_names = names if check_names is None else check_names
        check('eight_saved_PASS_gates_no_reexecution:'+task_id, len(names)==8 and names==check_names and
              all(x['PASS'] is True and x['phase_alignment'] is False for x in result['checks']))
        for axis in t['axes']:
            cost = axis['cost']['saved']; label=task_id+':'+axis['axis']
            result_cost=next(x for x in result['axis_costs'] if x['axis']==axis['axis'])
            check('axis_cost_identity:'+label, cost==result_cost and cost['event_digest']==result['event_digest'] and
                  cost['compiler_settings_hash']==binding['compiler'] and cost['preparation_hash']==result['preparation_hash'])
            guard_name=axis['guard_projection']; guard_files.add(guard_name)
            guard=read(RECEIVED/guard_name); g=guard['saved']; origin=original_by_id[guard['origin']]
            check('guard_original_locator:'+label, task_id in origin['locator'] and
                  origin['locator'].endswith('compiler_phase_guard_'+('0' if axis['axis']=='cosine' else '1')+'.json'))
            check('guard_saved_policy_and_flags:'+label, g['policy']==POLICY and g['unchanged_gate_stream'] is True and
                  g['state_fit'] is False and g['raw_validation_threshold_changed'] is False and bool(g['passes']))
            check('guard_saved_per_pass_flags:'+label,
                  all(v['policy']==POLICY and v['unchanged_gate_stream'] is True and
                      v['input_or_output_state_fit'] is False and v['postcompile_phase_correction'] is False
                      for v in g['passes']))
    check('exact_134_guard_files', len(guard_files)==134 and guard_files==
          {str(p.relative_to(RECEIVED)) for p in (RECEIVED/'compiler_guard_receipts').rglob('*.json')})

    schedule = read(RECEIVED/'rte_schedule_metadata.json')['tasks']
    check('schedule_same_task_ids', {t['task_id'] for t in schedule}=={t['task']['id'] for t in tasks})
    check('336_saved_piece_records_no_generation', sum(len(t['pieces']) for t in schedule)==336)
    check('piece_metadata_unconditioned_exploration', all(p['saved']['cohort']=='EXPLORATION' and
          p['saved']['conditioned'] is False for t in schedule for p in t['pieces']))

    signals = read(RECEIVED/'signal_records_2.json')
    check('signal_bundle_SOURCE_and_preparation', signals['header']['SOURCE']['saved']==ACQ and
          signals['header']['preparation_hash']['saved']==prep['cut1']['saved'] and
          signals['header']['full_preparation_hash']['saved']==prep['fullprep3']['saved'])
    check('exact_two_signal_candidates', [x['saved']['candidate']['id'] for x in signals['records']]==list(IDS))
    epath = AUDIT/'sources/8613b3ecda2030a677b1690fe7aa3da23dec4873/artifacts/resource_applicability/track_a_h246_saved_cost_results/2026-10-21/evaluation.json'
    saved_e=read(epath)
    for r in signals['records']:
        r=r['saved']; cid=r['candidate']['id']
        g=next(g for g in saved_e['groups'] if g['candidate']['id']==cid)
        shot=next(x for x in g['rows'] if x['epsilon']==Decimal('0.0005') and x['mode']=='USUAL_SIGNED')['shot']
        check('saved_E_signal_bias_logB_allowance_identity:'+cid,
              shot['saved_signed_bias']==r['signal']['total_signed_bias'] and shot['log_B']==r['signal']['log_B'] and
              shot['u']==r['empirical_eligibility']['u'] and shot['log_B_margin']==r['empirical_eligibility']['log_B_margin'])
    old = read(RECEIVED/'old_pf_receipts.json')
    old_result=read(AUDIT/'sources/829d697ae7905b3ea90a61da9d2d4da15d8b757a/artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/D_completed_run02/result.json')
    check('old_and_new_input_bindings_equal', old['binding']['saved']==old_result['binding']==binding)
    for r in old['rows']:
        original=next(v for v in old_result['measurements'] if v['cell_id']==r['candidate'])
        check('old_saved_measurement_identity:'+r['candidate'], r['measurement']['saved']==original)
        check('old_compiled_gate_remains_missing:'+r['candidate'], all(r[k] is None for k in
              ('compiled_action_phase_receipt','compiled_circuit_saved_locator','compiled_vector_saved_locator')))

    runtime=read(RECEIVED/'runtime_provider_manifest.json')
    check('new_runtime_declares_provider_c31ef424', runtime['runtime']['saved']['provider_SOURCE']==NEW_PROVIDER)
    historical=runtime['historical_sealed_files']; provider=[]; science=[]; requests=[]; expected=[]
    for row in historical:
        if row['saved']['path'].startswith('${NEW_PROVIDER}/'):
            provider.append(row)
            for commit in (NEW_PROVIDER,OLD_PROVIDER):
                item=next(x for x in row['provider_git_comparisons'] if x['commit']==commit)
                requests.append(commit+':'+item['path']); expected.append((row,item))
        elif row['saved']['path'].startswith('${ACQUISITION_WORKTREE}/'):
            science.append(row); requests.append(ACQ+':'+row['fixed_Git_path']);expected.append((row,None))
    check('provider_and_acquisition_seal_counts',len(provider)==278 and len(science)==70)
    for request,raw,(row,item) in zip(requests,git_batch(requests),expected):
        wanted = item['sha256'] if item else row['fixed_Git_sha256']
        check('sealed_Git_SOURCE:'+request, (sha(raw) if raw is not None else None)==wanted)
        if item and item['commit']==NEW_PROVIDER:
            check('new_provider_seal_matches_fixed_Git:'+item['path'],wanted==row['saved']['sha256'])
        if item is None:
            check('acquisition_SOURCE_seal_matches_fixed_Git:'+request,wanted==row['saved']['sha256'])
    old_comparisons=[next(x for x in r['provider_git_comparisons'] if x['commit']==OLD_PROVIDER) for r in provider]
    different=[x['path'] for x in old_comparisons if x['matches_saved'] is False]
    absent=[x['path'] for x in old_comparisons if x['sha256'] is None]
    check('two_existing_provider_differences_identified', set(different)=={'src/trotterlib/README.md','src/trottertracks/resource_applicability/h48_preflight_contract_v1.py'})
    check('sixteen_provider_paths_absent_from_old_Git',len(absent)==16)
    changed=[r for r in historical if r['matches_saved'] is not True]
    check('sender_current_vs_historical_runtime_difference_explicit',len(historical)==2012 and len(changed)==1 and 'liblzma' in changed[0]['saved']['path'])
    save(HERE/'provider_scope_update.json', dict(new_provider_SOURCE=NEW_PROVIDER,old_provider_SOURCE=OLD_PROVIDER,
        provider_saved_Git_matches=len(provider),old_existing_path_differences=different,old_missing_paths=absent,
        current_byte_checks_are_sender_observations_not_reperformed_here=True,
        historical_guard_file_hash_attestation=False,complete_actual_import_graph_attested=False))

    for entry in read(HERE/'supplemental_source_manifest.json'):
        raw=(HERE/entry['local_path']).read_bytes()
        check('supplemental_fixed_SOURCE:'+entry['path']+':'+entry['commit'][:8], sha(raw)==entry['sha256'] and raw==git('show',entry['commit']+':'+entry['path']))
        ast.parse(raw)
    def function(path,name):
        return next(x for x in ast.parse(path.read_bytes()).body if isinstance(x,ast.FunctionDef) and x.name==name)
    new_h48=HERE/'sources'/NEW_PROVIDER/'src/trottertracks/resource_applicability/h48_preflight_contract_v1.py'
    old_h48=AUDIT/'sources'/OLD_PROVIDER/'src/trottertracks/resource_applicability/h48_preflight_contract_v1.py'
    check('h4_plan_AST_unchanged',ast.dump(function(new_h48,'h4_plan'))==ast.dump(function(old_h48,'h4_plan')))
    new_native=HERE/'sources'/ACQ/'scripts/h46_new2_native.py'
    prepare_ast=function(new_native,'prepare')
    check('new_basis_lowering_override_is_explicit_SOURCE',
          'gaussian_operations' in ast.dump(prepare_ast) and 'U_to_qiskit_ops_jw' in ast.dump(prepare_ast) and
          any(isinstance(n,ast.Try) and n.finalbody for n in ast.walk(prepare_ast)))

    for name in ('completion_status.json','STOP.json'):
        s=read(RECEIVED/name)
        check('received_mandatory_STOP:'+name,s['mandatory_STOP'] is True if 'mandatory_STOP' in s else s['status']=='MANDATORY_STOP')
    check('all_new_artifacts_inside_two_authorized_directories',all(p.startswith((str(PREP.relative_to(ROOT))+'/',str(HERE.relative_to(ROOT))+'/'))
          for p in git('ls-files','--others','--exclude-standard').decode().splitlines()))
    save(HERE/'technical_verification.json',dict(status='STATIC_RECEIPT_INTAKE_PASS_WITH_EXPLICIT_GAPS',
         passed=len(checks),failed=0,checks=checks,received_files=150,task_count=67,axis_count=134,
         scientific_function_imports=0,scientific_function_calls=0,shot_calculations=0,cost_evaluations=0,
         original_private_files_reverified_here=0,sender_scientific_or_private_audits_reexecuted=False,
         scope='Git/projection byte hashes, saved metadata identity joins, coverage and static SOURCE only',
         independent_scientific_review=False,policy_adopted=False,science_GO=False))
    inventory=[]
    for f in sorted(HERE.rglob('*')):
        if f.is_file() and f.name!='file_inventory.json':
            raw=f.read_bytes();inventory.append(dict(path=str(f.relative_to(HERE)),bytes=len(raw),sha256=sha(raw)))
    # Include the received sender inventory too; exclude only our own inventory.
    f=RECEIVED/'file_inventory.json';raw=f.read_bytes()
    inventory.append(dict(path=str(f.relative_to(HERE)),bytes=len(raw),sha256=sha(raw)))
    save(HERE/'file_inventory.json',dict(scope='LOCAL_UNCOMMITTED_RECEIPT_UPDATE',self_excluded=True,files=inventory))
    print(json.dumps(dict(status='STATIC_RECEIPT_INTAKE_PASS_WITH_EXPLICIT_GAPS',passed=len(checks),failed=0,
         received_files=150,tasks=67,axes=134,scientific_calls=0,mandatory_STOP=True)))


if __name__=='__main__':
    main()
