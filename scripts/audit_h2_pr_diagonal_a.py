"""Bounded A-only SOURCE/input/serialized-field audit; scientific calls zero.

Uses explicit original catalogue paths and fixed prior artifact references.
Never evaluates a primitive or imports science. Public outputs are relative
locators/digests, not original private paths, raw events or circuit payloads.
"""
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import h2_pr_identity_recovery as r
import h2_pr_diagonal_a as a
import h2_pr_stream_conformance as stream
import audit_h2_pr_original_identity as original
import audit_h2_pr_equivalence as previous

BASE=a.BASE
OUT=r.ROOT/'artifacts/resource_applicability/track_a_h2_pr_diagonal_reuse_a/2026-10-23'
FILES=('scripts/h2_pr_diagonal_a.py','scripts/h2_pr_diagonal_a_backend.py','scripts/audit_h2_pr_diagonal_a.py',
 'tests/test_h2_pr_diagonal_a.py','scripts/h2_pr_saved_qpy.py','scripts/h2_pr_stream_conformance.py',
 'scripts/h2_pr_identity_recovery.py','scripts/audit_h2_pr_original_identity.py','scripts/audit_h2_pr_equivalence.py',
 'scripts/h2_pr_equivalence_contract.py','scripts/h2_accepted_compare_contract.py','scripts/h2_common_production_contract.py',
 'scripts/h2_accepted_diagnostics.py','scripts/h2_diagnostics_binding.py','src/trotterlib/h2_common_native_policy.py')


def write(name,x):
    with (OUT/name).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def read(path):return r.loads(Path(path).read_bytes())


class ProjectionCursor(stream.Cursor):
    def __init__(self,circuit,groups):
        super().__init__(circuit);self.groups=groups;self.saved={};self.rows=[];self.piece=None
    def diagonal(self,operations,fragment):
        value=super().diagonal(operations,fragment)
        block=a.BLOCKS[self.groups.index(operations)];self.saved[block]=value
        return value
    def directional(self,operations,saved):
        phase=super().directional(operations,saved)
        block=a.BLOCKS[self.groups.index(operations)]
        payload=dict(global_phase_hex=phase.hex(),rzz=[dict(wires=w,angle_hex=x.hex()) for w,x in saved[0]],
                    rz=[dict(wires=w,angle_hex=x.hex()) for w,x in saved[1]])
        self.rows.append(dict(piece_time_hex=self.piece,primitive_argument_hex=dict(a.TIMES)[self.piece],
             block=block,saved_parameter_fields_sha256=r.digest(payload),independent=False))
        return phase


def source_call_audit(acquisition,provider):
    records=[]
    targets=[(acquisition/'scripts/h46_m5_native.py','ACQUISITION','scripts/h46_m5_native.py','compose'),
      (provider/'src/trottertracks/resource_applicability/ax2b_native_df_v5.py','PROVIDER','src/trottertracks/resource_applicability/ax2b_native_df_v5.py','build_partial_native'),
      (provider/'src/trottertracks/resource_applicability/ax2b_native_df_v5.py','PROVIDER','src/trottertracks/resource_applicability/ax2b_native_df_v5.py','_append_block'),
      (provider/'src/trotterlib/df_trotter/ops.py','PROVIDER','src/trotterlib/df_trotter/ops.py','one_body_diagonal_primitives'),
      (provider/'src/trotterlib/df_trotter/ops.py','PROVIDER','src/trotterlib/df_trotter/ops.py','df_squared_diagonal_primitives')]
    for path,role,rel,name in targets:
        text=path.read_text();tree=ast.parse(text);node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
        calls=[]
        for n in ast.walk(node):
            if isinstance(n,ast.Call):
                fn=ast.unparse(n.func)
                if any(token in fn for token in ('build_partial_native','_append_block','diagonal_primitives')):
                    calls.append(dict(line=n.lineno,call=ast.unparse(n)))
        records.append(dict(root=role,path=rel,file_sha256=r.file_digest(path),function=name,
            function_sha256=hashlib.sha256(ast.get_source_segment(text,node).encode()).hexdigest(),calls=calls))
    r.require(any('delta / 2' in x['call'] for x in records[1]['calls']),'SOURCE_HALF_TIME_BOUNDARY')
    r.require(any("T=s['signed_time']" in x['call'] for x in records[0]['calls']),'SOURCE_PIECE_TIME')
    for rec in records[2:3]:r.require(any('time' in x['call'] for x in rec['calls']),'DIRECT_PRIMITIVE_TIME_FORWARD')
    return dict(status='VERIFIED_SINGLE_HALF_TIME_MAPPING',records=records,
       time_table=[dict(piece_time_hex=p,primitive_argument_hex=t,division_count=1) for p,t in a.TIMES],
       clarification='User explicitly corrected signed half-time label: listed hex values identify full pieces, primitive receives exactly piece/2. No scientific condition changed.',
       previous_DRAFT_half_time_label_incorrect=True,previous_artifacts_left_unchanged=True,
       primitive_functions_do_not_rehalve_time=True,new_wrapper_primitive_does_not_rehalve_time=True,
       call_budget_difference=dict(previous_unapproved_plan_primitive_evaluations=6,proposed=6,change=0),
       previous_execution_call_path='compose -> build_partial_native(q=1,T=piece) -> _append_block(time=piece/2) -> original primitive(time)',
       prospective_call_path='fixed plan primitive_argument_hex=piece/2 -> original primitive(argument); do not invoke _append_block or a circuit builder')


def run(config,source):
    source=subprocess.check_output(['git','rev-parse',source],cwd=r.ROOT,text=True).strip()
    for f in FILES:r.require((r.ROOT/f).read_bytes()==r.git_bytes(source,f),'PINNED_AUDIT_SOURCE')
    catalogue=r.Catalogue();old=catalogue.verify_config(config);root=config.parents[2]
    compare=r.loads(r.git_bytes(r.BASE,r.PREVIOUS+'COMPARE_contract_BLOCKED.json'))
    watch=previous.watch_set(catalogue,old,root,compare);paths=sorted(watch);before=previous.observe(paths)
    old_audit=r.loads(r.git_bytes('588d19544a2afdfb55ec4b40475a58114bd47b0b',
        'artifacts/resource_applicability/track_a_h2_pr_original_identity/2026-10-23/preservation_audit.json'))
    r.require(len(paths)==16804,'FIXED_PROTECTED_FILES')
    for path,item in zip(paths,old_audit['files']):r.require(before[path]==item['after'],'ORIGINAL_PROTECTED_INPUT_CHANGED')
    prep_path=Path(compare['accepted_diagnostics']['run_root'])/'results/preparation_1.json';prep=read(prep_path)
    snapshot_receipt=Path(compare['inputs']['snapshot_receipt'])
    binding=dict(snapshot=compare['binding']['snapshot'],snapshot_receipt=r.file_digest(snapshot_receipt),
        hamiltonian=compare['binding']['hamiltonian'],state=compare['binding']['state'],
        preparation=prep['preparation_sha256'],common_operations=prep['operations_sha256'])
    acquisition=Path(compare['acquisition_root']);provider=Path(compare['provider_root'])
    sources=dict(prepare=r.file_digest(acquisition/'scripts/h46_new2_native.py'),
        one_body=r.file_digest(provider/'src/trotterlib/df_trotter/ops.py'),
        df_fragment=r.file_digest(provider/'src/trotterlib/df_trotter/ops.py'))
    plan=a.draft(binding,sources);call=source_call_audit(acquisition,provider)
    storage={x['task_id']:x for x in catalogue.storage}
    groups=[x['operations'] for x in prep['operation_stream']];known={r.digest(x) for x in groups}
    def task(anchor):
        t=anchor['task'];tid=t['id'];folder=Path(old['output'])/tid
        events=read(folder/'events_PRIVATE.json');result=read(folder/'result.json')
        times=[float(x['saved']['signed_step_time']).hex() for x in catalogue.schedule[tid]['pieces']]
        r.check_events(events,anchor,[x['saved']['rte_steps'] for x in catalogue.schedule[tid]['pieces']])
        output=[]
        for i,axis in enumerate(('cosine','sine')):
            manifest=next(x['manifest']['saved'] for x in storage[tid]['axes'][i]['archives'] if x['manifest']['saved']['archive_file']=='wrapper.qpy.zlib')
            w=original.original_qpy(folder/('quarantine_'+axis)/'wrapper.qpy.zlib',manifest['codec_receipt'])
            validated=stream.recognize(w,events['events'],prep,axis,result['scalar'])
            r.require(validated['global_phase_relation_exact'],'SAVED_GLOBAL_PHASE_RELATION')
            cursor=ProjectionCursor(w,groups)
            for piece,event in zip(times,events['events']):
                a.time_pair(piece,dict(a.TIMES)[piece]);cursor.piece=piece
                one=cursor.diagonal(groups[0],False);fragment=cursor.diagonal(groups[1],True)
                cursor.tail(event,known);cursor.directional(groups[1],fragment);cursor.directional(groups[0],one)
            cursor.take('PhaseGate',[4],[-result['scalar']*.8])
            if axis=='sine':cursor.take('SdgGate',[4])
            cursor.take('HGate',[4]);r.require(cursor.pos==len(w['instructions'])-1,'COMPLETE_PARAMETER_PROJECTION')
            output.append(dict(task=tid,candidate=t['candidate']['id'],sample=t['sample'],axis=axis,
                cohort='EXPLORATION',wrapper_raw_sha256=manifest['codec_receipt']['original']['sha256'],
                event_digest=events['event_digest'],signed_piece_times=times,parameter_occurrences=cursor.rows,
                saved_structure='VERIFIED',absolute_deterministic_diagonal_vs_common_preparation='UNRESOLVED',
                independent_existing_receipt=None,strict_reuse='BLOCKED',alternative_adopted=False,
                reason='No independent common-H/preparation six primitive receipt in bounded evidence. These parameter digests originate in saved QPY and cannot feed independent producer.'))
        return output
    with ThreadPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:axes=[x for pair in pool.map(task,catalogue.tasks) for x in pair]
    variants=defaultdict(set);occurrences=Counter()
    for axis in axes:
        for entry in axis['parameter_occurrences']:
            key=(entry['piece_time_hex'],entry['block']);variants[key].add(entry['saved_parameter_fields_sha256']);occurrences[key]+=1
    r.require(len(variants)==6 and all(len(v)==1 for v in variants.values()),'SAVED_PARAMETER_CONSENSUS')
    after=previous.observe(paths);r.require(before==after,'ORIGINAL_EVIDENCE_CHANGED')
    r.require(not any(k.split('.')[0] in ('numpy','qiskit','scipy','trotterlib','trottertracks') for k in sys.modules),'NO_SCIENTIFIC_IMPORTS')
    write('time_and_call_audit.json',call)
    write('execution_contract_DRAFT.json',plan)
    write('metadata_preflight.json',a.preflight(plan,binding,sources))
    write('axis_parameter_scope_134.json',dict(rows=axes,axes=134,A_absolute_verified_axes=0,
        static_parameter_coverage_axes=134,real_primitive_evaluations=0,case='A2',science_authorized=False))
    write('six_saved_parameter_groups.json',dict(status='SAVED_QPY_PARAMETER_IDENTITY_ONLY_NOT_INDEPENDENT',
        rows=[dict(piece_time_hex=p,primitive_argument_hex=dict(a.TIMES)[p],block=b,
         saved_parameter_fields_sha256=next(iter(v)),saved_axis_piece_occurrences=occurrences[(p,b)],
         independent=False) for (p,b),v in sorted(variants.items())],
        all_same_group_saved_fields_match=True,independent_receipt_found=False,formula_evaluation=0))
    write('input_binding.json',dict(binding=binding,producer_sources=sources,original_O00002_sha256=r.file_digest(config),
        snapshot_receipt_sha256=r.file_digest(snapshot_receipt),common_preparation_receipt_sha256=r.file_digest(prep_path),
        original_scientific_scope_preserved=True,private_paths_exported=False))
    write('SOURCE.json',dict(base_commit=BASE,audit_SOURCE=source,files=[dict(path=f,sha256=r.file_digest(r.ROOT/f)) for f in FILES],
        strict_COMPARE_SOURCE_unchanged=compare['source_commit'],strict_contract_digest_unchanged=r.digest(compare),
        alternative_contract_sha256=hashlib.sha256(r.git_bytes(BASE,a.PRIOR+'alternative_contract_DRAFT.json')).hexdigest()))
    write('preservation_audit.json',dict(status='PASS',checked_files=16804,changes=0,
      before_sha256=r.digest(before),after_sha256=r.digest(after),reference_earlier_audit_sha256=r.digest(old_audit),
      original_SOURCE_input_runtime_result_grant_ledger_marker_STOP_unchanged=True,
      previous_A_B_alternative_artifacts_unchanged=True,new_grants=0,raw_copies_written=0))
    write('STOP.json',dict(status='MANDATORY_STOP_DIAGONAL_A_PREPARATION_ONLY',case='A2',
       independent_real_receipts=0,A_condition_resolved=False,A_verified_axes=0,COMPARE='BLOCKED',alternative_adopted=False,
       scientific_authorized=False,real_primitive_evaluations=0,sampling=0,H2_circuit_build=0,
       lowering=0,compile=0,transpile=0,simulation=0,u_N_G=0,allowance=0,ranking=0,
       scientific_imports=0,new_grants=0,old_grants_reused=0,push=False,detailed_scientific_review=False,
       final_evidence_freeze=False,mandatory_STOP=True))
    print(json.dumps(dict(status='A2_EXISTING_PRIMITIVES_REUSED_PREPARATION_ONLY',axes=134,
        distinct_saved_parameter_groups=len(variants),independent_receipts=0,science=0,
        protected_files=16804,time_halving_confirmed=True,COMPARE='BLOCKED')))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);p.add_argument('--source',required=True)
    args=p.parse_args();run(args.config.resolve(strict=True),args.source)
