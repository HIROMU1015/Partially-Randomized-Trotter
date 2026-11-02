"""Static/code, byte, schema and saved-record joins only; no scientific analysis.

No numpy/Qiskit/FLINT imports, model evaluation, grant, sampling, compile or N/G.
All output is exclusive and separate from the fixed historical namespaces.
"""
import ast
import base64
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
BASE='86b8b002704aced8ca239414080f53793de81658'
C_SOURCE='e80d7510880f8db993b777c627bb8e95b05a070d'
OLD_SOURCE='0c329b30a81accb35c212be8defa66fee0dd0f12'
EVIDENCE='artifacts/non_df_h4_c_reference_evidence/2026-10-23'
PILOT='artifacts/non_df_h4_pilot/2026-10-23/run01'
CASES=['pauli_jw_PF','pauli_jw_PR','df_native_PF','df_native_PR']
STATES={'VERIFIED','EVIDENCE_EXISTS_UNCHECKED','MISSING_EVIDENCE','FAILED_CONDITION','ASSUMPTION_DEPENDENT','UNKNOWN'}

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(name,x):
    with (HERE/name).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,sort_keys=True);f.write('\n')
def git_bytes(ref,name):return subprocess.check_output(['git','show',ref+':'+name],cwd=ROOT)
def blob(data):return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
def lines(name,needle):
    matches=[i for i,s in enumerate((ROOT/name).read_text().splitlines(),1) if needle in s]
    assert matches,(name,needle)
    return dict(path=name,lines=matches,SOURCE=C_SOURCE if name in ('src/trotterlib/h4_ball_evidence.py','src/trotterlib/h4_legacy_export.py') else OLD_SOURCE,
                SHA256=sha(ROOT/name),needle=needle)

def github():
    first=read(HERE/'github_retrieval.json');more=read(HERE/'github_more.json');legacy=read(HERE/'github_legacy.json')
    initial={x['kind']:x for x in first['initial']}
    commit=json.loads(initial['commit']['data']['content']);branch=json.loads(initial['branch']['data']['content'])
    assert commit['sha']==branch['object']['sha']==BASE
    assert subprocess.check_output(['git','rev-parse',BASE+'^{tree}'],cwd=ROOT,text=True).strip()==commit['tree']['sha']
    records=[]
    report=initial['report']['data'];path='docs/non_df_h4_c_reference_evidence_20261023.md';data=git_bytes(BASE,path)
    assert report['sha']==blob(data) and report['content'].replace('\r\n','\n')==data.decode().replace('\r\n','\n')
    records.append(dict(path=path,ref=BASE,git_blob=blob(data),SHA256=hashlib.sha256(data).hexdigest(),verification='Git blob plus normalized UTF8 connector content'))
    for item in first['files']+more['files']+legacy['files']:
        assert item['status']=='fulfilled' and item['isError'] is False
        file=item['data'];raw=base64.b64decode(file['content']);data=git_bytes(item['ref'],item['path'])
        assert raw==data and file['sha']==blob(data),item['path']
        records.append(dict(path=item['path'],ref=item['ref'],git_blob=file['sha'],SHA256=hashlib.sha256(data).hexdigest(),verification='Exact base64-decoded bytes'))
    extras={x['kind']:x for x in more['extras']}
    source=json.loads(extras['C_SOURCE_commit']['data']['content']);assert source['sha']==C_SOURCE
    assert subprocess.check_output(['git','rev-parse',C_SOURCE+'^{tree}'],cwd=ROOT,text=True).strip()==source['tree']['sha']
    listing=json.loads(extras['run_directory']['data']['content']);members=[]
    for item in listing:
        actual=subprocess.check_output(['git','rev-parse',BASE+':'+item['path']],cwd=ROOT,text=True).strip()
        assert actual==item['sha'],item['path']
        members.append(dict(path=item['path'],type=item['type'],git_object=item['sha'],size=item['size']))
    record=dict(status='PASS_FIXED_GITHUB_SOURCE_RESULTS_AND_RUN_DIRECTORY',base_commit=BASE,C_SOURCE=C_SOURCE,
        branch_at_retrieval=branch['object']['sha'],base_tree=commit['tree']['sha'],C_SOURCE_tree=source['tree']['sha'],
        retrieved_files=records,run_directory_members=members,
        large_model_contents_scope='run directory blob identity and local committed bytes; no remote large-model numerical evaluation')
    save('github_verification.json',record);return record

def static():
    export=ROOT/'src/trotterlib/h4_legacy_export.py';tree=ast.parse(export.read_text())
    calls=[x for x in ast.walk(tree) if isinstance(x,ast.Call)]
    target=[kw.value for x in calls for kw in x.keywords if kw.arg=='exact_event_range_proved']
    assert len(target)==1 and isinstance(target[0],ast.Compare)
    assert ast.unparse(target[0])=="representation == 'pauli_jw'"
    record=dict(status='PASS_FLAG_ORIGIN_NOT_AN_EVENT_NORM_TEST',
        flag_expression=ast.unparse(target[0]),
        flag_origin=lines('src/trotterlib/h4_legacy_export.py','exact_event_range_proved='),
        flag_copy=lines('src/trotterlib/h4_ball_evidence.py',"range_plus_minus_saved_B_certified=case['exact_event_range_proved']"),
        target_test=lines('src/trotterlib/h4_ball_evidence.py',"if not case['circuit_correspondence_proved']"),
        correspondence_flag=lines('src/trotterlib/h4_legacy_export.py','circuit_correspondence_proved=True'),
        range_flag_is_not_TARGET_MET_gate=True,
        scope_disclaimer=lines('src/trotterlib/h4_legacy_export.py',"case['correspondence_scope']="),
        basis_rounded_input=lines('src/trotterlib/h4_ball_evidence.py',"raw=op['matrix'];invraw=op['inverse_matrix']"),
        basis_norm_allowance=lines('src/trotterlib/h4_ball_evidence.py','defect=max(defect,frobenius(Ui*U-I))'),
        fixed_signal_refusal_to_certify=lines('src/trotterlib/fixed_signal_accounting.py','certificate_verified=False'),
        normalization_refusal_to_certify=lines('src/trotterlib/h4_audited_evaluation.py','normalization_certificate_verified=False'),
        mathematical_inference='DF false is an intentionally withheld proof flag for encoded local matrices; it is not a saved numerical violation, and literal binary ancilla outcomes remain ±1 in the ideal measurement model.',
        actual_H4_range_violation_established=False)
    save('static_flag_path.json',record);return record

def old_compilation():
    cases=[];all_paths=[];extensions=Counter()
    for case in CASES:
        folder=ROOT/PILOT/'run'/case;r=read(folder/'result.json');v=read(folder/'validation.json')
        count=r['circuit_cost']['sample_count'];joins=sorted((folder/'trajectories').glob('*/price_join.json'))
        assert len(joins)==count
        samples={x['trajectory_id']:x for x in r['circuit_cost']['samples']}
        assert len(samples)==count
        orders=Counter();components=set();fingerprints=[]
        for p in joins:
            x=read(p);record_hash=x.pop('trajectory_record_hash');assert digest(x)==record_hash,p
            base={k:v for k,v in x.items() if k not in {'compiled_RZ','compiler','wrapper_fingerprints','event_costs_additive'}}
            event_hash=base.pop('event_record_hash');assert digest(base)==event_hash,p
            event_file=read(p.parent/'events.json');stored_event_hash=event_file.pop('event_record_hash')
            assert stored_event_hash==event_hash and event_file==base
            assert base['case_id']==case and base['binding']['SOURCE']==OLD_SOURCE
            sample=samples[base['trajectory_id']]
            assert sample['trajectory_record_hash']==record_hash and sample['RZ']==x['compiled_RZ']
            assert sample['seed']==base['seed'] and sample['event_record_hash']==event_hash
            assert x['compiler']==r['compiler'] and set(x['wrapper_fingerprints'])=={'cosine','sine'}
            assert all(isinstance(f,str) and len(f)==64 for f in x['wrapper_fingerprints'].values())
            for ev in base['events']:
                orders[str(ev['taylor_order'])]+=1;components.update(ev['selected_component_ids'])
            fingerprints.append(dict(trajectory_id=base['trajectory_id'],wrapper_input_fingerprints=x['wrapper_fingerprints']))
            all_paths.extend([p,p.parent/'events.json'])
        # Directory scan limited to these four known case namespaces.
        for p in folder.rglob('*'):
            if p.is_file():extensions[p.suffix]+=1
        row=dict(case=case,sample_count=count,joined_trajectories_verified=len(joins),compiled_wrapper_requests=read(folder/'actual_call_accounting.json')['wrapper_compile_requests'],
            compiled_signal_max_error=r['circuit_cost']['compiled_signal_max_error'],
            saved_compiled_error_scope='maximum float probability difference over both axes of fixed saved trajectories and one saved input state; not a rigorous bound',
            trajectory_operator_checks=v['trajectory_operator_checks'],forced_order_checks=v['forced_order_checks'],
            sampled_event_orders=dict(orders),observed_component_ID_count=len(components),
            SOURCE=OLD_SOURCE,compiler=r['compiler'],thresholds=v['thresholds'],
            result_path=str((folder/'result.json').relative_to(ROOT)),result_SHA256=sha(folder/'result.json'),
            validation_path=str((folder/'validation.json').relative_to(ROOT)),
            wrapper_fingerprints=fingerprints,compiler_fingerprint_semantics='pretranspile wrapped circuit; not compiled output serialization or formal equivalence proof')
        cases.append(row)
    qpy_or_qasm=[str(p.relative_to(ROOT)) for case in CASES for p in (ROOT/PILOT/'run'/case).rglob('*') if p.is_file() and p.suffix.lower() in {'.qpy','.qasm','.qasm3'}]
    assert not qpy_or_qasm
    record=dict(status='PASS_EXISTING_EVENT_PRICE_COMPILER_JOINS',cases=cases,trajectories_verified=sum(x['sample_count'] for x in cases),
        known_case_file_extensions=dict(extensions),compiled_circuit_serializations_in_known_case_roots=qpy_or_qasm,
        serialization_scope='Four known pilot case directories only; no claim about the entire filesystem',
        joined_paths_SHA256={str(p.relative_to(ROOT)):sha(p) for p in all_paths},
        numerical_reanalysis=False,new_compile=False,sampling_or_RNG_replay=False)
    save('legacy_compiler_evidence.json',record);return record

def saved_C():
    rows=read(ROOT/EVIDENCE/'axis_outcomes.json')['rows']
    assert len(rows)==40 and len({(x['case'],x['epsilon_sig'],x['axis']) for x in rows})==40
    assert all(x['status']=='TARGET_MET' and x['precision_bits']==64 for x in rows)
    assert all(x['formal_certificate_accepted'] is False and x['old_N'] is None and x['old_G'] is None for x in rows)
    results={rep:read(ROOT/EVIDENCE/'run01'/(rep+'_result.json')) for rep in ('pauli_jw','df_native')}
    registry=[]
    for row in rows:
        rep,method=row['case'].rsplit('_',1);result=results[rep]
        axis=next(a for a in result['axes'] if (a['case'],a['epsilon_sig'],a['axis'])==(row['case'],row['epsilon_sig'],row['axis']))
        assert axis['target_exact']==row['target_exact'] and axis['total_error_upper']['upper']==row['total_error_upper']
        assert sha(ROOT/row['saved_result_path'])==row['saved_result_SHA256']
        gates=dict(numerical_enclosure='VERIFIED',reference_encoded_full_H='VERIFIED',bias_encoded_saved='VERIFIED',
            normalization_nominal_log_B='VERIFIED',measurement_binary_range='VERIFIED',
            encoded_event_norm='VERIFIED' if rep=='pauli_jw' else 'MISSING_EVIDENCE',
            sampling_law='ASSUMPTION_DEPENDENT' if method=='PR' else 'VERIFIED',
            wrapper_transfer='MISSING_EVIDENCE',compiler_transfer='MISSING_EVIDENCE',
            conditional_statistics='ASSUMPTION_DEPENDENT',provenance='VERIFIED',
            saved_B_to_shot_upper_binding='EVIDENCE_EXISTS_UNCHECKED' if method=='PR' else 'VERIFIED',
            certificate_acceptance_implementation='MISSING_EVIDENCE',
            final_combined_error_budget='EVIDENCE_EXISTS_UNCHECKED')
        assert set(gates.values())<=STATES
        registry.append(dict(case=row['case'],epsilon_sig=row['epsilon_sig'],axis=row['axis'],status=row['status'],bits=64,
            gates=gates,formal_certificate_accepted=False,old_N=None,old_G=None,
            fixed_C_result=row['saved_result_path'],fixed_C_result_SHA256=row['saved_result_SHA256'],
            evidence_scope='VERIFIED numerical fields mean existing fixed encoded-model candidate evidence, not a new certificate or completed wrapper/compiler transfer'))
    for rep,r in results.items():
        assert r['normalization']['PR']['saved_log_B_upper_encloses'] is True
        assert r['normalization']['PR']['range_plus_minus_saved_B_certified']==(rep=='pauli_jw')
        assert r['normalization']['PR']['formal_certificate_accepted'] is False
    record=dict(schema='H4_existing_evidence_certificate_gate_audit_v1',BASE=BASE,C_SOURCE=C_SOURCE,axis_count=40,
        status_vocabulary=sorted(STATES),rows=registry,formal_certificate_generated=False,formal_certificate_accepted=False,
        no_new_unconditional_RNG_independence_requirement=True,
        original_H4_native_completion='UNKNOWN_UNRECOVERED_RETAINED_SEPARATELY',new_C_native_completion='EXISTING_PASS_SEPARATE')
    save('certificate_gate_matrix.json',record)
    with (HERE/'certificate_gate_matrix.csv').open('x',newline='') as f:
        cols=['case','epsilon_sig','axis','status','bits',*registry[0]['gates'],'formal_certificate_accepted']
        writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader()
        for x in registry:writer.writerow({k:x[k] for k in ('case','epsilon_sig','axis','status','bits','formal_certificate_accepted')}|x['gates'])
    return record

def main():
    g=github();s=static();old=old_compilation();c=saved_C()
    before=read(HERE/'preservation_before.json')
    for name,digest_value in before['files'].items():assert sha(name)==digest_value,name
    save('structural_audit.json',dict(status='PASS_STATIC_SAVED_SCHEMA_HASH_JOINS',fixed_GitHub_files=len(g['retrieved_files']),
        run_directory_members=len(g['run_directory_members']),old_trajectory_joins=old['trajectories_verified'],
        C_axes=c['axis_count'],old_files_verified=before['files_verified'],old_files_changed=[],
        scientific_arrays_evaluated=0,reference_law_bias_recomputed=0,sampling_compile_calls=0,shot_G_calls=0,
        new_certificate_generated=False,formal_acceptance=False,grant_issued=False,push=False))
    print(json.dumps(dict(status='PASS',GitHub_files=len(g['retrieved_files']),old_joins=old['trajectories_verified'],C_axes=40,old_files=before['files_verified'])))

if __name__=='__main__':main()
