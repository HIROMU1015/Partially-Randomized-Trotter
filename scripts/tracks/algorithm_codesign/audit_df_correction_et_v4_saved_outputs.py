"""Post-STOP saved bytes/metadata audit; stdlib only, no circuit or math run."""
import ast
import csv
import hashlib
import json
from collections import Counter,defaultdict
from fractions import Fraction as F
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'artifacts/track_b_df_correction_et_resumption_result/2026-10-12/v1'
AUD=ROOT/'artifacts/track_b_df_correction_et_execution_audit/2026-10-12/v1'
ART=ROOT/'artifacts/track_b_df_correction_et_resumption_preparation/2026-10-12'
S='016e02b486cd4b182b654af18743ef1ef50fe7e6'
A='111199c45a9c37ac81821512ea8a44e6093f50cc'
checks=[]
referenced_chunks=set()
logical_receipts=[]


def check(name,ok):
    checks.append({'name':name,'PASS':bool(ok)})
    if not ok:raise AssertionError(name)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def save(name,value):
    raw=(json.dumps(value,indent=2)+'\n').encode()
    check('review size '+name,len(raw)<=524288)
    with (AUD/name).open('xb') as f:f.write(raw)


def load(path):
    path=Path(path);index=Path(str(path)+'.review.json')
    if not index.exists():
        raw=path.read_bytes()
    else:
        rec=json.loads(index.read_text());parts=[]
        check('logical index '+index.name,rec['logical_path']==path.name and rec['reconstruction_verified'] is True)
        permitted=(path.parent/(path.name+'.chunks')).resolve()
        for i,row in enumerate(rec['chunks']):
            p=(path.parent/row['path']).resolve()
            check('chunk bounds '+row['path'],p.is_relative_to(permitted) and row['order']==i and 0<row['bytes']<=262144)
            raw=p.read_bytes();raw.decode('utf-8')
            check('chunk identity '+row['path'],len(raw)==row['bytes'] and sha(raw)==row['sha256'])
            referenced_chunks.add(p);parts.append(raw)
        raw=b''.join(parts)
        check('complete reconstructed identity '+path.name,len(raw)==rec['original_bytes'] and sha(raw)==rec['original_sha256'])
    value=json.loads(raw)
    logical_receipts.append({'path':str(path.relative_to(ROOT)),'bytes':len(raw),'sha256':sha(raw),'storage':'chunks' if index.exists() else 'raw'})
    return value


def payload_hash(value):return sha(json.dumps(value,sort_keys=True,separators=(',',':')).encode())


def validate_inventory(inventory,name):
    ir=inventory['ordered_native_ir']
    check('IR digest '+name,payload_hash(ir)==inventory['ordered_native_ir_sha256'])
    counts=Counter();rz=[]
    for i,row in enumerate(ir['operations']):
        gate,wires,params=row['name'],row['qubits'],row['params_binary64_exact']
        if (row['position']!=i or gate not in ('rz','sx','x','cx') or len(wires)!=(2 if gate=='cx' else 1)
            or len(set(wires))!=len(wires) or any(type(q) is not int or not 0<=q<ir['num_qubits'] for q in wires)
            or len(params)!=(1 if gate=='rz' else 0)):
            check('IR semantics '+name,False)
        counts[gate]+=1
        if gate=='rz':rz.append((i,params[0]))
    check('IR/count/phase/angle binding '+name,
          dict(counts)==inventory['gate_counts'] and counts['cx']==inventory['CX'] and counts['rz']==inventory['total_RZ']
          and sum(v for k,v in counts.items() if k!='cx')==inventory['1Q']
          and rz==[(r['position'],r['raw_binary64_radians']) for r in inventory['occurrences']]
          and ir['global_phase_binary64_exact']==inventory['global_phase_raw'])


def main():
    contract=json.loads((ART/'contract_v4.json').read_text());auth=json.loads((ART/'authorization.json').read_text())
    manifest=json.loads((ART/'source_manifest_v4.json').read_text())
    check('audit begins at execution HEAD A',subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==A)
    check('A sole parent S',subprocess.check_output(['git','rev-list','--parents','-n','1',A],cwd=ROOT,text=True).split()==[A,S])
    check('authorization-only changed paths',set(subprocess.check_output(['git','diff','--name-only',S,A],cwd=ROOT,text=True).splitlines())=={str((ART/'authorization.json').relative_to(ROOT)),'docs/tracks/algorithm_codesign/df_correction_et_execution_authorization_v4.md'})
    check('authorization exact bounds',auth['source_commit']==S and auth['science_execution_authorized'] is True and auth['runs']==1 and auth['retries']==0 and auth['mandatory_STOP'] is True)
    for row in manifest['files']:
        raw=(ROOT/row['path']).read_bytes();check('protected '+row['path'],len(raw)==row['bytes'] and sha(raw)==row['sha256'])
    check('contract binding',sha((ART/'contract_v4.json').read_bytes())==auth['contract_sha256'])
    check('manifest binding',sha((ART/'source_manifest_v4.json').read_bytes())==auth['source_manifest_sha256'])
    inv=json.loads((AUD/'original_run_inventory.json').read_text())
    current_files={str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file()}
    check('original result inventory no addition/deletion',current_files=={r['path'] for r in inv['files']})
    for row in inv['files']:
        raw=(ROOT/row['path']).read_bytes();check('original result immutable '+row['path'],len(raw)==row['bytes'] and sha(raw)==row['sha256'])
        check('review file size '+row['path'],len(raw)<=524288)
    marker=load(OUT/'one_shot_marker.json');terminal=load(OUT/'terminal_status.json')
    marker_hash=sha((OUT/'one_shot_marker.json').read_bytes())
    source_info={k:marker[k] for k in ('source','authorization','authorization_sha256','contract_sha256','source_manifest_sha256','branch','remote_sha')}
    check('marker exact identities',marker['source']==S and marker['authorization']==A and marker['remote_sha']==A and marker['runs']==1 and marker['retries']==0 and marker['mandatory_STOP'] is True)
    check('marker auth hash',marker['authorization_sha256']==sha((ART/'authorization.json').read_bytes()))
    check('terminal provenance matches marker',terminal['provenance']==source_info and terminal['marker_SHA256']==marker_hash)
    outer=json.loads((AUD/'outer_process_receipt.json').read_text());claim=json.loads((AUD/'runner_invocation_claim.json').read_text())
    stdout=(AUD/'outer_runner_stdout.log').read_bytes();printed=json.loads(stdout)
    check('one external invocation',outer['run_invocations']==claim['run_invocations']==1 and outer['retries']==claim['retries']==0)
    check('outer status stdout identity',outer['exit_status']==2 and len(stdout)==outer['stdout_bytes'] and sha(stdout)==outer['stdout_sha256'])
    check('terminal/outer declared failure and STOP',terminal['classification']==printed['classification']=='TECHNICAL_INCONCLUSIVE' and terminal['mandatory_STOP'] is True and printed['mandatory_STOP'] is True)
    check('failure reason bound',terminal['technical_reason']==printed['reason']=='CapFailure: Output/log cap at metadata boundary')
    check('no science winner',terminal['scientific_winner'] is None)
    check('no T inventory or jobs',terminal['synthesis_inventory_binding'] is None and not any(OUT.glob('T-*.job.json')) and not (OUT/'synthesis_inventory.json').exists())
    check('CD not recomputed/no recipe calls',terminal['CD_recalculation_calls']==0 and terminal['E_recipe_generation_calls']==0)
    cd=load(OUT/'CD_saved_substage_receipt.json')
    check('CD source/bytes bound',cd['saved_rows']==6 and cd['CD_recomputed'] is False and cd['old_run_reclassified'] is False and cd['sha256']==contract['resumption']['saved_CD_sha256'] and cd['old_run_classification']=='TECHNICAL_INCONCLUSIVE' and cd['E_cost_multiplication_allowed'] is False)
    saved_input=load(OUT/'E-inputs.json')
    original_input=json.loads((ROOT/contract['resumption']['saved_E_inputs']['path']).read_text())
    check('new E saved input direct reuse',saved_input=={str(r):original_input[str(r)] for r in (3,6)})
    for r in (3,6):
        job=load(OUT/f'E-r{r}.job.json');started=load(OUT/f'E-r{r}.started.json')
        check('native job saved input binding '+str(r),job['r']==r and job['stage']=='native' and job['frames']==saved_input[str(r)] and job['marker_sha256']==marker_hash)
        check('native exclusive start '+str(r),started['job_sha256']==sha(json.dumps(job,sort_keys=True).encode()) and started['retries']==0)
    complete=load(OUT/'E-r3.json');p3=load(OUT/'E-r3.json.prefix.json')['prefix'];p6=load(OUT/'E-r6.json.prefix.json')['prefix']
    check('r3 completed/r6 incomplete not final E stage',complete['compile_calls']==8 and len(complete['native'])==8 and p3['compile_calls']==8 and p6['compile_calls']==2 and len(p6['native'])==1 and not (OUT/'E-r6.json.review.json').exists())
    native_rows=[]
    for r,data,scope in [(3,complete,'COMPLETE_CHILD_IN_INCOMPLETE_RUN'),(6,p6,'PARTIAL_CHILD_IN_INCOMPLETE_RUN')]:
        for i,row in enumerate(data['native']):
            inventory=row['inventory'];validate_inventory(inventory,f'r{r}/{i}')
            check('saved numeric guard '+str(r)+'/'+str(i),F(row['numeric_operator_error_upper'])<=F(contract['E']['local_numeric_allowance']))
            native_rows.append({'r':r,'arm':row['arm'],'controlled':row['controlled'],'label':row.get('label'),'scope':scope,
                'CX':inventory['CX'],'RZ':inventory['total_RZ'],'1Q':inventory['1Q'],'depth':row['depth'],
                'native_numeric_error_upper':row['numeric_operator_error_upper'],'ordered_native_ir_sha256':inventory['ordered_native_ir_sha256'],
                'historical_expected':row.get('historical_expected_fingerprint'),'historical_matches_expected':row.get('historical_matches_expected'),
                'actual_T':None,'final_scientific_comparison_usable':False})
    observations=[]
    # Every committed observation must reconstruct, but no uncommitted generation
    # is parsed as a result. Payloads/array bytes are checked without regeneration.
    raw_observations={p for p in OUT.glob('*.raw-provenance-*.json') if not p.name.endswith('.review.json')}
    for logical in sorted({Path(str(p)[:-len('.review.json')]) for p in OUT.glob('*.raw-provenance-*.review.json')} | raw_observations):
        row=load(logical)
        check('raw payload hash '+logical.name,row['actual_fingerprint']==payload_hash(row['recursive_payload']))
        check('raw payload exception absence '+logical.name,row['fingerprint_exception'] is None and row['fingerprint_attempts']==1)
        check('observation provenance '+logical.name,all(row['provenance'][k]==source_info[k] for k in ('source','authorization','authorization_sha256','contract_sha256','source_manifest_sha256')) and row['provenance']['marker_sha256']==marker_hash)
        for name,array in row['input_arrays'].items():
            raw=bytes.fromhex(array['raw_bytes_hex']);check('saved array bytes '+logical.name+'/'+name,len(raw)==array['bytes'] and sha(raw)==array['sha256'])
        observations.append({'logical_path':str(logical.relative_to(ROOT)),'phase':row['phase'],'fingerprint_attempts':row['fingerprint_attempts'],
            'transpile_attempts':row['transpile_attempts'],'actual_fingerprint':row['actual_fingerprint'],'expected_fingerprint':row['expected_fingerprint'],
            'historical_matches_expected':row.get('matches_expected'),'context':row['context'],'fingerprint_exception':row['fingerprint_exception']})
    returned=[r for r in observations if r['phase']=='COST_RETURNED_BEFORE_IDENTITY_GUARD']
    precompile=[r for r in observations if r['phase']=='FINGERPRINT_COMPLETE']
    check('nine returned native observations one pending controlled',len(returned)==9 and len(precompile)==1 and precompile[0]['context']['active_modes']==6 and precompile[0]['context']['controlled'] is True and precompile[0]['transpile_attempts']==0)
    for index in OUT.glob('*.review.json'):
        logical=Path(str(index)[:-len('.review.json')])
        if not any(r['path']==str(logical.relative_to(ROOT)) for r in logical_receipts):load(logical)
    orphan=defaultdict(lambda:{'bytes':0,'files':[]})
    for p in OUT.rglob('chunk-*.txt'):
        if p.resolve() not in referenced_chunks:
            logical=p.parent.parent.name
            orphan[logical]['bytes']+=p.stat().st_size;orphan[logical]['files'].append(str(p.relative_to(ROOT)))
    total=sum(r['bytes'] for r in inv['files']);logs=sum(r['bytes'] for r in inv['files'] if r['path'].endswith('.log'))
    accounting={'original_output_bytes_after_terminal':total,'nominal_output_cap_bytes':contract['caps']['output_bytes'],
        'active_guard_output_cap_excluding_terminal_reserve':contract['caps']['output_bytes']-262144,'log_bytes':logs,'log_cap_bytes':contract['caps']['log_bytes'],
        'unreferenced_incomplete_generation_bytes':sum(v['bytes'] for v in orphan.values()),'unreferenced_generation_by_logical':dict(orphan),
        'original_inventory':str((AUD/'original_run_inventory.json').relative_to(ROOT)),
        'evidence_scope':'post-termination disk snapshot, not exact byte count at first guard trip; nonreferenced chunks never treated as complete results',
        'committed_logical_artifacts':logical_receipts,
        'technical_inference':'r6 controlled raw observation replacement held prior 32718729-byte committed generation while a new generation was being written. Static synchronous save precedes native transpile. Current committed state remains FINGERPRINT_COMPLETE, transpile_attempts=0. No historical-hash rejection is recorded.'}
    check('output rather than log cap',total>contract['caps']['output_bytes'] and logs==0)
    check('unreferenced partial replacement only controlled observation',set(orphan)=={'E-r6.json.raw-provenance-A-1.json.chunks'})
    summary={'classification':terminal['classification'],'technical_reason':terminal['technical_reason'],'source_S':S,'authorization_A':A,'execution_branch':source_info['branch'],
        'marker_sha256':marker_hash,'completed_stages':terminal['completed_stages'],'completed_workers':terminal['resource_receipt']['completed_labels'],
        'worker_exit_statuses':terminal['resource_receipt']['child_receipts'],'outer_exit_status':outer['exit_status'],
        'compile_reservations_saved':10,'native_cost_return_observations':9,'E_r3_saved_native_rows':8,'E_r6_saved_native_rows':1,
        'E_r6_controlled_transpile_reached':False,'E_r6_controlled_internal_recursive_definition_acquired':True,
        'synthesis_keys':0,'synthesis_inventory_created':False,'CD_recomputed':False,'saved_CD_rows':6,
        'source_guard_resource_receipt':terminal['resource_receipt'],'outer_process_receipt':outer,
        'original_output_bytes':total,'numeric_or_fingerprint_error_recorded':False,
        'original_run_retry':0,'this_run_retry':0,'runner_invocations':1,'mandatory_STOP':True,'scientific_winner':None,
        'prefix_as_final_positive_or_negative_evidence':False,'source_contract_manifest_authorization_marker_unchanged':True,
        'G10_RA_D0_molecule_NPZ_GPU_LP_sampling_trajectory':0,
        'interpretation':'technical output failure only; no A/B research acceptance or further execution authorized'}
    save('stage_observation_summary.json',{'observations':observations,'native_identity_rows':native_rows,'all_rows_diagnostic_only':True})
    save('storage_accounting.json',accounting)
    save('technical_summary.json',summary)
    save('STOP.json',{'status':'MANDATORY_STOP_AFTER_ONE_E_T_V4_RUN','classification':'TECHNICAL_INCONCLUSIVE','marker_sha256':marker_hash,'run_invocations':1,'retries':0,'additional_registered_computation':0,'next_stage_authorized':False,'scientific_winner':None})
    save('saved_output_audit.json',{'checks':checks,'PASS':len(checks),'FAIL':0,'auditor_scope':'stdlib saved bytes/JSON/IR/Fraction metadata only; no matrix/circuit/compile/synthesis/LP/science call','source_manifest_files':369,'original_result_files':len(inv['files']),'mandatory_STOP':True})
    print(json.dumps({'audit_PASS':len(checks),'classification':summary['classification'],'original_output_bytes':total,'orphan_bytes':accounting['unreferenced_incomplete_generation_bytes'],'native_observations':len(returned),'synthesis':0,'mandatory_STOP':True},indent=2))


if __name__=='__main__':main()
