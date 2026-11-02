"""One-shot authority, immutable inputs and native audit for B scalar analysis.

The grant is valid for forty saved scalar diagnostics only. It cannot authorize
C, molecular signals, certificates, shots, costs, sampling or compilation.
"""
import csv
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
BASE_REF='ebe20403bcb5c4439162614f64d991a769d7ed68'
SCIENCE_SOURCE='0c329b30a81accb35c212be8defa66fee0dd0f12'
BASE_PATH='artifacts/non_df_h4_precision_bc/2026-10-23'
INPUT_PATHS=(
 'artifacts/non_df_h4_evidence_investigation/2026-10-23/precision_state_inventory.json',
 'artifacts/non_df_h4_pilot/2026-10-23/run01/contract.json',
 *('artifacts/non_df_h4_pilot/2026-10-23/run01/run/'+c+'/precision_rows.json' for c in
   ('pauli_jw_PF','pauli_jw_PR','df_native_PF','df_native_PR')),
 *('artifacts/non_df_h4_pilot/2026-10-23/run01/run/'+c+'/result.json' for c in
   ('pauli_jw_PF','pauli_jw_PR','df_native_PF','df_native_PR')))
CODE_FILES=('src/trotterlib/h4_precision_budget.py','src/trotterlib/h4_precision_control.py',
 'src/trotterlib/h4_enclosure_prototype.py','scripts/run_h4_precision_b.py',
 'tests/test_h4_precision_bc.py','tests/test_h4_precision_b_authority.py')


def load_module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


budget=load_module('b_scalar_budget',Path(__file__).with_name('h4_precision_budget.py'))
require=budget.require


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args])


def now():
    return datetime.now(timezone.utc).isoformat()


def runtime():
    paths={str(Path(sys.executable).resolve())}
    for name in ['fractions','decimal','_decimal','json','_json','math']:
        spec=importlib.util.find_spec(name)
        if spec and spec.origin and Path(spec.origin).is_file():
            paths.add(spec.origin)
    return dict(python=sys.version,files={p:sha(p) for p in sorted(paths)})


def source_manifest():
    commit=git('rev-parse','HEAD').decode().strip()
    files={}
    for name in CODE_FILES:
        committed=git('show',commit+':'+name)
        require((ROOT/name).read_bytes()==committed,'UNCOMMITTED_B_CODE:'+name)
        files[name]=sha(ROOT/name)
    return dict(SOURCE=commit,files=files,runtime=runtime(),scope='B_SCALAR_ONLY_C_ARTIFICIAL_PROTOTYPES_ONLY')


def collect_inputs():
    files={}
    for name in INPUT_PATHS:
        original=git('show',BASE_REF+':'+name)
        require((ROOT/name).read_bytes()==original,'OLD_INPUT_CHANGED:'+name)
        files[name]=sha(ROOT/name)
    inventory=read(ROOT/INPUT_PATHS[0]);plan=read(ROOT/INPUT_PATHS[1]);rows=[]
    require(plan['numerical']['maximum_u_over_headroom']==.01,'FIXED_GUARD')
    for row in inventory['rows']:
        case=row['case']
        folder=ROOT/'artifacts/non_df_h4_pilot/2026-10-23/run01/run'/case
        original=next(x for x in read(folder/'precision_rows.json') if x['epsilon_sig']==row['epsilon_sig'])
        original_result=read(folder/'result.json')
        require(original['shots']['N_total'] is None and original['total_RZ'] is None,'ORIGINAL_N_G_NULL')
        require(original_result['signal']['signed_bias']==inventory['cases'][case]['signed_bias'],'SIGNED_BIAS_NOT_ORIGINAL')
        require(original_result['normalization']==inventory['cases'][case]['normalization'],'NORMALIZATION_NOT_ORIGINAL')
        for axis,part in zip(budget.AXES,('real','imag')):
            saved=row['axes'][axis]['saved']
            require(saved==original['shots']['axes'][axis],'SAVED_AXIS_NOT_ORIGINAL')
            rows.append(dict(case=case,epsilon_sig=row['epsilon_sig'],axis=axis,
                epsilon_axis=saved['epsilon'],signed_bias=inventory['cases'][case]['signed_bias'][part],
                absolute_saved_bias=saved['bias'],saved_u=saved['allowance'],saved_headroom=saved['headroom'],
                saved_status=saved['status'],saved_shots=saved['shots'],saved_N_total=row['stored_N_total'],
                saved_G=row['stored_G'],group='R' if saved['headroom']>0 else 'H',
                normalization_as_saved=inventory['cases'][case]['normalization']))
    budget.validate_registry(rows)
    return dict(source_commit=BASE_REF,scientific_SOURCE=SCIENCE_SOURCE,files=files,rows=rows,rho=.01,
        scope='Forty saved scalars only; no arrays, signal or certificate calculation')


def prepare(pack, authorization):
    pack=Path(pack).resolve();authorization=Path(authorization)
    require(pack.parent==ROOT/BASE_PATH and pack.name=='B_run01','FIXED_B_NAMESPACE')
    source=source_manifest();inputs=collect_inputs()
    approved_bytes=authorization.read_bytes();approved_text=approved_bytes.decode('utf-8')
    require('段階Bの固定40軸' in approved_text and '段階C' in approved_text,'EXPLICIT_B_AUTHORIZATION_REQUIRED')
    authorization_sha=sha(authorization)
    authority=digest(dict(SOURCE=source['SOURCE'],inputs=digest(inputs),authorization_sha256=authorization_sha))
    claim=ROOT/'.h4_precision_b_claims'/authority
    require(not pack.exists() and not claim.exists(),'B_AUTHORITY_OR_NAMESPACE_ALREADY_RESERVED_NO_RETRY')
    pack.parent.mkdir(parents=True,exist_ok=True);pack.mkdir()
    claim.parent.mkdir(parents=True,exist_ok=True);claim.mkdir()
    save(claim/'marker.json',dict(authority=authority,pack=str(pack),UTC=now(),refundable=False))
    save(pack/'SOURCE.json',source);save(pack/'scalar_inputs.json',inputs)
    (pack/'authorization.txt').write_bytes(approved_bytes)
    approval=dict(authorization_sha256=authorization_sha,SOURCE=source['SOURCE'],scope='B_FIXED40_SAVED_SCALAR_DIAGNOSTICS',C_real_authorized=False)
    save(pack/'approval.json',approval)
    contract=dict(schema='h4_precision_B_one_shot_v1',namespace=str(pack),SOURCE=source['SOURCE'],
        inputs_sha256=sha(pack/'scalar_inputs.json'),old_input_commit=BASE_REF,scientific_SOURCE=SCIENCE_SOURCE,
        authority=authority,claim_directory=str(claim),B_authorized=True,C_real_authorized=False,
        rho=.01,arithmetic='exact binary64 inputs as Fraction; independent Decimal100; decimal display80 inward',
        budget=dict(axis_diagnostics=40,independent_scalar_checks=40,exact_boundary_identity_checks=40,
            R_H_group_summaries=2,R_axes=22,H_axes=18,
            SCF=0,FCI=0,DF_generation=0,state_calculation=0,signal_calculation=0,
            sampling=0,compilation=0,real_H4_interval_calculation=0,certificate_generation=0,
            shot_recalculation=0,G_reaccounting=0),automatic_retry=False,mandatory_STOP=True)
    save(pack/'contract.json',contract)
    grant=dict(kind='B_ANALYSIS_GRANT_NOT_A_C_SCIENCE_GRANT',grant_id=authority,namespace=str(pack),
        SOURCE=source['SOURCE'],source_sha256=sha(pack/'SOURCE.json'),
        inputs_sha256=sha(pack/'scalar_inputs.json'),contract_sha256=sha(pack/'contract.json'),
        approval_sha256=sha(pack/'approval.json'),B_valid=True,C_valid=False,one_shot=True,automatic_retry=False)
    save(pack/'grant.json',grant)
    return grant


def check_authority(pack, *, allow_consumed=False):
    pack=Path(pack).resolve();source=read(pack/'SOURCE.json');inputs=read(pack/'scalar_inputs.json')
    contract=read(pack/'contract.json');grant=read(pack/'grant.json');approval=read(pack/'approval.json')
    require(str(pack)==contract['namespace']==grant['namespace'],'B_NAMESPACE_CHANGED')
    require(grant['kind']=='B_ANALYSIS_GRANT_NOT_A_C_SCIENCE_GRANT' and grant['B_valid'] and not grant['C_valid'] and not contract['C_real_authorized'],'B_ONLY_SCOPE')
    require(grant['SOURCE']==contract['SOURCE']==source['SOURCE']==approval['SOURCE'],'B_SOURCE_BINDING')
    for file,key in [('SOURCE.json','source_sha256'),('scalar_inputs.json','inputs_sha256'),('contract.json','contract_sha256'),('approval.json','approval_sha256')]:
        require(sha(pack/file)==grant[key],'B_PACK_BYTES_CHANGED:'+file)
    require(sha(pack/'authorization.txt')==approval['authorization_sha256'],'B_AUTHORIZATION_BYTES')
    for name,expected in source['files'].items():
        require(sha(ROOT/name)==expected,'B_SOURCE_FILE_CHANGED:'+name)
        require(hashlib.sha256(git('show',source['SOURCE']+':'+name)).hexdigest()==expected,'B_SOURCE_NOT_COMMITTED')
    require(runtime()==source['runtime'],'B_RUNTIME_CHANGED')
    for name,expected in inputs['files'].items():
        require(sha(ROOT/name)==expected,'OLD_INPUT_CHANGED:'+name)
    budget.validate_registry(inputs['rows'])
    require(not (pack/'CONSUMED.json').exists() or allow_consumed,'B_ALREADY_CONSUMED_NO_RETRY')
    require(read(Path(contract['claim_directory'])/'marker.json')['pack']==str(pack),'B_CLAIM_BINDING')
    return pack,inputs,contract,grant


def run(pack):
    pack,inputs,contract,grant=check_authority(pack)
    save(pack/'CONSUMED.json',dict(grant_id=grant['grant_id'],UTC=now(),pid=os.getpid(),
        reserved_axis_diagnostics=40,scope='B_SCALAR_ONLY',refundable=False,C_real_calls=0))
    results=[];attempted=0
    try:
        (pack/'axes').mkdir()
        for index,row in enumerate(inputs['rows']):
            attempted+=1
            value=budget.diagnose_axis(row,rho=inputs['rho'])
            save(pack/'axes'/f'{index:02d}.json',value);results.append(value)
        budget.validate_registry(results)
        save(pack/'diagnostics.json',dict(status='B_NECESSARY_ERROR_BUDGETS_ONLY',rows=results,
            SOURCE=grant['SOURCE'],grant_id=grant['grant_id'],original_N_G_remain_null=True,C_real_calls=0))
        columns=['case','epsilon_sig','axis','group','saved_u','signed_bias','saved_headroom',
            'u_max_conservative','required_u_reduction','required_u_reduction_percent','required_u_improvement_factor',
            'fixed_point_bias_makes_guard_impossible','official_status','official_N','official_G']
        with (pack/'diagnostics.csv').open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
            for row in results:
                writer.writerow({k:row[k]['decimal'] if isinstance(row[k],dict) else row[k] for k in columns})
        summary=budget.summarize(results);save(pack/'summary.json',summary)
        lines=['# B：40軸の必要誤差予算（certificateではない）','','元の全20行N/G=null・UNDETERMINEDを維持。実H4信号・区間計算・certificate・shot/G計算なし。','',
            '|case|ε_sig|axis|群|現在u|必要u上限|削減率%|改善倍率|','|---|---:|---|---|---:|---:|---:|---:|']
        for row in results:
            f=lambda key:row[key]['decimal'][:18] if row[key] is not None else '不可'
            lines.append(f"|{row['case']}|{row['epsilon_sig']}|{row['axis']}|{row['group']}|{row['saved_u']}|{f('u_max_conservative')}|{f('required_u_reduction_percent')}|{f('required_u_improvement_factor')}|")
        (pack/'diagnostics.md').write_text('\n'.join(lines)+'\n')
        check_authority(pack,allow_consumed=True)
        save(pack/'verification.json',dict(status='PASS_B_SCALAR_DIAGNOSTICS',axis_diagnostics=40,
            independent_arithmetic_checks=40,missing_or_duplicate_axes=0,old_input_hashes_unchanged=True,
            original_N_G_remain_null=True,official_status='UNDETERMINED',C_real_calls=0,
            original_native_completion_verified=False))
        save(pack/'STOP.json',dict(status='B_COMPLETE_MANDATORY_STOP',scope='NEW_B_ANALYSIS_ONLY',
            actual_axis_diagnostics=len(results),automatic_retry=False,C_real_calls=0,
            original_native_completion_verified=False,UTC=now()))
        names=['diagnostics.json','diagnostics.csv','diagnostics.md','summary.json','verification.json','STOP.json',
               *('axes/'+f'{i:02d}.json' for i in range(40))]
        save(pack/'output_inventory.json',dict(files={n:sha(pack/n) for n in names}))
    except BaseException:
        if not (pack/'STOP.json').exists():
            save(pack/'STOP.json',dict(status='B_FAILED_NO_RETRY_MANDATORY_STOP',scope='NEW_B_ANALYSIS_ONLY',
                completed_axes=len(results),axis_diagnostics_started=attempted,traceback=traceback.format_exc(),automatic_retry=False,C_real_calls=0,
                original_native_completion_verified=False))
        raise
    return summary


def launch(pack):
    pack,_,_,_=check_authority(pack)
    save(pack/'LAUNCH_CLAIM.json',dict(pid=os.getpid(),UTC=now(),scope='B_ONLY',one_shot=True))
    with (pack/'stdout.log').open('x') as out,(pack/'stderr.log').open('x') as err:
        process=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/run_h4_precision_b.py'),
            '--run','--pack',str(pack)],cwd=ROOT,stdout=out,stderr=err)
    save(pack/'native_process.json',dict(exit_code=process.returncode,direct_child_waited=True,
        UTC=now(),scope='NEW_B_ANALYSIS_ONLY_NOT_ORIGINAL_H4',original_native_completion_verified=False))
    require(process.returncode==0,'B_NATIVE_FAILURE_NO_RETRY')
    return verify(pack)


def verify(pack):
    pack,_,_,grant=check_authority(pack,allow_consumed=True)
    require(read(pack/'CONSUMED.json')['grant_id']==grant['grant_id'],'B_CONSUMPTION_BINDING')
    require(read(pack/'STOP.json')['status']=='B_COMPLETE_MANDATORY_STOP','B_NOT_COMPLETE')
    for name,expected in read(pack/'output_inventory.json')['files'].items():
        require(sha(pack/name)==expected,'B_OUTPUT_CHANGED:'+name)
    require(read(pack/'native_process.json')['exit_code']==0,'B_NATIVE_EXIT_NOT_ZERO')
    data=read(pack/'diagnostics.json');budget.validate_registry(data['rows'])
    require(all(x['official_N'] is None and x['official_G'] is None and x['official_status']=='UNDETERMINED' for x in data['rows']),'OLD_STATUS_CHANGED')
    return dict(status='PASS_B_SAVED_OUTPUT_AND_NATIVE_AUDIT',axes=40,grant_consumed=True,
        C_real_calls=0,original_native_completion_verified=False)
