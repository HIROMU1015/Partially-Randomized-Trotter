"""Verify saved L evidence without repeating a gate, basis or physics call.

Reuses the sealed static plan builder with numerical functions blocked. Only
saved rational bounds are compared; no new Arb/H4 matrix evaluation is run.
"""
from collections import Counter
from decimal import Decimal, localcontext, ROUND_CEILING
from fractions import Fraction as Q
from pathlib import Path
from unittest.mock import patch
import csv, datetime, hashlib, importlib.util, json, os, shutil

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
NS=HERE/'run01'
OUT=HERE/'audit02'
OUT.mkdir(exist_ok=False)

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

c=load('saved_L_audit_control','src/trotterlib/h4_local_control.py')
n=load('saved_L_audit_static','src/trotterlib/h4_local_enclosure.py')

def read(path):
    return c.read(path)

def require(condition,message):
    c.require(condition,message)

def upper(text):
    require(type(text) is str,'EXACT_RATIONAL_STRING')
    value=Q(text)
    require(value>=0 and value.denominator>0,'NONNEGATIVE_FINITE_BOUND')
    require(value.denominator & (value.denominator-1)==0,'DYADIC_OUTWARD_BOUND')
    return value

def display(q):
    # Rounded upward text for display; exact rational remains authoritative.
    with localcontext() as ctx:
        ctx.prec=12;ctx.rounding=ROUND_CEILING
        return str(Decimal(q.numerator)/Decimal(q.denominator))

def exact_product(errors):
    bound=Q(1)
    for error in errors:bound*=1+error
    return bound-1

def save_csv(name,rows):
    with (OUT/name).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

contract,seal=read(NS/'contract.json'),read(NS/'seal.json')
c.validate_contract(contract,active=True);c.verify_seal(seal);c.routing(contract,seal)
require(c.sha(NS/'seal.json')==contract['technical_seal_sha256'],'SEAL_BINDING')
model,registry=read(contract['model_path']),read(contract['registry_path'])
with patch.object(n,'ideal_gate',side_effect=AssertionError('NO_NEW_NUMERIC_CALL')), \
     patch.object(n,'discrepancy',side_effect=AssertionError('NO_NEW_NUMERIC_CALL')), \
     patch.object(n,'product_bound',side_effect=AssertionError('NO_NEW_NUMERIC_CALL')), \
     patch.object(n,'diagnose',side_effect=AssertionError('NO_NEW_NUMERIC_CALL')):
    plan=n.plan_from_saved(model,registry)
require((len(plan['gates']),len(plan['bases']),len(plan['supports']))==(148,11,324),'TARGET_COUNTS')

locals_by_gate={};local_rows=[];call_result_paths=[]
for gate_hash,gate in sorted(plan['gates'].items()):
    locals_by_gate[gate_hash]={}
    for direction in ('forward','inverse'):
        p=NS/f'local_{gate_hash}_{direction}.json';record=read(p);call_result_paths.append(p)
        require(record['kind']=='LOCAL_GATE_DISCREPANCY' and record['matrix_hash']==gate_hash and record['direction']==direction,'LOCAL_IDENTITY')
        require(record['precision_bits']==128 and record['literal_decomposition_or_compiler_proved'] is False,'LOCAL_SCOPE')
        value=upper(record['discrepancy_upper_exact']);locals_by_gate[gate_hash][direction]=value
        local_rows.append(dict(gate_hash=gate_hash,direction=direction,gate_name=gate['descriptor']['name'],parameters_json=json.dumps(gate['descriptor']['parameters']),matrix_dimension=len(gate[direction]),actual_matrix_SHA256=n.matrix_hash(gate[direction]),norm_interval_lower_exact='0',norm_interval_upper_exact=str(value),upper_decimal_ceiling_display=display(value),precision_bits=128,guarantee_status='LOCAL_OPERATOR_NORM_ENCLOSURE_ONLY',SOURCE=seal['SOURCE'],model_SHA256=c.sha(contract['model_path']),result_path=str(p.relative_to(ROOT)),result_SHA256=c.sha(p)))
require({p.name for p in NS.glob('local_*.json')}=={p.name for p in call_result_paths},'LOCAL_EXTRA_OR_MISSING')
save_csv('local_gate_bounds.csv',local_rows)

basis_pairs={};basis_rows=[]
for basis_hash,operations in sorted(plan['bases'].items()):
    p=NS/f'basis_{basis_hash}.json';record=read(p);call_result_paths.append(p)
    order=[op['gate_hash'] for op in operations]
    require(record['basis_hash']==basis_hash and record['forward_gate_order']==order and record['inverse_gate_order']==order[::-1],'ORDERED_BASIS_IDENTITY')
    forward,inverse=upper(record['forward_upper_exact']),upper(record['inverse_upper_exact'])
    require(forward>=exact_product([locals_by_gate[h]['forward'] for h in order]),'FORWARD_OUTWARD_COMPOSITION')
    require(inverse>=exact_product([locals_by_gate[h]['inverse'] for h in order[::-1]]),'INVERSE_OUTWARD_COMPOSITION')
    pair=upper(record['control_zero_upper_exact'])
    require(pair==upper(record['control_one_upper_exact']) and pair>=forward+inverse+forward*inverse,'CONTROL_BRANCH_BOUND')
    basis_pairs[basis_hash]=pair
    basis_rows.append(dict(basis_hash=basis_hash,local_operation_count=len(order),forward_upper_exact=str(forward),inverse_upper_exact=str(inverse),control_zero_upper_exact=str(pair),control_one_upper_exact=str(pair),control_upper_decimal_ceiling_display=display(pair),precision_bits=128,forward_operations_json=json.dumps(operations),inverse_gate_order_json=json.dumps(order[::-1]),guarantee_status='ORDERED_BASIS_LOCAL_TRANSFER_ONLY',result_path=str(p.relative_to(ROOT)),result_SHA256=c.sha(p)))
require({p.name for p in NS.glob('basis_*.json')}=={f'basis_{h}.json' for h in plan['bases']},'BASIS_EXTRA_OR_MISSING')
save_csv('basis_bounds.csv',basis_rows)

support_rows=[]
for comp in plan['supports']:
    p=NS/f"support_{comp['index']}.json";record=read(p);call_result_paths.append(p)
    require(all(record[key]==value for key,value in comp.items()),'SUPPORT_IDENTITY')
    require(record['ideal_diagonal_involution'] is True and record['DF_fragment_lambda_A_squared_is_involution'] is False,'SUPPORT_NOT_FRAGMENT_INVOLUTION')
    require(comp['basis_hash'] in basis_pairs,'SUPPORT_BASIS_COVERAGE')
    support_rows.append(dict(component_id=comp['index'],basis_hash=comp['basis_hash'],sign=comp['sign'],qubit_support_json=json.dumps(comp['support']),identity_component=not comp['support'],control_zero_ideal_identity=True,control_one_diagonal='signed Z/ZZ/I',event_orders_json='[0,2,4]',q=8,r=4,coverage_status='FULL_FIXED_SYMBOLIC_REGISTRY_NOT_OBSERVED_MC_SAMPLE',result_path=str(p.relative_to(ROOT)),result_SHA256=c.sha(p)))
require({p.name for p in NS.glob('support_*.json')}=={f"support_{x['index']}.json" for x in plan['supports']},'SUPPORT_EXTRA_OR_MISSING')
save_csv('component_support.csv',support_rows)

result=read(NS/'worker_result.json');require(result['precision_bits']==128 and result['domain']=='SAVED_H4','RESULT_DOMAIN_PRECISION')
uniform=[]
for method in ('PF','PR'):
    record=result['uniform_basis_bounds'][method];count=8*(2*plan['block_counts'][method]+(20 if method=='PR' else 0))
    require(record['maximum_conjugations']==count and record['measurement_scale_B_not_applied'] is True,'UNIFORM_SCOPE')
    require(record['event_orders']==([0,2,4] if method=='PR' else []),'EVENT_FAMILY')
    maximum=max(basis_pairs[h] for h in plan['case_bases'][method]);bound=upper(record['local_basis_transfer_upper_exact'])
    require(bound>=exact_product([maximum]*count),'UNIFORM_OUTWARD_BOUND')
    uniform.append(dict(method='df_native_'+method,maximum_conjugations=count,operator_bound_upper_exact=str(bound),operator_bound_decimal_ceiling_display=display(bound),precision_bits=128,B_scale_applied=False,combined_with_C=False))
save_csv('uniform_basis_transfer.csv',uniform)

ledger=read(NS/'call_ledger.json');limits=dict(forward=148,inverse=148,basis=11,support=324)
require(ledger['limits']==ledger['used']==limits,'CALL_BUDGET_EXACT')
require(len(ledger['events'])==sum(limits.values())==len(call_result_paths)==631,'RESERVATION_AND_RESULT_COUNT')
require(dict(Counter(x['call'] for x in ledger['events']))==limits,'RESERVATION_CATEGORIES')
expected_calls=['forward','inverse']*148+['basis']*11+['support']*324
worker=read(NS/'WORKER_CLAIM.json')
for event,kind,path in zip(ledger['events'],expected_calls,call_result_paths):
    require(event['call']==kind and event['reserved_before_call'] is True and event['pid']==worker['pid'],'RESERVATION_IDENTITY_ORDER')
    # Filesystem mtime is not a strict realtime event clock (audit01 records
    # submillisecond inversions). Ordering evidence is sealed take->call->emit
    # SOURCE plus fsync ledger and exact event/result correspondence.
    require(path.is_file(),'DURABLE_RESULT_AFTER_RESERVED_CALL')
require(read(NS/'CONSUMED.json')['grant_id']==read(NS/'grant.json')['grant_id'],'CONSUMED_GRANT_ID')
require(read(NS/'grant.json')['contract_sha256']==c.sha(NS/'contract.json'),'IMMUTABLE_CONTRACT_HASH')
require(read(NS/'grant.json')['seal_sha256']==c.sha(NS/'seal.json'),'IMMUTABLE_SEAL_HASH')

native=read(NS/'native/worker_native_exit.json');native_stop=read(NS/'native/STOP.json');witness=read(NS/'witness_exit.json');stop=read(NS/'STOP.json')
require(native['pid']==worker['pid'] and native['exit_code']==0 and native['reaped'] is True and native['scope']=='NEW_L_WORKER_ONLY','NATIVE_L_SCOPE')
require(native_stop['status']=='NATIVE_EXIT_REAP_RECORDED' and native_stop['all_owned_reaped'] is True and native_stop['descendants_terminated'] is False and native_stop['original_H4_native_verified'] is False,'NATIVE_REAP_STOP')
require(worker['pid'] in native_stop['reaped_pids'] and witness['exit_code']==0 and witness['reaped'] is True,'WITNESS_REAP')
require(c.C._proc_identity(worker['pid'])!=read(NS/'native/worker_identity.json'),'OWNED_WORKER_STILL_LIVE')
require(stop['mandatory_STOP'] is True and stop['native_exit']==0 and stop['witness_reaped'] is True,'FINAL_STOP')
require(all(stop[x] is False for x in ('execution_authorized','grant_valid','dispatch_enabled','formal_certificate_accepted','N_G_authorized')),'STOP_AUTHORITY_FALSE')
require(result['coverage_ledger']['status']=='COMBINATION_BLOCKED' and result['coverage_ledger']['simple_addition_permitted'] is False and result['compiler_transfer']=='MISSING_EVIDENCE' and result['formal_certificate_accepted'] is False,'SCIENTIFIC_GATES_UNCHANGED')
refusals=[]
for worker_mode in (False,True):
    try:c.authority(NS,worker=worker_mode)
    except ValueError as error:
        require(str(error)=='L_ALREADY_STOPPED','UNEXPECTED_AUTHORITY_REFUSAL');refusals.append(dict(worker_mode=worker_mode,reason=str(error),check='Read-only authority check only; no relaunch/retry was attempted'))
    else:raise AssertionError('STOP_AUTHORITY_NOT_REJECTED')

before=read(HERE/'preservation_before.json')
for path,digest in before['files'].items():require(c.sha(path)==digest,'OLD_EVIDENCE_CHANGED:'+path)
c.verify_seal(seal)
c.save(HERE/'preservation_after.json',dict(status='PASS',UTC=c.utc(),files_verified=before['file_count'],all_existing_bytes_unchanged=True,old_grants_ledgers_markers_STOP_unchanged=True,C40_and_old_N_G_unchanged=True,original_H4_native_completion='UNKNOWN_UNCHANGED',K_full_guarantee='MISSING_EVIDENCE_UNCHANGED',new_native_receipts_not_applied_to_original_H4=True))
c.save(HERE/'coverage_ledger.json',dict(result['coverage_ledger'],new_L_result_ref='run01/worker_result.json',new_L_bound_reference='audit02/uniform_basis_transfer.csv',old_C_total_changed=False,new40_axis_combined_bounds_created=False,followup_only='Separately justify primitive-disjoint accounting or joint enclosure before C/L combination; no subtraction or simple addition here'))
summary=dict(status='PASS_LOCAL_NUMERICAL_EVIDENCE_SAVED_COMBINATION_BLOCKED',UTC=c.utc(),SOURCE=seal['SOURCE'],precision_bits=128,completed_calls=limits,local_results=296,basis_results=11,support_results=324,all_bounds_nonnegative_finite_dyadic=True,saved_scalar_bounds_outward_composition_checked=True,no_duplicate_physical_numeric_evaluations=True,technical_audit01_failure_preserved=True,file_mtime_strict_order_not_claimed=True,all324_symbolic_supports_covered=True,finite_observed_MC_events_used_as_coverage_proof=False,max_local_forward_upper_exact=str(max(v['forward'] for v in locals_by_gate.values())),max_local_inverse_upper_exact=str(max(v['inverse'] for v in locals_by_gate.values())),max_basis_control_upper_exact=str(max(basis_pairs.values())),uniform_basis_transfer=uniform,unresolved_local_gate_results=0,unsupported_gate_results=0,scientific_grants_issued=1,grants_consumed=1,automatic_retries=0,precision_escalations=0,formal_certificates_accepted=0,N_G_accounting_calls=0,forbidden_calls=contract['forbidden_calls'],C_L_combination='COMBINATION_BLOCKED',compiler_transfer='MISSING_EVIDENCE',literal_wrapper_transfer='INCOMPLETE_DIAGONAL_CONTROL_SYNTHESIS_PHASE_AND_B_SCALE',native_worker_exit_code=0,worker_reaped=True,witness_reaped=True,old_files_preserved=before['file_count'],mandatory_STOP=True)
c.save(HERE/'result_summary.json',summary)
c.save(HERE/'verification.json',dict(status='PASS_LOCAL_SCOPE_ONLY',UTC=c.utc(),checks=['Static sealed descriptor/registry match with numerical functions mocked to raise','148 forward and148 inverse exact result identities/SHA/precision/dyadic endpoints','11 ordered forward/inverse scalar outward compositions','Both control branches saved pair-bound inequality','324 exact symbolic support IDs/signs/bases/qubits; complete fixed registry','Uniform PF176/PR192 saved rational inequalities; no new gate/basis evaluation','Sealed SOURCE take->call->emit and631 ordered nonrefundable reservations matched to631 durable results; no mtime clock ordering claim','Native wait/reap and terminal STOP for NEW_L only','Authority refuses stopped run for launcher and worker; no actual relaunch','Post-run seal13/runtime1050/input5 hashes and2486 historical bytes'],authority_refusals=refusals,raw_namespace_modified_by_audit=False,prior_audit01_failure='Preserved filesystem mtime ordering assertion; corrected audit only, no science rerun',additional_scientific_calls=0,formal_certificate_accepted=False,independent_scientific_review=False,verification_owner='Primary agent'))
mem={x.split(':')[0]:x.split(':')[1].strip() for x in Path('/proc/meminfo').read_text().splitlines()}
c.save(HERE/'resource_after.json',dict(UTC=c.utc(),MemAvailable=mem['MemAvailable'],filesystem_free_bytes=shutil.disk_usage(ROOT).free,pressure={name:Path('/proc/pressure',name).read_text() for name in ('cpu','memory','io')},shared_environment_modified=False))
# This inventory is a byte preservation record, not a final scientific evidence freeze.
c.save(HERE/'raw_result_inventory.json',dict(kind='THIS_L_RUN_BYTE_INVENTORY_NOT_FINAL_EVIDENCE_FREEZE',files={str(p.relative_to(ROOT)):c.sha(p) for p in sorted(NS.rglob('*')) if p.is_file()},scientific_scope='L-only',formal_certificate_accepted=False))
print(json.dumps({k:summary[k] for k in ('status','completed_calls','uniform_basis_transfer','native_worker_exit_code','old_files_preserved')},ensure_ascii=False))
