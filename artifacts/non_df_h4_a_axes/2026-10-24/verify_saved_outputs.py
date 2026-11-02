"""Read-only SHA/metadata/saved-moment audit; no QPY load or new statistics."""
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT=Path(__file__).resolve().parents[3]
BASE=Path(__file__).resolve().parent
NS=BASE/'run01'
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=types.ModuleType('axes_audit_private');p.__path__=[str(ROOT/'src/trotterlib')];sys.modules[p.__name__]=p
    from axes_audit_private import h4_a_axes as D
    checks={}
    def check(name,value):
        checks[name]=bool(value)
        if not value:raise AssertionError(name)
    seal,contract,grant=read(BASE/'seal.json'),read(BASE/'active_contract.json'),read(NS/'grant.json')
    D.verify_seal(seal);D.scope(contract)
    check('frozen_SOURCE_inputs_runtime_and_old_preservation',True)
    check('grant_digest',grant['grant_id']==D.C.digest({k:v for k,v in grant.items() if k!='grant_id'}))
    check('grant_binding',grant['SOURCE']==seal['SOURCE']==contract['science_SOURCE']
          and grant['contract_sha256']==sha(BASE/'active_contract.json')
          and grant['seal_sha256']==contract['seal_sha256']==sha(BASE/'seal.json')
          and grant['approval_sha256']==contract['approval_sha256']==seal['approval_sha256'])
    check('one_shot_consumed',read(NS/'CONSUMED.json')['grant_id']==grant['grant_id']
          and read(NS/'CONSUMED.json')['nonrefundable'] is True)
    ledger,stop=read(NS/'call_ledger.json'),read(NS/'STOP.json')
    check('finite_scientific_budget',ledger['used']==ledger['limits']==D.LIMITS==grant['call_limits'])
    counts=Counter(r['call'] for r in ledger['events'])
    check('all_calls_reserved_before_execution',all(r['reserved_before_call'] for r in ledger['events'])
          and all(counts[k]==v for k,v in D.LIMITS.items()))
    check('mandatory_STOP_complete_no_next_stage',stop['status']=='A_AXES_COMPLETE_MANDATORY_STOP'
          and stop['mandatory_STOP'] and stop['automatic_next_stage'] is False
          and stop['ledger']==ledger and stop['preservation_error'] is None)
    engine=read(NS/'ENGINE_STOP.json')
    check('256_unique_successes_256_attempts_no_scientific_interruption',engine['new_unique_successes']==256
          and engine['total_compile_attempts']==256 and engine['technical_interruption'] is False and engine['recovery_runs']==0)
    check('legacy_unknown_and_certificate_unpromoted',stop['original_H4_native_verified'] is False and stop['formal_certificate_accepted'] is False)
    check('engine_native_exit_reaped',stop['native_engine_exit']['exit_code']==0
          and stop['native_engine_exit']['reaped'] and stop['all_owned_reaped'])
    manifest=read(BASE/'case_inputs.json')
    check('saved_257_cosine_and_reused_PR1_sine_metadata',manifest==D.input_manifest())
    rows=read(NS/'paired_costs.json')['rows'];stats=read(NS/'statistics.json')
    check('paired_257_unique_ID_rows',len(rows)==257
          and len({(r['case'],r['trajectory_id']) for r in rows})==257
          and sum(r['case']=='PR' for r in rows)==256 and sum(r['case']=='PF' for r in rows)==1)
    check('only_256_new_sine_targets',{p.name for p in (NS/'trajectories').iterdir()}==set(D.CASES)
          and 'PR0001' not in D.CASES and stop['native_task_count']==256 and stop['all_tasks_exit0_reaped'])
    schema=read(BASE/'output_schema.json');errors=[];identities=[read(NS/'native/worker_identity.json')]
    for target,pair in zip(manifest['rows'],rows):
        cos=read(ROOT/target['cosine_folder']/'metrics.json')
        good=(pair['case']==target['method'] and pair['index']==target['trajectory_index']
              and pair['trajectory_id']==target['trajectory_id'] and pair['event_record_hash']==target['event_record_hash']
              and pair['trajectory_record_hash']==target['trajectory_record_hash']
              and all(pair['cosine'][k]==cos[k] for k in pair['cosine']))
        if target['key']=='PR0001':
            sine=read(ROOT/target['reused_sine_folder']/'metrics.json')
            good=good and pair['sine_reused'] is True and sine['RZ']==1129
        else:
            folder=NS/'trajectories'/target['key'];task=NS/'native_tasks'/('compile_'+target['key'])
            sine,inp,binding=read(folder/'metrics.json'),read(folder/'input.json'),read(folder/'source_binding.json')
            receipt,reservation=read(task/'native_exit.json'),read(task/'reservation.json')
            good=(good and all((folder/p).is_file() for p in schema['per_new_target'])
                  and all(k in sine for k in schema['metrics']) and sine['target']==binding['target']==target
                  and sine['SOURCE']==inp['SOURCE']==seal['SOURCE'] and inp['trajectory_id']==target['trajectory_id']
                  and sine['axis']==inp['axis']=='sine' and sine['case']==target['key'] and sine['method']==target['method']
                  and binding['status']=='PASS_UNCHANGED_CONTROLLED_BODY' and binding['ancilla']==8
                  and binding['body_replay']==binding['central_control_build']==0 and binding['new_certificate'] is False
                  and binding['direct_saved_body']==bool(target['body_qpy'])
                  and len(binding['body_signature_sha256'])==len(binding['cosine_signature_sha256'])==64
                  and sine['compiler']==cos['compiler']==D.H.fixed()['compiler']
                  and sine['input_qpy_sha256']==inp['input_qpy_sha256']==sha(folder/'input.qpy')
                  and sine['output_qpy_sha256']==sha(folder/'output.qpy')
                  and sine['input_fingerprint']==inp['input_fingerprint'] and len(sine['output_fingerprint'])==64
                  and sine['pretranspile_histogram']==inp['pretranspile_histogram']
                  and sine['RZ']==sine['posttranspile_histogram'].get('rz',0)
                  and sine['CX']==sine['posttranspile_histogram'].get('cx',0)
                  and receipt['exit_code']==0 and receipt['native_wait'] and receipt['reaped']
                  and reservation['grant_id']==grant['grant_id'] and reservation['case']==target['key'])
            identities.append(read(task/'launch.json')['identity'])
        good=good and all(pair['sine'][k]==sine[k] for k in pair['sine'])
        if not good:errors.append(target['key'])
    check('all_target_phase_QPY_SHA_fingerprint_profile_pair_and_native',not errors)
    check('price_inputs_only_no_NG_or_certificate',stats['status']=='COMPLETE_257_PAIRED_PR256_PF1'
          and stats['new_N_G'] is False and stats['precision_or_G_guarantee'] is False
          and stats['compiled_error_certificate'] is False and stats['population_mean_upper'] is None
          and stats['price_inputs_for_conditional_CNG_complete'] is True and stats['cross_representation_pairing'] is False)
    pr=[r for r in rows if r['case']=='PR'];x=[r['cosine']['RZ'] for r in pr];y=[r['sine']['RZ'] for r in pr]
    check('saved_exact_mean_moment_and_difference_identities',Fraction(sum(x),256)==Fraction(stats['PR']['axes']['cosine']['mean_exact'])
          and Fraction(sum(y),256)==Fraction(stats['PR']['axes']['sine']['mean_exact'])
          and (sum(a*b for a,b in zip(x,y))-Fraction(sum(x)*sum(y),256))/255==Fraction(stats['PR']['sample_covariance_RZ_exact'])
          and Fraction(sum(b-a for a,b in zip(x,y)),256)==Fraction(stats['PR']['paired_RZ_difference']['mean_exact']))
    check('saved_equal_unequal_counts',sum(a==b for a,b in zip(x,y))==stats['PR']['RZ_equal_count']
          and sum(a!=b for a,b in zip(x,y))==stats['PR']['RZ_unequal_count'])
    inventory=read(NS/'output_inventory.json')['files']
    check('run_inventory_SHA',all(sha(NS/p)==h for p,h in inventory.items()))
    check('run_inventory_complete',set(inventory)=={str(p.relative_to(NS)) for p in NS.rglob('*') if p.is_file() and p.name!='output_inventory.json'})
    previous=json.loads(subprocess.check_output(['git','show','7c5e25dce757b7daa165095839ca879ed0bb0c10:artifacts/validation_manifest.json'],cwd=ROOT))
    current=read(ROOT/'artifacts/validation_manifest.json')
    check('prior_manifest_entries_unchanged',all(current[k]==v for k,v in previous.items()))
    remaining=[r for r in identities if D.C._proc_identity(r['pid'])==r]
    check('no_recorded_owned_process_remaining',not remaining)
    return dict(status='PASS',scope='PRIMARY_AUTHOR_SAVED_METADATA_SHA_MOMENT_IDENTITIES_NOT_NEW_STATISTICS_OR_CERTIFICATE',
        SOURCE=seal['SOURCE'],checks=checks,failed_targets=errors,remaining_owned_processes=remaining,
        counts=dict(code_files=len(seal['code_files']),input_files=len(seal['input_files']),runtime_files=len(read(ROOT/seal['runtime_manifest'])['files']),
                    preserved_files=len(read(BASE/'preservation_before.json')),run_inventory_files=len(inventory),
                    new_sine_targets=256,reused_cosine_rows=257,reused_sine_rows=1,paired_rows=257,prior_manifest_entries=len(previous)),
        fingerprint_scope='Saved metadata binding and QPY SHA; no QPY reload',new_scientific_calls=0,new_statistics_batches=0)


if __name__=='__main__':print(json.dumps(main(),ensure_ascii=False,indent=2,sort_keys=True))
