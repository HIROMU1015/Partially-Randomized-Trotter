"""Future E/T-only one-shot. Separate explicit authorization required."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import subprocess

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.df_correction_et_resumption.launch import ART,RESULT,verify_launch,sha
from trottertracks.algorithm_codesign.df_correction_comparison.guard import Supervisor,Limits,exclusive_json
from trottertracks.algorithm_codesign.df_correction_comparison.storage import dump_json,load_json,artifact_identity
from trottertracks.algorithm_codesign.df_correction_comparison.inventory_integrity import freeze_inventory,verify_inventory,verify_synthesis_result


def main():
    # No CLI overrides for fixture, caps, precision, grid, resource coordinate.
    if len(sys.argv)!=1:raise PermissionError('Runner takes no scientific overrides')
    provenance=verify_launch(ROOT)  # Fails on current preparation source; marker stays absent.
    contract=json.loads((ROOT/ART/'contract_v4.json').read_text())
    from trottertracks.algorithm_codesign.df_correction_et_resumption.worker import runtime_check
    runtime_check(contract,'native')  # Coordinator uses the fixed numeric runtime.
    runtime_env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1')
    for stage in ('native','synthesis'):
        section=contract['runtimes']['synthesis' if stage=='synthesis' else 'numeric']
        code=('import json,sys;from trottertracks.algorithm_codesign.df_correction_et_resumption.worker import runtime_check;'
              'runtime_check(json.load(open(sys.argv[1])),sys.argv[2])')
        subprocess.run([section['python'],'-B','-c',code,str(ROOT/ART/'contract_v4.json'),stage],
                       cwd=ROOT,env=runtime_env,check=True,timeout=30)
    # Runtime/code identity is checked BEFORE consuming the science marker.
    out=ROOT/RESULT;out.mkdir(parents=True,exist_ok=True)
    marker=out/'one_shot_marker.json'
    exclusive_json(marker,dict(provenance,runs=1,retries=0,mandatory_STOP=True,created_unix=time.time()))
    marker_hash=sha(marker)
    os.environ['PYTHONPATH']=str(ROOT/'src')
    caps=contract['caps']
    supervisor=Supervisor(Limits(caps['wall_seconds_total'],caps['CPU_seconds_total'],caps['aggregate_RSS_MiB'],caps['aggregate_RSS_MiB']),out,
                          output_cap=caps['output_bytes']-262144,log_cap=caps['log_bytes'])
    completed_stages=[];classification='TECHNICAL_INCONCLUSIVE';reason=None;binding=None
    def make(label,stage,**payload):
        output=out/(label+'.json');job=out/(label+'.job.json')
        exclusive_json(job,dict(payload,stage=stage,output=str(output),marker_sha256=marker_hash))
        runtime=contract['runtimes']['synthesis' if stage=='synthesis' else 'numeric']['python']
        return {'label':label,'output':str(output),'command':[runtime,'-B','-m','trottertracks.algorithm_codesign.df_correction_et_resumption.worker',str(job)]}
    def run(jobs,phase,workers,per_key=False):
        c=caps[phase]
        limits=Limits(c['wall_seconds'],c['CPU_seconds'],c['aggregate_RSS_MiB'],c['worker_RSS_MiB'],c.get('per_key_wall_seconds',0),c.get('per_key_CPU_seconds',0))
        return supervisor.phase(jobs,limits,workers,per_key=per_key)
    try:
        from trottertracks.algorithm_codesign.df_correction_et_resumption.saved_data import load_saved_inputs,saved_cd_receipt
        # Read-only previous CD substage; not a new science task or reclassification.
        cd_receipt=saved_cd_receipt(ROOT,contract)
        dump_json(out/'CD_saved_substage_receipt.json',cd_receipt)
        inputs=load_saved_inputs(ROOT,contract)
        dump_json(out/'E-inputs.json',inputs)
        completed_stages.append('CD_SAVED_REFERENCE_ONLY')
        run([make('E-r'+str(r),'native',r=r,frames=inputs[str(r)]) for r in (3,6)],'native',2)
        completed_stages.append('E_NATIVE')
        from fractions import Fraction as F
        from trottertracks.algorithm_codesign.df_correction_comparison.intervals import configure
        from trottertracks.algorithm_codesign.df_correction_comparison.angles import Angle
        from trottertracks.algorithm_codesign.df_correction_comparison.native import select_synthesis_inventory,t_completion,local_t_separation
        configure()
        circuits={};bs={};native={}
        for r in (3,6):
            native[r]=load_json(out/('E-r'+str(r)+'.json'))
            if sum(native[k]['compile_calls'] for k in native)>22:raise RuntimeError('Compile call cap')
            for row in native[r]['native']:
                if row['arm']=='A':circuits[(r,row['controlled'])]=row['inventory']
            bs[r]=[Angle.load(a) for a in native[r]['T_semantic_angles']]
        inventory=select_synthesis_inventory(circuits,bs)
        from trottertracks.algorithm_codesign.df_correction_et_resumption.acceptance import bind_native_results,verify_native_bindings
        inventory['native_result_bindings']=bind_native_results(out,native)
        verify_native_bindings(out,inventory)
        binding=freeze_inventory(out,inventory,provenance,marker_hash)  # Original bytes before ANY call.
        jobs=[make('T-'+str(i),'synthesis',angle=a,**binding) for i,a in enumerate(inventory['unique_keys'])]
        if len(jobs)>64:raise RuntimeError('Synthesis call cap')
        key_by_label={j['label']:inventory['unique_keys'][i]['key'] for i,j in enumerate(jobs)}
        bset=set(inventory['B_required_keys']);pset=set(inventory['phase_required_keys'])-bset
        batches=[[j for j in jobs if key_by_label[j['label']] in bset],
                 [j for j in jobs if key_by_label[j['label']] in pset],
                 [j for j in jobs if key_by_label[j['label']] not in bset|pset]]
        t_start_cpu=supervisor.cpu();t_start=time.monotonic()
        original_wall=caps['synthesis']['wall_seconds'];original_cpu=caps['synthesis']['CPU_seconds']
        for batch in batches:
            if not batch:continue
            verify_inventory(out,binding,marker_sha256=marker_hash,provenance=provenance)
            verify_native_bindings(out,inventory)
            caps['synthesis']['wall_seconds']=original_wall-(time.monotonic()-t_start)
            caps['synthesis']['CPU_seconds']=original_cpu-(supervisor.cpu()-t_start_cpu)
            run(batch,'synthesis',2,per_key=True)
        completed_stages.append('T_KEYS')
        verify_inventory(out,binding,marker_sha256=marker_hash,provenance=provenance)
        sequences={}
        synthesis_result_identities=[]
        for j in jobs:
            row=load_json(j['output'])
            expected=inventory['unique_keys'][int(j['label'].split('-')[1])]
            verify_synthesis_result(row,expected,binding)
            from trottertracks.algorithm_codesign.df_correction_et_resumption.accounting import validate_saved_sequence
            validate_saved_sequence(row,contract)
            synthesis_result_identities.append(artifact_identity(j['output']))
            sequences[row['angle']['key']]=row
        accounting=[]
        for r in (3,6):
            for control in (False,True):
                a=t_completion(circuits[(r,control)],sequences)
                # B arbitrary-angle keys are shared by ALL labels; basis/phase Clifford T=0.
                rows=[]
                for event in native[r]['all_B_event_costs']:
                    if event['controlled']!=control:continue
                    sign=event['sign'];ell=F(native[r]['ell_computational'])
                    scales=(-sign,sign) if control else (-2*sign,)
                    keys=[Angle('atan',ell/F(128),F(s)).key for s in scales]
                    cost=sum(sequences[k]['T_count'] for k in keys) if all(k in sequences for k in keys) else None
                    rows.append(dict(event,actual_T=cost))
                complete=(bool(rows) and all(x['actual_T'] is not None for x in rows)
                    and F(native[r]['B_local_numeric_uniform_upper'])<=F(contract['E']['local_numeric_allowance'])
                    and all(x['total_RZ']*F(contract['T']['gate_precision'])<=F(contract['E']['local_synthesis_allowance']) for x in rows))
                expected=sum(F(x['executed_conditional_probability'])*x['actual_T'] for x in rows) if complete else None
                b={'actual_total_T':max(x['actual_T'] for x in rows) if complete else None,
                   'expected_conditional_T':str(expected) if complete else None,'all_B_events_covered':complete}
                accounting.append({'r':r,'controlled':control,'A':a,'B':b,
                    # A lower greater than the WORST B event gives a local fixed-plan separation.
                    **local_t_separation(a,b)})
        verify_inventory(out,binding,marker_sha256=marker_hash,provenance=provenance)
        dump_json(out/'T_accounting.json',{'rows':accounting,'CD_PR_total_winner':None,
            **binding,'synthesis_result_identities':synthesis_result_identities,
            'native_result_bindings':inventory['native_result_bindings']})
        supervisor.check_total()
        verify_native_bindings(out,inventory)
        classification='COMPLETE_E_T_COMPARISON_NO_AUTOMATIC_GO'
    except BaseException as exc:
        reason=type(exc).__name__+': '+str(exc)
    finally:
        inventory_integrity=None
        if binding is not None:
            try:
                verify_inventory(out,binding,marker_sha256=marker_hash,provenance=provenance)
                verify_native_bindings(out,inventory)
                inventory_integrity=True
            except BaseException as exc:
                inventory_integrity=False;classification='TECHNICAL_INCONCLUSIVE'
                reason=(reason+'; ' if reason else '')+'inventory final guard: '+type(exc).__name__+': '+str(exc)
        terminal={'classification':classification,'technical_reason':reason,
            'completed_stages':completed_stages,'provenance':provenance,'marker_SHA256':marker_hash,
            'synthesis_inventory_binding':binding,'synthesis_inventory_integrity':inventory_integrity,
            'resource_receipt':supervisor.receipt(),'retries':0,'mandatory_STOP':True,
            'scientific_winner':None,'G10_RA_D0_molecule_GPU_LP_trajectory_calls':0,
            'CD_recalculation_calls':0,'E_recipe_generation_calls':0,
            'historical_fingerprint_policy_modified':False}
        try:
            dump_json(out/'terminal_status.json',terminal)
        except BaseException as exc:
            classification='TECHNICAL_INCONCLUSIVE'
            reason='terminal storage failure: '+type(exc).__name__+': '+str(exc)
            # Reserved metadata only, no result salvage or repeat calculation.
            # If even this write fails, the outer process must record nonzero
            # exit and consumed marker; no partial result is accepted.
            exclusive_json(out/'terminal_storage_failure.json',{
                'classification':classification,'technical_reason':reason[:4096],
                'provenance':provenance,'marker_SHA256':marker_hash,
                'retries':0,'mandatory_STOP':True,'partial_scientific_outcome_allowed':False})
        print(json.dumps({'classification':classification,'mandatory_STOP':True,'reason':reason}))
    return 0 if classification.startswith('COMPLETE') else 2


if __name__=='__main__':raise SystemExit(main())
