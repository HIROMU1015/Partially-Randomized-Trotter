"""Freeze saved-only S2 models and conditional S4 hypotheses, without compile.

Old profile identities remain separate. Auditing a transfer hypothesis is not
proof of compiler output equivalence or a bound outside the registered axes.
This command never issues a grant and requires fresh, exclusive destinations.
"""
import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import prepare_track_a_v1 as base
import track_a_v1_costs as costs

r=base.r;c=r.contract;ROOT=base.ROOT
ART=ROOT/'artifacts/resource_applicability'
B0_PATH='artifacts/resource_applicability/track_a_h6_exploration_summary_v1/2026-10-12/wrapper_metrics_v1.csv'
B0_RESULT='df99d6ee28942df8d12aac3c4db7cd202df55d77'
B0_SOURCE='54986a9f54515c8c2289e38fa16fd81e2b6008c2'
CORE=('src/trottertracks/resource_applicability/ax2b_native_df_v5.py',
      'src/trottertracks/resource_applicability/ax2a_control_plan.py',
      'src/trottertracks/resource_applicability/ax2b_stream_fingerprint_v5.py',
      'src/trottertracks/resource_applicability/h6_matched_port_v1.py',
      'src/trotterlib/df_partial_s2.py','src/trotterlib/df_trotter/ops.py',
      'src/trotterlib/rpe_hadamard_interrogation.py')
SOURCES=base.SOURCES+('scripts/prepare_track_a_v1_phase_b.py','scripts/launch_track_a_v1_phase_b.py')


def blob(commit,path):
    return subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)


def put(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def freeze(old):
    """Only parse saved JSON/CSV and source bytes; never load a science array."""
    binding=old['binding'];compiler=binding['compiler']
    phase=ART/'track_a_v1_phase_a_run02/2026-10-16'
    import gzip,hashlib,io
    signals=json.loads(gzip.decompress((phase/'signal_results_v1.json.gz').read_bytes()))
    terminal=r.read_json(phase/'execution_terminal_v1.json')
    if signals['status']!='COMPLETE_MANDATORY_STOP' or terminal.get('exit_code')!=0:
        raise ValueError('COMPLETE_PHASE_A_REQUIRED')
    eligible={x['cell']['id'] for x in signals['signals'] if any(
        y['1.0']['cost_acquisition_eligible'] for y in x['precision_rows'].values())}
    b0_bytes=blob(B0_RESULT,B0_PATH)
    provenance=r.read_json(ART/'track_a_d0_saved_value_reanalysis_result_v1/2026-10-14/provenance_v1.json')
    expected=next(x for x in provenance['input_receipts'] if x['alias']=='h6_exploration_wrappers.csv')
    if hashlib.sha256(b0_bytes).hexdigest()!=expected['sha256']:raise ValueError('B0_RAW_INPUT_RECEIPT')
    raw=list(csv.DictReader(io.StringIO(b0_bytes.decode())))
    b0=[x for x in raw if x['cell_id'].startswith('B0_') and x['phase']=='exploration']
    if len(b0)!=16 or any(x['source_commit']!=B0_SOURCE or x['control']!='symmetric_directional'
                         or x['compiler_hash']!=compiler for x in b0):raise ValueError('B0_ANCHOR_IDENTITIES')
    family=ART/'track_a_h6_native_b0i_full_family_result_v1/2026-10-15'
    b0i=list(csv.DictReader((family/'axis_compiled_metrics_v1.csv').open()))
    # Exclude the three legacy pilot cases rather than pooling their profiles.
    b0i_new=[x for x in b0i if x['acquisition']=='NEW_MEASURED']
    if len(b0i_new)!=12 or any(x['origin_SOURCE']!='ac6d28e35cf01398f169ce59577182a38da9b959'
            or x['compiler_hash']!=compiler or x['native_exit_code']!='0' for x in b0i_new):
        raise ValueError('B0I_ANCHOR_IDENTITIES')
    same_sources=[]
    for name in CORE:
        oldblob=blob(B0_SOURCE,name);current=blob(c.PROVIDER,name)
        if oldblob!=current or hashlib.sha256(current).hexdigest()!=old['provider_hashes'][name]:
            raise ValueError('NATIVE_SCHEDULE_PHASE_COMPILER_SOURCE_CHANGED:'+name)
        same_sources.append(dict(path=name,sha256=hashlib.sha256(current).hexdigest()))
    archive=r.read_json(old['archive']['path'])
    if r.sha(old['archive']['path'])!=old['archive']['sha256']:raise ValueError('ARCHIVE_HASH')
    codec=archive['codec'];node=lambda ref:codec['nodes'][ref['ref']]
    whole=node(codec['roots']['faithful:p19'])['fields']
    blocks=node(whole['deterministic_blocks'])['items']
    for prefix in (5,10,15):
        original=node(codec['roots']['discard:p'+str(prefix)])['fields']
        if node(original['deterministic_blocks'])['items']!=blocks[:prefix+1]:
            raise ValueError('SAVED_PREFIX_BLOCK_IDENTITY')
    export=archive['export_receipt']
    if export['deterministic_preparation_unchanged'] is not True:raise ValueError('DETERMINISTIC_EXPORT_IDENTITY')
    bounds=[]
    for ref in blocks:
        item=node(ref);f=item['fields'];n=f['num_system_qubits']
        pairs=0 if item['type']=='DFDeterministicOneBodySpec' else n*(n-1)//2
        bounds.append(4*len(node(f['runtime_basis_operations'])['items'])+4*n+6*pairs+1)
    axes=c.select_validation_axes(eligible)
    admission=[dict(cell_id=x['cell']['id'],axis=x['axis'],wrapper_instruction_bound=
        3*x['cell']['q']*sum(bounds[:x['cell']['prefix']+1])+5) for x in axes]
    if any(x['wrapper_instruction_bound']>c.phase_b_plan()['native_instruction_cap'] for x in admission):
        raise ValueError('STATIC_INSTRUCTION_ADMISSION')
    profile_b0=c.digest(dict(kind='LEGACY449_BINARY_PROFILE_NOT_INVENTORIED',source=B0_SOURCE,
        compiler_hash=compiler,receipt=expected['sha256']))
    profile_b0i=r.read_json(ART/'track_a_native_df_metadata_binding/2026-10-16/binding_inventory_v1.json')[0]['profile']
    transfer=dict(schema='V1_conditional_transfer_audit_v1',
        status='AUDITED_CONDITIONAL_TRANSFER_REQUIRES_COMPILE_VALIDATION',
        core_source_byte_identity=same_sources,old_B0_SOURCE=B0_SOURCE,native_provider=c.PROVIDER,
        compiler_hash=compiler,control='symmetric_directional',
        scalar_policy='single ancilla phase after whole schedule; B0I adds saved full-Fock tail C',
        S4_policy='three unmerged signed Yoshida pieces; wrapper and scalar once',
        prefix_blocks_equal_saved_discard_roots=True,archive_sha256=old['archive']['sha256'],
        phase_a_signal_sha256=hashlib.sha256(gzip.decompress((phase/'signal_results_v1.json.gz').read_bytes())).hexdigest(),
        B0_anchor_profile=dict(stratum_receipt_digest=profile_b0,binary_profile_available=False),
        B0I_anchor_profile=profile_b0i,archive_export_profile=old['profile_reference']['sha256'],
        current_runtime_closure_digest=c.digest(old['runtime_file_hashes']),
        old_compiler_output_bit_identity_verified=False,
        transfer_equivalence_certified=False,population_error_bound=False,
        conditions='Separate source/profile fits; target-profile transfer remains a falsifiable hypothesis until max-four-axis validation. No matched B0/B0I causal interpretation.',
        paired_axes_separately_measured=True,structural_admission=admission)
    strata=[];anchors=[]
    for method,data,profile in (('B0',b0,profile_b0),('B0I',b0i_new,profile_b0i)):
        for axis in c.AXES:
            stratum=dict(method=method,formula='2nd',axis=axis,control='symmetric_directional',
                compiler_hash=compiler,profile_hash=profile,phase_policy='single_scalar_on_ancilla')
            rows=[]
            for x in data:
                if x['axis']!=axis:continue
                if method=='B0':
                    cid=x['cell_id'];parts=cid.split('_');prefix=int(parts[1][1:]);q=int(parts[2][1:])
                else:
                    prefix=int(x['prefix']);q=int(x['q']);cid=f'B0I_p{prefix}_q{q}_2nd'
                rows.append(dict(id=cid+':'+axis,cell_id=cid,prefix=prefix,q=q,RZ=int(x['RZ']),
                    status='MEASURED_COMPLETE',stratum=stratum,
                    origin_SOURCE=x.get('source_commit',x.get('origin_SOURCE')),
                    fingerprint=x.get('fingerprint',x.get('numeric_fingerprint'))))
            strata.append(dict(stratum=stratum,rows=rows));anchors.extend(rows)
    frozen=costs.freeze_models(strata,contract=c,eligible_ids=eligible,
        transfer_receipts=[dict(verified=True,receipt_sha256=c.digest(transfer),
                               meaning='audit of declared conditional transfer, not binary equivalence')])
    frozen.update(phase_a_result_commit='bf865de1efae51ce1663c878ad1e9daa5bb915c2',
        transfer_equivalence_certified=False,scope='H6 linear 1Angstrom STO-3G DF19 T=.8; RZ P=0',
        S2_application_prefixes=list(range(6,16)),S4_application_prefixes=list(range(11,16)),
        q_domain=list(c.QS),conservative_bias_certificate=None)
    coverage=[]
    for rec in signals['signals']:
        coverage.append(dict(cell_id=rec['cell']['id'],method=rec['cell']['method'],formula=rec['cell']['formula'],
            measured_axis_cost_available=False,
            eligible={e:{f:v['cost_acquisition_eligible'] for f,v in factors.items()}
                      for e,factors in rec['precision_rows'].items()},
            proposed_cost_kind='FROZEN_S2_MODEL' if rec['cell']['formula']=='2nd' else 'UNVALIDATED_S4_HYPOTHESIS'))
    return frozen,transfer,dict(anchors=anchors,additional_candidates=coverage,
        old_B0_raw_receipt=expected,old_B0_result_commit=B0_RESULT,
        B0I_csv_sha256=r.sha(family/'axis_compiled_metrics_v1.csv'),
        excluded_B0I_legacy_axes=len(b0i)-len(b0i_new),new_compile=0,new_five_precision_cost_evaluation=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-config',type=Path,required=True)
    p.add_argument('--private-directory',type=Path,required=True)
    p.add_argument('--public-directory',type=Path,required=True)
    a=p.parse_args();old=r.read_json(a.old_config)
    for path in (a.private_directory,a.public_directory):
        if not path.is_absolute() or path.exists() or not path.is_relative_to('/home/AbeHiromu/projects'):
            raise ValueError('NEW_ABSOLUTE_DESTINATION_REQUIRED')
    frozen,transfer,coverage=freeze(old)
    a.private_directory.mkdir(mode=0o700);a.public_directory.mkdir(parents=True)
    for name,data in (('frozen_model.json',frozen),('transfer_receipt.json',transfer),('saved_cost_coverage.json',coverage),('plan.json',c.phase_b_plan())):
        put(a.public_directory/name,data)
    config=deepcopy(old)
    config.pop('phase_a_parallel',None)
    config.update(phase_b=True,plan_sha256=c.digest(c.phase_b_plan()),resources=c.phase_b_plan()['resources'],
        source_hashes={name:r.sha(ROOT/name) for name in SOURCES},
        frozen_model_sha256=r.sha(a.public_directory/'frozen_model.json'),frozen_model_digest=c.digest(frozen),
        control=str(a.private_directory/'control01'),outputs={'compile_validation':str(a.private_directory/'output01')},
        locks={'compile_validation':str(a.private_directory/'compile_ONE_SHOT')},approved=False,
        runtime_authorization=False,allowed_cpus=[],absolute_launch_command=None)
    signal_dir=Path(old['outputs']['signal'])
    for name,path in [('frozen_model.json',a.public_directory/'frozen_model.json'),
                      ('transfer_receipt.json',a.public_directory/'transfer_receipt.json'),
                      ('V1_signal_complete.json',signal_dir/'V1_signal_complete.json'),
                      ('V1_signal_terminal.json',signal_dir/'terminal.json')]:
        config['input_mapping'][name]=str(path)
        config['files'].append(dict(logical_name=name,path=str(path),bytes=path.stat().st_size,sha256=r.sha(path)))
    if r.sha(signal_dir/'V1_signal_complete.json')!=transfer['phase_a_signal_sha256']:
        raise ValueError('PUBLIC_PRIVATE_SIGNAL_IDENTITY')
    put(a.private_directory/'config_draft.json',config)
    put(a.public_directory/'preparation_binding.json',dict(plan_sha256=config['plan_sha256'],
        frozen_model_sha256=config['frozen_model_sha256'],frozen_model_digest=config['frozen_model_digest'],
        private_config_sha256=r.sha(a.private_directory/'config_draft.json'),source_hashes=config['source_hashes'],
        runtime_closure_digest=c.digest(config['runtime_file_hashes']),
        grant_issued=False,compile_started=False,next_stage_authorized=False))
    print(json.dumps(dict(models=len(frozen['models']),anchors=len(coverage['anchors']),
        predictions=frozen['predictions'],frozen_model_sha256=config['frozen_model_sha256']),indent=2))


if __name__=='__main__':main()
