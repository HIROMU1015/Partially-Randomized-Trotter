"""Proposed two-prefix menu metadata only. Never run a scientific provider.

Reuses B's published fixed-A adapter/identity; integer schedule generation is
metadata, not an evaluation of a mean, normalizer, N, risk or cost.
"""
import ast
from copy import deepcopy
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import pr_rpe_rmse_s0_v2 as fixed
from trottertracks import pr_rpe_signal_plan as metadata
from trottertracks.pr_rpe_signal_provider import numerical_recipe

B='437ba4acdb82708b6fc98e3cd119a2a734f7cf64'
SIGNAL='fefd548ecd56f60d72031b3d385f6f5e6ee95d06'
S0='7da8a35f68e9fdc2b46c361dd2c68e2a799b1e75'
INDEX=dict(commit='7a8bc7f132e166390c943e6b32c92cc6abe1baf4',
    path='artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/SOURCE_INDEX.json')
PHYSICAL=dict(commit='bba150834ca30bdb186fbfd58c55a8d51177b855',
    path='docs/research/h4_pr_physical_error_contract_20261113/PHYSICAL_INPUTS.json')
IMPLEMENTATION=dict(commit=SIGNAL,path='src/trottertracks/pr_rpe_signal_provider.py',
    recipe='DIRECT_EXPM_HEAD_PAIRED_TAYLOR_MATRIX_POWER_V1')


def prepare():
    store=fixed.p.SavedArtifacts(ROOT)
    s0=store.read(dict(commit=S0,path=fixed.S0_DIR+'CONFIGURATION.json'))
    original=store.read(dict(commit='3216d27a69ac8532d2a22e78f5f94491d202d238',
        path='docs/research/h4_s0_v2_integration_20261113/run01/CONFIGURATION_V2.json'))
    index=store.read(INDEX);desc=index['T20_reference_files']['H4']
    data=Path(desc['path']).read_bytes()
    if hashlib.sha256(data).hexdigest()!=desc['sha256']:raise ValueError('Saved tail descriptor changed')
    context=json.loads(data)
    if context['binding']!=s0['inputs']['design']['binding']:raise ValueError('Saved binding differs')
    # Reuse the existing narrow B metadata loader, in an owned namespace with
    # its B source pinned to the user's latest release; no module globals mutate.
    source=subprocess.check_output(['git','show',SIGNAL+':src/trottertracks/pr_rpe_rmse_s0_v2.py'],cwd=ROOT)
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='b_metadata')
    scope=dict(vars(fixed),B_RELEASE=B)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(OUT/'PREPARE_MENU.py')+':B_metadata','exec'),scope)
    b=scope['b_metadata'](ROOT)
    physical=store.read(PHYSICAL)
    if len(physical['rounds'])!=13:raise ValueError('13 saved reference descriptors required')
    archive_source=dict(commit='155529e20a8fac39e00786c40b57830b71203c4f',
        path='artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/signal_records.json.gz')
    archive=json.loads(gzip.decompress(subprocess.check_output(
        ['git','show',archive_source['commit']+':'+archive_source['path']],cwd=ROOT)))
    common_keys=('M','round_count_M_plus_one','h','h_exact','kappa_paper_exact','kappa_paper_approx',
        'target_rmse_ha_exact','qpe_budget_ha_exact','trot_budget_ha_exact',
        'target_rmse_ha','qpe_budget_ha','trot_budget_ha','error_rule','normalization_rule')
    plan=dict(status='PROPOSED_FINITE_MENU_AWAITING_USER_EXECUTION_PERMISSION',
        B_source=B,signal_source=IMPLEMENTATION,
        conditions={k:deepcopy(s0['conditions'][k]) for k in common_keys},
        target=deepcopy(physical['target']),fixed_S0_source=dict(commit=S0,path=fixed.S0_DIR+'CONFIGURATION.json'),
        changing_design_variable='L_D_ONLY_R_DERIVED_FROM_UNCHANGED_COMMON_PAPER_KAPPA',
        candidates=[],deferred_prefixes=[3,5,7,9],unsupported_saved_prefixes=[4,6,8],
        scientific_calls=0,saved_signal_reanalysis=0,
        sufficient_N=None,global_RMSE_bound=None,G=None,winner=None,
        limits=dict(round_candidates=26,complex_outputs_raw_corrected_exact_tail=78,
            finite_normalizer_piece_evaluations=26,new_reference_actions=0,new_signal_sampling=0,
            new_state_or_hamiltonian_generation=0,new_N=0,new_C=0,new_G=0),
        baseline_reference_compatible_but_old_means_not_relabelled=True,mandatory_stop=True)
    summary=[]
    for prefix in (1,2):
        inputs=deepcopy(s0['inputs']);inputs['design']['prefix']=prefix
        inputs['design']['source']=dict(input_source=deepcopy(s0['inputs']['design']['source']),
            composition_recipe=deepcopy(IMPLEMENTATION),status='PROPOSED_PREFIX_COMPOSITION_NOT_SCIENCE_RESULT')
        tail_source=dict(index=INDEX,context_sha256=desc['sha256'],selector=['prefixes',str(prefix)])
        inputs['tail_norm'].update(prefix=prefix,lambda_R=context['prefixes'][str(prefix)]['lambda_R'],source=tail_source)
        config=b.generate(b.a_spec(inputs,'PR'));b.validate(config)
        rows={r['m']:r['physical_candidate'] for r in config['rounds']}
        pending=metadata.plan(config,rows,b_validate=b.validate,
            b_exact_scalar=lambda v:fixed.Fraction(v['numerator'],v['denominator']) if isinstance(v,dict) else fixed.Fraction(v),
            numerical_recipe=numerical_recipe(IMPLEMENTATION))
        if prefix==1:
            if [r['physical_candidate'] for r in config['rounds']]!=[r['physical_candidate'] for r in original['rounds']]:
                raise ValueError('Baseline physical schedule differs from published S0 V2')
        if config['spec']['kappa']!=original['spec']['kappa']:raise ValueError('Common paper kappa changed')
        for job in pending['jobs']:
            m=job['m'];c=rows[m]
            ref=physical['rounds'][m]
            if c['T']!=ref['physical_candidate']['T']:raise ValueError('Saved reference time differs')
            if not {'reference','independent_reference','signal_difference'}<=set(ref['diagnostics']['reference_pair']):
                raise ValueError('Missing saved reference pair field')
            if job['status']!='MISSING_MEAN_REQUIRES_PERMISSION':raise ValueError('Unexpected automatic acceptance')
            summary.append(dict(L_D=prefix,m=m,T=c['T'],q_RPE=job['q_RPE'],q_PF=c['q'],delta=c['delta'],
                R_outer=c['R_outer'],events=c['q']*c['R_outer'],allocation=json.dumps(job['physical_context']['structure']['allocation']),
                physical_ID=c['id'],B_candidate_ID=job['B_candidate_id'],B_configuration_ID=config['id']))
        name='CONFIGURATION_LD'+str(prefix)+'.json'
        (OUT/name).write_text(json.dumps(config,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        plan['candidates'].append(dict(label='PR-LD'+str(prefix),L_D=prefix,configuration_file=name,
            configuration_id=config['id'],tail_descriptor=deepcopy(context['prefixes'][str(prefix)]),tail_source=tail_source,
            new_round_candidates=13,new_complex_outputs=39,new_normalizers=13,
            saved_reference_pairs=13,source_reuse='INPUTS_TAIL_METADATA_FULL_H_REFERENCE_ONLY',
            published_physical_match_rounds=[m for m,c in rows.items()
                if c==physical['rounds'][m]['physical_candidate']],
            track_a_archived_physical_match_rounds=[m for m,c in rows.items()
                if c['id'] in archive and archive[c['id']]['candidate']==c],
            track_a_archive_source=archive_source,direct_fefd_mean_accepted_rounds=[],
            baseline_physical_recipe_preserved=prefix==1))
    (OUT/'MENU.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    with (OUT/'SCHEDULE.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n');writer.writeheader();writer.writerows(summary)
    assert not any(v in sys.modules for v in ('numpy','scipy','qiskit'))
    print(json.dumps(dict(status=plan['status'],configurations=[r['configuration_id'] for r in plan['candidates']],
        rounds=len(summary),metadata_only=True,scientific_calls=0,numpy_scipy_qiskit_loaded=False),indent=2))


if __name__=='__main__':prepare()
