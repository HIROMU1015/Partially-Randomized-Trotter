"""Passive whole-schedule signal requests; not a candidate generator/optimizer.

The caller supplies B's canonical validator and physical recipes. No signal,
normalizer, N/C/G, scientific inventory or prioritization is evaluated here.
"""
from copy import deepcopy
from . import pf_pr_rpe_pipeline as common


def plan(config, physical_by_round, *, b_validate, b_exact_scalar, numerical_recipe, saved_inventory=()):
    """A metadata bridge over B-owned identities; emits ALL M+1 round requests.

    saved_inventory contains contexts, field NAMES and SOURCE/status only.
    Matching a raw context is not acceptance of a mean or a new N packet.
    """
    b_validate(config)
    spec=config['spec'];rows=config['rounds'];M=spec['M']
    if spec['method']!='PR' or config['schema']!='PR_RPE_RMSE_V2':
        raise ValueError('Canonical PR V2 whole schedule required')
    if (len(rows)!=M+1 or [r['m'] for r in rows]!=list(range(M+1))
            or set(physical_by_round)!=set(range(M+1))):
        raise ValueError('All rounds and physical recipes required; no terminal-only plan')
    if not numerical_recipe.get('source') or numerical_recipe.get('deterministic_PF')!='DIRECT_EXPM':
        raise ValueError('Explicit source-bound direct PF numerical recipe required')
    if numerical_recipe.get('sampling') is not False:
        raise ValueError('Finite mean planner has no trajectory sampler')
    if not numerical_recipe.get('finite_RTE'):
        raise ValueError('Explicit finite-K numerical recipe required')
    jobs=[]
    for row in rows:
        c=deepcopy(physical_by_round[row['m']]);st=common.structure(c)
        T=b_exact_scalar(row['T']);delta=b_exact_scalar(row['actual_delta_pf'])
        if (c['id']!=common.candidate_id(c) or c['method']!='PR'
                or c['system']!=spec['binding']['system']
                or b_exact_scalar(c['T'])!=T
                or b_exact_scalar(c['delta'])!=delta
                or (c['formula'],c['prefix'],c['K'],c['q'],c['R_outer'])!=(
                    spec['formula'],spec['L_D'],spec['finite_K'],row['q_PF'],row['R_outer'])
                or not common.same(st,row['structure'])):
            raise ValueError('Physical recipe differs from B round')
        if row['q_RPE']!=1<<row['m']:
            raise ValueError('q_RPE and q_PF are distinct fields')
        # Full-H pure RTE and full PF endpoints are separate comparison methods.
        if not 1<=c['prefix']<spec['binding']['rank']:
            raise ValueError('Nonempty saved DF tail and deterministic prefix required')
        context=dict(physical_candidate=c,structure=deepcopy(st),binding=deepcopy(spec['binding']),
            profile=deepcopy(spec['profile']),split=deepcopy(spec['split']),law=deepcopy(spec['law']),
            numerical_recipe=deepcopy(numerical_recipe))
        matches=[x for x in saved_inventory if common.same(x.get('physical_context'),context)]
        if len(matches)>1:raise ValueError('Ambiguous saved evidence; select SOURCE explicitly')
        evidence=matches[0] if matches else None
        if evidence and (not evidence.get('source') or any(k in evidence for k in ('N_X','N_Y','G'))):
            raise ValueError('Supply source-bound mean metadata, not old N/G records')
        fields=set((evidence or {}).get('signal_fields',[]))
        required={'raw','corrected','exact_tail','reference','log_B','allowance_evidence'}
        status=('SAVED_FAILURE_NO_AUTOMATIC_RETRY' if evidence and evidence.get('status')=='NUMERICAL_FAILURE'
                else 'SAVED_MEAN_METADATA_NEEDS_ACCEPTANCE' if evidence and required<=fields
                else 'SAVED_EVIDENCE_INCOMPLETE_NO_FALLBACK' if evidence else 'MISSING_MEAN_REQUIRES_PERMISSION')
        jobs.append(dict(m=row['m'],q_RPE=row['q_RPE'],q_PF=row['q_PF'],T=deepcopy(row['T']),
            B_configuration_id=config['id'],B_candidate_id=row['candidate']['id'],
            B_identity=deepcopy(row['identity']),B_law_binding=deepcopy(row['law_binding']),
            physical_context=context,status=status,saved_source=deepcopy((evidence or {}).get('source')),
            missing_fields=sorted(required-fields),N_X=None,N_Y=None,C=None,G=None,
            scientific_PASS_FAIL=None,automatic_retry=False))
    return dict(status='WHOLE_SCHEDULE_METADATA_ONLY',B_configuration_id=config['id'],
        jobs=jobs,round_count=M+1,science_calls=0,external_provider_calls=0,
        cost_optimizer_implemented=False,scientific_range_adopted=False,global_RMSE_bound=None,
        sufficient_N=None,winner=None,mandatory_stop=True)
