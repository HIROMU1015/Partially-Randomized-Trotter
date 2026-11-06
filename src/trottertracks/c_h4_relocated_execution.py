"""Fixed-three-candidate glue; passive preflight never decodes scientific data.

execute() is a dormant science entry requiring separate user permission. The
preparation CLI exposes preflight only. No approval artifact is required.
"""
from pathlib import Path
import math
from . import c1_stage1_execution as io
from . import c_h4_relocated_downstream as bridge

SCOPE='C_H4_RELOCATED_THREE_FIXED_JU_DOWNSTREAM_V1'
CONTRACT='docs/research/research_c_h4_relocated_downstream_contract_unapproved_20261106.json'


def preflight(root,path=CONTRACT):
    root=Path(root).resolve();c=io.read(root/path)
    if (c['scope']!=SCOPE or c['relocation']['version']!=bridge.VERSION
            or c['r_grid']!=list(bridge.R_VALUES) or c['retry']!=0 or c['resume']!=0
            or not c['mandatory_stop'] or set(c['candidates'])!={'P','E1','E2'}
            or c['readiness']!='TECHNICALLY_PREPARED_REQUIRES_SCIENTIFIC_ADOPTION_AND_EXECUTION_PERMISSION'):
        raise ValueError('FIXED_RELOCATION_SCOPE')
    if io.sha(root/c['SOURCE'])!=c['SOURCE_sha256']:raise ValueError('SOURCE_MANIFEST')
    for group in (io.read(root/c['SOURCE'])['files'],c['inputs']):
        for p,w in group.items():
            if io.sha(root/p)!=w:raise ValueError('SOURCE_OR_INPUT:'+p)
    for row in c['candidates'].values():
        if (c['inputs'][row['path']]!=row['file_sha256']
                or set(row['array_identity'])!={'J','U'}):raise ValueError('CANDIDATE_IDENTITY')
    if c['inputs'][c['original_input']]!=c['input_identity']:raise ValueError('ORIGINAL_INPUT_IDENTITY')
    output=(root/c['fresh_output']).resolve()
    if not output.is_relative_to(root):raise ValueError('LOCAL_FRESH_OUTPUT')
    if output.exists():raise FileExistsError('FRESH_OUTPUT_RETRY_RESUME_ZERO')
    return c


def comparisons(c,plan,finals,states):
    """Registered A/B differences; selection-frozen best PR/D secondaries.

    All stochastic intervals come from the existing Welch/Student-t helpers.
    Missing/ineligible policies stay unresolved. No final-based reselection.
    """
    from . import c_near_r32_statistics as stats
    out={}
    for regime,right in (('A_common_r32','E1'),('B_fair_retuned','E2')):
        rs=(32,32) if right=='E1' else (plan['tuned_r']['P'],plan['tuned_r'][right])
        keys=(('P',rs[0]),(right,rs[1]))
        if not all(k in finals for k in keys):
            out[regime]=dict(status='UNRESOLVED_MISSING_FINAL_PAIR');continue
        x,y=(finals[k] for k in keys)
        policy=dict(stats.POLICY,**c['statistics']['primary'])
        out[regime]=stats.primary(x['rows'],y['rows'],x['bindings']['ideal_exact_native'],
            y['bindings']['ideal_exact_native'],expected_n=64,statistics=policy)
    for right in ('E1','E2'):
        choices=[plan['best_execution'][x] for x in ('P',right)]
        key='secondary_best_policy_P_vs_'+right
        if None in choices:
            out[key]=dict(status='UNRESOLVED_NO_ELIGIBLE_POLICY');continue
        def side(label,choice):
            if choice=='D':return states[label]['D']['ideal_exact_native']['cost']['totals']
            return finals[label,plan['tuned_r'][label]]
        left,rhs=(side(x,ch) for x,ch in zip(('P',right),choices))
        if choices==['PR','PR']:
            value=stats.primary(left['rows'],rhs['rows'],left['bindings']['ideal_exact_native'],
                rhs['bindings']['ideal_exact_native'],expected_n=64,statistics=dict(stats.POLICY,samples=64,confidence=.95))
        elif choices==['D','D']:
            value=dict(status='CONDITIONAL_DETERMINISTIC_POINT_COMPARISON',difference=left['rz_count']-rhs['rz_count'],
                left=left,right=rhs,interval=None)
        else:
            random=left if choices[0]=='PR' else rhs
            fixed=rhs if choices[0]=='PR' else left
            value=stats.shared.price_statistics(random['rows'],dict(random['bindings']['ideal_exact_native'],D=fixed),
                expected_n=64,statistics=dict(stats.POLICY,samples=64,confidence=.95))
            value['random_side']='P' if choices[0]=='PR' else right
        out[key]=dict(choices=choices,selection_frozen=True,individual_nominal_only=True,result=value)
    return out


def execute(root,path=CONTRACT):
    """Call ONLY after explicit permission for the fixed contract, never in prep."""
    c=preflight(root,path);root=Path(root).resolve();contract_sha=io.sha(root/path)
    import importlib.metadata
    import threading
    from collections import Counter
    from qiskit import qpy
    from . import c_pr_aware_df_stage_b_runner as loader
    from . import c_pr_aware_df_stage_b2 as b2
    from .c1_stage1 import car_bilinears
    from .c1_h4_native_fair import profile_dict
    if profile_dict()!=c['compiler']:raise ValueError('COMMON_COMPILER_PROFILE')
    for p,v in c['packages'].items():
        if importlib.metadata.version(p)!=v:raise ValueError('NUMERICAL_PACKAGE:'+p)
    states={};finals={};counts=Counter();lock=threading.Lock()
    def call(kind,label,fn):
        if call.stop.is_set():raise RuntimeError('DEPENDENT_WORK_STOPPED')
        try:value=fn()
        except BaseException:
            call.stop.set();raise
        with lock:counts[kind]+=1
        return value
    call.stop=threading.Event()
    def work(task,output):
        # All numeric loads are inside the fresh one-shot task, after preflight.
        io.save(output/'contract.json',c);io.save(output/'SOURCE.json',io.read(root/c['SOURCE']))
        exported=loader.load(root/c['kappa_export'],('kappa','constant'))
        for name,x in exported.items():
            if io.array_record(x)!=c['kappa_array_identity'][name]:raise ValueError('EXISTING_KAPPA_SCALAR_IDENTITY')
        shell_reports=io.read(root/c['shell_reports']);shells={}
        for label,paths in c['shells'].items():
            qs={}
            for kind,p in paths.items():
                with (root/p).open('rb') as stream:items=qpy.load(stream)
                if len(items)!=1 or items[0].num_qubits!=9:raise ValueError('SHELL_SCHEMA')
                qs[kind]=items[0]
            shells[label],_=b2.shell_assessment(qs['raw'],qs['compiled'],label)
        car=call('car_context','shared immutable H4 CAR',lambda:car_bilinears(8))
        runtime=dict(source_manifest_sha256=c['SOURCE_sha256'],shells=c['shells'])
        for label in ('P','E1','E2'):
            row=c['candidates'][label];d=loader.load(root/row['path'],('J','U'))
            for name,value in d.items():
                if io.array_record(value)!=row['array_identity'][name]:raise ValueError('SAVED_ARRAY_IDENTITY')
            sector=loader.load(root/row['reference_path'],('J','U','generators','H'))
            rel,a,report=bridge.prepare_candidate(exported['kappa'],float(exported['constant']),**d,
                saved_sector=sector,physical=row['physical'],candidate=row['file_sha256'],input_identity=c['input_identity'],
                role='proposed' if label=='P' else 'baseline',out=output/label/'A',call=call,shell_reports=shell_reports)
            if a is None:
                identity=rel.arm.model.source_identity
                states[label]=dict(a=dict(candidate_identity=identity),rows=[dict(
                    candidate_sha256=identity,r=r,split='selection',samples=32,actual_samples=0,
                    correctness='PASS',eligible=False,total_rz=None) for r in bridge.R_VALUES],
                    D=dict(ideal_exact_native=dict(cost=dict(totals=None))),policies={},prepared=None,
                    qualification=report)
                io.save(output/label/'unqualified.json',report)
            else:states[label]=bridge.candidate_selection(root,runtime,output/label/'selection',call,rel,a,car=car,shells=shells)
        ds={}
        for label,state in states.items():
            cost=state['D']['ideal_exact_native']['cost']['totals']
            ds[label]=dict(candidate_sha256=state['a']['candidate_identity'],eligible=cost is not None,
                total_rz=None if cost is None else cost['rz_count'])
        plan=bridge.final_plan({k:v['rows'] for k,v in states.items()},ds)
        io.save(output/'frozen_selection.json',plan)
        for label,r in plan['PR_final_pairs']:
            finals[label,r]=bridge.final_pair(root,runtime,output/label/f'final_r{r}',call,states[label],r)
        result=dict(status='REGISTERED_RELOCATED_DOWNSTREAM_COMPLETE',selection=plan,
            comparisons=comparisons(c,plan,finals,states),call_counts=dict(counts),
            actual_CA_achieved=None,whole_law_synthesis_bound=None,implemented_law_error=None,
            FT_resources=None,quantum_shots_drawn=0,mandatory_stop=True)
        preflight_bytes(root,c)
        if io.sha(root/path)!=contract_sha:raise ValueError('FIXED_CONTRACT_CHANGED')
        return result
    return bridge.run_fresh(root/c['fresh_output'],[dict(id='comparison')],work)


def preflight_bytes(root,c):
    """Check only the scientific SOURCE/inputs actually used by this run."""
    if io.sha(root/c['SOURCE'])!=c['SOURCE_sha256']:raise ValueError('SOURCE_MANIFEST_CHANGED')
    for group in (io.read(root/c['SOURCE'])['files'],c['inputs']):
        for p,w in group.items():
            if io.sha(root/p)!=w:raise ValueError('SOURCE_OR_INPUT_CHANGED:'+p)
