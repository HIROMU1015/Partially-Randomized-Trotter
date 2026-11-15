"""Adopted PF6 beta=3/4 cost menu: exact preparation, at most 12 Q3 calls."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


S = load_script('_budget_observed_runner', 'scripts/run_full_h_kappa2_scale3o2_q3_fixed.py')
M, R = S.M, S.R
P, T, H = M.P, M.T, M.H
D = load_script('_budget_display', 'scripts/summarize_pf_shortlist_total_q3_fixed.py')
PF_METADATA, PF_CHECK, SAVED_WORKER = P.metadata, P.check_source, M.worker
PLAN = '7f27e18f06dde12127c85be898777b685d29bacc'
PF = '8dba096dc39788b8948af9eafe394f03d386c520'
B2 = '1002f1dfea34de439ca1b704a0192bca473c5387'
PF_DIR = 'docs/research/pf_shortlist_q3_fixed_results_20261115/'
B2_DIR = 'docs/research/pf6_allround_cost_20261115/'
DIRECTORY = 'docs/research/pf6_cost_budget_q3_results_20261115'
INPUT = DIRECTORY+'/INPUTS.json'
CALLER = 'scripts/run_pf6_cost_budget_q3_fixed.py'
N0 = [59,55,51,47,43,39,35,31,27,23,19,15,11]
PASS = 'MODEL_BOUND_MEETS_ADOPTED_THRESHOLD'


def document(commit, path):
    return json.loads(R.blob(commit, path))


def ref(commit, path, selector=None):
    return dict(commit=commit, path=path, **({} if selector is None else dict(selector=selector)))


def check_source():
    PF_CHECK()
    for path in ('scripts/run_pf_shortlist_total_q3_fixed.py',
                 'scripts/run_pr_df_fullh_total_q3_fixed.py',
                 'scripts/run_full_h_kappa2_scale3o2_q3_fixed.py',
                 'scripts/summarize_pf_shortlist_total_q3_fixed.py'):
        if (ROOT/path).read_bytes() != R.blob(PLAN, path):
            raise ValueError('Reused worker/observation SOURCE differs: '+path)


def exact(value):
    return T.B.a.exact_value(value)


def integer(value):
    if type(value) is not int or value <= 0:
        raise ValueError('Positive exact integer required')
    return value


def generate(caps, prices, budget, rule):
    """Only the adopted finite breakpoint chain; never evaluate risk."""
    if len(caps) != len(prices) or not caps:
        raise ValueError('Price/vector length mismatch')
    caps = [integer(n) for n in caps]
    prices = [integer(c) for c in prices]
    budget = Fraction(budget)
    weights = [Fraction(n) if rule == 'A_SHAPE' else Fraction(n,c)
               for n,c in zip(caps,prices)]
    if rule not in ('A_SHAPE','B_PRICE'):
        raise ValueError('Unadopted allocation rule')
    points = sorted({Fraction(k)/w for n,w in zip(caps,weights) for k in range(1,n+1)})

    def at(lam):
        shots = [max(1,min(n,(lam*w).numerator//(lam*w).denominator))
                 for n,w in zip(caps,weights)]
        return shots, sum(n*c for n,c in zip(shots,prices))

    if sum(prices) > budget:
        return None
    lo, hi = 0, len(points)
    while lo < hi:
        mid = (lo+hi)//2
        if at(points[mid])[1] <= budget:
            lo = mid+1
        else:
            hi = mid
    index = max(0,lo-1)
    shots,cost = at(points[index])
    if cost > budget:
        raise ValueError('Breakpoint budget inconsistency')
    return dict(rule=rule,lambda_value=exact(points[index]),shots=shots,
        conditional_cost=cost,unused_budget=exact(budget-cost),budget=exact(budget),
        next_breakpoint_cost=None if index+1==len(points) else at(points[index+1])[1],
        finite_breakpoints=len(points))


def count_bounds(shots, frontier=8192):
    nx,ny = ([r['N_'+a] for r in shots] for a in ('X','Y'))
    if nx == ny:
        return H.finite_bounds(nx,frontier)
    cells = [(integer(x)+1)*(integer(y)+1) for x,y in zip(nx,ny)]
    states,pairs = min(cells[0],frontier),cells[0]
    for c in cells[1:]:
        pairs += states*c
        states = min(frontier,states*c)
    return dict(maximum_round_outcomes=max(cells),
        verification_pairs_upper=sum(a*b for a,b in zip(cells,cells[1:])),
        transition_pairs_upper=pairs,
        rule='FULL_XY_CARTESIAN_COUNTS_AND_FRONTIER_UPPER_BOUND_NO_PROBABILITIES')


def metadata(case):
    return PF_METADATA(case)


def build_inputs():
    check_source()
    old = document(PF,PF_DIR+'INPUTS.json')
    baseline = document(PF,PF_DIR+'SUMMARY.json')
    price_rows = document(B2,B2_DIR+'AXIS_TERMS.json')['rows']
    totals = {r['label']:r['total'] for r in document(B2,B2_DIR+'RESULT.json')['rows']}
    if [c['label'] for c in old['cases']] != P.LABELS or len(price_rows) != 156:
        raise ValueError('Fixed six/156-axis input mismatch')
    if {p:sum(r['price_class']==p for r in price_rows) for p in
            ('DIRECT_COMPILED_DETERMINISTIC','CONDITIONAL_FIXED_COMPILER_CLOSURE_DERIVATION')} != {
            'DIRECT_COMPILED_DETERMINISTIC':96,'CONDITIONAL_FIXED_COMPILER_CLOSURE_DERIVATION':60}:
        raise ValueError('Price domain partition differs')
    bases, cases, proposals = [], [], []
    for original in old['cases']:
        label = original['label']
        binding,view = metadata(original)
        rows = [(i,r) for i,r in enumerate(price_rows) if r['label']==label]
        rows.sort(key=lambda v:(v[1]['m'],v[1]['axis']))
        if [(r['m'],r['axis']) for _,r in rows] != [(m,a) for m in range(13) for a in ('X','Y')]:
            raise ValueError('Missing/duplicate price axes: '+label)
        for i,r in rows:
            m = r['m']
            physical = view['signal_packet']['means'][m]['provider_binding']['physical_context']['physical_candidate']
            if (r['original_configuration_id'] != original['configuration_id']
                    or r['total_configuration_id'] != original['total_view_id']
                    or r['physical_id'] != physical['id']
                    or r['original_candidate_id'] != binding['original_configuration']['rounds'][m]['candidate']['id']
                    or r['total_candidate_id'] != binding['total_configuration']['rounds'][m]['candidate']['id']
                    or r['N'] != N0[m]):
                raise ValueError('Physical/canonical/axis price join differs: '+label)
            integer(r['C'])
        prior = next(r for r in baseline['rows'] if r['label']==label)
        if prior['status'] != PASS:
            raise ValueError('Fixed baseline status differs')
        bases.append(dict(label=label,method=T.B.comparison_method(binding['original_configuration']['spec']),
            configuration_id=original['configuration_id'],total_view_id=original['total_view_id'],
            allocation=deepcopy(original['allocation']),conditional_cost=totals[label],
            status=prior['status'],MSE_lower=prior['MSE_lower'],MSE_upper=prior['MSE_upper'],
            Q3_source=ref(PF,PF_DIR+prior['result_path']),cost_source=ref(B2,B2_DIR+'RESULT.json'),
            evidence_status=prior['evidence_status'],original_API_return_saved=False))
        benchmark = totals['FULL_PF_S2_16h_over_7' if label.startswith('FULL_') else 'B0I_P7_S2_16h_over_7']
        budget = Fraction(3,4)*benchmark
        seen = {}
        for rule in ('A_SHAPE','B_PRICE'):
            allocation_id = label+'_'+rule+'_COST_BUDGET_3O4_01'
            g = generate([N0[r['m']] for _,r in rows],[r['C'] for _,r in rows],budget,rule)
            if g is None:
                proposals.append(dict(base_label=label,allocation_id=allocation_id,rule=rule,
                    disposition='BUDGET_MENU_EMPTY',scientific_calls=0))
                continue
            shots = [dict(m=m,N_X=g['shots'][2*m],N_Y=g['shots'][2*m+1]) for m in range(13)]
            vector = tuple(g['shots'])
            disposition = 'NEW_EVALUATION'
            reused = None
            if vector == tuple(n for n in N0 for _ in ('X','Y')):
                disposition,reused = 'REUSE_BASELINE',label
            elif vector in seen:
                disposition,reused = 'SAME_VECTOR_AS_OTHER_RULE',seen[vector]
            new = deepcopy(original)
            new.update(label=label+'__'+rule,base_label=label,method=bases[-1]['method'],
                allocation=dict(allocation_id=allocation_id,shot_policy='USER_SUPPLIED',shots=shots),
                rule=rule,generation={k:v for k,v in g.items() if k!='shots'},
                benchmark_cost=benchmark,conditional_cost=g['conditional_cost'],
                unused_budget=g['unused_budget'],
                price_terms=[dict(m=r['m'],axis=r['axis'],N=n,C=r['C'],N_times_C=n*r['C'],
                    price_class=r['price_class'],price_source=ref(B2,B2_DIR+'AXIS_TERMS.json',['rows',i]))
                    for (i,r),n in zip(rows,g['shots'])])
            bounds = count_bounds(shots)
            options = old['options']
            if (bounds['maximum_round_outcomes']>options['maximum_round_outcomes']
                    or bounds['verification_pairs_upper']>options['maximum_verification_pairs']
                    or bounds['transition_pairs_upper']>options['total_state_refinement']['maximum_transition_pairs']):
                raise ValueError('Count-only scope cannot contain concrete allocation')
            new['finite_scope_preflight'] = bounds
            proposals.append(dict(base_label=label,label=new['label'],allocation_id=allocation_id,
                disposition=disposition,reused_result=reused,rule=rule,allocation=deepcopy(new['allocation']),
                conditional_cost=new['conditional_cost'],budget=g['budget'],unused_budget=g['unused_budget']))
            if disposition == 'NEW_EVALUATION':
                cases.append(new)
                seen[vector] = new['label']
    if len(cases)>12 or any(sum(c['base_label']==b['label'] for c in cases)>2 for b in bases):
        raise ValueError('Adopted scientific call limit exceeded')
    contract = deepcopy(old['statistical_contract'])
    contract['id'] = 'PF6_CURRENT_WIDTH_COST_BUDGET_3O4_TWO_SHAPES_TOTAL_Q3'
    options = deepcopy(old['options'])
    options['model_loss_contract']['scope'] = 'PF6_CURRENT_WIDTH_TWO_SHAPES_COST_BUDGET_3O4'
    concurrency = min(len(cases),max(1,len(os.sched_getaffinity(0))//(old['acceleration']['workers_per_configuration']+1)))
    return dict(fixed_plan=ref(PLAN,'docs/research/pf_joint_width_shot_plan_20261115/REPORT.md'),
        fixed_prices=ref(B2,B2_DIR+'AXIS_TERMS.json'),fixed_Q3=ref(PF,PF_DIR+'SUMMARY.json'),
        beta=exact(Fraction(3,4)),baseline=bases,proposals=proposals,cases=cases,
        maximum_scientific_calls=12,actual_new_call_limit=len(cases),held=[],
        reused_worker=ref(PLAN,'scripts/run_pr_df_fullh_total_q3_fixed.py'),
        kernel_source=old['kernel_source'],C_source=old['C_source'],
        acceleration=deepcopy(old['acceleration']),concurrent_configurations=concurrency,
        concurrency_basis='available CPU affinity divided by 8 DP children plus one parent; no OS caps',
        statistical_contract=contract,options=options,adopted_evaluation=deepcopy(old['adopted_evaluation']),
        price_policy='PF_FIXED_COMPILER_CLOSURE_CONDITIONAL_60_AXES_V1',
        duplicate_check='SEE_SCOPE_AND_PREFLIGHT.md',signal_recomputed=False,
        sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None)


def worker(case, inputs, source, output):
    with patch.object(M,'metadata',metadata),patch.object(M,'check_source',check_source):
        return SAVED_WORKER(case,inputs,source,output)


def prepare():
    inputs = build_inputs()
    (ROOT/DIRECTORY).mkdir(exist_ok=True)
    R.write_new(ROOT/INPUT,inputs)
    for p in inputs['proposals']:
        print(p['base_label'],p['rule'],p['disposition'],p.get('conditional_cost'),flush=True)
    print('New Q3 upper bound:',len(inputs['cases']),'parallel parents:',inputs['concurrent_configurations'],flush=True)


@contextmanager
def installed_workers(output):
    with patch.object(M,'worker',worker),S.observation(output):
        yield


def execute():
    check_source()
    sha = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for path in (INPUT,CALLER):
        if (ROOT/path).read_bytes() != R.blob(sha,path):
            raise ValueError('Caller/INPUTS not committed')
    inputs = json.loads((ROOT/INPUT).read_text())
    cases = inputs['cases']
    if len(cases)>12 or len({c['label'] for c in cases})!=len(cases):
        raise ValueError('Fixed call identities differ')
    output = ROOT/DIRECTORY/'run01'
    output.mkdir()
    for c in cases:(output/c['label']).mkdir()
    source = dict(commit=sha,path=INPUT,caller=CALLER)
    R.write_new(output/'EXECUTION.json',dict(source=source,started_UTC=R.utc(),
        maximum_scientific_calls=12,actual_new_call_limit=len(cases)))
    with installed_workers(output):
        result = M.supervise(cases,inputs,source,output)
    exits = [c for p in output.glob('PROCESS_EXITS_*.jsonl') for line in p.read_text().splitlines()
             for c in json.loads(line)['children']]
    exit_gate = dict(recorded_children=len(exits),all_exit_codes_zero=bool(exits) and all(c['exitcode']==0 for c in exits))
    status = ('FIXED_SCIENTIFIC_WORKERS_SAVED' if not result['errors'] and not result['monitor_errors']
        and exit_gate['all_exit_codes_zero'] and len(result['receipts'])==len(cases)
        and all(r['status']!='UNKNOWN' for r in result['receipts']) else 'UNKNOWN_STOP_NO_RETRY')
    R.write_new(output/'COMPLETION.json',dict(status=status,**result,finished_UTC=R.utc(),
        process_exit_gate=exit_gate,mandatory_stop=True,sufficient_N=None,G=None,G_qualified=None))
    print('Finished:',status,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','execute'))
    (prepare if parser.parse_args().mode=='prepare' else execute)()
