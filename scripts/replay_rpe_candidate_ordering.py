"""Joint wide-schedule/shot ordering replay; outcomes revealed only after choice.

Reuses the fixed wide mapper/planner, resumable V2 and the prior replay clock
interface. No signal, calibration, compiler or scientific Q3 is executed.
"""
import argparse
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from trottertracks import rpe_method_optimizer as O
from trottertracks import rpe_priority_search as P
from trottertracks import rpe_schedule_search as W
from trottertracks.rpe_candidate_ordering import Selector
from replay_rpe_adaptive_throughput import SavedClock, make_index
import rpe_round_tools as T

NAME = 'PR_PAULI'
ROUND05_SOURCE = 'd8e806265c3ded8e1184181c37c2d7ec71f08ade'
PREPARED_POINTS = 'docs/research/rpe_method_optimizer_h4_20261116/kappa_abs_round04/run01/KAPPA_POINTS.json'
LEVELS = [Fraction(x) for x in ('501233190.43', '437894306.43', '414301229.37', '410199237')]


def qid(p):
    return 'Q:' + p['configuration_id'] + ':' + ','.join(map(str, p['vector']))


class Censored(RuntimeError):
    def __init__(self, reason, action):
        self.reason, self.action = reason, action
        super().__init__(reason)


class BudgetStop(RuntimeError):
    pass


class ReplayClock(SavedClock):
    """A hidden service-time oracle, not a source of selector predictions.

    No global mean from future jobs. A missing result (including a speculative
    launch) censors immediately. Missing recorded service time remains unknown.
    """
    def __init__(self, index, seeds, candidates, slots, maximum, fixed_seconds=None):
        super().__init__({}, seeds, slots, maximum)
        self.index = index;self.mean_seconds = None;self.fixed_seconds = fixed_seconds
        self.candidates = {c['configuration_id']: c for c in candidates}
        self.opened = {};self.values = {};self.observations = [];self.events = []
        self.curve = [];self.demand = None;self.unknown_requests = []
        self.profile = dict(slots=slots, dp_workers=8, prefix_reuse=False)
        self.initial_best = min((O.cost(v, self.candidates[c]['prices']) for (c, v), r in seeds.items()
                                if r['status'] == O.PASS), default=None)
        self.best_all = self.initial_best
        for (cid, vector), r in seeds.items():
            self.observations.append(dict(id=qid(dict(configuration_id=cid, vector=vector)), kind='SHOT_Q3', method=NAME,
                configuration_id=cid, status=r['status'], MSE_upper=r.get('MSE_upper'), finished_at=0,
                seconds=None, profile=self.profile, source='DECLARED_INITIAL_C_RECORD'))
        self.curve.append(dict(seconds=0., launched=0, best_all=self.best_all, job=None))

    def launch(self, proposal):
        key = self.key(proposal)
        if key in self.seeds or key in self.finishes:return True
        if len(self.finishes) >= self.maximum or self.active() >= self.slots:return False
        self.events.append(dict(event='SELECT_FOR_LAUNCH', id=qid(proposal), seconds=self.now,
                                speculative=self.demand is None or key != self.key(self.demand)))
        row = self.index.get(key)
        if row is None:
            self.unknown_requests.append(qid(proposal))
            raise Censored('UNSAVED_Q3_INCLUDING_SPECULATION', dict(id=qid(proposal), kind='SHOT_Q3'))
        seconds = self.fixed_seconds if self.fixed_seconds is not None else row.get('seconds')
        if seconds is None:
            raise Censored('RECORDED_Q3_SERVICE_TIME_MISSING', dict(id=qid(proposal), kind='SHOT_Q3'))
        self.finishes[key] = self.now + seconds
        if self.demand is None or key != self.key(self.demand):self.speculative.add(key)
        return True

    def advance(self, until):
        for key, end in sorted(self.finishes.items(), key=lambda x: (x[1], x[0])):
            if end > until or key in self.values:continue
            row = self.index[key]
            if 'value' in row:  # explicitly artificial fixture only
                value = dict(row['value'])
            else:
                raw = row['outcome'].read_bytes()
                if hashlib.sha256(raw).hexdigest() != row['sha256']:raise ValueError('Changed fixed outcome')
                value = json.loads(raw)
            result = dict(status=value['status'], MSE_lower=None if value.get('MSE_lower') is None else O.frac(value['MSE_lower']),
                MSE_upper=None if value.get('MSE_upper') is None else O.frac(value['MSE_upper']),
                source='REVEALED_AFTER_SELECTED_LAUNCH_AND_COMPLETION', calls=1)
            self.values[key] = result;self.opened[key] = str(row.get('outcome', 'ARTIFICIAL'))
            duration = self.fixed_seconds if self.fixed_seconds is not None else row['seconds']
            self.observations.append(dict(id=qid(dict(configuration_id=key[0], vector=key[1])), kind='SHOT_Q3', method=NAME,
                configuration_id=key[0], status=result['status'], MSE_upper=result['MSE_upper'], seconds=duration,
                finished_at=end, profile=self.profile, source='COMPLETED_REPLAY_JOB'))
            self.events.append(dict(event='REVEAL_COMPLETED', id=qid(dict(configuration_id=key[0], vector=key[1])), seconds=end))
            if result['status'] == O.PASS:
                cost = O.cost(key[1], self.candidates[key[0]]['prices'])
                self.best_all = cost if self.best_all is None else min(cost, self.best_all)
            self.curve.append(dict(seconds=end, launched=sum(t <= end for t in self.launch_times.values()),
                                   best_all=self.best_all, job=qid(dict(configuration_id=key[0], vector=key[1]))))
        self.now = until

    @property
    def launch_times(self):
        return {(e['id']):e['seconds'] for e in self.events if e['event']=='SELECT_FOR_LAUNCH' and e['id'] not in self.unknown_requests}

    def prefetch(self, picks):
        # Same HYBRID generator and settings. The selected answer is requested
        # first, followed by the unchanged generated preview, with key dedup.
        ordered = ([self.demand] if self.demand is not None else []) + list(picks)
        seen = set()
        for proposal in ordered:
            key = self.key(proposal)
            if key in seen:continue
            seen.add(key)
            if not self.launch(proposal):break

    def answer(self, proposal):
        key = self.key(proposal)
        if key in self.seeds:return dict(self.seeds[key], calls=0)
        if key not in self.finishes:
            if self.active() >= self.slots:self.advance(min(t for t in self.finishes.values() if t > self.now))
            if not self.launch(proposal):raise BudgetStop()
        self.advance(max(self.now, self.finishes[key]))
        return dict(self.values[key])

    def adopt(self, row):
        self.adopted.add((row['configuration_id'], tuple(row['vector'])))


def best_by_configuration(knowledge, clock):
    out = {}
    for (cid, v), record in knowledge.items():
        if record['status'] == O.PASS:
            t = O.cost(v, clock.candidates[cid]['prices'])
            if cid not in out or t < out[cid][0]:out[cid] = (t, record)
    return out


class JointReplay:
    """Existing W batch phases and a live P closure, with a common action menu."""
    def __init__(self, candidates, seeds, families, sched, clock, search, tau, generate, policy,
                 wide=None, batch=64, max_cycles=200):
        self.candidates = deepcopy(candidates);self.by_id = {c['configuration_id']:c for c in self.candidates}
        self.knowledge = deepcopy(seeds);self.active = {cid for cid, v in seeds}
        self.families, self.sched, self.clock = families, sched, clock
        self.search, self.tau, self.generate = search, tau, generate
        self.selector = Selector(policy);self.wide = wide or {};self.batch = batch;self.max_cycles = max_cycles
        self.points = [];self.trace = [];self.adopted_trace = [];self.plans = []
        self.stop = None;self.censor = None;self.current_frame = None
        for cid in sorted(self.active):
            if cid in sched:
                ld, R, k = sched[cid]
                self.points.append(dict(family=ld,kappa=k,status='IN_PROGRESS',best=None,cid=cid))
        self.refresh_points()

    def refresh_points(self, result=None):
        visible = dict(self.knowledge);visible.update(self.clock.values)
        best = {}
        for (cid, vector), rec in visible.items():
            if rec['status']==O.PASS:
                cost=O.cost(vector,self.by_id[cid]['prices']);best[cid]=min(best.get(cid,cost),cost)
        states={r['configuration_id']:r for r in (result or {}).get('by_configuration',[])}
        for point in self.points:
            cid=point['cid']
            if cid in best:point.update(status='FEASIBLE',best=best[cid])
            elif cid in states:
                point.update(status='NO_FEASIBLE_FOUND' if not states[cid]['promising'] else 'IN_PROGRESS',best=None)
            lows=[r['MSE_lower']/self.tau for (c,v),r in visible.items()if c==cid and r.get('MSE_lower')is not None]
            if lows:point['min_lower_over_tau']=min(lows)

    def outer(self):
        self.refresh_points()
        floors=[(p['family'],self.families[p['family']].R(p['kappa']),sum(self.by_id[p['cid']]['prices'])) for p in self.points]
        def floor(ld,R):
            values=[price for family,r,price in floors if family==ld and all(a<=b for a,b in zip(r,R))]
            return max(values) if values else None
        proposals, decisions, report=W.plan_wide(self.families,self.points,self.wide,floor)
        self.plans.append(dict(proposals=deepcopy(proposals),report=report))
        return proposals

    def schedule_action(self, proposal, ordinal):
        ld,k=proposal['family'],proposal['kappa'];R=self.families[ld].R(k)
        cid=next((c for c,(family,r,kap) in self.sched.items() if family==ld and r==R),None)
        prepared=cid is not None and self.by_id[cid]['ready']
        return dict(id='S:'+str(ld)+':'+','.join(map(str,R)),kind='NEW_SCHEDULE',a_rank=[0,ordinal],method=NAME,L_D=ld,
            width='BINARY64_H',kappa=str(k),physical_R=list(R),configuration_id=cid,cost=None,
            price_floor=sum(self.by_id[cid]['prices'])if prepared else None,untried=True,profile=self.clock.profile,
            preparation=dict(signal_reuse=prepared,price_reuse=prepared,requires_calibration=False if prepared else None,
                             seconds=0. if prepared else None,source='DECLARED_ARCHIVED_PREPARED_INPUTS'if prepared else'UNSAVED_INPUT_PREPARATION'),
            origin=proposal['reason'],proposal=proposal)

    def shot_actions(self, frame):
        if frame is None:return []
        known=best_by_configuration(self.knowledge,self.clock);default=frame['default_choice'];out=[]
        for i,p in enumerate(frame['eligible']):
            rec=known.get(p['configuration_id']);is_default=(p['configuration_id'],p['vector'])==(default['configuration_id'],default['vector'])
            out.append(dict(id=qid(p),kind='SHOT_Q3',a_rank=[1,0 if is_default else 1,i],method=p['method'],
                L_D=self.by_id[p['configuration_id']]['key'].get('L_D'),width='BINARY64_H',configuration_id=p['configuration_id'],
                cost=p['cost'],parent_cost=None if rec is None else rec[0],parent_MSE_upper=None if rec is None else rec[1].get('MSE_upper'),
                untried=False,profile=self.clock.profile,preparation=dict(signal_reuse=True,price_reuse=True,requires_calibration=False,seconds=0),
                origin=p['origin'],cls=p['cls'],proposal=p))
        return out

    def session(self):
        search=dict(self.search,max_steps_per_method=self.batch,on_missing='STOP')
        return P.priority_session(NAME,self.candidates,self.knowledge,self.clock,search,self.tau,self.generate,
            self.clock.maximum,spent=len(self.clock.finishes),active_configurations=self.active,
            free=lambda p:self.clock.key(p)in self.clock.seeds or self.clock.key(p)in self.clock.finishes,
            prefetch=self.clock.prefetch,prefetch_state=self.clock.state,log=self.on_adopt)

    def on_adopt(self,row):
        self.clock.adopt(row);self.adopted_trace.append(dict(row,seconds=self.clock.now,launched=len(self.clock.finishes)))

    def activate_schedule(self, action):
        """Charge a declared preparation model; never perform a scientific stage."""
        cid=action['configuration_id']
        if cid is None or not self.by_id[cid]['ready']:
            raise Censored('UNSAVED_SCHEDULE_PREPARATION',{k:v for k,v in action.items()if k!='proposal'})
        prep=action['preparation']
        reused=prep.get('signal_reuse')is True and prep.get('price_reuse')is True and prep.get('requires_calibration')is False
        seconds=0. if reused else prep.get('seconds')
        if seconds is None or not math.isfinite(float(seconds)) or seconds<0:
            raise Censored('INPUT_PREPARATION_TIME_OR_RESULT_UNRECORDED',{k:v for k,v in action.items()if k!='proposal'})
        self.clock.advance(self.clock.now+seconds)
        self.active.add(cid)
        self.points.append(dict(family=action['L_D'],kappa=Fraction(action['kappa']),status='IN_PROGRESS',best=None,cid=cid))

    def run(self):
        try:
            if self.clock.maximum==0:raise BudgetStop()
            for cycle in range(self.max_cycles):
                proposed=self.outer()
                pending=[self.schedule_action(p,i)for i,p in enumerate(proposed)]
                session=self.session();frame=None;ended=None
                try:frame=next(session)
                except StopIteration as end:ended=end.value
                while pending or frame is not None:
                    actions=(pending if len(self.clock.finishes)<self.clock.maximum else [])+self.shot_actions(frame)
                    if not actions:
                        if len(self.clock.finishes)>=self.clock.maximum:self.stop='Q3_BUDGET_EXHAUSTED'
                        break
                    selected=self.selector.choose(actions,method_best=self.clock.best_all,tau=self.tau,
                                                  now=self.clock.now,observations=self.clock.observations)
                    action=selected['chosen'];record=selected['trace']
                    self.trace.append(dict(cycle=cycle+1,seconds=self.clock.now,selected=action['id'],kind=action['kind'],
                        pool_ids=[a['id']for a in actions],active=sorted(self.active),selection=record))
                    if action['kind']=='NEW_SCHEDULE':
                        self.activate_schedule(action)
                        pending=[p for p in pending if p['id']!=action['id']]
                        if frame is not None:
                            try:frame=session.send({'refresh':True})
                            except StopIteration as end:frame=None;ended=end.value
                        else:
                            session=self.session()
                            try:frame=next(session)
                            except StopIteration as end:frame=None;ended=end.value
                    else:
                        self.clock.demand=action['proposal']
                        try:frame=session.send(action['proposal'])
                        except StopIteration as end:frame=None;ended=end.value
                self.refresh_points(ended)
                if ended and ended['stop_reason']=='Q3_BUDGET_EXHAUSTED':
                    self.stop='Q3_BUDGET_EXHAUSTED'
                if self.stop:break
                if not pending and ended and ended['stop_reason']in('ONLY_SMALL_OR_DEFERRED_LEFT','NO_CANDIDATE_LEFT_IN_RANGE','NO_FEASIBLE_POINT_IN_RANGE'):
                    if not proposed:self.stop=ended['stop_reason'];break
            if self.stop is None:self.stop='CYCLE_LIMIT'
            if self.clock.finishes:self.clock.advance(max(self.clock.finishes.values()))
        except BudgetStop:
            self.stop='Q3_BUDGET_EXHAUSTED'
            if self.clock.finishes:self.clock.advance(max(self.clock.finishes.values()))
        except Censored as censor:
            self.stop='CENSORED_'+censor.reason;self.censor=dict(reason=censor.reason,action=censor.action,seconds=self.clock.now)
        return self.project()

    def project(self):
        initial=self.clock.initial_best
        curve=[dict(row,adopted=row['job']in {qid(dict(configuration_id=c,vector=v))for c,v in self.clock.adopted})for row in self.clock.curve]
        adopted=min([O.cost(v,self.by_id[c]['prices'])for(c,v),r in self.knowledge.items()if r['status']==O.PASS],default=None)
        reaches=[]
        for target in LEVELS:
            reach=next((dict(seconds=r['seconds'],launched=r['launched'],cost=r['best_all'])for r in curve if r['best_all']is not None and r['best_all']<=target),None)
            reaches.append(dict(cost_level=target,reach=reach,right_censored=self.censor is not None))
        return dict(policy=self.selector.policy,stop=self.stop,censor=self.censor,seconds=self.clock.now,
            initial_best=initial,best_adopted=adopted,best_all_completed=self.clock.best_all,Q3_launched=len(self.clock.finishes),
            unknown_Q3_requests=len(self.clock.unknown_requests),completed=len(self.clock.values),
            revealed_records=len(self.clock.opened),adopted=len(self.clock.adopted),
            completed_not_adopted=len(set(self.clock.values)-self.clock.adopted),
            curve=curve,adopted_trace=self.adopted_trace,decisions=self.trace,target_reach=reaches,plans=self.plans,
            selector_state=self.selector.snapshot(),events=self.clock.events,
            timing_qualification='SIMULATED_RECORDED_OLD_SERVICE'if self.clock.fixed_seconds is None else'SIMULATED_FIXED_SERVICE_ASSUMPTION',
            physical_scientific_calls=0)


def compare(candidates,seeds,families,sched,index,search,tau,generate,*,slots=32,lookahead=16,budget=400,wide=None,fixed_seconds=None):
    search=dict(search,lookahead=lookahead,lookahead_mode='HYBRID',adaptive_hybrid=False,allocation_exchange=None)
    results={}
    for policy in('ROUND05','IMPROVEMENT_PER_TIME_V1'):
        clock=ReplayClock(index,deepcopy(seeds),candidates,slots,budget,fixed_seconds)
        results[policy]=JointReplay(candidates,seeds,families,sched,clock,search,tau,generate,policy,wide).run()
    aa,bb=(results[p]['decisions']for p in results)
    divergence=None
    for i,(a,b)in enumerate(zip(aa,bb)):
        if a['selected']!=b['selected']:
            divergence=dict(index=i,A=a,B=b,same_pool=a['pool_ids']==b['pool_ids']);break
    horizon=min(results[p]['seconds']for p in results)
    return dict(conditions=dict(slots=slots,lookahead=lookahead,lookahead_mode='HYBRID',DP_workers=8,prefix_reuse=False,
                               shot_exchange=False,Q3_budget=budget,initial_records=len(seeds),width='BINARY64_H'),
                first_divergence=divergence,paired_observable_horizon=horizon,policies=results,
                qualification='Retrospective round05-rule replay with limited initial C knowledge, not actual round05 initial state')


def save_context(cache, data):
    """Preserve prices; O.jsonable's report projection intentionally drops them."""
    cache.parent.mkdir(parents=True,exist_ok=True)
    tmp=cache.with_suffix(cache.suffix+'.tmp')
    tmp.write_text(json.dumps(data,default=O.jsonable,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(cache)


def fixed_metadata_inputs(baseline, destination, prepared_points=None):
    """Materialize only fixed input blobs; never follow a live round05 plan.

    Other worktrees may advance freely. Scientific evidence referenced by these
    documents still uses the existing pinned readers and source/input checks.
    """
    reader=O.Reader(baseline);destination.mkdir(parents=True,exist_ok=True)
    def document(path):
        return reader.document(ROUND05_SOURCE,path)
    def optional_document(path, default):
        if Path(path).is_absolute():raise ValueError('Expected repository-relative fixed metadata: '+path)
        found=subprocess.check_output(['git','ls-tree',ROUND05_SOURCE,'--',path],cwd=baseline)
        return document(path)if found else default
    def local_registry(path):
        rows=optional_document(path,{})
        local=destination/(hashlib.sha256(path.encode()).hexdigest()[:12]+'.json')
        local.write_text(json.dumps(rows))
        return str(local.resolve())
    problem=document('configs/rpe_method_optimizer/h4_df10_problem.json')
    spec=document('configs/rpe_method_optimizer/rounds/h4_pr_pauli_schedule_wide_round05.json')
    for methods in (problem.get('methods',{}),spec.get('method_overrides',{})):
        for name in spec['methods']:
            generation=(methods.get(name,{})or{}).get('generation',{})
            points=generation.pop('kappa_points_file',None)
            if points:
                # A later untracked B2 run is not additional replay input.
                generation['kappa_points']=list(generation.get('kappa_points')or[])+optional_document(points,[])
    if prepared_points is not None:
        # Declare archived prepared-input availability separately from initial
        # numerical knowledge. Never seed the corresponding Q3 answers.
        generation=spec.setdefault('method_overrides',{}).setdefault(NAME,{}).setdefault('generation',{})
        generation['kappa_points']=list(generation.get('kappa_points')or[])+document(prepared_points)
    for routes in (problem.get('execution_routes',{}),spec.get('routes_override',{})):
        for rules in routes.values():
            for rule in rules:
                if rule.get('configurations_file'):rule['configurations_file']=local_registry(rule['configurations_file'])
    for prices in (problem.get('price_rules',{}),spec.get('price_rules_overrides',{})):
        if prices.get('registry_file'):prices['registry_file']=local_registry(prices['registry_file'])
    problem_path=destination/'PROBLEM.json';round_path=destination/'ROUND.json'
    problem_path.write_text(json.dumps(problem));round_path.write_text(json.dumps(spec))
    state=document('docs/research/rpe_method_optimizer_h4_20261116/kappa_abs_round04/run01/KAPPA_STATE.json')
    initial=document('docs/research/rpe_method_optimizer_h4_20261116/kappa_wide_round05/first_decision/KAPPA_STATE.json')
    constants=dict(state['family_constants']);constants.update({k:v for k,v in initial['family_constants'].items()if k not in constants})
    return problem_path,round_path,constants


def context_inputs(baseline, cache, preparation):
    if cache is not None and cache.exists():
        data=json.loads(cache.read_text())
        if data['round05_source']!=ROUND05_SOURCE:raise ValueError('Different declared metadata source')
        candidates=data['candidates']
        if any('prices' not in c or len(c['prices'])!=26 for c in candidates):
            raise ValueError('Metadata cache lacks complete price vectors; rebuild this technical cache')
        for c in candidates:c['prices']=[Fraction(v)for v in c['prices']]
        seeds={}
        for row in data['initial_records']:
            r=row['record']
            for key in ('MSE_lower','MSE_upper','cost'):
                if r.get(key)is not None:r[key]=O.frac(r[key])
            seeds[(row['configuration_id'],tuple(row['vector']))]=r
        families={int(ld):W.Family(int(ld),rows)for ld,rows in data['family_constants'].items()}
        generator=O.pinned_function(O.Reader(baseline),data['generator'])
        print('CONTEXT_CACHE_REUSED',cache,flush=True)
        return data['spec'],data['search'],candidates,seeds,families,Fraction(data['tau']),generator
    started=time.perf_counter()
    print('CONTEXT_START metadata/initial C only',flush=True)
    problem_path,round_path,constants=fixed_metadata_inputs(baseline,preparation/'fixed_inputs',PREPARED_POINTS)
    oldT,oldC=T.ROOT,T.CLI.ROOT
    try:
        T.ROOT=baseline;T.CLI.ROOT=baseline
        spec,problem,ctx,routes=T.round_context(problem_path,round_path,preparation)
    finally:T.ROOT,T.CLI.ROOT=oldT,oldC
    cands=[c for c in ctx['candidates']if c['method']==NAME and c['ready']]
    seeds=O.initial_knowledge(ctx['saved'],cands,'all')
    families={int(ld):W.Family(int(ld),rows)for ld,rows in constants.items()}
    settings=dict(problem['search'],**spec['search_overrides'])
    if cache is not None:
        keys=('method','family','key','configuration_id','name','ready','caps','needs','labels','price_status','signal_source','kappa_value')
        reduced=[dict({k:c[k]for k in keys if k in c},prices=[str(v)for v in c['prices']])for c in cands]
        data=dict(round05_source=ROUND05_SOURCE,spec=spec,search=settings,candidates=reduced,
            prepared_points_metadata_source=dict(commit=ROUND05_SOURCE,path=PREPARED_POINTS,scope='Declared input metadata, no numerical outcomes'),
            initial_records=[dict(configuration_id=c,vector=list(v),record=r)for(c,v),r in seeds.items()],
            family_constants=constants,tau=str(ctx['tau']),generator=problem['generator'],metadata_seconds=time.perf_counter()-started,
            limitation='Only initial C numerical records; no4783 future outcomes in this cache. Static archived prepared-input metadata is declared available.')
        save_context(cache,data)
    print('CONTEXT_END seconds',round(time.perf_counter()-started,3),flush=True)
    return spec,settings,cands,seeds,families,ctx['tau'],ctx['generate']


def equal_elapsed(comparison, times=(0,60,120,300,600,1200,2400,3600,7200,9000,12000,14400)):
    times=sorted(set(times)|{comparison['paired_observable_horizon']})
    rows=[]
    for t in times:
        row=dict(seconds=t,paired_observable=t<=comparison['paired_observable_horizon'],policies={})
        for policy,r in comparison['policies'].items():
            if r['censor']is not None and t>r['censor']['seconds']:
                row['policies'][policy]=dict(status='UNKNOWN_AFTER_CENSOR',best_all=None)
            else:
                event=next(x for x in reversed(r['curve'])if x['seconds']<=t)
                row['policies'][policy]=dict(status='OBSERVED_REPLAY_PREFIX',best_all=event['best_all'],launched=event['launched'])
        rows.append(row)
    return rows


def compact_output(output):
    """Keep causal choice/baseline reasons without repeating every score row.

    Exact computations are unchanged. Very large diagnostic MSE rationals use
    the existing outward-30-digit display; raw fixed outcomes remain intact.
    """
    output=deepcopy(output);seen=set()
    def decision(d):
        if id(d) in seen:return
        seen.add(id(d))
        selection=d['selection'];wanted={selection['baseline_id'],selection['chosen_id']}
        selection['scored_candidates']=len(selection['ranks'])
        selection['ranks']=[r for r in selection['ranks']if r['id']in wanted]
        selection['omitted_rank_rows']='Reproducible from candidate pool and visible observations; chosen and A-baseline rows retained.'
    for comparison in output['results'].values():
        comparison['equal_elapsed']=equal_elapsed(comparison)
        for result in comparison['policies'].values():
            for row in result['decisions']:decision(row)
        if comparison['first_divergence']:
            for side in('A','B'):decision(comparison['first_divergence'][side])
    for row in output.get('budget_scan',{}).values():
        for comparison in row.values():
            comparison['equal_elapsed']=equal_elapsed(comparison)
            for result in comparison['policies'].values():
                for d in result['decisions']:decision(d)
            if comparison['first_divergence']:
                for side in('A','B'):decision(comparison['first_divergence'][side])
    return T.CLI.slim(output)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--baseline-worktree',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--budget',type=int,default=400)
    ap.add_argument('--context-cache',type=Path)
    ap.add_argument('--scan-budgets',nargs='*',type=int,default=[4,8,16,32,64])
    a=ap.parse_args()
    if a.budget<0:ap.error('--budget must be nonnegative')
    a.output.mkdir(parents=True,exist_ok=False);baseline=a.baseline_worktree.resolve();started=time.perf_counter()
    cache_was_input=a.context_cache is not None and a.context_cache.exists()
    spec,settings,cands,seeds,families,tau,generate=context_inputs(baseline,a.context_cache,a.output/'preparation')
    sched={c['configuration_id']:(c['key']['L_D'],families[c['key']['L_D']].R(Fraction(str(c['kappa_value']))),Fraction(str(c['kappa_value'])))for c in cands if c.get('kappa_value')is not None and c['key']['L_D']in families}
    roots=[(baseline/p).resolve()for p in spec['executor']['prior_run_dirs']]
    index,missing=make_index(baseline/'docs/research/rpe_method_optimizer_h4_20261116/kappa_wide_round05/ledger/LEDGER.csv',roots)
    print('INDEX_END',len(index),'initial numeric records',len(seeds),flush=True)
    settings['small_and_deferred']='STOP'
    results={}
    for name,lds,wide,slots,lookahead in [('MAIN_ROUND05_PROFILE',None,settings['schedule_wide'],32,16),
             ('MAIN_NO_SPECULATION',None,settings['schedule_wide'],1,1),
             ('LD5_KAPPA_LE64_NO_SPECULATION',{5},dict(settings['schedule_wide'],max_kappa='64'),1,1)]:
        print('COMPARE_START',name,flush=True)
        candidates=[c for c in cands if lds is None or c['key']['L_D']in lds]
        allowed={c['configuration_id']for c in candidates}
        sub_seeds={k:v for k,v in seeds.items()if k[0]in allowed}
        result=compare(candidates,sub_seeds,{ld:f for ld,f in families.items()if lds is None or ld in lds},
            {c:v for c,v in sched.items()if c in allowed},index,settings,tau,generate,slots=slots,lookahead=lookahead,budget=a.budget,wide=wide)
        results[name]=result
        for policy,r in result['policies'].items():print(name,policy,r['stop'],'answers',r['completed'],'seconds',r['seconds'],flush=True)
    scans={}
    candidates=[c for c in cands if c['key']['L_D']==5];allowed={c['configuration_id']for c in candidates}
    for cap in a.scan_budgets:
        if cap<0:raise ValueError('Nonnegative replay budget required')
        print('BUDGET_SCAN_START',cap,flush=True)
        scans[str(cap)]=dict(LD5_KAPPA_LE64_NO_SPECULATION=compare(candidates,{k:v for k,v in seeds.items()if k[0]in allowed},
            {ld:f for ld,f in families.items()if ld==5},{c:v for c,v in sched.items()if c in allowed},
            index,settings,tau,generate,slots=1,lookahead=1,budget=cap,wide=dict(settings['schedule_wide'],max_kappa='64')))
    output=dict(schema='RPE_JOINT_CANDIDATE_ORDERING_REPLAY_V1',base_commit='e839b0d3f5469c21f5944aec5d978f91d193efc8',
        source_round05='d8e806265c3ded8e1184181c37c2d7ec71f08ade',archive_index_size=len(index),missing_archive_locators=missing,
        metadata_input=dict(kind='DECLARED_CONTEXT_CACHE'if cache_was_input else'FIXED_GIT_INPUTS_AND_PINNED_PREPARATION',
            context_cache_sha256=None if a.context_cache is None else hashlib.sha256(a.context_cache.read_bytes()).hexdigest(),
            note='Cache bytes are the replay input; its header states origin, not independent verification of its contents.'),
        initial_numeric_records=len(seeds),initial_record_ids=[qid(dict(configuration_id=c,vector=v))for c,v in seeds],
        static_prepared_inputs=dict(source=dict(commit=ROUND05_SOURCE,path=PREPARED_POINTS),ready_candidates=len(cands),
            qualification='Declared archived configuration/signal/price cache; outcomes are not initially known'),
        results=results,budget_scan=scans,replay_compute_seconds_before_serialization=time.perf_counter()-started,new_real_scientific_calls=0,
        limits=['Every first unsaved chosen action, including speculative launch, stops that branch.',
                'Unknown preparation/service times are not zero or an all-archive mean.',
                'Only the horizon before the earlier branch censor supports a paired comparison.',
                'Historical raw numerical results and original STOPs are read-only.'])
    (a.output/'COMPARISON.json').write_text(json.dumps(O.jsonable(compact_output(output)),ensure_ascii=False,indent=2)+'\n')
    print('DONE seconds',round(time.perf_counter()-started,3),flush=True)


if __name__=='__main__':main()
