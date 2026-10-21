#!/usr/bin/env python3
"""Saved-value displays only: no science providers or scientific accounting.

Old A and outer32 have separate panels/denominators. B uses saved formula bests.
C uses only the saved identity-bound m5 subset, with a separate noncancellation
full-m5 diagnostic. Missing values never participate in minima or denominators.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from fractions import Fraction as F
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys

P = Path(__file__).resolve().parent
ROOT = P.parents[4]
CHECK_ONLY = False
EPS = ('0.05', '0.01', '0.005', '0.001', '0.0005')
SYSTEMS = ('H2', 'H4', 'H6', 'H8')
MODES = ('USUAL_SIGNED', 'NONCANCELLATION')
OLD_SERIES = ('PR S2 (A)', 'full PF S2', 'full PF S4', 'B0 S2', 'B0 S4', 'B0+I S2', 'B0+I S4')
PR_SERIES = ('PR S2', 'PR Yoshida4', 'PR m5_best', 'PR high best')
DET_SERIES = ('full PF m5', 'B0 m5', 'B0+I m5')
PRE = 'artifacts/resource_applicability/'
NEW = PRE + 'track_a_h246_saved_cost_results/2026-10-21/'
READY = PRE + 'track_a_h246_higher_pf_confirmation_readiness/2026-10-21/'
OLD = 'docs/progress/track_a/figures/all_methods_hchain_20261022/'
OLD_COMMIT = '5d53698534a1f325c8b7ed7be1bca6fa5192a77d'
EVAL_COMMIT = '8613b3ecda2030a677b1690fe7aa3da23dec4873'
READY_COMMIT = 'c7c1d8229508253390212588ae2b25e504eb4915'
SPECS = {
    **{f'old_{name}': (OLD_COMMIT, OLD + name + '.json') for name in
       ('figure_data', 'provenance', 'verification', 'STOP')},
    **{name: (EVAL_COMMIT, NEW + name + '.json') for name in
       ('evaluation', 'input_provenance', 'policy_adoption', 'end_audit', 'structure_checks')},
    **{name: (EVAL_COMMIT, NEW + name + '.csv') for name in
       ('candidate_precision_costs_1000', 'formula_comparison_30', 'cost_decomposition_60',
        'unit_statistics_36', 'observed_strata_144')},
    'evaluation_report': (EVAL_COMMIT, 'docs/research/track_a_h246_saved_cost_results_20261021.md'),
    'readiness_report': (READY_COMMIT, 'docs/research/track_a_h246_higher_pf_confirmation_readiness_20261021.md'),
    'fixed_candidates_8': (READY_COMMIT, READY + 'fixed_candidates_8.csv'),
    'fixed_shots_6': (READY_COMMIT, READY + 'fixed_shots_6.csv'),
    'stage_readiness': (READY_COMMIT, READY + 'stage_readiness.json'),
    'h2_acquisition_report': ('a05f4ed19e87e1eb65f7fc346385b5b4eae69845',
        'docs/research/track_a_h2_outer32_run01_results_20261021.md'),
    'h46_acquisition_report': ('605ca07a653241db49348fbd7f4e091f9f18ced7',
        'docs/research/track_a_h46_pr_exploration_run01_results_20261021.md'),
    'h2_acquisition_summary': ('a05f4ed19e87e1eb65f7fc346385b5b4eae69845',
        PRE + 'track_a_h2_outer32_run01/2026-10-21/summary.json'),
    'h46_acquisition_summary': ('605ca07a653241db49348fbd7f4e091f9f18ced7',
        PRE + 'track_a_h46_pr_exploration_run01/2026-10-21/run_summary.json'),
}


def sha(b): return hashlib.sha256(b).hexdigest()
def dump(name, obj):
    text=json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if CHECK_ONLY and name in ('figure_data.json','coverage_audit.json','group_minima_audit.json','compatibility_audit.json'):
        assert (P/name).read_text()==text, 'Saved display data differs from pinned sources: '+name
    else:(P/name).write_text(text)
def csv_rows(b): return list(csv.DictReader(io.StringIO(b.decode())))
def write_csv(name, rr):
    out = io.StringIO(newline=''); w = csv.DictWriter(out, fieldnames=list(rr[0]), lineterminator='\n')
    w.writeheader()
    for r in rr:
        w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k,v in r.items()})
    if CHECK_ONLY:assert (P/name).read_text()==out.getvalue(), 'CSV differs from saved JSON/source extraction: '+name
    else:(P/name).write_text(out.getvalue())


def freeze():
    (P/'sources').mkdir(exist_ok=True); inventory=[]
    for name,(commit,path) in SPECS.items():
        b=subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)
        local='sources/'+name+Path(path).suffix
        f=P/local
        if f.exists(): assert f.read_bytes()==b
        else: f.write_bytes(b)
        inventory.append(dict(label=name,commit=commit,source_path=path,local_path=local,
            sha256=sha(b),bytes=len(b),url=f'https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/{commit}/{path}'))
    dump('input_sources.json',inventory)


def load_sources():
    index=json.loads((P/'input_sources.json').read_text()); d={}
    for x in index:
        b=(P/x['local_path']).read_bytes(); assert sha(b)==x['sha256']
        d[x['label']]=json.loads(b) if x['source_path'].endswith('.json') else (
            csv_rows(b) if x['source_path'].endswith('.csv') else b.decode())
    return d,{x['label']:x for x in index}


def exact(row):
    v=row.get('total_RZ_exact')
    return str(F(v['numerator'],v['denominator'])) if v is not None else None


def key(domain,mode,e,s,series): return '|'.join((domain,mode,e,s,series))


def build_data():
    d,index=load_sources(); ev=d['evaluation']; old=d['old_figure_data']['rows']
    assert ev['status']=='SAVED_VALUE_EVALUATION_PASS_MANDATORY_STOP'
    assert ev['independent_confirmation'] is False and ev['cohort_pooling'] is False
    assert len(ev['groups'])==100 and len(ev['summary'])==30
    assert d['input_provenance']['stronger_existing_S2_S4_controls']=='PRESERVED_NOT_REPLACED_NOT_IDENTITY_JOINED'
    assert d['input_provenance']['H2_deterministic_comparison']=='NULL_NO_IDENTITY_BOUND_SAME_SERIES_SIGNAL_SET'
    records={}; minima=[]
    bindings={x['system']:x for x in d['input_provenance']['input_metadata']}
    old_names=('PR','full PF, S2','full PF, S4','B0, S2','B0, S4','B0+I, S2','B0+I, S4')
    def ref(label,locator):
        x=index[label];return dict(label=label,commit=x['commit'],path=x['source_path'],sha256=x['sha256'],locator=locator)
    def add(domain,mode,e,s,series,row=None,*,locator='',reason='',pool=None,record_copy=None):
        k=key(domain,mode,e,s,series)
        if record_copy is not None:
            r=dict(record_copy);r.update(record_id=k,series=series,selection='DISPLAY_MINIMUM_OF_TWO_SAVED_HIGH_ORDER_FAMILY_BESTS')
            records[k]=r;return r
        if row is not None:
            cost=exact(row);candidates=[x for x in (pool or []) if x['status']=='ELIGIBLE' and exact(x) is not None]
            ties=sorted(x['candidate_id'] for x in candidates if exact(x) is not None and F(exact(x))==F(cost)) if cost else []
            reason='' if cost else row['status']
        else:cost=None;ties=[]
        rr=dict(record_id=k,domain=domain,mode=mode,epsilon=e,system=s,series=series,
            cost_exact=cost,candidate_id=row['candidate_id'] if row else None,
            formula=row['formula'] if row else None,prefix=row['prefix'] if row else None,
            q=row['q'] if row else None,delta_exact=str(F(4,5)/row['q']) if row else None,
            K=4 if row and row['method']=='PR' else None,
            r_policy='ABS_TIME_PER_OUTER' if row and row['method']=='PR' else None,
            r_value=32 if row and row['method']=='PR' else None,
            n=row['n'] if row else None,saved_SE=row['total_RZ_SE'] if row else None,
            saved_status=row['status'] if row else 'MISSING',missing_reason=reason,
            exact_ties=ties,selection='COPY_SAVED_FAMILY_SELECTION' if domain=='OUTER32' else 'DISPLAY_METHOD_MINIMUM_OF_SAVED_ELIGIBLE_TOTALS',
            cohort='NEW_EXPLORATION_SEPARATE_FROM_PILOT_AND_CONFIRMATION' if domain=='OUTER32' else 'SAVED_SAME_NATIVE_M5_DETERMINISTIC',
            independent_confirmation=False,population_guarantee=False,
            model='linear H-chain, adjacent 1 Angstrom, STO-3G, tol-only 1e-8',
            DF_rank={'H2':3,'H4':10,'H6':19,'H8':28}[s],T='0.8',P=0,
            alpha_axis='0.025',allowance_factor='1',cost_scope='Saved analytic shots x full cosine/sine Hadamard wrapper compiled RZ; no preparation',
            input_binding=bindings[s]['binding'] if s in bindings else None,
            acquisition_SOURCE=bindings[s]['SOURCE'] if s in bindings else None,
            scientific_SOURCE=ev['SOURCE'],source=ref('evaluation',locator) if locator else None,
            saved_shot_status=row['shot']['status'] if row else None,
            saved_shots={'N_cos':row['shot']['N_cos'],'N_sin':row['shot']['N_sin']} if row else None)
        records[k]=rr;return rr
    for i,r in enumerate(old):
        series=OLD_SERIES[old_names.index(r['series'])];k=key('LEGACY_A','USUAL_SIGNED',r['epsilon'],r['system'],series)
        records[k]=dict(record_id=k,domain='LEGACY_A',mode='USUAL_SIGNED',epsilon=r['epsilon'],system=r['system'],
            series=series,cost_exact=r['cost_exact'],candidate_id=r['candidate_id'],formula=r['PF_order'],prefix=r['L_D'],
            q=r['q'],delta_exact=r['delta'],K=r['K'],r_policy='LEGACY_REGISTERED_GRID',r_value=r['r'],n=r['n'],
            saved_SE=r['saved_PR_SE'],saved_status='ELIGIBLE' if r['cost_exact'] else 'INELIGIBLE',
            missing_reason='' if r['cost_exact'] else 'SAVED_ALL_REGISTERED_CANDIDATES_INELIGIBLE',exact_ties=r['exact_ties'],
            selection='COPY_FROZEN_5D536985_DISPLAY_POINT',cohort=r['cohort'],independent_confirmation=False,
            population_guarantee=False,model=r['model'],DF_rank=r['DF_rank'],T=r['T'],P=r['P'],alpha_axis=r['alpha_axis'],
            allowance_factor=r['allowance_factor'],cost_scope=r['cost_scope'],input_binding={'hamiltonian':r['hamiltonian_hash'],
                'state':r['state_hash'],'snapshot':r['snapshot_hash'],'compiler':r['compiler_settings_hash']},
            acquisition_SOURCE=None,scientific_SOURCE=r['scientific_SOURCE'],source=ref('old_figure_data',f'/rows/{i}'),
            saved_shot_status=None,saved_shots=None)
    lookup={(g['candidate']['id'],r['mode'],str(r['epsilon'])):(gi,ri,r)
            for gi,g in enumerate(ev['groups']) for ri,r in enumerate(g['rows'])}
    assert len(lookup)==1000
    for mode in MODES:
        for e in EPS:
            for s in SYSTEMS:
                sm=next(((i,x) for i,x in enumerate(ev['summary']) if x['system']==s and x['mode']==mode and str(x['epsilon'])==e),None)
                for formula,series in zip(('2nd','4th','4th(m5_best)'),PR_SERIES[:3]):
                    pool=[r for g in ev['groups'] if g['candidate']['system']==s and g['candidate']['method']=='PR'
                          and g['candidate']['formula']==formula for r in g['rows'] if r['mode']==mode and str(r['epsilon'])==e]
                    selected=sm[1]['families'][formula]['best'] if sm else None
                    if selected:
                        gi,ri,saved=lookup[(selected['candidate_id'],mode,e)];assert selected==saved
                        assert selected['status']=='ELIGIBLE' and selected['independent_confirmation'] is False
                        assert all(F(exact(selected))<=F(exact(r)) for r in pool if exact(r))
                        loc=f'/summary/{sm[0]}/families/{formula}/best'
                    else:loc=''
                    add('OUTER32',mode,e,s,series,selected,locator=loc,reason='NO_SAVED_H8_OUTER32_DATA',pool=pool)
                    minima.append(dict(domain='OUTER32',mode=mode,epsilon=e,system=s,series=series,
                        eligible=sum(exact(r) is not None for r in pool),ineligible=sum(r['status']=='INELIGIBLE' for r in pool),
                        missing_components=sum(r['status']=='MISSING_COMPONENTS' for r in pool),
                        registered=len(pool),selected_candidate=selected['candidate_id'] if selected else None,
                        selected_exact=exact(selected) if selected else None,rule='COPY_SAVED_FORMULA_BEST'))
                high=[records[key('OUTER32',mode,e,s,x)] for x in PR_SERIES[1:3]]
                high=[x for x in high if x['cost_exact']]
                if high:
                    best=min(high,key=lambda x:(F(x['cost_exact']),x['candidate_id']))
                    rr=add('OUTER32',mode,e,s,PR_SERIES[3],record_copy=best)
                    rr['exact_ties']=sorted(x['candidate_id'] for x in high if F(x['cost_exact'])==F(best['cost_exact']))
                    rr['high_order_minimum_excludes_S2']=True
                else:add('OUTER32',mode,e,s,PR_SERIES[3],reason='NO_SAVED_H8_OUTER32_DATA')
                for method,series in zip(('FULL','B0','B0I'),DET_SERIES):
                    pool=[r for g in ev['groups'] if g['candidate']['system']==s and g['candidate']['method']==method
                          for r in g['rows'] if r['mode']==mode and str(r['epsilon'])==e]
                    eligible=[r for r in pool if r['status']=='ELIGIBLE' and exact(r) is not None]
                    selected=min(eligible,key=lambda r:(F(exact(r)),r['candidate_id'])) if eligible else None
                    loc=''
                    if selected:
                        gi,ri,_=lookup[(selected['candidate_id'],mode,e)];loc=f'/groups/{gi}/rows/{ri}'
                    reason=('MISSING_COMPONENTS' if pool and all(r['status']=='MISSING_COMPONENTS' for r in pool)
                            else 'SAVED_BUT_INELIGIBLE' if pool else 'NO_IDENTITY_BOUND_DETERMINISTIC_SIGNAL_SET' if s=='H2'
                            else 'NO_SAVED_H8_OUTER32_DATA')
                    add('M5_SUBSET',mode,e,s,series,selected,locator=loc,reason=reason,pool=pool)
                    minima.append(dict(domain='M5_SUBSET',mode=mode,epsilon=e,system=s,series=series,
                        eligible=len(eligible),ineligible=sum(r['status']=='INELIGIBLE' for r in pool),
                        missing_components=sum(r['status']=='MISSING_COMPONENTS' for r in pool),registered=len(pool),
                        selected_candidate=selected['candidate_id'] if selected else None,
                        selected_exact=exact(selected) if selected else None,rule='DISPLAY_METHOD_MINIMUM_OF_SAVED_TOTALS'))
    # Record the apparent hash agreement without promoting it to a complete join.
    compatibility=[]
    for s in SYSTEMS[:3]:
        a=records[key('LEGACY_A','USUAL_SIGNED','0.0005',s,OLD_SERIES[0])]['input_binding'];b=bindings[s]['binding']
        equal={field:a[field]==b[field] for field in ('hamiltonian','state','snapshot','compiler')}
        compatibility.append(dict(system=s,fields_equal=equal,scientific_comparison_status='not comparable',
            reason='Pinned saved evaluation explicitly did not identity-join stronger legacy S2/S4 controls; matching four hashes does not certify full accounting/runtime policy',
            action='Separate panels and denominators; no high-order/legacy-PF ratio',
            same_native_deterministic_scope='M5_SUBSET_H4_H6_ONLY' if s!='H2' else 'NO_JOINED_DETERMINISTIC_SET'))
    dump('compatibility_audit.json',compatibility);dump('group_minima_audit.json',minima)
    return d,records,minima


def panels(records):
    figs=[];draw=[]
    def make(group,mode,e,kind,panels,title):
        stem=f'{group}_{mode.lower()}_{kind}_eps_{e.replace(".","p")}'
        spec=dict(stem=stem,group=group,mode=mode,epsilon=e,kind=kind,title=title,panels=[])
        for pi,opts in enumerate(panels):
            pp=dict(opts);pp['points']=[]
            for series,domain in opts['series']:
                for s in opts['systems']:
                    r=records[key(domain,mode,e,s,series)]
                    denom=None
                    if kind=='relative':
                        if opts['denominator']=='LEGACY_FULL':
                            choices=[records[key('LEGACY_A',mode,e,s,t)] for t in OLD_SERIES[1:3]]
                        elif opts['denominator']=='PR_S2':choices=[records[key('OUTER32',mode,e,s,PR_SERIES[0])]]
                        else:
                            den_series=DET_SERIES if opts['denominator']=='M5_SUBSET' else DET_SERIES[:1]
                            choices=[records[key('M5_SUBSET',mode,e,s,t)] for t in den_series]
                        choices=[x for x in choices if x['cost_exact']]
                        if choices:denom=min(choices,key=lambda x:F(x['cost_exact']))
                        assert not (domain!='LEGACY_A' and denom and denom['domain']=='LEGACY_A'), 'Forbidden unjoined ratio'
                    value=(str(F(r['cost_exact'])/F(denom['cost_exact'])) if kind=='relative' and r['cost_exact'] and denom
                           else r['cost_exact'] if kind=='absolute' else None)
                    point=dict(figure=stem,group=group,panel=pi,mode=mode,epsilon=e,system=s,series=series,
                        record_id=r['record_id'],candidate_id=r['candidate_id'],formula=r['formula'],prefix=r['prefix'],
                        q=r['q'],delta_exact=r['delta_exact'],n=r['n'],cost_exact=r['cost_exact'],display_exact=value,
                        display_value=float(F(value)) if value else None,
                        denominator_record=denom['record_id'] if denom else None,
                        denominator_cost_exact=denom['cost_exact'] if denom else None,
                        denominator_scope=opts.get('denominator') if kind=='relative' else None,
                        saved_status=r['saved_status'],missing_reason=r['missing_reason'],
                        evidence_level='SAVED_VALUE_EVALUATION_OF_NEW_EXPLORATION' if domain!='LEGACY_A' else 'LEGACY_A_SAVED_DISPLAY',
                        source=r['source'])
                    pp['points'].append(point);draw.append(point)
            spec['panels'].append(pp)
        figs.append(spec)
    for e in EPS:
        for kind in ('absolute','relative'):
            make('A','USUAL_SIGNED',e,kind,[
                dict(title='Legacy A',systems=SYSTEMS,series=[(s,'LEGACY_A') for s in OLD_SERIES],
                     denominator='LEGACY_FULL',ylabel='Cost / best full PF' if kind=='relative' else 'Total compiled RZ cost (including shots)'),
                dict(title='Outer32',systems=SYSTEMS,series=[(PR_SERIES[0],'OUTER32'),(PR_SERIES[3],'OUTER32')],
                     denominator='PR_S2',ylabel='Cost / PR S2 (outer32)' if kind=='relative' else '')],
                'Separate saved domains')
    for mode in MODES:
        for e in EPS:
            for kind in ('absolute','relative'):
                make('B',mode,e,kind,[dict(title='Outer32 PR',systems=SYSTEMS[:3],
                    series=[(s,'OUTER32') for s in PR_SERIES[:3]],denominator='PR_S2',
                    ylabel='Cost / PR S2 (outer32)' if kind=='relative' else 'Total compiled RZ cost (including shots)')],
                    'PR internal / '+('signed' if mode=='USUAL_SIGNED' else 'noncancellation'))
                make('C',mode,e,kind,[dict(title='Same-native m5 subset' if mode=='USUAL_SIGNED' else 'Full m5 only',
                    systems=SYSTEMS[1:3],series=[(PR_SERIES[0],'OUTER32'),(PR_SERIES[3],'OUTER32')]+[(s,'M5_SUBSET') for s in DET_SERIES],
                    denominator='M5_SUBSET' if mode=='USUAL_SIGNED' else 'FULL_M5_ONLY',
                    ylabel=('Cost / best saved m5 subset' if mode=='USUAL_SIGNED' else 'Cost / full PF m5') if kind=='relative'
                           else 'Total compiled RZ cost (including shots)')],
                    'Saved subset / '+('signed' if mode=='USUAL_SIGNED' else 'noncancellation'))
    assert len(figs)==50 and len(draw)==740
    return figs,draw


def coverage(records,minima):
    cov=[];wanted=OLD_SERIES[:1]+('PR high best',)+OLD_SERIES[1:]
    def status(r):
        if r['cost_exact']:return 'plotted'
        return 'saved-but-ineligible' if r['saved_status']=='INELIGIBLE' or r['missing_reason']=='SAVED_BUT_INELIGIBLE' else 'missing'
    def emit(scope,mode,e,s,method,r,*,state=None,reason=None,display=None):
        st=state or status(r);pub=(st=='plotted') if display is None else display
        cov.append(dict(scope=scope,mode=mode,system=s,epsilon=e,method=method,status=st,
            display_status='plotted' if pub else 'not plotted',saved_record_id=r['record_id'] if r else None,
            saved_status=r['saved_status'] if r else 'MISSING',
            reason=reason if reason is not None else r['missing_reason'],
            independently_confirmed=False,
            independent_confirmation_status='missing' if s=='H8' else 'not yet independently confirmed' if r and r['domain']=='OUTER32' else
                'deterministic; not applicable' if r and r['domain']=='M5_SUBSET' else 'legacy exploration value; separate legacy P not used',
            candidate_id=r['candidate_id'] if r else None,cost_exact=r['cost_exact'] if r else None,
            panel='outer32' if r and r['domain']=='OUTER32' else 'legacy' if scope=='A' else 'm5 subset'))
    for e in EPS:
        for s in SYSTEMS:
            for method in wanted:
                domain='OUTER32' if method=='PR high best' else 'LEGACY_A';r=records[key(domain,'USUAL_SIGNED',e,s,method)]
                if method=='PR high best' and r['cost_exact']:
                    emit('A','USUAL_SIGNED',e,s,method,r,state='not comparable',display=True,
                         reason='Shown in separate outer32 panel; legacy PF not identity-joined; no legacy-PF denominator')
                else:emit('A','USUAL_SIGNED',e,s,method,r)
    for mode in MODES:
        for e in EPS:
            for s in SYSTEMS:
                for series in PR_SERIES[:3]:emit('B',mode,e,s,series,records[key('OUTER32',mode,e,s,series)])
                for series in (PR_SERIES[0],PR_SERIES[3])+DET_SERIES:
                    domain='OUTER32' if series.startswith('PR') else 'M5_SUBSET';r=records[key(domain,mode,e,s,series)]
                    if s=='H2' and r['cost_exact']:
                        emit('C',mode,e,s,series,r,state='not comparable',display=False,
                             reason='No identity-bound H2 deterministic signal set in pinned evaluation')
                    else:emit('C',mode,e,s,series,r)
                for series in ('full PF S2','B0 S2','B0+I S2'):
                    legacy=records[key('LEGACY_A','USUAL_SIGNED',e,s,series)]
                    emit('C',mode,e,s,series,legacy,state='not comparable' if s!='H8' else 'missing',display=False,
                         reason='No same-native S2 deterministic subset; legacy controls not identity-joined')
    assert len(cov)==600
    dump('coverage_audit.json',dict(schema='SOURCE_SCOPED_COVERAGE_V1',rows=cov,
        status_axes_separate=True,independent_confirmation_state='Readiness only at c7c1d822; no completed higher-PR confirmation',
        group_minima=minima));write_csv('coverage_matrix.csv',cov)
    return cov


def style(series):
    styles={
        'PR S2 (A)':('#0072B2','o','-',True), 'PR S2':('#0072B2','o','-',True),
        'PR Yoshida4':('#CC79A7','v','--',False),'PR m5_best':('#6F42A6','P','-.',False),
        'PR high best':('#A23B90','*','-',True),
        'full PF S2':('#4A5568','s','-',True),'full PF S4':('#4A5568','s','--',False),
        'B0 S2':('#D55E00','^','-',True),'B0 S4':('#D55E00','^','--',False),
        'B0+I S2':('#009E73','D','-',True),'B0+I S4':('#009E73','D','--',False),
        'full PF m5':('#4A5568','s','--',False),'B0 m5':('#D55E00','^','--',False),'B0+I m5':('#009E73','D','--',False)}
    return styles[series]


def mpl_init():
    os.environ['MPLCONFIGDIR']=str(P/'.mpl-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.labelsize':14,
        'axes.titlesize':17,'legend.fontsize':11,'axes.spines.top':False,'axes.spines.right':False,
        'svg.fonttype':'none','svg.hashsalt':'higher-pr-display-20261022','pdf.fonttype':42,
        'savefig.facecolor':'white'})
    return plt


def save(fig,name,checked):
    fig.canvas.draw();rr=fig.canvas.get_renderer();bounds=fig.bbox;clipped=[]
    for item in checked:
        b=item.get_window_extent(rr)
        if item.get_visible() and not all(bounds.contains(*pt) for pt in b.get_points()):
            clipped.append(item.get_text() if hasattr(item,'get_text') else 'legend')
    assert not clipped,(name,clipped)
    outputs=[]
    for ext in ('png','svg','pdf'):
        kw={'dpi':190}
        if ext=='svg':kw['metadata']={'Date':None}
        if ext=='pdf':kw['metadata']={'CreationDate':None,'ModDate':None}
        f=P/(name+'.'+ext);fig.savefig(f,**kw);outputs.append(dict(file=f.name,sha256=sha(f.read_bytes()),bytes=f.stat().st_size))
    return dict(figure=name,status='PASS',canvas_clipping=[],outputs=outputs)


def render_one(spec):
    plt=mpl_init()
    from matplotlib.ticker import NullFormatter,FixedLocator,FuncFormatter
    from matplotlib.lines import Line2D
    multi=len(spec['panels'])==2
    fig,aa=plt.subplots(1,len(spec['panels']),figsize=(16,6.8) if multi else (10.8,6.5),squeeze=False)
    fig.subplots_adjust(left=.075 if multi else .13,right=.975,bottom=.28,top=.80,wspace=.27)
    fig.text(.075 if multi else .13,.94,spec['title'],fontsize=13,color='#536171')
    fig.text(.075 if multi else .13,.88,r'$\epsilon = '+spec['epsilon']+'$',fontsize=21)
    checked=list(fig.texts);all_values=[]
    for panel,ax in zip(spec['panels'],aa[0]):
        names=[x[0] for x in panel['series']];offsets={s:(i-(len(names)-1)/2)*.035 for i,s in enumerate(names)}
        for series,_ in panel['series']:
            rr=[next(x for x in panel['points'] if x['system']==s and x['series']==series) for s in panel['systems']]
            vals=[x['display_value'] if x['display_value'] is not None else math.nan for x in rr]
            c,m,ls,fill=style(series)
            ax.plot([i+offsets[series] for i in range(len(rr))],vals,color=c,marker=m,linestyle=ls,
                markerfacecolor=c if fill else 'white',markeredgewidth=1.5,markersize=10 if m=='*' else 7.5,
                linewidth=2.3 if series.startswith('PR') else 1.5,label=series,zorder=6 if series.startswith('PR') else 4)
        ax.set_yscale('log');ax.set_xticks(range(len(panel['systems'])),panel['systems'])
        ax.set_xlim(-.35,len(panel['systems'])-.65);ax.set_xlabel('H-chain',labelpad=11)
        ax.set_ylabel(panel['ylabel'],labelpad=9);ax.set_title(panel['title'],loc='left',pad=12)
        vals=[x['display_value'] for x in panel['points'] if x['display_value'] is not None];assert vals
        all_values.extend(vals)
        ax.set_ylim(min(vals)/1.4,max(vals)*1.4)
        if spec['kind']=='relative':
            ticks=[x for x in (.1,.2,.3,.5,1,2,3,5,10) if min(vals)/1.4<=x<=max(vals)*1.4]
            ax.yaxis.set_major_locator(FixedLocator(ticks));ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}'))
            ax.axhline(1,color='#9AA1AB',linewidth=.8,zorder=1)
        ax.yaxis.set_minor_formatter(NullFormatter());ax.grid(axis='y',color='#D9DEE5',linewidth=.7);ax.set_axisbelow(True)
        handles,labels=ax.get_legend_handles_labels()
        if len(handles)==7:
            handles.insert(1,Line2D([],[],linestyle='none'));labels.insert(1,'')
            ncols=4
        else:ncols=3 if len(handles)==5 else len(handles)
        legend=ax.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,-.25),ncol=ncols,
                         frameon=False,handlelength=2.3,columnspacing=1.2)
        checked.extend([ax.xaxis.label,ax.yaxis.label,ax.title,legend,*ax.get_xticklabels()])
    if multi and spec['kind']=='absolute':
        for ax in aa[0]:ax.set_ylim(min(all_values)/1.5,max(all_values)*1.5)
    result=save(fig,spec['stem'],checked);plt.close(fig);return result


def render_coverage(cov):
    plt=mpl_init();from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    import numpy as np
    methods=[OLD_SERIES[0],'PR high best',*OLD_SERIES[1:]]
    states=['plotted','saved-but-ineligible','missing','not comparable']
    colors=['#66B99E','#E7B96A','#E5E7EB','#B6A0D1'];fig,ax=plt.subplots(figsize=(13,9.8))
    grid=[[states.index(next(x for x in cov if x['scope']=='A' and x['system']==s and x['epsilon']==e and x['method']==m)['status'])
          for m in methods] for e in EPS for s in SYSTEMS]
    ax.imshow(np.array(grid),aspect='auto',cmap=ListedColormap(colors),vmin=-.5,vmax=3.5)
    ax.set_xticks(range(8),['PR S2 A','PR high','full S2','full S4','B0 S2','B0 S4','B0+I S2','B0+I S4'],rotation=25,ha='right')
    ax.set_yticks(range(20),[s+' / '+e for e in EPS for s in SYSTEMS],fontsize=11)
    for i,row in enumerate(grid):
        for j,x in enumerate(row):ax.text(j,i,['OK','I','M','NC'][x],ha='center',va='center',fontsize=11)
    ax.set_title('Coverage / source-aware comparison',loc='left',pad=15)
    leg=fig.legend(handles=[Patch(color=c,label=s) for c,s in zip(colors,states)],loc='lower center',ncol=4,frameon=False,fontsize=11)
    fig.subplots_adjust(left=.18,right=.98,top=.92,bottom=.16)
    r1=save(fig,'D_coverage_legacy_vs_higher',[ax.title,leg,*ax.get_xticklabels(),*ax.get_yticklabels()]);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10.8,6.2));grid=[[0 if s!='H8' else 1 for _ in EPS] for s in SYSTEMS]
    ax.imshow(np.array(grid),aspect='auto',cmap=ListedColormap(['#E7B96A','#E5E7EB']),vmin=-.5,vmax=1.5)
    ax.set_xticks(range(5),EPS);ax.set_yticks(range(4),SYSTEMS)
    for i,row in enumerate(grid):
        for j,v in enumerate(row):ax.text(j,i,'Pending' if v==0 else 'Missing',ha='center',va='center',fontsize=12)
    ax.set_xlabel(r'$\epsilon$',labelpad=12);ax.set_title('Outer32 PR / independent confirmation',loc='left',pad=15)
    leg=fig.legend(handles=[Patch(color='#E7B96A',label='not yet independently confirmed'),Patch(color='#E5E7EB',label='missing')],
                   loc='lower center',ncol=2,frameon=False,fontsize=11)
    fig.subplots_adjust(left=.12,right=.97,top=.86,bottom=.22)
    r2=save(fig,'D_higher_pr_confirmation_coverage',[ax.title,ax.xaxis.label,leg,*ax.get_xticklabels(),*ax.get_yticklabels()]);plt.close(fig)
    return [r1,r2]


def verify_sources_and_data(d,records,draw):
    def resolve(obj,loc):
        for part in loc.strip('/').split('/'):
            obj=obj[int(part)] if isinstance(obj,list) else obj[part]
        return obj
    for r in records.values():
        if not r['cost_exact']:continue
        source=resolve(d[r['source']['label']],r['source']['locator'])
        token=source['cost_exact'] if r['domain']=='LEGACY_A' else exact(source)
        assert token==r['cost_exact'];assert source.get('candidate_id',source.get('candidate_id'))==r['candidate_id']
    for x in draw:
        r=records[x['record_id']];assert x['cost_exact']==r['cost_exact']
        if x['denominator_record']:
            dr=records[x['denominator_record']];assert dr['system']==r['system'] and dr['epsilon']==r['epsilon'] and dr['mode']==r['mode']
            assert not (r['domain']!='LEGACY_A' and dr['domain']=='LEGACY_A')
            if x['display_exact']:assert F(x['display_exact'])==F(r['cost_exact'])/F(dr['cost_exact'])
    strict={s:next(x for x in d['evaluation']['summary'] if x['system']==s and x['mode']=='USUAL_SIGNED' and x['epsilon']==.0005)
            for s in SYSTEMS[:3]}
    for s,formula in [('H2','4th'),('H4','4th(m5_best)'),('H6','4th(m5_best)')]:
        r=records[key('OUTER32','USUAL_SIGNED','0.0005',s,'PR high best')];assert r['formula']==formula
        assert r['candidate_id']==strict[s]['families'][formula]['best']['candidate_id']
    for sm in d['evaluation']['summary']:
        s,e,mode=sm['system'],str(sm['epsilon']),sm['mode']
        baseline=records[key('OUTER32',mode,e,s,PR_SERIES[0])]
        for formula,series in zip(('4th','4th(m5_best)'),PR_SERIES[1:3]):
            saved=sm['higher_PF_vs_S2'][formula]['total_cost_ratio_exact']
            higher=records[key('OUTER32',mode,e,s,series)]
            assert F(higher['cost_exact'])/F(baseline['cost_exact'])==F(saved['numerator'],saved['denominator'])
        if mode=='USUAL_SIGNED' and s in ('H4','H6'):
            det=[records[key('M5_SUBSET',mode,e,s,x)] for x in DET_SERIES]
            best=min((r for r in det if r['cost_exact']),key=lambda r:F(r['cost_exact']))
            assert best['candidate_id']==sm['deterministic_m5_subset']['best']['candidate_id']
            assert best['cost_exact']==exact(sm['deterministic_m5_subset']['best'])
    assert d['stage_readiness']['science_started'] is False
    assert not d['stage_readiness']['valid_grants']
    assert all(not x['independent_confirmation'] for x in records.values())


def main():
    global CHECK_ONLY
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--freeze-sources',action='store_true');ap.add_argument('--check',action='store_true');args=ap.parse_args()
    CHECK_ONLY=args.check
    if args.freeze_sources:freeze()
    d,records,minima=build_data();figs,draw=panels(records);cov=coverage(records,minima)
    verify_sources_and_data(d,records,draw)
    dump('figure_data.json',dict(schema='HIGHER_PR_SOURCE_SEPARATED_DISPLAY_V1',primary_mode='USUAL_SIGNED',
        noncancellation_separate=True,records=records,figures=figs,display_points=draw,
        no_unjoined_high_order_legacy_PF_ratios=True));write_csv('figure_data.csv',draw)
    if not args.check:
        # Warm fonts and verify one reference display before parallel export.
        mpl_init();preflight=next(x for x in figs if x['group']=='B' and x['mode']=='USUAL_SIGNED' and x['epsilon']=='0.0005' and x['kind']=='absolute')
        results=[render_one(preflight)];pending=[x for x in figs if x is not preflight]
        workers=min(len(os.sched_getaffinity(0)),len(pending))
        print(json.dumps(dict(preflight='PASS',available_cpus=os.cpu_count(),workers=workers,pending_figures=len(pending))),flush=True)
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork')) as pool:
            futures=[pool.submit(render_one,x) for x in pending]
            for f in as_completed(futures):results.append(f.result())
        results.extend(render_coverage(cov));results.sort(key=lambda x:x['figure'])
        dump('rendering_checks.json',dict(status='PASS',figure_count=52,export_count=156,figures=results,
            workers=workers,preflight=preflight['stem'],shared_font_cache='local packet only',
            scientific_calls=0,no_artificial_resource_caps=True))
    rc=json.loads((P/'rendering_checks.json').read_text());assert len(rc['figures'])==52
    from PIL import Image
    import xml.etree.ElementTree as ET
    for f in rc['figures']:
        for x in f['outputs']:
            b=(P/x['file']).read_bytes();assert sha(b)==x['sha256']
            if x['file'].endswith('.png'):Image.open(P/x['file']).verify()
            elif x['file'].endswith('.svg'):ET.fromstring(b)
            else:assert b.startswith(b'%PDF-') and b.rstrip().endswith(b'%%EOF')
    dump('verification.json',dict(status='PASS',source_snapshots=len(SPECS),canonical_records=len(records),
        cost_figures=50,coverage_figures=2,exports=156,display_points=len(draw),coverage_rows=len(cov),
        selected_exact_tokens_equal_pinned_sources=True,all_display_ratios_use_comparable_same_scope_denominators=True,
        legacy_record_count=140,legacy_points_unchanged=True,higher_PR_has_no_H8_values=True,
        formula_bests_reused=True,method_minimum_is_saved_value_display_extraction_only=True,
        actual_new_scientific_calls=dict(sampling=0,compile=0,signal=0,fit=0,bootstrap=0,
            shot_recalculation=0,total_cost_accounting=0,scientific_reranking=0,reoptimization=0),
        independent_confirmation_used=False,independent_scientific_review=False))
    print(json.dumps(dict(status='PASS',figures=52,exports=156,records=len(records),display_points=len(draw),coverage_rows=len(cov))))


if __name__=='__main__':main()
