"""Read completed one-shot outputs; arithmetic and plotting only, no science calls."""
from fractions import Fraction as F
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/resource_applicability/pr_pf_s2_recomputed/2026-10-22'
OUT=ROOT/'docs/progress/track_a/figures/pr_pf_s2_recomputed_20261022'
OLD=ROOT/'docs/progress/track_a/figures/pr_pf_s2_theory_compiled_main_20261022'
EPS=['0.05','0.01','0.005','0.001','0.0005']
SYSTEMS=['H2','H4','H6','H8']


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,sort_keys=True,allow_nan=False)+'\n')
def unit(system,cid,stage):return BASE/'units'/system/cid/stage


def completed(system,cid,stage):
    p=unit(system,cid,stage)
    if not (p/'STOP.json').exists() or read(p/'STOP.json')['status']!='COMPLETE_MANDATORY_STOP_UNIT':return None
    result=p/'result.json'
    if not result.exists():return None
    stop=read(p/'STOP.json')
    assert sha(result)==stop['result_files']['result.json']
    return read(result)


def weighted(units,shots,bound):
    return sum((F(units[a][bound])*shots[a] for a in ['cosine','sine']),F())


def physical_observations(system):
    p=unit(system,'SYSTEM','native_physics_validation')
    if not (p/'STOP.json').exists() or not (p/'result.json').exists():return None
    result=read(p/'result.json');stop=read(p/'STOP.json')
    checks=[x for row in result['blocks'] for x in row['checks']]
    actual=[x for x in checks if x['kind'] in ['one','squared_G']]
    return dict(raw_unit_status=stop['status'],raw_result_status=result['status'],
        full_PASS=stop['status']=='COMPLETE_MANDATORY_STOP_UNIT' and result['status']=='PASS',
        hamiltonian_primitive_checks_PASS=bool(actual) and all(x['PASS'] for x in actual),
        failed_checks=[dict(block=row['block'],**x) for row in result['blocks'] for x in row['checks'] if not x['PASS']],
        maximum_error=result['maximum_error'],result_sha256=sha(p/'result.json'),STOP_sha256=sha(p/'STOP.json'),
        interpretation='Failed diagnostic remains failed. Costs are conditional if only auxiliary unweighted-G checks failed; no threshold change or scientific acceptance.')


def summary(prices,shots):
    from scipy.stats import t
    values=[shots['cosine']*x['cosine']+shots['sine']*x['sine'] for x in prices]
    n=len(values);mean=F(sum(values),n)
    var=sum((F(x)-mean)**2 for x in values)/(n-1) if n>1 else F()
    se=math.sqrt(float(var)/n);half=float(t.ppf(.975,n-1))*se if n>1 else 0.
    return dict(mean=float(mean),mean_exact=str(mean),SE=se,half95=half,n=n,
                relative_halfwidth=half/float(mean),target_met=half<=.01*float(mean),
                cosine_mean=str(F(sum(x['cosine'] for x in prices),n)),
                sine_mean=str(F(sum(x['sine'] for x in prices),n)))


def recover_h2(old):
    src=ROOT/'docs/progress/track_a/figures/pr_pf_theory_compiled_ratio_20261022/sources'
    A,D=read(src/'h2_a.json'),read(src/'h2_d.json')
    identity=read(unit('H2','SYSTEM','input_preparation')/'identity.json')
    physics=completed('H2','SYSTEM','native_physics_validation')
    assert physics is not None and physics['status']=='PASS'
    assert identity['original_hamiltonian_equal'] and identity['original_state_equal']
    old_rte=ROOT/'artifacts/resource_applicability/track_a_current_s2_theory_compiled/2026-10-22/run01/fixed_source/rte.py'
    new_rte=BASE/'fixed_provider/src/trotterlib/rte.py'
    assert sha(old_rte)==sha(new_rte)
    results={};audits=[]
    for original in [r for r in old if r['system']=='H2']:
        eps=original['epsilon'];roles={}
        for role,source in [('PR',A),('PF',D)]:
            cid=original[role+'_candidate_id'];theory=completed('H2',cid,'theory_counter')
            assert theory is not None
            candidates=source['measurements']+source.get('reused_FP_measurements',[])
            values=[x for x in candidates if x['cell_id']==cid]
            assert len(values)==(8 if role=='PR' else 1)
            expected=identity['preparation_hashes'][str(original[role+'_L_D'])]
            assert all(x['preparation_hash']==expected==theory['preparation_hash'] for x in values)
            assert source['binding']['hamiltonian']==identity['hamiltonian_hash']
            assert source['binding']['state']==identity['state_hash']
            saved_signal=next(x for x in source['signals'] if x['cell']['id']==cid)
            new_signal=completed('H2',cid,'signal')
            assert new_signal is not None and saved_signal['log_B']==new_signal['log_B']
            signal_gap=max(abs(new_signal['corrected'][axis]-saved_signal['signals']['corrected'][axis]) for axis in ['real','imag'])
            assert signal_gap<1e-12
            for x in values:
                assert all(a['compiler_hash']==source['binding']['compiler'] for a in x['axes'].values())
                assert x['axes']['cosine']['event_digest']==x['axes']['sine']['event_digest']==x['event_digest']
                assert x['axes']['cosine']['seed']==x['axes']['sine']['seed']==x['seed']
            shots=original[role+'_shots'];prices=[dict(cosine=x['axes']['cosine']['RZ'],sine=x['axes']['sine']['RZ']) for x in values]
            stat=summary(prices,shots)
            assert F(stat['mean_exact'])==F(original['G_'+role+'_compiled'])
            roles[role]=dict(theory=theory['units'],shots=shots,compiled=stat,
                law=theory.get('law'),source='EXACT_RECOVERED_ORIGINAL_H2',sample_kind='SAVED_EXPLORATION_N8' if role=='PR' else 'SAVED_DIRECT_DETERMINISTIC',
                preparation_hash=expected,old_compiled_reused=True)
            roles[role]['native_physics_validated']=True
            roles[role]['hamiltonian_primitive_checks_PASS']=True
            roles[role]['selection_effect_corrected']=False
            audits.append(dict(epsilon=eps,role=role,candidate=cid,hamiltonian_hash=identity['hamiltonian_hash'],
                state_hash=identity['state_hash'],preparation_hash=expected,matching_measurements=len(values),
                exact_saved_G_match=True,same_rte_source_bytes=True,same_saved_axis_shots=True,
                saved_normalization_log_B_bit_match=True,corrected_signal_max_gap=signal_gap,
                law_recovery='Regenerated from fixed original finite-RTE source and exact preparation; saved normalization bit matches',
                no_new_counter_sampling_or_compile=True,theory_reused_from_successful_unit=str(unit('H2',cid,'theory_counter').relative_to(ROOT)),
                compiler_hash=source['binding']['compiler'],original_result_SOURCE=source['SOURCE']))
        results[eps]=roles
    write(OUT/'original_H2_identity_audit.json',dict(status='PASS_EXACT_INPUT_PREPARATION_AND_LAW_BINDING',
        audit=audits,files={str(src.relative_to(ROOT)/name):sha(src/name) for name in ['h2_a.json','h2_d.json']},
        original_MC_remains_n8=True,new_H2_failed_units_unchanged=True,failed_units_retried=False,
        statement='Recovered original-cohort accounting; not successful new H2 MC or a precision-certified population mean.'))
    return results


def new_role(original,role):
    system=original['system'];cid=original[role+'_candidate_id'];eps=original['epsilon']
    th=completed(system,cid,'theory_counter');shot=completed(system,cid,'shot_accounting_'+eps)
    measured=completed(system,cid,'MC_main' if role=='PR' else 'compiled_cost')
    result=dict(theory=None if th is None else th['units'],shots=None,compiled=None,
                source='RECOMPUTED_NATIVE_COHORT',old_compiled_reused=False)
    physics=physical_observations(system)
    result['physical_observations']=physics
    result['native_physics_validated']=physics is not None and physics['full_PASS']
    result['hamiltonian_primitive_checks_PASS']=physics is not None and physics['hamiltonian_primitive_checks_PASS']
    if shot is not None:
        result['eligibility']=shot;result['shots']=None if not shot['usable'] else dict(cosine=shot['N_cos'],sine=shot['N_sin'])
    if measured is not None and result['shots'] is not None:
        result['compiled']=summary(measured['values'],result['shots'])
        if role=='PR':
            result['rare_event_diagnostics']=measured['rare_event_diagnostics']
            law=measured['rare_event_diagnostics']['law'];n=measured['n']
            # The raw adapter's k=2 absence field uses any positive order.
            # Preserve raw bytes and report the literal per-order quantity here.
            for k in [2,4]:
                p=law['order_probabilities'].get(str(k),0.)
                probability=-math.expm1(law['events_per_trajectory']*math.log1p(-p))
                result['rare_event_diagnostics']['strata'][str(k)].update(
                    probability_any_this_order=probability,
                    probability_absent_in_main=(1-probability)**n,
                    expected_trajectories_with_this_order=n*probability)
            result['rare_event_diagnostics']['per_order_absence_recomputed_from_saved_law']=True
            result['main_statistics_match']=abs(result['compiled']['mean']-measured['statistics'][eps]['mean'])<1e-6
            assert result['main_statistics_match']
    if th is not None:result.update(law=th.get('law'),preparation_hash=th['preparation_hash'])
    return result


def make_row(original,roles):
    row={k:original[k] for k in ['system','epsilon','PR_candidate_id','PF_candidate_id','PF_method','DF_rank','PR_L_D','PR_q','PR_r','PR_K','PF_L_D','PF_q']}
    row.update(PR=roles['PR'],PF=roles['PF'],cohort=roles['PR']['source'],status='MISSING_COMPARABLE_COSTS')
    for role in ['PR','PF']:
        v=roles[role]
        if v.get('shots') is not None:
            row[role+'_N_cos']=v['shots']['cosine'];row[role+'_N_sin']=v['shots']['sine']
        if v.get('theory') is not None and v.get('shots') is not None:
            row['G_'+role+'_theory_lower']=str(weighted(v['theory'],v['shots'],'lower'))
            row['G_'+role+'_theory_upper']=str(weighted(v['theory'],v['shots'],'upper'))
            for axis in ['cosine','sine']:
                for bound in ['lower','upper']:row[role+'_C_'+axis+'_theory_'+bound]=v['theory'][axis][bound]
        if v.get('compiled') is not None:
            row['G_'+role+'_compiled']=v['compiled']['mean_exact']
            row[role+'_n']=v['compiled']['n'];row[role+'_MC_SE']=v['compiled']['SE']
            row[role+'_MC_relative_halfwidth']=v['compiled']['relative_halfwidth']
            row[role+'_MC_target_met']=v['compiled']['target_met'] if role=='PR' else None
            for axis in ['cosine','sine']:row[role+'_C_'+axis+'_compiled']=v['compiled'][axis+'_mean']
    needed=['G_'+role+'_'+kind for role in ['PR','PF'] for kind in ['compiled','theory_lower','theory_upper']]
    if not all(k in row for k in needed):return row
    full_physics=all(roles[role].get('native_physics_validated') for role in ['PR','PF'])
    if not all(roles[role].get('hamiltonian_primitive_checks_PASS') for role in ['PR','PF']):
        row['status']='MISSING_OR_FAILED_HAMILTONIAN_PRIMITIVE_VALIDATION';return row
    pr,pf=F(row['G_PR_compiled']),F(row['G_PF_compiled'])
    lo=F(row['G_PR_theory_lower'])/F(row['G_PF_theory_upper'])
    hi=F(row['G_PR_theory_upper'])/F(row['G_PF_theory_lower'])
    row.update(status='COMPLETE_PAIRED_WITHIN_COHORT' if full_physics else 'CONDITIONAL_PAIRED_AUXILIARY_PHYSICS_FAILURE',
        executed_technical_checks_complete=full_physics,rho_theory_lower=float(lo),rho_theory_upper=float(hi),
        rho_compiled=float(pr/pf),rho_compiled_exact=str(pr/pf),
        rho_compiled_half95=roles['PR']['compiled']['half95']/float(pf))
    for role,value in [('PR',pr),('PF',pf)]:
        tl,tu=map(F,[row['G_'+role+'_theory_lower'],row['G_'+role+'_theory_upper']])
        row[role+'_reduction_lower']=float(100*(1-value/tl))
        row[role+'_reduction_upper']=float(100*(1-value/tu))
        row[role+'_reduction_half95']=100*roles[role]['compiled']['half95']/float((tl+tu)/2)
    return row


def plot(rows,directory,cohort_filter=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','pdf.fonttype':42})
    directory.mkdir(parents=True,exist_ok=True);names=[]
    for eps in EPS:
        subset=[next(r for r in rows if r['system']==s and r['epsilon']==eps) for s in SYSTEMS]
        valid=[r['status'] in ['COMPLETE_PAIRED_WITHIN_COHORT','CONDITIONAL_PAIRED_AUXILIARY_PHYSICS_FAILURE'] and (cohort_filter is None or r['cohort']==cohort_filter) for r in subset]
        x=np.arange(4)
        for kind in ['A','B']:
            fig,ax=plt.subplots(figsize=(8.4,5.1),layout='constrained')
            if cohort_filter is None:
                ax.axvspan(-.45,.45,color='#fff1d3',zorder=-5)
                ax.axvline(.5,color='#b4b4b4',lw=.8,ls=':')
                subtitle='H2: recovered original n=8  |  H4/H6/H8: independent recomputation'
            elif cohort_filter=='EXACT_RECOVERED_ORIGINAL_H2':subtitle='Recovered original H2 cohort; saved exploration n=8'
            else:subtitle='Recomputed cohort only; no old compiled samples pooled'
            if kind=='A':
                theory=np.array([(r.get('rho_theory_lower',np.nan)+r.get('rho_theory_upper',np.nan))/2 if v else np.nan for r,v in zip(subset,valid)])
                comp=np.array([r.get('rho_compiled',np.nan) if v else np.nan for r,v in zip(subset,valid)])
                ce=np.array([r.get('rho_compiled_half95',0) if v else np.nan for r,v in zip(subset,valid)])
                te=np.array([(r.get('rho_theory_upper',0)-r.get('rho_theory_lower',0))/2 if v else np.nan for r,v in zip(subset,valid)])
                ax.errorbar(x-.055,theory,yerr=te,color='#222222',marker='s',ls='none',capsize=4,label='Theory (native law expectation)')
                ax.errorbar(x+.055,comp,yerr=ce,color='#0072b2',marker='o',ls='none',capsize=4,label='Compiled (nominal 95% MC interval)')
                ax.axhline(1,color='#555555',ls='--',lw=1)
                ax.set_ylabel('PR / PF total RZ cost')
                ymax=max([1.1]+[float(v) for a in [theory+te,comp+ce] for v in a if math.isfinite(v)])
                ax.set_ylim(0,ymax*1.19)
                ax.text(.99,.03,'Below 1: PR cheaper',transform=ax.transAxes,ha='right',fontsize=10,color='#444444')
            else:
                tops=[];bottoms=[]
                for offset,role,color,label in [(-.18,'PR','#0072b2','PR S2'),(.18,'PF','#d55e00','Fixed PF S2')]:
                    vals=[(r.get(role+'_reduction_lower',np.nan)+r.get(role+'_reduction_upper',np.nan))/2 if v else np.nan for r,v in zip(subset,valid)]
                    errors=[r.get(role+'_reduction_half95',0) if v else np.nan for r,v in zip(subset,valid)]
                    ax.bar(x+offset,vals,width=.34,color=color,label=label,yerr=errors,capsize=4)
                    tops += [v+e for v,e in zip(vals,errors) if math.isfinite(v+e)]
                    bottoms += [v-e for v,e in zip(vals,errors) if math.isfinite(v-e)]
                ax.axhline(0,color='#555555',lw=.8)
                ax.set_ylabel('100 (1 - compiled / theory) [%]')
                ax.set_ylim(min([0.]+bottoms)*1.12,max([1.]+tops)*1.32)
            for i,v in enumerate(valid):
                if not v:ax.text(i,.5,'Not available',rotation=90,ha='center',va='center',transform=ax.get_xaxis_transform(),color='#888888')
            ax.set_xticks(x,[r['system']+('*' if v and r['status']=='CONDITIONAL_PAIRED_AUXILIARY_PHYSICS_FAILURE' else '')+'\n'+r['PF_method'] for r,v in zip(subset,valid)]);ax.set_xlim(-.5,3.5)
            footnotes=[]
            if kind=='B':footnotes.append('PR difference includes finite-sample MC effects')
            if any(r['status']=='CONDITIONAL_PAIRED_AUXILIARY_PHYSICS_FAILURE' and v for r,v in zip(subset,valid)):
                footnotes.append('* Auxiliary G diagnostic failed; conditional cost comparison (see notes)')
            if footnotes:fig.supxlabel('\n'.join(footnotes),fontsize=9,color='#6c4a25')
            ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(loc='upper left',fontsize=9)
            ax.set_title(('A: PR/PF cost ratio' if kind=='A' else 'B: Theory-to-compiled cost decrease')+'    epsilon = '+eps,loc='left',fontweight='bold',pad=24)
            ax.text(0,1.025,subtitle,transform=ax.transAxes,fontsize=9,color='#555555')
            name=kind+'_eps_'+eps.replace('.','p')
            for suffix in ['png','svg','pdf']:fig.savefig(directory/(name+'.'+suffix),dpi=190)
            plt.close(fig);names.append(name)
    return names


def main():
    originals=read(OLD/'figure_data.json')['rows'];OUT.mkdir(parents=True,exist_ok=True)
    recovered=recover_h2(originals)
    rows=[];new_rows=[]
    for original in originals:
        new={role:new_role(original,role) for role in ['PR','PF']}
        new_rows.append(make_row(original,new))
        roles=recovered[original['epsilon']] if original['system']=='H2' else new
        rows.append(make_row(original,roles))
    metadata=dict(schema='COHORT_SEPARATED_FIXED_S2_COMPARISON_V1',rows=rows,new_cohort_rows=new_rows,
        complete_pairs=sum(r['status']=='COMPLETE_PAIRED_WITHIN_COHORT' for r in rows),
        numerical_pairs=sum('rho_compiled' in r for r in rows),
        conditional_pairs=sum(r['status']=='CONDITIONAL_PAIRED_AUXILIARY_PHYSICS_FAILURE' for r in rows),
        original_H2_pairs=sum(r['system']=='H2' and r['status']=='COMPLETE_PAIRED_WITHIN_COHORT' for r in rows),
        new_cohort_pairs=sum('rho_compiled' in r for r in new_rows),
        new_cohort_fully_validated_pairs=sum(r['status']=='COMPLETE_PAIRED_WITHIN_COHORT' for r in new_rows),
        scope='Fixed-time signal T=.8, two Hadamard axes, P=0; not full RPE all-round total cost or a final general PR advantage',
        fixed_candidates=True,theory_side_reoptimization=False,cohort_mix_within_pair=False,
        conditions=dict(model='linear hydrogen chain',adjacent_Angstrom=1.,basis='STO-3G',DF='tol-only1e-8 ranks3/10/19/28; signed generation order',
            T=.8,control='symmetric_directional',scalar='original_plus_tail_once',reference_rotated=False,
            compiler='Qiskit1.3.0 rz/sx/x/cx opt1 seed17 topology-none; exact gate-stream preserving scalar bookkeeping',
            allowance='usual signed bias; numerical factor1; empirical allowance, not rigorous certificate'),
        mc_note='New samples are independent pilot/main; original H2 retains historical n8 and wider nominal uncertainty. Rare unobserved mass is not certified by ordinary MC intervals.')
    write(OUT/'figure_data.json',metadata)
    write(OUT/'physical_validation_audit.json',{s:physical_observations(s) for s in SYSTEMS})
    fields=['system','epsilon','cohort','status','PR_candidate_id','PF_candidate_id','DF_rank','PR_L_D','PR_q','PR_r','PR_K','PF_L_D','PF_q',
        'G_PR_theory_lower','G_PR_theory_upper','G_PF_theory_lower','G_PF_theory_upper','G_PR_compiled','G_PF_compiled',
        'rho_theory_lower','rho_theory_upper','rho_compiled','rho_compiled_half95','PR_reduction_lower','PR_reduction_upper','PF_reduction_lower','PF_reduction_upper']
    fields += [role+'_'+name for role in ['PR','PF'] for name in ['N_cos','N_sin','n','MC_SE','MC_relative_halfwidth','MC_target_met']]
    fields += [role+'_C_'+axis+'_'+kind for role in ['PR','PF'] for axis in ['cosine','sine'] for kind in ['theory_lower','theory_upper','compiled']]
    with (OUT/'figure_data.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    plot(rows,OUT/'overview')
    plot(new_rows,OUT/'recomputed_only','RECOMPUTED_NATIVE_COHORT')
    plot(rows,OUT/'recovered_H2','EXACT_RECOVERED_ORIGINAL_H2')
    lines=['# 条件別coverage','', '|系|ε|cohort|比較|PR本評価n|名目95%相対半幅|','|---|---|---|---|---:|---:|']
    for r in rows:
        s=r['PR'].get('compiled') or {}
        lines.append('|'+ '|'.join(map(str,[r['system'],r['epsilon'],r['cohort'],r['status'],s.get('n','—'),s.get('relative_halfwidth','—')]))+'|')
    (OUT/'coverage_matrix.md').write_text('\n'.join(lines)+'\n')
    receipts=[]
    for p in sorted((BASE/'units').rglob('STOP.json')):
        d=read(p);receipts.append(dict(path=str(p.relative_to(ROOT)),status=d['status'],reason=d.get('reason')))
    write(OUT/'execution_unit_audit.json',dict(units=receipts,failed_units=sum(r['status']=='FAILED_NO_RETRY' for r in receipts),automatic_retry=False))
    write(OUT/'verification.json',dict(status='TECHNICAL_ACCOUNTING_CHECKED_PHYSICS_CAVEATS_RETAINED',complete_pairs=metadata['complete_pairs'],
        numerical_pairs=metadata['numerical_pairs'],conditional_pairs=metadata['conditional_pairs'],
        original_H2_identity_audit='original_H2_identity_audit.json',new_MC_pilot_pooled=False,missing_zero_filled=False,
        all_successful_result_STOP_hashes_checked=True,failed_diagnostic_bytes_hashed_separately=True,
        exact_rational_total_accounting=True,figures=30,formats=['PNG','SVG','PDF'],
        independent_scientific_review=False))
    write(OUT/'provenance.json',dict(SOURCE_initial=read(BASE/'SOURCE.json'),SOURCE_unexecuted_systems=read(BASE/'SOURCE_v2.json'),
        SOURCE_physics_validation=read(BASE/'SOURCE_physics_validation.json'),
        plotting_source_sha256=sha(__file__),original_fixed_result_commit='2d4ff27cb630939f87e08ab5881404c72ad2fa19',
        output_inventory={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='provenance.json'}))
    print(json.dumps({k:metadata[k] for k in ['complete_pairs','numerical_pairs','conditional_pairs','original_H2_pairs','new_cohort_pairs']},indent=2))


if __name__=='__main__':main()
