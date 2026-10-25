#!/usr/bin/env python3
"""H2-only saved accounting/comparison, figures and completed-run verification."""
import csv,gzip,hashlib,importlib.util,io,json,math
from pathlib import Path
from fractions import Fraction as Q
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('h2_final_driver',ROOT/'scripts/run_h2_pauli_rte_validation.py');R=importlib.util.module_from_spec(s);s.loader.exec_module(R)
OUT=R.OUT;read,save=R.read,R.save
def csvsave(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',newline='') as f:w=csv.DictWriter(f,keys,lineterminator='\n');w.writeheader();w.writerows(rows)
def ratio(a,b):return str(Q(a['G_exact'])/Q(b['G_exact'])) if a and b and R.finite(a) and R.finite(b) else None
def compare(h,r):
    if not r or not R.finite(h) or not R.finite(r):return 'UNRESOLVED'
    if R.near(h,r):return 'NEAR_TIE_OR_UNRESOLVED'
    return 'RTE_LOWER_EMPIRICAL_G' if Q(r['G_exact'])<Q(h['G_exact']) else 'HYBRID_LOWER_EMPIRICAL_G'
def flat(x):
    cfg=x['config'];st=x['price_statistics'];target=read(ROOT/x['target_path'])
    out=dict(molecule='H2',candidate_id=x['candidate_id'],phase=x['phase'],cohort=x['cohort'],epsilon_sig=x['epsilon_sig'],L_D=0,q=cfg['q'],r=cfg['r'],K=cfg['K'],s=x['s'],
        eligibility=x['eligibility'],B_exact=x['B_exact'],price_n=st['n'],G_exact=x['G_exact'],G_SE=x['G_SE'],G_nominal95_halfwidth=x.get('G_nominal95_halfwidth'),N_origin=x['N_origin'],
        source_file=x['target_path'],source_commit=read(OUT/'SOURCE.json')['commit'],actual_rho=None,hypothetical_rho='0',population_price_tail='UNGUARANTEED')
    for a,k in [('cosine','cos'),('sine','sin')]:
        out.update({f'N_{k}':x[f'N_{k}'],f'C_{k}_exact':st['mean_exact'][a],f'C_{k}_SE':st['SE'][a],f'bias_{k}_exact':target['axes'][a]['signed_bias_exact'],f'u_{k}_exact':target['axes'][a]['u_enc_exact'],f'rho_guard_{k}_exact':x['axes'][a]['rho_guard_exact']})
    return out
def verify(data):
    from qiskit import qpy
    source=R.check();rawcount=0;targets=0;decoded=0;qpys=0;orders={};cases=[]
    all_old=read(OUT/'prior_evidence_selection.json')
    for rel,meta in all_old['selected_members'].items():assert R.sha(OUT/'prior_evidence'/rel)==meta['sha256']
    plan=read(OUT/'study_plan.json');reference=read(OUT/'H2/inputs/reference.json')
    for c in plan['candidates']:
        cid=c['candidate_id'];d=R.case(cid)
        if not (d/'target.json').exists():continue
        cases.append(cid);t=read(d/'target.json');assert not t['state_renormalized'] and t['B_applied_count']==1 and Q(t['raw_norm_exact'])==Q(reference['raw_norm_exact'])
        if c['phase']=='A':
            original=OUT/'prior_evidence'/c['old_case']/'target.json';assert original.read_bytes()==(d/'target.json').read_bytes()
        else:
            assert (d/'target_call_started.json').exists();targets+=1
        spec=read(d/'spec.json')
        for cohort in ['exploration','confirmation']:
            files=sorted((d/'prices'/cohort).glob('*/result.json'))
            if not files:continue
            assert len(files)==8 if cohort=='exploration' else len(files) in [96,192,384]
            counts={k:0 for k in [0,2,4]}
            for i,p in enumerate(files):
                x=read(p);assert x['index']==i and x['seed']==R.H.seeds(spec,cohort,i)
                assert len(x['events'])==len(x['orders'])==x['event_count']==c['q']*c['r']
                assert all(k in [0,2,4] and k<=c['K'] for k in x['orders'])
                for k in counts:counts[k]+=x['orders'].count(k)
                for a in R.AXES:
                    for kind in ['input','compiled']:
                        b=(p.parent/(a+'_'+kind+'.qpy.gz')).read_bytes();meta=x['axes'][a][kind+'_qpy'];assert hashlib.sha256(b).hexdigest()==meta['gzip_sha256'];raw=gzip.decompress(b);assert hashlib.sha256(raw).hexdigest()==meta['raw_sha256'];qpys+=1
                        if kind=='compiled':
                            circuit=qpy.load(io.BytesIO(raw))[0];ops=circuit.count_ops();assert circuit.num_qubits==5
                            assert set(ops)<=set(('rz','sx','x','cx','measure','barrier'))
                            assert ops.get('rz',0)==x['axes'][a]['rz_count'] and ops.get('cx',0)==x['axes'][a]['cx_count'];decoded+=1
                rawcount+=1
            orders[cid+':'+cohort]=dict(n=len(files),event_order_counts=counts)
    for x in data:
        if x['N_total'] is not None:
            for a in R.AXES:
                shot=x['axes'][a]['rho_zero'];assert Q(shot['N'])*Q(shot['headroom_exact'])**2>=Q(x['axes'][a]['scale']['A_upper'])
        if x['G_exact'] is not None:
            st=x['price_statistics'];assert Q(x['G_exact'])==sum(Q(x['N_cos' if a=='cosine' else 'N_sin'])*Q(st['mean_exact'][a]) for a in R.AXES)
    native=[x for p in list((OUT/'stages').glob('**/native.json'))+list((OUT/'confirmations').glob('**/native.json')) for x in read(p)['results']]
    assert all(x['exit_code']==0 and x['reaped'] for x in native)
    use=read(OUT/'usage.json');assert rawcount==use['exploration_trajectories']+use['confirmation_trajectories'] and targets==use['new_targets']
    assert len(native)==rawcount+targets
    assert rawcount<=plan['limits']['total_trajectories'] and targets<=plan['limits']['new_targets']
    return dict(status='PASS',new_target_state_actions=targets,new_trajectory_count=rawcount,new_axis_compiles=2*rawcount,native_child_count=len(native),all_native_exit_zero_and_reaped=True,QPY_compressed_and_raw_SHA_checked=qpys,all_compiled_QPY_structure_decoded=decoded,Phase_A_saved_targets_and_N_reused=True,original_H2_H4_H6_evidence_preserved=True,restored_old_members_checked=len(all_old['selected_members']),observed_orders=orders,new_Hamiltonian_state_reference=0,new_H4_H6_or_Hybrid_science=0,actual_rho_guaranteed=False,population_tail_guaranteed=False,formal_certificate_accepted=False,checking_scope='Primary Codex technical checks; not an independent scientific review')
def plots(trends,hybrid):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none'})
    f=OUT/'figures';f.mkdir(exist_ok=False)
    fig,axs=plt.subplots(1,5,figsize=(17,4),layout='constrained')
    pairs=sorted({(int(r['q']),int(r['K'])) for r in trends})
    colors={q:plt.get_cmap('tab10')(j) for j,q in enumerate([1,2,4,8])}
    for ax,e in zip(axs,R.EPS):
        for q,K in pairs:
            for cohort,marker in [('exploration','x'),('confirmation','o')]:
                rr=sorted([r for r in trends if r['epsilon_sig']==e and int(r['q'])==q and int(r['K'])==K and r['cohort']==cohort and R.finite(r)],key=lambda r:int(r['s']))
                if rr:ax.errorbar([int(r['s']) for r in rr],[float(Q(r['G_exact'])) for r in rr],yerr=[2*float(r['G_SE']) for r in rr],color=colors[q],marker=marker,linestyle='-' if K==4 else '--',alpha=.9 if cohort=='confirmation' else .45,label=f'q{q} K{K} {cohort}; 2SE')
        h=next(x for x in hybrid if x['epsilon_sig']==e);ax.axhline(float(Q(h['G_exact'])),color='black',linestyle=':',label='Saved Hybrid n96')
        ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xlabel('s=q*r (actual q/r retained)');ax.set_title('epsilon='+e);ax.grid(alpha=.2)
    axs[0].set_ylabel('Conditional G (RZ x shot)');axs[-1].legend(fontsize=5)
    fig.suptitle('H2 pure RTE: hypothetical rho=0; separate cohorts; lines are visual guides; finite price samples')
    for ext in ['png','svg','pdf']:fig.savefig(f/('H2_G_vs_s.'+ext),dpi=160)
    plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for q,K in pairs:
        rr=sorted([r for r in trends if r['epsilon_sig']=='0.0005' and int(r['q'])==q and int(r['K'])==K and R.finite(r)],key=lambda r:int(r['s']))
        unique={}
        for x in rr:
            key=int(x['s'])
            if key not in unique or x['cohort']=='confirmation':unique[key]=x
        rr=list(unique.values())
        if not rr:continue
        for ax,key in [(axs[0],'B_exact'),(axs[1],'C_cos_exact')]:ax.plot([int(r['s']) for r in rr],[float(Q(r[key])) for r in rr],marker='.',color=colors[q],linestyle='-' if K==4 else '--',label=f'q{q} K{K}')
        axs[2].plot([int(r['s']) for r in rr],[int(r['N_cos'])+int(r['N_sin']) for r in rr],marker='.',color=colors[q],linestyle='-' if K==4 else '--',label=f'q{q} K{K}')
    for ax,title in zip(axs,['Normalization B','Single-wrapper C_cos (RZ)','Conditional N_cos+N_sin at epsilon=.0005']):ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xlabel('s=q*r');ax.set_title(title);ax.grid(alpha=.2);ax.legend(fontsize=6)
    fig.suptitle('H2 saved B/N/C: confirmation price if available, otherwise exploration; no price-tail guarantee')
    for ext in ['png','svg','pdf']:fig.savefig(f/('H2_B_N_C_vs_s.'+ext),dpi=160)
    plt.close(fig)
def main():
    R.check();data=R.rows();save(OUT/'analysis_SOURCE.json',dict(scientific_SOURCE=read(OUT/'SOURCE.json')['commit'],analysis_code_sha256=R.sha(Path(__file__)),input_snapshot_sha256=R.sha(OUT/'candidate_results.json')))
    v=verify(data);new=[flat(x) for x in data];prior=R.prior_quotes()
    old=[dict(x,s=int(x['q'])*int(x['r']),phase='OLD') for x in prior if x['group']=='RTE'];hybrid=[x for x in prior if x['group']=='Hybrid' and x['cohort']=='confirmation']
    trends=old+new;csvsave(OUT/'candidate_precision_table.csv',new);csvsave(OUT/'event_cost_trends.csv',trends)
    comparison=[];details=[]
    for e in R.EPS:
        h=next(x for x in hybrid if x['epsilon_sig']==e);ro=R.minimum([x for x in old if x['epsilon_sig']==e and x['cohort']=='confirmation']);aa=R.minimum([x for x in new if x['epsilon_sig']==e and x['phase']=='A' and x['cohort']=='confirmation']);be=R.minimum([x for x in new if x['epsilon_sig']==e and x['phase']=='B' and x['cohort']=='exploration']);bc=R.minimum([x for x in new if x['epsilon_sig']==e and x['phase']=='B' and x['cohort']=='confirmation']);updated=R.minimum([x for x in trends if x['epsilon_sig']==e and x['cohort']=='confirmation'])
        c=dict(molecule='H2',epsilon_sig=e,Hybrid_id=h['candidate_id'],Hybrid_G_exact=h['G_exact'],Hybrid_G_SE=h['G_SE'],Hybrid_n=h['price_n'],old_RTE_id=ro['candidate_id'],old_RTE_G_exact=ro['G_exact'],old_Hybrid_over_RTE_exact=ratio(h,ro),Phase_A_RTE_id=aa['candidate_id'] if aa else None,Phase_A_RTE_G_exact=aa['G_exact'] if aa else None,Phase_A_Hybrid_over_RTE_exact=ratio(h,aa),Phase_B_exploration_id=be['candidate_id'] if be else None,Phase_B_exploration_G_exact=be['G_exact'] if be else None,Phase_B_confirmation_id=bc['candidate_id'] if bc else None,Phase_B_Hybrid_over_RTE_exact=ratio(h,bc),updated_RTE_id=updated['candidate_id'],updated_RTE_G_exact=updated['G_exact'],updated_RTE_G_SE=updated['G_SE'],updated_RTE_n=updated['price_n'],Hybrid_over_RTE_exact=ratio(h,updated),RTE_over_Hybrid_exact=ratio(updated,h),classification=compare(h,updated),finite_price_hypothetical_rho='0',population_tail_guaranteed=False,global_optimality_proved=False)
        comparison.append(c);details.append(dict(comparison=c,Hybrid=h,old_RTE=ro,Phase_A_RTE=aa,Phase_B_exploration=be,Phase_B_confirmation=bc,updated_RTE=updated))
    csvsave(OUT/'comparison_by_precision.csv',comparison);save(OUT/'comparison_details.json',details)
    coverage=[]
    for cid in sorted({x['candidate_id'] for x in old}):
        oo=next(x for x in old if x['candidate_id']==cid);nn=[x for x in new if x['candidate_id']==cid and x['cohort']=='confirmation'];coverage.append(dict(candidate_id=cid,q=oo['q'],r=oo['r'],K=oo['K'],s=oo['s'],old_exploration_n=next(x['price_n'] for x in old if x['candidate_id']==cid and x['cohort']=='exploration'),old_confirmation_n=next((x['price_n'] for x in old if x['candidate_id']==cid and x['cohort']=='confirmation'),None),new_confirmation_n=nn[0]['price_n'] if nn else None,Phase_A_status='ACQUIRED' if nn else 'OLD_CONFIRMED_REUSED' if any(x['candidate_id']==cid and x['cohort']=='confirmation' for x in old) else 'NOT_NOMINATED_SAVED_G_CLEARLY_HIGHER',target_recomputed=False))
    csvsave(OUT/'old_candidate_confirmation_map.csv',coverage);plots(trends,hybrid);save(OUT/'verification.json',v)
    save(OUT/'result_summary.json',dict(status='H2_CONDITIONAL_COMPARISON_COMPLETE',counts={k:v[k] for k in ['new_target_state_actions','new_trajectory_count','new_axis_compiles','native_child_count']},comparisons=comparison,actual_rho_guaranteed=False,population_tail_guaranteed=False,mandatory_STOP=True))
    save(OUT/'STUDY_STOP.json',dict(mandatory_STOP=True,science_complete=True,no_resume=True,no_H2_H4_H6_major_claim_adoption_or_detailed_review=True))
    print(json.dumps(dict(counts=read(OUT/'result_summary.json')['counts'],comparisons=comparison),indent=2))
if __name__=='__main__':main()
