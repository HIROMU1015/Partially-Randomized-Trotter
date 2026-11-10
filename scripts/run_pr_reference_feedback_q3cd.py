"""One fixed Q3-C/D saved-evidence and native-cost campaign; no quantum runs."""
from __future__ import annotations
from fractions import Fraction as F
from functools import lru_cache
import hashlib, io, json, lzma, os, platform, subprocess, sys, tarfile, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE='1795d72b704cfd89aeac7c0075ebb576406ba624'
SCIENCE='f2638a7845685a119619382b850441f5d8d33713'
OUT=ROOT/'artifacts/pr_reference_feedback_q3cd/2026-11-10/run01'
INPUTS={}


def require(value,reason):
    if not value:raise RuntimeError(reason)


@lru_cache(None)
def raw(path):
    data=subprocess.check_output(['git','show',BASE+':'+path],cwd=ROOT)
    INPUTS[path]=dict(commit=BASE,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    return data


def read(path):
    data=raw(path)
    return json.loads(lzma.decompress(data) if path.endswith('.xz') else data)


def archive(path):
    with tarfile.open(fileobj=io.BytesIO(raw(path)),mode='r:xz') as stream:
        return [json.load(stream.extractfile(m)) for m in stream if m.isfile()]


def save(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,sort_keys=True)+'\n')


def pair(lo,hi=None):return [F(lo),F(lo if hi is None else hi)]
def add(ps):
    ps=list(ps)
    return [sum((p[i] for p in ps),F()) for i in (0,1)]
def scale(n,p):return [n*p[0],n*p[1]]
def subtract(a,b):return [a[0]-b[1],a[1]-b[0]]
def pack(p):return dict(exact=[str(x) for x in p],display=[format(float(x),'.15g') for x in p])


def c0():
    bp=archive('artifacts/pr_reference_bprime/2026-11-07/Bprime_local_catalogue.tar.xz')
    ref=[d for d in archive('artifacts/pr_reference_transfer_qualification/2026-11-06/local_catalogue_boundary.tar.xz')
         if d['request'][0]=='reference']
    require(len(bp)==len(ref)==4,'Catalogue context coverage')
    checks=[];selected={}
    for arm,contexts in [('Bprime',bp),('Reference_B1',ref)]:
        selected[arm]={r['name']:r for r in next(d for d in contexts if d['request'][-2:]==[True,True])['records']}
        require(not any('x' in r['ops'] for d in contexts for r in d['records']),
                'Native X exists: additive SX-only size formula must not be used')
        for rare in (False,True):
            rows=[next(d for d in contexts if d['request'][-2:]==[rare,y]) for y in (False,True)]
            require(set(rows[1]['post_unroll'])==set(rows[0]['post_unroll'])|{'sdg'},'Unexpected Y context delta')
            tables=[{r['name']:r for r in d['records']} for d in rows]
            require(set(tables[1])-set(tables[0])=={'sdg'},'Unexpected Y record delta')
            require(set(tables[0])<=set(tables[1]),'Missing Y records')
            require(all(r==tables[1][k] for k,r in tables[0].items()),'Core/H/P lowering changed across Y contexts')
            for table in tables:
                require(table['p_symbolic']['lowering']==[['rz',[0],['symbolic_correction']]],'P angle not preserved')
                require(table['h']['lowering']==[['rz',[0],[float.fromhex('0x1.921fb54442d18p+0')]],
                        ['sx',[0],[]],['rz',[0],[float.fromhex('0x1.921fb54442d18p+0')]]],'H lowering changed')
            require(tables[1]['sdg']['lowering']==[['rz',[0],[-float.fromhex('0x1.921fb54442d18p+0')]]],'Sdg template changed')
            for d in rows:
                for rec,pre in zip(d['records'],d['pre_last_cancel']):
                    if rec['meta']['kind'] in ('basis','basis_inverse','D_half','D_full','baseline_D'):
                        require(not pre['exposed_cancelable_prefix'] and not pre['exposed_cancelable_suffix'],
                                'D/frame fence not retained')
            checks.append(dict(arm=arm,rare=rare,common_records=len(tables[0]),
                Q_post_unroll=rows[0]['post_unroll'],core_lowerings_identical=True,
                P_signed_parameter_preserved=True,H_sdg_fences=True,no_native_X=True))
    q2=read('artifacts/pr_reference_feedback_q2/2026-11-10/run02/RESULTS.json')
    require(len(q2['rows'])==16 and all(r['passed'] for r in q2['rows']),'Q2 evidence incomplete')
    for row in q2['rows']:
        arm=row['identity']['arm'];rare=bool(row['identity']['rare'])
        context=next(d for d in (bp if arm=='Bprime' else ref) if d['request'][-2:]==[rare,False])
        post=next(t for t in row['compiler_trace'] if t['pass_name']=='UnrollCustomDefinitions')
        require(set(post['ops'])-{'measure','barrier'}==set(context['post_unroll']),'Q2 Q/context inventory mismatch')
    symbolic=read('artifacts/pr_reference_feedback_q3b/2026-11-10/run01/POST_B2_TECHNICAL.json')
    require(symbolic['same_signed_axis_parameter'] and symbolic['parameter_unbound'],'Missing symbolic P(-phi) evidence')
    source_paths=['src/trottertracks/'+p+'.py' for p in ('track_c_compiler_a','pr_reference_bprime',
        'pr_reference_interference_pilot','pr_reference_transfer_qualification')]
    for path in source_paths:
        require((ROOT/path).read_bytes()==raw(path)==subprocess.check_output(['git','show',SCIENCE+':'+path],cwd=ROOT),
                'Fixed physical SOURCE changed: '+path)
    result=dict(status='ALL_CONTEXT_NATIVE_TRANSFER_CLOSED_IN_FIXED_MODEL',checks=checks,
        Q2_saved_rows_reused=16,scientific_build_compile_operator_calls=0,
        C1='NOT_NEEDED: same source basis, identical core templates, structural norm proof',
        proof='docs/research/pr_reference_feedback_q3cd_20261110/NATIVE_GUARANTEE.md',
        general_native_error='old 2sqrt(2)E raw disk retained once; virtual-state norm lemma',
        additional_readout_angle_error_rad='0, analytic P->RZ identity in ideal-native model')
    save('C0.json',result)
    return selected


def d_stage():
    from trottertracks import pr_reference_feedback_q3b as b
    q0=read('artifacts/pr_reference_q0/2026-11-09/run01/INPUTS.json')
    q0checks=read('artifacts/pr_reference_q0/2026-11-09/run01/CHECKS.json')
    b1=read('artifacts/pr_reference_feedback_q3b/2026-11-10/run01/B1.json')
    b2=read('artifacts/pr_reference_feedback_q3b/2026-11-10/run01/B2.json')
    require(q0checks['status']=='Q0_POSTRUN_CHECKS_PASS','Q0 checks not passed')
    require(b1['all_requested_domains_covered'] and all(r['status']=='PASS' for r in b2['rows']),'Q3-B not passed')
    fe=F(b1['tables']['347']['max_feedback_error'])
    require(fe<b.ARITHMETIC_ANGLE_LIMIT<b.q1.FEEDBACK_CAP,'Additional angle cap exceeded')
    require(F(34,720)+F(1,360)==F(1,20),'Failure budget changed')
    t17=(1<<17)*b.q1.REFERENCE_TIME
    require(b.q1.CAP/t17<F('0.00015936'),'Energy accuracy insufficient')
    prices=read('artifacts/pr_reference_bprime/2026-11-07/TOTALS_18ROUND.json')
    proof=read('artifacts/pr_reference_transfer_qualification/2026-11-06/STAGE3_ALL_SUPPORT.json')
    q0r=read('artifacts/pr_reference_v6p/2026-11-09/P_run01/RESULTS.json')
    rows=[]
    for arm in ('Bprime','Reference_B1'):
        model=q0['arms'][arm]
        saved=prices['rows'][17] if arm=='Bprime' else q0r['conditional_reference']
        require(all(model[k]==saved[k] for k in ('raw_radius_lower','statistical_phase_lower','total_raw_disk_upper')),
                'Q0 arm/model no longer matches saved science')
        row=next(r for r in b2['rows'] if r['arm']==arm)
        kappa=F(row['kappa_Ha']['exact'])
        lhs=(kappa+b.OMEGA_E)*t17+b.OMEGA_Q0
        reserve=F(row['delta_lo'])-F(row['delta_safe'])
        require(lhs==F(row['lhs_rad']['exact']) and lhs<reserve,'Hull rounding reserve fails')
        nseq=b.q1._saved_prefix_shots(arm)
        require(nseq==tuple(r['n'] for r in b1['per_arm_round_bounds'][arm]),'Prefix N binding changed')
        require(all(a['all_support_sufficient'] for r in proof['rows'][:17] for a in r['arms'].values()),'Prefix guarantees absent')
        rows.append(dict(arm=arm,status='GUARANTEE_CLOSED_IN_FIXED_MODEL',N17=340,
            rho_lower=model['raw_radius_lower'],bias_upper_exact=str(b.q1.CAP-F.from_float(model['statistical_phase_lower'])),
            total_raw_disk_upper=model['total_raw_disk_upper'],feedback_arithmetic_upper=str(fe),
            pure_added_native_angle_error='0 (analytical identity; H/core errors retained in raw disk)',
            shared_angle_cap=str(b.q1.FEEDBACK_CAP),hull_rounding_reserve=str(reserve),
            hull_rounding_charge=str(lhs),remaining_reserve=str(reserve-lhs),
            prefix_shots_axis=list(nseq),prefix_trial_count=sum(2*n for n in nseq),
            all_trial_count=sum(2*n for n in nseq)+340,
            final_energy_halfwidth_upper_Ha=str(b.q1.CAP/t17)))
    result=dict(status='GUARANTEE_CLOSED_IN_FIXED_MODEL',rows=rows,
        probability='conditional on each prefix history: fresh 340 IID +/-1 trials; 34/720+1/360=1/20',
        scope='fixed ideal-native / independent finite-bit / IEEE / 192-bit directed-interval model',
        formal_research_adoption=False,hardware_success_evaluated=False,
        Q0_Q2_Q3B_campaigns_rerun=False)
    save('D.json',result)
    return prices,q0r


def auxiliary_sx(row,depth,records,arm):
    """Add surviving SX gates from saved templates and exact saved law moments.

    Source cancellation removes no XX+YY, CZ or CSWAP in this grammar; native
    final cancellation removes only X/CX. All saved native X counts are zero.
    """
    q,r=row['q'],row['r'];a=records[arm]
    count=lambda name:F(a[name]['ops'].get('sx',0))
    if arm=='Bprime':
        D=count('left_all')+count('right_all')+(q-1)*(count('left_no0')+count('right_no0')+count('boundary_control'))
    else:
        C=lambda k:count(f'block:{k}:half')
        D=2*sum(C(k) for k in range(4)) if q==1 else 4*C(0)+(q-2)*count('block:0:full')+2*q*sum(C(k) for k in (1,2,3))
        D+=16*count('cswap')
    term=depth['axes']['X']['depth_per_shot']['tail_Bprime' if arm=='Bprime' else 'tail_reference']
    p2=F(row['contributions']['p_order2']);support=F(row['contributions']['expected_support'])
    frames=q*sum((count('basis:'+bid)+count('basis_inverse:'+bid))*(F(pi)+(r*(1+2*p2)-1)*F(pi)*(1-F(pi)))
                 for bid,pi in term['basis_probabilities'].items())
    # CRZ/Bprime/RZZ/rotation kernels have no SX. Only controlled products do.
    products=2*q*r*p2*support*(count('cz') if arm=='Bprime' else count('z'))
    return D+frames+products+2*count('h')


def cost_stage(prices,v6p,records):
    depth=read('artifacts/pr_reference_depth_v5/2026-11-07/DEPTH_18ROUND.json')
    oldref=read('artifacts/pr_reference_transfer_qualification/2026-11-06/STAGE4_RPE.json')
    require(len(prices['rows'])==len(depth['rows'])==len(oldref['rows'])==18,'Round price coverage')
    accum={arm:{mode:{metric:[] for metric in ('RZ','CX','size','depth')} for mode in ('XY','Q')}
           for arm in ('Bprime','Reference_B1')}
    output=[]
    for row,dep,ref in zip(prices['rows'],depth['rows'],oldref['rows']):
        m=row['m'];require(dep['m']==ref['m']==m,'Price/depth round binding')
        for arm in ('Bprime','Reference_B1'):
            n=row['shots_per_axis'] if arm=='Bprime' else ref['arms']['reference']['shots_per_axis']
            if m==17:n=626
            require(n==row['shots_per_axis'],'B1 versus Bprime prefix N changed')
            sx=auxiliary_sx(row,dep,records,arm)
            axes={}
            for axis in 'XY':
                p=row['price_per_axis'][axis];da='Bprime' if arm=='Bprime' else 'reference'
                rz=pair(p[da+'_RZ_lower'],p[da+'_RZ_upper'])
                cx=(pair(p['Bprime_CX_lower'],p['Bprime_CX_upper']) if arm=='Bprime' else
                    pair(F(row['contributions']['D_reference_CX_including_CSWAP'])-128+F(row['contributions']['shared_basis_CX']),
                         p['reference_CX_upper']))
                # The lower drops all rotation and CSWAP CX analytically;
                # protected D/frame CX remain. It is not a missing-price fill.
                require(cx[0]>0 and cx[1]>=cx[0] and rz[1]>=rz[0]>0,'Invalid qualified price bounds')
                size=[rz[i]+cx[i]+sx+1+(8 if arm=='Reference_B1' else 0) for i in (0,1)]
                dd=dep['axes'][axis]['depth_per_shot'][da]
                axes[axis]=dict(RZ=rz,CX=cx,size=size,depth=pair(dd['lower'],dd['upper']))
            require(axes['Y']['RZ']==[x+1 for x in axes['X']['RZ']] and axes['X']['CX']==axes['Y']['CX'],
                    'Readout structural delta not consistent')
            # All Q gate names/wires equal Y; the ancilla RZ parameter alone changes.
            qprice=axes['Y']
            old={k:scale(n,add(axes[a][k] for a in 'XY')) for k in axes['X']}
            new={k:(scale(340,qprice[k]) if m==17 else old[k]) for k in old}
            saving={}
            for metric in old:
                # Correlated same-price saving: P-Q = n*X+(n-340)*Y.
                # Cancels unchanged prefix exactly; no independent subtraction.
                saving[metric]=(add((scale(n,axes['X'][metric]),scale(n-340,axes['Y'][metric])))
                                if m==17 else pair(0))
                for mode,value in [('XY',old[metric]),('Q',new[metric])]:accum[arm][mode][metric].append(value)
            output.append(dict(m=m,arm=arm,q=row['q'],r=row['r'],bits=row['bits'],
                old_N_per_axis=n,new_physical_shots=340 if m==17 else 2*n,
                SX_per_shot_exact=str(sx),per_shot={a:{k:pack(v) for k,v in ps.items()} for a,ps in axes.items()},
                Q_per_shot={k:pack(v) for k,v in qprice.items()} if m==17 else None,
                XY_cost={k:pack(v) for k,v in old.items()},Q_cost={k:pack(v) for k,v in new.items()},
                paired_saving={k:pack(v) for k,v in saving.items()},
                extra_P_cost_included='Q uses Y price: +1 RZ/+1 size relative to X',
                A_native_outside_interrogation='0: no additional quantum operation outside the counted circuits'))
    totals={a:{mode:{k:add(ps) for k,ps in metrics.items()} for mode,metrics in modes.items()} for a,modes in accum.items()}
    # Reproduce already-published RZ totals with the B1 change applied only at m17.
    accounts=v6p['accounting_RZ_only']
    require(totals['Bprime']['XY']['RZ']==list(map(F,accounts['Bprime_total_unchanged'])),'Saved Bprime total mismatch')
    require(totals['Reference_B1']['XY']['RZ']==list(map(F,accounts['conditional_reference_total'])),'Saved B1 total mismatch')
    comparisons={}
    for arm in totals:
        last=next(x for x in output if x['arm']==arm and x['m']==17)
        comparisons[arm]={}
        for metric in ('RZ','CX','size','depth'):
            saving=list(map(F,last['paired_saving'][metric]['exact']))
            require(saving[0]>0,'No guaranteed own-arm saving')
            comparisons[arm][metric]=dict(status='TOTAL_COST_REDUCTION_ESTABLISHED_IN_FIXED_MODEL',
                m17_saving=pack(saving),all18_saving=pack(saving),
                reduction_fraction_all18=pack([saving[0]/totals[arm]['XY'][metric][1],
                    saving[1]/totals[arm]['XY'][metric][0]]),
                reduction_fraction_m17=pack([saving[0]/F(last['XY_cost'][metric]['exact'][1]),
                    saving[1]/F(last['XY_cost'][metric]['exact'][0])]))
    cross={}
    for metric in ('RZ','CX','size','depth'):
        delta=subtract(totals['Reference_B1']['Q'][metric],totals['Bprime']['Q'][metric])
        cross[metric]=dict(reference_minus_Bprime=pack(delta),
            reference_upper_below_Bprime_lower=delta[1]<0,
            reference_saving_fraction_lower=str(-delta[1]/totals['Bprime']['Q'][metric][0]) if delta[1]<0 else None)
    result=dict(status='TOTAL_COST_REDUCTION_ESTABLISHED_IN_FIXED_MODEL',
        rows=output,totals={a:{mode:{k:pack(v) for k,v in d.items()} for mode,d in modes.items()} for a,modes in totals.items()},
        comparisons=comparisons,Reference_Q_vs_Bprime_Q=cross,
        expected_price_bounds_not_feasible_endpoint_policies=True,
        metrics='RZ,CX,size,shot-weighted native DAG depth separately; ideal all-to-all gates',
        qubits=dict(Bprime=9,Reference_B1=17,new_readout_extra=0),
        m17_shots=dict(old=1252,new=340),
        classical_feedback='one 17-round interval-prefix and angle computation per RPE attempt; unpriced, no runtime advantage asserted',
        calibration='none in ideal-native model; device calibration is outside the qualified cost model',
        compiler_generation='one fresh canonical trajectory/circuit per physical shot; classical CPU/RAM cost unpriced',
        Clifford_T='UNQUALIFIED: no feedback word, angle-wide synthesis policy or all18 total T evaluation',
        Full_Truncated='same opportunity must be granted, no PR-versus-Full/Truncated superiority claimed')
    save('COST.json',result)
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();stop=dict(status='MANDATORY_STOP',next_owner='GPT/user')
    paths=['scripts/run_pr_reference_feedback_q3cd.py','src/trottertracks/pr_reference_feedback_q3cd.py',
           'tests/test_pr_reference_feedback_q3cd.py','docs/research/pr_reference_feedback_q3cd_20261110/NATIVE_GUARANTEE.md']
    source=dict(base_commit=BASE,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        environment=dict(host=platform.node(),python=sys.version,executable=sys.executable,
            affinity=len(os.sched_getaffinity(0)),load=os.getloadavg(),
            memory=Path('/proc/meminfo').read_text(),cpu_pressure=Path('/proc/pressure/cpu').read_text(),
            memory_pressure=Path('/proc/pressure/memory').read_text()),
        parallelism='small saved rational tables; no heavy independent quantum calculation to parallelize',
        scientific_calls=dict(C0_saved_evidence_campaign=1,C1_build_compile_operator=0,D_saved_boundary_check=1,
            cost_all18_saved_reaccount=1,old_campaign_reruns=0,real_H4_cache_sampling=0,
            m17_whole_build_compile=0,Clifford_T=0),old_SOURCE_results_STOP_shared_environment_changed=False)
    try:
        selected=c0()
        prices,v6p=d_stage()
        cost_stage(prices,v6p,selected)
        save('RESULTS.json',dict(native='ALL_CONTEXT_NATIVE_TRANSFER_CLOSED_IN_FIXED_MODEL',
            RPE='GUARANTEE_CLOSED_IN_FIXED_MODEL',cost='TOTAL_COST_REDUCTION_ESTABLISHED_IN_FIXED_MODEL',
            formal_research_adoption=False,physical_hardware_or_total_T_claim=False))
        stop['reason']='Q3-C0, D and qualified cost comparison complete; C1 not needed'
        print('Q3-C/D and fixed-model native-cost comparison complete',flush=True)
    except BaseException as exc:
        stop.update(reason='Fixed-scope mismatch: no scope expansion',error=repr(exc))
        raise
    finally:
        source['inputs']=INPUTS;source['elapsed_seconds']=time.monotonic()-started
        save('SOURCE.json',source);save('STOP.json',stop)


if __name__=='__main__':main()
