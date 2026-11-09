"""Approved fixed Q3-B1/B2 finite count audit; no circuit or old campaign."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction as F
import hashlib
import json
import lzma
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from importlib.metadata import version
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import pr_reference_feedback_q3b as b

BASE='c624114d99056373c7cbf5e933fb8ee786259c42'
SCI='f2638a7845685a119619382b850441f5d8d33713'
Q0='34327b904540556bce7509f2a3168cbff36daab7'
OUT=ROOT/'artifacts/pr_reference_feedback_q3b/2026-11-10/run01'
FILES={'Bprime':'artifacts/pr_reference_bprime/2026-11-07/ERROR_RPE.json',
       'Reference_B1':'artifacts/pr_reference_transfer_qualification/2026-11-06/STAGE4_RPE.json'}
INPUTS={}


def require(condition,reason):
    if not condition:raise RuntimeError(reason)


def save(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')


def raw(ref,path):
    payload=subprocess.check_output(['git','show',ref+':'+path],cwd=ROOT)
    INPUTS[ref+':'+path]=dict(commit=ref,path=path,sha256=hashlib.sha256(payload).hexdigest())
    return payload


def read(ref,path):return json.loads(raw(ref,path))


def fraction_record(value):
    q=b.q1._q0()
    return dict(exact=str(value),display=q.mp.nstr(q.mp.mpf(value.numerator)/value.denominator,24))


def audit_row(task):
    n,kx,feedback=task
    q=b.q1._q0();xa=b.q1.AxisCounts(n,kx)
    maximum=None;max_feedback=None;phase_witness=None;feedback_witness=None
    nonzero=zero=0;digest=hashlib.sha256()
    try:
        for ky in range(n+1):
            ya=b.q1.AxisCounts(n,ky)
            if not xa.mean and not ya.mean:
                try:b.count_phase(b.q1.RoundCounts(0,xa,ya))
                except b.q1.UnqualifiedPrefix:pass
                else:raise RuntimeError('Zero vector was accepted')
                zero+=1;digest.update(f'{ky}:ZERO\n'.encode());continue
            p=b.count_phase(b.q1.RoundCounts(0,xa,ya));lo,hi=b.bounds(p)
            w=hi-lo
            # Same exact direction, with integer coordinates rather than
            # rounded rational input endpoints. This is a consistency check;
            # inclusion relies on the unchanged directed-interval model.
            a,c=2*kx-n,2*ky-n
            ref=-q.mp.iv.pi if c==0 and a<0 else q.mp.iv.atan2(q.I(c),q.I(a))
            rlo,rhi=b.bounds(ref)
            require(max(lo,rlo)<=min(hi,rhi),'Integer/rational arg intervals disagree')
            require(-b.bounds(q.mp.iv.pi)[1]<=lo<=hi<=b.bounds(q.mp.iv.pi)[1], 'Principal arg range')
            if maximum is None or w>maximum:
                maximum=w;phase_witness=dict(kx=kx,ky=ky,phase=q.pack(p),integer_phase=q.pack(ref))
            f=None
            if feedback:
                f=b.q1._feedback_phase(p)
                fl,fh=b.bounds(f.reduced_ideal_interval);binary=F.from_float(f.parameter)
                exact_error=max(abs(binary-fl),abs(binary-fh))
                require(f.arithmetic_error_upper>=exact_error,'Feedback error does not cover interval endpoints')
                require(f.arithmetic_error_upper<=b.ARITHMETIC_ANGLE_LIMIT,'Feedback exceeds fixed analytic arithmetic bound')
                require(f.arithmetic_error_upper<=b.q1.FEEDBACK_CAP,'Feedback arithmetic exceeds Q0 cap')
                require(abs(f.period_shift)<=2 and abs(binary)<4,'Feedback representative outside fixed arithmetic domain')
                if max_feedback is None or f.arithmetic_error_upper>max_feedback:
                    max_feedback=f.arithmetic_error_upper
                    feedback_witness=dict(kx=kx,ky=ky,parameter_hex=f.parameter.hex(),
                        period_shift=f.period_shift,ideal=q.pack(f.reduced_ideal_interval),
                        exact_endpoint_error=str(exact_error))
            entry=[ky,[str(lo),str(hi)],None if f is None else [f.parameter.hex(),str(f.arithmetic_error_upper)]]
            digest.update((json.dumps(entry,separators=(',',':'))+'\n').encode())
            nonzero+=1
        return dict(status='PASS',n=n,kx=kx,ky_domain=[0,n],nonzero=nonzero,zero=zero,
            all_nonzero_cases_pass=True,phase_width_upper=str(maximum),phase_witness=phase_witness,
            feedback_error_upper=None if max_feedback is None else str(max_feedback),
            feedback_witness=feedback_witness,case_stream_sha256=digest.hexdigest())
    except Exception as exc:
        return dict(status='STOP_WITNESS',n=n,kx=kx,ky=ky,nonzero_completed=nonzero,
                    error=repr(exc),case_stream_sha256=digest.hexdigest())


def b2(b1):
    q=b.q1._q0()
    inp=read(Q0,'artifacts/pr_reference_q0/2026-11-09/run01/INPUTS.json')
    payload=raw(Q0,'artifacts/pr_reference_q0/2026-11-09/run01/RESULTS.json.xz')
    results=json.loads(lzma.decompress(payload))
    checks=read(Q0,'artifacts/pr_reference_q0/2026-11-09/run01/CHECKS.json')
    require(checks['status']=='Q0_POSTRUN_CHECKS_PASS','Q0 check status mismatch')
    require(hashlib.sha256(lzma.decompress(payload)).hexdigest()==checks['raw_result_sha256'],'Q0 complete result hash mismatch')
    require(inp['target']==[17,131072,32,88],'Q0 target changed')
    # Coarse source-backed 192-bit rounding envelope. At most 64 elementary
    # operations per endpoint chain, magnitudes <2^16, outward relative error
    # <=2^-191. Fraction conversions are bounded in relative form first.
    input_slack=64*F(1<<16)*F(1,1<<191)  # 2^-169
    root_slack=F(1,1<<84)  # sqrt(2*input_slack)
    inverse_slack=64*(root_slack+input_slack)
    require(inverse_slack<b.OMEGA_Q0,'Coarse inversion arithmetic envelope')
    time17=(1<<17)*b.q1.REFERENCE_TIME
    records=[]
    for arm in results['arms']:
        name=arm['arm'];model=inp['arms'][name]
        rho=F.from_float(model['raw_radius_lower'])
        bias=b.q1.CAP-F.from_float(model['statistical_phase_lower'])
        require(F(1,2)<=rho<=1 and 0<=bias<=F(1,4),'Q0 inverse magnitude-domain premise failed')
        item=arm['feedback_margins']['340']
        require(item['status']=='BRACKETED' and item['lower_is_sufficient'],'Q0 feedback lower bound not qualified')
        require(next(c for c in checks['checks'] if c['arm']==name)['feedback_brackets_verified_all_counts'], 'Missing all-count bracket check')
        delta_lo=F(item['bracket'][0]);reserve=delta_lo-b.q1.FEEDBACK_CAP
        kappa=F(b1['per_arm_round_bounds'][name][-1]['energy_center_width_upper'])
        lhs=kappa*time17+b.OMEGA_E*time17+b.OMEGA_Q0
        records.append(dict(arm=name,delta_lo=str(delta_lo),delta_safe=str(b.q1.FEEDBACK_CAP),
            reserve=fraction_record(reserve),kappa_Ha=fraction_record(kappa),
            omega_E_Ha=fraction_record(b.OMEGA_E),omega_Q0_rad=fraction_record(b.OMEGA_Q0),
            time17_exact=str(time17),lhs_rad=fraction_record(lhs),
            residual_rad=fraction_record(reserve-lhs),status='PASS' if lhs<=reserve else 'FAIL',
            new_general_signal_or_native_error_assumed_zero=False))
    require(all(r['status']=='PASS' for r in records),'B2 rounding margin insufficient')
    return dict(status='PASS_FIXED_INTERVAL_MODEL',rows=records,
        inverse_rounding_derivation=dict(input_slack=str(input_slack),root_slack=str(root_slack),
            composed_inverse_slack=str(inverse_slack),adopted_upper=str(b.OMEGA_Q0),
            source_model='192-bit directed basic and elementary interval operations; not a new formal library verification'),
        Q0_scan_inversion_bisection_rerun=False,all_native_contexts_qualified=False,
        implemented_m17_success_or_total_cost_qualified=False)


def run():
    require(not OUT.exists(),'Fresh Q3-B result namespace required')
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==BASE,'Source base changed')
    require(version('mpmath')=='1.3.0' and version('numpy')=='1.26.4' and sys.version_info[:2]==(3,12),'Fixed runtime mismatch')
    OUT.mkdir(parents=True)
    start=time.monotonic();q=b.q1._q0();schedules={}
    stop=dict(status='MANDATORY_STOP',next_owner='GPT/user')
    source=dict(base_commit=BASE,precision_bits=192,python=sys.version,executable=sys.executable,
        packages={p:version(p) for p in ('mpmath','numpy')},environment=dict(host=platform.node(),affinity=len(os.sched_getaffinity(0)),
            load=os.getloadavg(),memory=Path('/proc/meminfo').read_text(),cpu_PSI=Path('/proc/pressure/cpu').read_text(),
            memory_PSI=Path('/proc/pressure/memory').read_text()),
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
            'scripts/run_pr_reference_feedback_q3b.py','src/trottertracks/pr_reference_feedback_q3b.py',
            'tests/test_pr_reference_feedback_q3b.py','src/trottertracks/pr_reference_feedback_q1.py',
            'scripts/diagnose_pr_reference_q0.py')},inputs=INPUTS)
    try:
        for arm,path in FILES.items():
            d=read(SCI,path)
            require([r['m'] for r in d['rows'][:17]]==list(range(17)),'Saved prefix round mismatch')
            schedules[arm]=[r['shots_per_axis'] if arm=='Bprime' else r['arms']['reference']['shots_per_axis'] for r in d['rows'][:17]]
            require(all(type(n) is int and n>0 for n in schedules[arm]),'Invalid saved N')
        feedback_ns={ns[-1] for ns in schedules.values()}
        unique_ns=sorted({n for ns in schedules.values() for n in ns})
        tasks=[(n,kx,n in feedback_ns) for n in unique_ns for kx in range(n+1)]
        save('INPUTS.json',dict(scientific_commit=SCI,schedules=schedules,unique_denominators=unique_ns,
            feedback_denominators=sorted(feedback_ns),zero_is_not_success=True,
            unique_full_grid_cases=sum((n+1)**2 for n in unique_ns),
            mapped_arm_round_cases=sum((n+1)**2 for ns in schedules.values() for n in ns),
            phase_width_limit=str(b.PHASE_WIDTH_LIMIT),center_width_limit=str(b.CENTER_WIDTH_LIMIT),
            kappa_limit=str(b.KAPPA_LIMIT),arithmetic_angle_limit=str(b.ARITHMETIC_ANGLE_LIMIT)))
        save('B0.json',dict(status='ARTIFICIAL_AND_ANALYTIC_BASE_PASS_PENDING_B1_COVERAGE',artificial_tests=6,
            selector='strict interval nearest-integer cells; signed unwrapped branches',legacy_basis=False,
            source='src/trottertracks/pr_reference_feedback_q3b.py',old_SOURCE_changed=False))
        workers=min(len(tasks),len(os.sched_getaffinity(0)));row_results=[]
        # All independent fixed rows are submitted once. In-flight work is
        # preserved on failure; no new domain or retry is introduced.
        with lzma.open(OUT/'B1_ROWS.jsonl.xz','wt') as journal:
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork')) as pool:
                futures=[pool.submit(audit_row,t) for t in tasks]
                for future in as_completed(futures):
                    row=future.result();journal.write(json.dumps(row,sort_keys=True)+'\n');row_results.append(row)
                    if row['status']!='PASS':
                        save('B1_FAILURE.json',row)
                        for f in futures:f.cancel()
                        raise RuntimeError('B1 numerical witness; no scope expansion')
                    if len(row_results)%128==0:
                        print('B1 fixed rows completed',len(row_results),'/',len(tasks),flush=True)
        tables={}
        for n in unique_ns:
            rows=sorted((r for r in row_results if r['n']==n),key=lambda r:r['kx'])
            require([r['kx'] for r in rows]==list(range(n+1)),'B1 row coverage incomplete')
            require(sum(r['nonzero']+r['zero'] for r in rows)==(n+1)**2,'B1 pair coverage incomplete')
            w=max(rows,key=lambda r:F(r['phase_width_upper']))
            fe=max((r for r in rows if r['feedback_error_upper'] is not None),key=lambda r:F(r['feedback_error_upper']),default=None)
            tables[str(n)]=dict(n=n,rows=len(rows),all_pairs=(n+1)**2,nonzero=sum(r['nonzero'] for r in rows),zero=sum(r['zero'] for r in rows),
                max_phase_width=w['phase_width_upper'],phase_witness=w['phase_witness'],
                max_feedback_error=None if fe is None else fe['feedback_error_upper'],
                feedback_witness=None if fe is None else fe['feedback_witness'])
        pi_width=b.width(q.mp.iv.pi);per_arm={}
        for arm,ns in schedules.items():
            rows=[]
            for m,n in enumerate(ns):
                phase_width=F(tables[str(n)]['max_phase_width']);power=1<<m
                wc=phase_width if m==0 else phase_width/power+2*pi_width+b.BASIC_SLACK
                cell_width=None if m==0 else (power*b.CENTER_WIDTH_LIMIT+phase_width)/6+b.BASIC_SLACK
                require(wc<b.CENTER_WIDTH_LIMIT,'Uniform center width bound failed')
                require(cell_width is None or cell_width<F(1,1<<130),'Uniform cell slack failed')
                # Different stages: center division error here, final energy
                # hull restoration error separately in B2. Neither is PF/disk.
                kappa=wc/b.q1.REFERENCE_TIME+b.OMEGA_E
                require(kappa<b.KAPPA_LIMIT,'Uniform energy-center bound failed')
                rows.append(dict(m=m,n=n,q=power,phase_width_upper=str(phase_width),
                    base_center_width_upper=str(wc),cell_width_upper=None if cell_width is None else str(cell_width),
                    energy_center_width_upper=str(kappa)))
            per_arm[arm]=rows
        b1=dict(status='PASS_FIXED_DIRECTED_INTERVAL_MODEL',tables=tables,per_arm_round_bounds=per_arm,
            workers=workers,audited_unique_rows=len(row_results),all_requested_domains_covered=True,
            primitive_inclusion_basis='preserved 192-bit directed interval model + exact Fraction inputs; integer-coordinate overlap checked for every nonzero pair',
            all_history_acceptance='proved by B0 induction using these finite phase-width bounds; not history-product enumeration',
            old_binary_ranking_required=False,mapped_arms=list(schedules),
            native_total_feedback_guarantee=False)
        save('B1.json',b1)
        b2_result=b2(b1);save('B2.json',b2_result)
        save('RESULTS.json',dict(status='B0_B1_B2_PASS_IN_FIXED_INTERVAL_MODEL',
            branch_selection='PROVED_ACCEPTANCE_IN_FIXED_SUCCESS_AND_INTERVAL_MODEL',
            finite_domain_coverage=True,feedback_arithmetic_and_energy_rounding='PASS',
            native_all_contexts='UNQUALIFIED',implemented_340_shot_RPE='CONDITIONAL_NOT_ADOPTED',
            total_18round_cost='NOT_EVALUATED',B1='B1.json',B2='B2.json'))
        stop.update(reason='Fixed B0-B2 and technical checks complete; native all-context qualification remains')
        print('B0/B1/B2 fixed-model PASS; native all-context guarantee remains UNQUALIFIED',flush=True)
    except BaseException as exc:
        stop.update(reason='Fixed-scope scientific/technical STOP',error=repr(exc))
        raise
    finally:
        source['elapsed_seconds']=time.monotonic()-start
        source['prohibited_calls']=dict(old_Q0_Q2_campaign=0,H4_ground_or_cache=0,circuit_build_compile=0,sampling=0,
            RPE=0,total_resource_accounting=0,Clifford_T_guard_synthesis_price=0)
        source['shared_environment_changed']=False;source['old_source_evidence_STOP_changed']=False
        save('SOURCE.json',source);save('STOP.json',stop)


if __name__=='__main__':run()
