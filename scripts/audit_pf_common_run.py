"""Read-only numerical/byte/ledger ending audit and portable result projection."""
import argparse
import csv
from decimal import Decimal,localcontext
from fractions import Fraction
import gzip
import hashlib
import math
from pathlib import Path
import pf_common_run_contract as c

def audit(private,output):
    private=Path(private);output=Path(output);c.require(not output.exists(),'NEW_AUDIT_EXPORT_ONLY')
    plan=c.read(private/'plan_APPROVED.json');c.verify(plan)
    stage_results={};files=[];grants=[]
    for name in ('SIGNAL','QUALIFY','COST','ACCOUNT'):
        root=private/name;receipt=c.read(root/'completion.json');c.require(receipt['status']=='PASS' and not (root/'STOP.json').exists(),'SUCCESSFUL_STAGE:'+name)
        grant=c.read(root/'grant.json');marker=c.read(root/'CONSUMED.json')
        c.require(marker['grant']==grant and grant['plan_sha256']==c.sha(private/'plan_APPROVED.json')
            and grant['SOURCE']==plan['SOURCE'],'GRANT_SOURCE_PLAN_MARKER:'+name)
        counts={k:0 for k in grant['caps']}
        for p in (root/'ledger').glob('*.json'):
            row=c.read(p);c.require(row['entered_nonrefundable'] is True,'NONREFUNDABLE_LEDGER');counts[row['kind']]+=row['count']
        c.require(counts==receipt['actual_calls']==c.read(root/'reserved_counts.json') and all(counts[k]<=grant['caps'][k] for k in counts),'LEDGER_COUNTER_CAP:'+name)
        for p in (root/'grant.json',root/'CONSUMED.json',root/'completion.json'):
            files.append(dict(path=str(p.relative_to(private)),bytes=p.stat().st_size,sha256=c.sha(p)))
        stage_results[name]=receipt;grants.append(dict(stage=name,grant_id=grant['grant_id'],grant_sha256=c.sha(root/'grant.json'),
            marker_sha256=c.sha(root/'CONSUMED.json'),ledger_entries=len(list((root/'ledger').glob('*.json'))),SOURCE=grant['SOURCE']))
    c.require(len({x['grant_id'] for x in grants})==4,'NO_GRANT_REUSE')
    new=c.read(private/'COST/selected_candidates.json');native=[];archives=[];maxima={}
    for row in new:
        root=private/'COST/output'/c.cid(row);result=c.read(root/'result.json')
        c.require(result['status']=='PASS' and result['candidate']==row and result['SOURCE']==plan['SOURCE'],'NATIVE_SOURCE_CANDIDATE')
        c.require(result['hamiltonian']==plan['binding']['hamiltonian'] and result['state']==plan['binding']['state'],'NATIVE_INPUT')
        for check in result['checks']:
            threshold=1e-7 if check['check'].startswith('compiled_vector_') else 1e-9
            c.require(check['PASS'] and check['phase_alignment'] is False and check['strictly_below']==threshold
                and math.isfinite(check['error']) and 0<=check['error']<threshold,'RAW_NATIVE_GATE')
            maxima[check['check']]=max(maxima.get(check['check'],0.),check['error'])
        for ax in ('cosine','sine'):
            price=result['prices'][ax]
            c.require(price['compiler']==c.COMPILER and price['preparation_hash']==result['preparation_hash'],'PAIRED_COMPILER_PREPARATION')
            if plan['system']=='H2':c.require(price['phase_guard']['unchanged_gate_stream'],'H2_PHASE_POLICY')
        for e in result['archives']:
            path=root/e['path'];blob=path.read_bytes();data=gzip.decompress(blob)
            c.require(len(data)==e['original_bytes'] and hashlib.sha256(data).hexdigest()==e['original_sha256']
                and hashlib.sha256(blob).hexdigest()==e['archive_sha256'],'RAW_ARCHIVE_BYTE_RESTORATION')
            archives.append(dict(candidate=c.cid(row),**e))
        native.append(dict(candidate=row,result_sha256=c.sha(root/'result.json'),preparation_hash=result['preparation_hash'],
            wall_seconds=result['wall_seconds'],peak_RSS_bytes=result['peak_RSS_bytes'],pid=result['pid']))
    worker=c.read(private/'COST/worker_exit_receipt.json')
    c.require(worker['same_parent_join'] and worker['remaining_owned_workers']==0 and all(x['exit_code']==0 and x['reaped'] for x in worker['workers']),'NATIVE_EXIT_REAP')
    endpoints=c.read(private/'ACCOUNT/output/PF_endpoint_costs.json');comparisons=c.read(private/'ACCOUNT/output/comparisons.json')
    c.require(len(endpoints)==len(plan['rows'])*5 and len(comparisons)==20,'ENDPOINT_COMPARISON_COVERAGE')
    integer_checks=0
    for ep in endpoints:
        if ep['status']=='ELIGIBLE':
            for ax,N in [('real',ep['N_cos']),('imag',ep['N_sin'])]:
                axis=ep['axes'][ax];h=axis['headroom'];L=ep['log_B']+ep['log_B_margin']
                # Separate integer-ceiling inequality, no new shot policy or qualification.
                bound=2*math.exp(L)**2/h**2*math.log(80.)
                c.require(N-1<bound<=N,'SAVED_INTEGER_CEILING_INEQUALITY');integer_checks+=1
            price=ep['price'];total=Fraction(ep['N_cos'])*price['cosine']+Fraction(ep['N_sin'])*price['sine']
            c.require(total==Fraction(ep['total_RZ_exact']),'SEPARATE_INTEGER_COST_CONSTRUCTION')
    for comparison in comparisons:
        formula=comparison['formula'];ep=comparison['epsilon'];all_ep=[x for x in endpoints if x['epsilon']==ep and (formula=='ALL_FORMULAS' or x['candidate']['formula']==formula)]
        pf=comparison['PF'];available=[x for x in all_ep if x['status']=='ELIGIBLE']
        c.require(not any(x['status']=='UNDETERMINED' for x in all_ep),'UNRESOLVED_PF_GRID_NOT_DROPPED')
        if available:
            c.require(pf['best'] in available and all(Fraction(pf['best']['total_RZ_exact'])<=Fraction(x['total_RZ_exact']) for x in available),'COMPLETE_GRID_MINIMUM')
        pr=comparison['PR_saved']
        if pr and pf['best']:
            with localcontext() as ctx:
                ctx.prec=80
                u=Decimal(pr['total_RZ_numerator'])/Decimal(pr['total_RZ_denominator']);v=Decimal(pf['best']['total_RZ_exact'])
                delta=float((1-u/v)*100)
            c.require(abs(delta-comparison['PR_reduction_percent'])<=1e-10*max(1,abs(delta)),'DECIMAL_COMPARISON_RATIO')
    output.mkdir(parents=True)
    def save(name,value):c.save(output/name,value)
    save('comparisons.json',comparisons);save('stage_receipts.json',stage_results)
    save('grants_and_markers.json',grants);save('native_result_inventory.json',native);save('raw_archive_inventory.json',archives)
    save('source_input_identity.json',dict(SOURCE=plan['SOURCE'],base_commit=plan['base_commit'],binding=plan['binding'],
        input_seals=[{k:v for k,v in e.items() if k!='path'}|dict(role=e['role']) for e in plan['closure'] if e['role'].startswith('scientific_input:')],
        original_snapshot_restored=plan['system']=='H6',regenerated_container_scientific_arrays_old_hashes_identical=plan['system']!='H6',
        source_closure_sha256=c.digest(plan['closure']),source_files=[dict(path=e['role'][7:],sha256=e['sha256']) for e in plan['closure'] if e['role'].startswith('SOURCE:')],
        previous_confirmation_provider_runtime_hashes_reverified=True))
    with (output/'PF_endpoint_costs.json.gz').open('xb') as f:f.write(gzip.compress(c.encoded(endpoints),compresslevel=1,mtime=0))
    columns=['id','system','method','prefix','q','formula','epsilon','status','u','log_B_margin','N_cos','N_sin','price_kind','cosine_RZ','sine_RZ','total_RZ_exact']
    with (output/'PF_endpoint_costs.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,lineterminator='\n');w.writeheader()
        for ep in endpoints:
            price=ep['price'] or {}
            w.writerow(dict(ep['candidate'],id=c.cid(ep['candidate']),**{k:ep[k] for k in ('epsilon','status','u','log_B_margin','N_cos','N_sin','total_RZ_exact')},
                price_kind=price.get('kind'),cosine_RZ=price.get('cosine'),sine_RZ=price.get('sine')))
    rows=[]
    for row in plan['rows']:
        identity=c.cid(row);ep=[e for e in endpoints if c.cid(e['candidate'])==identity]
        rows.append(dict(row,id=identity,signal='NEW' if identity in {c.cid(x) for x in plan['signal_base']+plan['signal_overlays']} else 'REUSED',
            eligible_epsilons=[e['epsilon'] for e in ep if e['status']=='ELIGIBLE'],cost_kind=(ep[0]['price'] or {}).get('kind','NOT_REQUIRED_INELIGIBLE'),
            status='COMPLETE' if all(e['status'] in ('ELIGIBLE','INELIGIBLE') and (e['status']=='INELIGIBLE' or e['total_RZ_exact'] is not None) for e in ep) else 'UNRESOLVED'))
    save('candidate_status.json',rows);save('worker_exit_receipt.json',worker);save('old_cost_identity_audit.json',c.read(private/'cost_identity_audit.json'))
    result=dict(status='PASS_MANDATORY_STOP',SOURCE=plan['SOURCE'],system=plan['system'],registered_candidates=len(rows),completed_candidates=sum(x['status']=='COMPLETE' for x in rows),
        new_signal_candidates=stage_results['SIGNAL']['new_candidate_signals'],signal_actual=stage_results['SIGNAL']['actual_calls'],
        qualification_endpoints=len(endpoints),new_compile_candidates=len(new),new_compile_axes=2*len(new),
        used_saved_compiled_pairs=stage_results['COST']['reused_pairs'],archives_verified=len(archives),native_gate_maxima=maxima,
        independent_integer_ceil_checks=integer_checks,integer_costs_and_Decimal_ratios_PASS=True,all_native_workers_exit0_reaped=True,
        retry=False,grant_reuse=False,PR_new_sampling=0,PR_new_prices_or_statistics=0,source_files_modified=False,
        detailed_scientific_review=False,external_reproduction=False,GitHub_push=False,mandatory_STOP=True)
    save('end_audit.json',result);save('MANDATORY_STOP.json',dict(status='MANDATORY_STOP',reason='Approved system acquisition and fixed PR/PF comparison finished',next_stage_started=False))
    save('control_inventory.json',files)
    save('file_inventory.json',dict(excludes_self=True,files=[dict(path=str(p.relative_to(output)),bytes=p.stat().st_size,sha256=c.sha(p)) for p in sorted(output.iterdir()) if p.is_file()]))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--private',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(c.encoded(audit(a.private,a.output)).decode())
