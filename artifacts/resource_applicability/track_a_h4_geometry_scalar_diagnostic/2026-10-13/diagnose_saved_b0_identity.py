"""Source-gated scalar-only B0+I sensitivity. No chemistry/array/quantum imports."""
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import signal
import struct
import subprocess
import time
import zipfile
from collections import defaultdict

SOURCE = 'd0726e946f89f8c30a08d383fb8ff48b363629ec'
BASE = '7c4483378047b57f514c47dad98905e3f2a9a756'
MAX_MEMBER = 8192


def require(value, reason):
    if not value:raise ValueError(reason)


def scalar_member(archive, name):
    info = archive.getinfo(name+'.npy')
    require(info.file_size <= MAX_MEMBER, 'member cap')
    raw = archive.read(info)
    require(raw[:8] == b'\x93NUMPY\x01\x00', 'NPY version')
    n = int.from_bytes(raw[8:10], 'little')
    require(n <= 2048 and 10+n <= len(raw), 'NPY header size')
    header = ast.literal_eval(raw[10:10+n].decode('ascii'))
    require(set(header) == {'descr','fortran_order','shape'}, 'NPY header keys')
    require(header['descr'] == '<f8' and header['fortran_order'] is False, 'NPY real C-order f8')
    require(header['shape'] == {'G':(12,8,8),'lambdas':(12,),'energy':()}[name], 'NPY shape')
    count = math.prod(header['shape'])
    require(len(raw)-10-n == 8*count, 'NPY payload bytes')
    values = struct.unpack('<'+str(count)+'d',raw[10+n:])
    require(all(math.isfinite(x) for x in values), 'finite coefficients')
    return values, {'member':info.filename, 'bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest(),
                    'shape':header['shape'], 'dtype':header['descr']}


def coefficient(lam, trace, trace_square):
    # Source diagonal_coefficients: lam/2 sum eta^2 + lam/2 sum_{i<j} eta_i eta_j.
    # G here means the symmetric matrix defined by saved lower triangle,
    # exactly the triangle consumed by the source's eigh(..., UPLO='L').
    # lam/4 [(Tr G)^2 + Tr(G^2)] needs no eigenvectors or new diagonalization.
    return lam/4*(trace*trace+trace_square)


def rotate(z, time_value, tail):
    theta = -time_value*tail
    return z*complex(math.cos(theta),math.sin(theta))


def artificial_checks():
    count = 0
    for eta in [(1.,2.),(-2.,.5,3.),(0.,0.),(.2,-.7,1.3)]:
        for lam in (.7,-.4):
            oracle = lam*math.fsum(x*x for x in eta)/2
            oracle += math.fsum(lam*eta[i]*eta[j]/2 for i in range(len(eta)) for j in range(i+1,len(eta)))
            require(math.isclose(oracle,coefficient(lam,math.fsum(eta),math.fsum(x*x for x in eta)),abs_tol=1e-14),'identity formula')
            count += 1
    z = complex(math.cos(.3),math.sin(.3))
    require(abs(rotate(z,1.,.2)-complex(math.cos(.1),math.sin(.1)))<1e-14,'phase sign');count += 1
    require(abs(abs(rotate(z,.8,-.7))-abs(z))<1e-14,'phase modulus');count += 1
    require(rotate(z,.8,0.)==z,'zero scalar');count += 1
    G2,G0,N=200.,140.,30
    require(G0+N*((G2-G0)/N)==G2,'common additional-cost break-even');count += 1
    eta=(2-math.sqrt(5),2+math.sqrt(5))
    oracle=.7*math.fsum(x*x for x in eta)/2+.7*eta[0]*eta[1]/2
    require(math.isclose(oracle,coefficient(.7,4.,18.),abs_tol=1e-14),'saved lower triangle convention');count += 1
    return count


def sha_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(1024*1024):h.update(block)
    return h.hexdigest()


def main():
    resource.setrlimit(resource.RLIMIT_AS,(256*2**20,256*2**20))
    resource.setrlimit(resource.RLIMIT_CPU,(45,45));signal.alarm(60)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--raw-run',type=Path,required=True)
    parser.add_argument('--supplied-review',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();started=time.monotonic()
    require(args.output.resolve().is_relative_to(Path('/home/AbeHiromu')),'home output only')
    args.output.mkdir(exist_ok=False)
    repo=args.repo
    pub=repo/'artifacts/resource_applicability/track_a_h4_geometry_results/2026-10-13'
    planpath=repo/'artifacts/resource_applicability/track_a_h4_production_resume02/2026-10-11/plan_authorized_v1.json'
    plan=json.loads(planpath.read_text());require(plan['source_commit']==SOURCE,'fixed science source')
    values=json.loads((pub/'saved_candidates_v1.json').read_text())
    summary=json.loads((pub/'summary.json').read_text())
    require(summary['fixed_review_artifact']=='81c78b366d2cf2c699bb31977f72ae9deab4f1d4','fixed review')
    spec=importlib.util.spec_from_file_location('saved_arithmetic',pub/'verify_saved_result_projection.py')
    arithmetic=importlib.util.module_from_spec(spec);spec.loader.exec_module(arithmetic)
    groups=defaultdict(list)
    for row in values['candidates']:
        row['_statistics']=arithmetic.statistics(row['saved']['paired_costs']);groups[row['geometry']].append(row)
    supplied=json.loads((args.supplied_review/'selected_points_arithmetic_audit.json').read_text())
    require(len(supplied['rows'])==18,'provided arithmetic coverage')
    for point in supplied['rows']:
        official=next(x for x in summary['selected_precision_comparisons'] if float(x['geometry_A'])==point['geometry'] and x['epsilon']==point['epsilon'])
        for k,other in [('RZ_work','G'),('SE','SE'),('best_deterministic_RZ_work','G_det'),('normalization','B')]:
            arithmetic.near(official[k],point[other])
        require(official['axis_shots']==[point['N_cos'],point['N_sin']],'review/source shot agreement')
    sources=[]
    for short in ('signal.py','circuits.py','inputs.py','execution.py'):
        relative='src/trottertracks/resource_applicability/h4_geometry/'+short
        blob=subprocess.check_output(['git','-C',str(repo),'show',SOURCE+':'+relative])
        require(blob==(repo/relative).read_bytes(),'science source changed')
        sources.append({'path':relative,'commit':SOURCE,'sha256':hashlib.sha256(blob).hexdigest()})
    input_audit=[];scalar_values=[];b0_signals=[];auxiliary=defaultdict(list)
    raw_bytes_read=0
    for g,rows in sorted(groups.items()):
        record=plan['inputs'][g];inp=Path(plan['input_root'])/record['file']
        stat_before=inp.stat();digest=sha_file(inp);raw_bytes_read+=stat_before.st_size
        require(digest==record['bytes_sha256'],'frozen input byte SHA mismatch')
        with zipfile.ZipFile(inp) as archive:
            require(len(set(archive.namelist()))==len(archive.namelist()),'duplicate NPZ member')
            G,Gmeta=scalar_member(archive,'G');lams,Lmeta=scalar_member(archive,'lambdas');energy,Emeta=scalar_member(archive,'energy')
        require(inp.stat().st_mtime_ns==stat_before.st_mtime_ns and inp.stat().st_size==stat_before.st_size,'input changed')
        identity=values['geometry_identities'][g]
        require(all(identity[k]==record[k] for k in ('H','DF','state','input')),'input identity match')
        coeffs=[];asym=0.;weighted_asym=0.
        for k,lam in enumerate(lams):
            at=lambda i,j:G[k*64+i*8+j]
            asym=max(asym,max(abs(at(i,j)-at(j,i)) for i in range(8) for j in range(8)))
            weighted_asym=max(weighted_asym,abs(lam)*max(abs(at(i,j)-at(j,i)) for i in range(8) for j in range(8)))
            trace=math.fsum(at(i,i) for i in range(8))
            trace2=math.fsum(at(i,i)**2 for i in range(8))+2*math.fsum(at(i,j)**2 for i in range(8) for j in range(i))
            coeffs.append(coefficient(lam,trace,trace2))
        input_audit.append({'geometry':g,'file':record['file'],'bytes':stat_before.st_size,'sha256':digest,
                            'input_identity':identity,'read_members':[Gmeta,Lmeta,Emeta],
                            'G_max_asymmetry':asym,'G_max_abs_lambda_times_asymmetry':weighted_asym,
                            'lower_triangle_convention':'matches numpy.linalg.eigh UPLO=L; does not assume saved upper/lower agreement',
                            'H_blocks_states_MOs_unread':True})
        tails={str(p):math.fsum(coeffs[p:]) for p in (0,3,4,5,6,9,12)}
        scalar_values.append({'geometry':g,'energy':energy[0],'T':.8,'DF_fragment_identity_coefficients':coeffs,
                              'tail_identity_by_prefix':tails,'units':'Hartree, atomic time',
                              'rounding_limit':'trace identity exact in real arithmetic; no bitwise eigensolver reconstruction or rigorous numerical certificate'})
        target=complex(math.cos(-.8*energy[0]),math.sin(-.8*energy[0]))
        for row in rows:
            if not row['template_id'].startswith(('B0-','PM1-B0-')):continue
            tid=row['template_id'];path=args.raw_run/f'signal-{g}-{tid}.json'
            raw=path.read_bytes();raw_bytes_read+=len(raw)
            require(hashlib.sha256(raw).hexdigest()==row['original_signal_sha256'] and len(raw)==row['original_signal_bytes'],'raw/public signal SHA match')
            d=json.loads(raw);require(d['identity']['input']==record['input'],'raw input binding')
            require(d['signal']['normalization']==d['saved']['normalization']==1.,'B0 normalization')
            z=complex(*d['signal']['corrected']);require(d['signal']['raw']==d['signal']['corrected'],'det raw/corrected')
            old_bias={a:abs(x-y) for a,x,y in [('cosine',z.real,target.real),('sine',z.imag,target.imag)]}
            for a in old_bias:
                require(math.isclose(old_bias[a],row['saved']['bias'][a],rel_tol=1e-10,abs_tol=1e-12),'saved target/bias mismatch')
            rank=int(tid.split('-rank')[1].split('-')[0]);tail=tails[str(rank)]
            zi=rotate(z,.8,tail)
            require(abs(abs(zi)-abs(z))<1e-12,'unit scalar preserves modulus')
            new_bias={'cosine':abs(zi.real-target.real),'sine':abs(zi.imag-target.imag)}
            aux={'geometry':g,'template_id':tid+'+I','original_template_id':tid,
                 'saved':{'bias':new_bias,'normalization':1.,'paired_costs':row['saved']['paired_costs']},
                 '_statistics':row['_statistics']}
            auxiliary[g].append(aux)
            b0_signals.append({'geometry':g,'original_template_id':tid,'prefix':rank,
                'original_signal_sha256':row['original_signal_sha256'],
                'target':[target.real,target.imag],'saved_corrected_signal':[z.real,z.imag],
                'scalar_phase_counterfactual':[zi.real,zi.imag],'tail_identity':tail,
                'phase_angle':-.8*tail,'original_bias':row['saved']['bias'],'counterfactual_bias':new_bias,
                'normalization_unchanged':1.,'compiled_cost':'UNMEASURED: old B0 cost used only as sensitivity intercept'})
    grid=sorted(set([.005*20**(i/300) for i in range(301)]+[.05]));grid[0]=.005;grid[-1]=.1
    def best(rows,epsilon):
        pairs=[(r,arithmetic.evaluate(r,epsilon)) for r in rows]
        return min(((r,p) for r,p in pairs if p is not None),key=lambda x:x[1]['work'],default=None)
    envelope=[];primary=[]
    for g,rows in sorted(groups.items()):
        b2=[r for r in rows if r['template_id'].startswith('B2-')]
        b0=[r for r in rows if r['template_id'].startswith(('B0-','PM1-B0-'))]
        for e in grid:
            x2=best(b2,e);x0=best(b0,e);xi=best(auxiliary[g],e)
            r2,p2=x2
            d={'geometry':g,'epsilon':e,'B2_template':r2['template_id'],'B2_G':p2['work'],'B2_SE':p2['SE'],
               'B2_N_total':sum(p2['shots']),'original_B0_template':x0[0]['template_id'] if x0 else None,
               'original_B0_G':x0[1]['work'] if x0 else None,
               'B0_I_reference_template':xi[0]['template_id'] if xi else None,
               'B0_I_reference_G_zero_extra':xi[1]['work'] if xi else None,
               'B0_I_N_total':sum(xi[1]['shots']) if xi else None,
               'common_extra_RZ_per_shot_break_even':(p2['work']-xi[1]['work'])/sum(xi[1]['shots']) if xi else None,
               'zero_extra_cost_reference_below_B2':bool(xi and xi[1]['work']<p2['work'])}
            envelope.append(d)
            if e in (.005,.05,.1):
                extra={**d,'B2_axis_shots':p2['shots'],'B0_I_axis_shots':xi[1]['shots'] if xi else None,
                       'B0_I_effective_reference_one_shot_RZ':xi[1]['work']/sum(xi[1]['shots']) if xi else None}
                primary.append(extra)
    potential=[r for r in envelope if r['zero_extra_cost_reference_below_B2']]
    tests=artificial_checks()
    decision={'status':'SCIENTIFIC_DECISION_STOP_B0_IDENTITY_SENSITIVITY' if potential else 'B0_IDENTITY_REFERENCE_NO_GRID_REVERSAL',
              'original_registered_result_unchanged':True,'B0_I_actual_compiled_cost':'UNMEASURED',
              'counterfactual_cost_intercept':'old B0 C_cos/C_sin; not a measured B0+I cost',
              'cost_model':'G_I = N_I_cos*(C_old_cos+delta_cos) + N_I_sin*(C_old_sin+delta_sin)',
              'common_delta_threshold':'(G_B2 - G_I(delta=0)) / (N_I_cos+N_I_sin)',
              'zero_extra_reference_reversals':len(potential),'evaluated_geometry_precision_points':1812,
              'reversals_by_geometry':{g:sum(r['zero_extra_cost_reference_below_B2'] for r in envelope if r['geometry']==g) for g in groups},
              'representative_18_points':primary,'artificial_scalar_checks_passed':tests,
              'GPT_18_point_arithmetic_matches_original_saved_summary':True,
              'new_quantum_signal_trajectory_eigensolver_Hamiltonian_circuit_transpile':0,
              'scalar_phase_operations_on_saved_complex_signals':len(b0_signals),
              'large_scientific_array_reads':0,'NPZ_member_data_limit':MAX_MEMBER,
              'raw_bytes_streamed_or_selected':raw_bytes_read,'H8_worktree_runner_plan_grant_changed':False,
              'A_C_D_full_analysis_and_manuscript_integration':'NOT_RUN_AFTER_PRIORITY_B_STOP' if potential else 'PENDING',
              'next_science_authorized':False,'old_grants_or_budget_reused':False,
              'runtime_seconds':time.monotonic()-started,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    require(raw_bytes_read<128*2**20,'analysis I/O cap')
    outputs={'diagnostic_summary_v1.json':decision,'source_input_provenance_v1.json':{'base_commit':BASE,'sources':sources,
         'public_projection_sha256':sha_file(pub/'saved_candidates_v1.json'),'public_summary_sha256':sha_file(pub/'summary.json'),
         'authorized_plan_sha256':sha_file(planpath),'inputs':input_audit},
         'tail_scalar_metadata_v1.json':scalar_values,'saved_b0_phase_sensitivity_v1.json':b0_signals}
    for name,value in outputs.items():
        (args.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (args.output/'scalar_reference_envelope_1812.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(envelope[0]),lineterminator='\n');w.writeheader();w.writerows(envelope)
    require(sum(x.stat().st_size for x in args.output.iterdir())<8*2**20,'analysis output cap')
    print(json.dumps({'status':decision['status'],'reference_reversals':len(potential),'by_geometry':decision['reversals_by_geometry'],
                      'scalar_tests':tests,'runtime_seconds':decision['runtime_seconds'],'peak_RSS_KiB':decision['peak_RSS_KiB']},ensure_ascii=False))


if __name__=='__main__':main()
