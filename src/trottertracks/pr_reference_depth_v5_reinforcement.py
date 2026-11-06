"""Fixed V5 reinforcement: actual SOURCE inverses and X-free lower chains."""
from __future__ import annotations
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import hashlib,inspect,io,json,lzma,os,platform,subprocess,sys
from importlib.metadata import version
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import CXGate,XGate,XXPlusYYGate,RZGate,RZZGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import InverseCancellation
from trotterlib.df_partial_s2 import QiskitDFPartialS2CircuitBuilder
from .track_c_h4 import ROOT,common,save_json
from .track_c_compiler_a import compile_reference,PROFILE
from .pr_reference_transfer_catalogue import fence
from . import pr_reference_depth_v5 as old

BASE='285b77023542c3b39caea767e2a937dffa046a3b'
OUT=ROOT/'artifacts/pr_reference_depth_v5_reinforcement/2026-11-07'
V5='artifacts/pr_reference_depth_v5/2026-11-07/'
BLOCKS=((0,.01),(1,.01),(2,.01),(3,.01),(0,.02))
CONTEXTS=((False,False),(False,True),(True,False),(True,True))
def data(path):return subprocess.check_output(['git','show',BASE+':'+path],cwd=ROOT)
def attach():
    if (OUT/'STOP.json').exists():raise RuntimeError('Mandatory STOP')
    os.environ['TRACK_C_H4_COMMON_CACHE']=str(OUT/'common.pkl')
def initialize():
    if OUT.exists():raise RuntimeError('Fresh reinforcement output required')
    OUT.mkdir(parents=True)
    raw=lzma.decompress(data(old.PILOT+'common.pkl.xz'));(OUT/'common.pkl').write_bytes(raw);attach()
    save_json(OUT/'INPUTS.json',dict(base=BASE,common_sha256=hashlib.sha256(raw).hexdigest(),
        new_Hamiltonian_ground_PF_law=0,old_outputs_read_only=True))
    save_json(OUT/'ENVIRONMENT.json',dict(host=platform.node(),platform=platform.platform(),python=sys.version,
        executable=sys.executable,packages={p:version(p) for p in ('qiskit','numpy','scipy','mpmath')},
        affinity=len(os.sched_getaffinity(0)),meminfo=Path('/proc/meminfo').read_text(),
        cpu_pressure=Path('/proc/pressure/cpu').read_text(),memory_pressure=Path('/proc/pressure/memory').read_text(),
        profile=PROFILE,shared_settings_changed=False))
    save_json(OUT/'PREDECLARED.json',dict(stage_A=dict(blocks=[dict(block=k,time=t,direction='actual source QuantumCircuit.inverse') for k,t in BLOCKS],
        reference_contexts=CONTEXTS,inverse_records=20,compile_batches=4,
        comparison='actual inverse native span <= saved positive span, no integer tolerance',
        revision='max of saved G D span and validated F inverse D span; no new candidate'),
        stage_B=dict(contexts=CONTEXTS,records='all saved Bprime D/frame; flag exact V5 lower-bound dependencies',
        comparison='non-X protected subcircuit delta versus old delta',new_scientific_compile=0),
        stage_C='no all18 recomputation if A/B bounds unchanged; otherwise substitute changed local bounds only',
        new_trajectory_whole_compile_sampling_simulation_synthesis=0,review_text='not provided at known path/base tree'))

@lru_cache(None)
def saved():return json.loads(lzma.decompress(data(V5+'LOCAL_FRONTIERS.json.xz')))
@lru_cache(None)
def totals():return json.loads(data(V5+'DEPTH_18ROUND.json'))

def instructions(q):
    return [[i.operation.name,[q.find_bit(z).index for z in i.qubits],
        [str(v) if getattr(v,'parameters',None) else float(v) for v in i.operation.params]] for i in q.data]
def source_key(seq):
    return [(g.name,tuple(qs),tuple(float(x).hex() for x in g.params)) for g,qs in seq]

def inverse_source_checks():
    attach();p=common()['preps'][3];definitions={f'D{k}':b.runtime_basis_operations for k,b in enumerate(p.deterministic_blocks)}
    definitions.update({bid:d.runtime_operations for bid,d in p.rte_preparation.basis_registry._definitions.items()})
    checks=[]
    for name,seq in definitions.items():
        assert set(g.name for g,_ in seq)<={'rz','xx_plus_yy'}
        inv=[(g.inverse(),qs) for g,qs in reversed(seq)]
        twice=[(g.inverse(),qs) for g,qs in reversed(inv)]
        assert source_key(seq)==source_key(twice)
        for arm in ('Bprime','reference'):
            assert 'basis:'+name in saved()['records'][arm] and 'basis_inverse:'+name in saved()['records'][arm]
        checks.append(dict(basis=name,source_gate_families=sorted(set(g.name for g,_ in seq)),
            inverse_key=source_key(inv),double_inverse_exact_binary_parameters=True,saved_both_directions=True))
    functions={'QuantumCircuit.inverse':QuantumCircuit.inverse,'XXPlusYYGate.inverse':XXPlusYYGate.inverse,
        'RZGate.inverse':RZGate.inverse,'RZZGate.inverse':RZZGate.inverse}
    source={k:dict(file=inspect.getsourcefile(f),source_sha256=hashlib.sha256(inspect.getsource(f).encode()).hexdigest(),
        first_line=inspect.getsourcelines(f)[1],lines=len(inspect.getsourcelines(f)[0])) for k,f in functions.items()}
    save_json(OUT/'SOURCE_INVERSE_CHECKS.json',dict(basis_closure=checks,installed_Qiskit_primary_functions=source,
        operation_semantics='reverse SOURCE order, invert individual SOURCE operations; not inverse compiled native',
        q1='F is reverse D half sequence, so F inverse uses the same 4 inverse D blocks in reversed order',
        fused_lanes='source inverse reverses existing fused D0 full blocks and half endpoints; multiplicities unchanged',passed=True))
    return checks

def stage_b():
    attach();_,contexts=old.catalogues();prior=saved()['records']['Bprime']
    bids=set().union(*(set(row['axes'][axis]['depth_per_shot']['tail_Bprime']['basis_probabilities'])
        for row in totals()['rows'] for axis in 'XY'))
    used={'left_all','left_no0','right_all','right_no0','boundary_control'}|{prefix+b for b in bids for prefix in ('basis:','basis_inverse:')}
    checks=[];changes={}
    for row in contexts:
        seen=set()
        for rec,pre in zip(row['records'],row['pre_last_cancel']):
            if rec['meta']['kind'] not in ('basis','basis_inverse','baseline_D','D_half','D_full'):continue
            x=[dict(index=i,wires=qs) for i,(name,qs,_) in enumerate(rec['lowering']) if name=='x']
            protected=[i for i in rec['lowering'] if i[0]!='x'];s=old.stats(dict(rec,lowering=protected))
            actual=old.stats(rec);assert actual['matrix']==prior[rec['name']]['matrix']
            selected=rec['name'] in used
            if selected:
                seen.add(rec['name']);assert not pre['exposed_cancelable_prefix'] and not pre['exposed_cancelable_suffix']
                assert s['system_min_progress']<=actual['system_min_progress']
                value=s['system_min_progress'];changes.setdefault(rec['name'],value);assert changes[rec['name']]==value
            checks.append(dict(context=row['request'],name=rec['name'],used_in_Bprime_lower=selected,
                local_final_X_positions=x,old_delta=actual['system_min_progress'],protected_nonX_delta=s['system_min_progress'],
                local_final_matrix=actual['matrix'],protected_nonX_matrix=s['matrix'],pre_last_fence=pre,
                pre_last_internal_X_count='NOT_RECORDED: not inferred from post-final absence',passed=True))
        assert seen==used
    revised=any(changes[k]!=prior[k]['system_min_progress'] for k in used)
    result=dict(status='REVISED_BOUND' if revised else 'PASS',checks=checks,used_record_names=sorted(used),
        delta_map=changes,all_checked_local_final_records_X_free=all(not x['local_final_X_positions'] for x in checks),
        proof='CX pass acts before X. Fences prevent cross-chunk CX deletion. Later X pass cannot delete retained non-X nodes; their wire-order subsequence contains the saved X-free local chain. Extra X can only add depth.',
        initial_source_pass='SOURCE Gaussian frames contain only RZ/XXPlusYY; directional D CX are internal and frames separate chunk boundaries; old Bprime SOURCE lemma reused',
        no_whole_cohort_generalization=True,new_scientific_compile=0,passed=True)
    save_json(OUT/'STAGE_B.json',result);return result

def source_blocks():
    p=common()['preps'][3];builder=QiskitDFPartialS2CircuitBuilder();blocks=[]
    for k,t in BLOCKS:
        q=QuantumCircuit(8);req=SimpleNamespace(preparation=p,controlled=False,ancilla_qubit=None,step_time=2*t)
        builder._append_block(q,p.deterministic_blocks[k],req,evolution_time=t)
        name=f'block:{k}:'+('full' if t==.02 else 'half')
        blocks.append((name,q))
    return blocks

def stage_a_one(request):
    attach();rare,y=request;batch=QuantumCircuit(9);sources=[]
    for name,forward in source_blocks():
        inverse=forward.inverse();assert instructions(inverse.inverse())==instructions(forward)
        batch.compose(inverse,qubits=range(8),inplace=True);batch.barrier()
        sources.append(dict(name=name,forward_SOURCE=instructions(forward),inverse_SOURCE=instructions(inverse),
            forward_global_phase=float(forward.global_phase),inverse_global_phase=float(inverse.global_phase)))
    # Same finite reference post-unroll families as the fixed catalogue.
    batch.h(0);batch.cswap(0,1,2);batch.p(Parameter('symbolic_correction'),0)
    if rare:batch.z(0)
    if y:batch.sdg(0)
    batch.barrier();trace={}
    def callback(**kw):
        name=type(kw['pass_']).__name__;dag=kw['dag']
        if name=='UnrollCustomDefinitions':trace['post_unroll']=sorted(set(dag.count_ops())-{'barrier'})
        if name=='BasisTranslator':
            slices=[];current=[]
            for node in dag.topological_op_nodes():
                if node.op.name=='barrier':slices.append(current);current=[]
                else:current.append([node.op.name,[dag.find_bit(z).index for z in node.qargs],[]])
            assert len(slices)==6 and not current
            trace['pre_last_fences']=[fence(s,8) for s in slices[:5]]
            trace['pre_last_matrices']=[old.transfer_matrix(s,8) for s in slices[:5]]
    native=compile_reference(batch,callback=callback);slices=[];current=[];locals=[]
    for inst in native.data:
        if inst.operation.name=='barrier':slices.append(current);current=[]
        else:current.append(inst)
    assert len(slices)==6 and not current
    old_context=next(row for row in old.catalogues()[0] if row['request']==['reference',True,rare,y])
    assert trace['post_unroll']==old_context['post_unroll']
    for src,insts,pre_m,pre_f in zip(sources,slices[:5],trace['pre_last_matrices'],trace['pre_last_fences']):
        q=QuantumCircuit(8)
        for inst in insts:q.append(inst.operation,[native.find_bit(z).index for z in inst.qubits])
        rec=dict(name=src['name'],n=8,lowering=instructions(q),ops=dict(q.count_ops()))
        s=old.stats(rec);d,levels=old.frontier_depth(q);assert s['span']==q.depth()==d
        previous=saved()['records']['reference'][src['name']]
        locals.append(dict(**src,native=rec,pre_last_matrix=pre_m,pre_last_fence=pre_f,
            inverse_frontier=s,positive_saved_span=previous['span'],span_upper_holds=s['span']<=previous['span'],
            positive_saved_matrix=previous['matrix'],matrix_equal_to_positive=s['matrix']==previous['matrix'],
            native_Qiskit_depth=d,wire_frontier=levels))
    result=dict(context=request,post_unroll=trace['post_unroll'],records=locals,
        inverse_records=5,compile_batches=1,forward_recompiles=0,whole_compile=0)
    save_json(OUT/'stage_A_raw'/f'rare{int(rare)}_Y{int(y)}.json',result)
    return result

def stage_a():
    attach();assert json.loads((OUT/'PREDECLARED.json').read_text())['stage_A']['compile_batches']==4
    if (OUT/'stage_A_raw').exists():raise RuntimeError('Already started: no automatic duplicate compile')
    with ProcessPoolExecutor(max_workers=min(len(CONTEXTS),len(os.sched_getaffinity(0)))) as pool:rows=list(pool.map(stage_a_one,CONTEXTS))
    values={}
    for row in rows:
        for rec in row['records']:
            name=rec['name'];s=rec['inverse_frontier']['span']
            values[name]=max(values.get(name,0),s)
    result=dict(status='PASS' if all(r['span_upper_holds'] for row in rows for r in row['records']) else 'REVISED_BOUND',
        inverse_span_upper=values,contexts=[r['post_unroll'] for r in rows],actual_inverse_records=20,
        compile_batches=4,old_positive_spans={k:saved()['records']['reference'][k]['span'] for k in values},
        all_inverse_matrices_equal_positive=all(r['matrix_equal_to_positive'] for row in rows for r in row['records']),
        rows=rows,passed=True)
    save_json(OUT/'STAGE_A.json',result);return result

def synthetic_tests():
    attach();checks=[]
    q=QuantumCircuit(2);q.x(0);q.cx(0,1);q.cx(0,1);q.x(0);q.rz(.17,0)
    mid=PassManager([InverseCancellation([CXGate()])]).run(q)
    full=PassManager([InverseCancellation([CXGate(),XGate()])]).run(q)
    assert mid.count_ops().get('cx',0)==0 and mid.count_ops().get('x',0)==2
    assert full.count_ops()=={'rz':1}
    checks.append(dict(name='CX deletion exposes X-X cancellation',before=instructions(q),
        after_CX=instructions(mid),after_CX_then_X=instructions(full),passed=True))
    q=QuantumCircuit(2);q.rz(.17,0);q.cx(0,1);q.rz(.23,1)
    inv=q.inverse();expected=[['rz',[1],[-.23]],['cx',[0,1],[]],['rz',[0],[-.17]]]
    assert instructions(inv)==expected and instructions(inv.inverse())==instructions(q)
    checks.append(dict(name='SOURCE inverse reverses order and angles',forward=instructions(q),inverse=instructions(inv),passed=True))
    # Retaining only non-X nodes is deliberately conservative. Inserting
    # extra X may increase a path, but an X pass cannot erase retained RZ/SX.
    pure=[['rz',[0],[.1]],['cx',[0,1],[]],['rz',[1],[.2]]]
    decorated=[['x',[0],[]],*pure[:1],['x',[1],[]],*pure[1:]]
    kept=[x for x in decorated if x[0]!='x'];assert kept==pure
    for start in ([0,0],[11,3],[2,17]):
        p=old.apply_ops(pure,start);d=old.apply_ops(decorated,start);assert all(x<=y for x,y in zip(p,d))
    checks.append(dict(name='non-X chain subsequence lower bound',protected=pure,with_extra_X=decorated,passed=True))
    save_json(OUT/'SYNTHETIC_TESTS.json',dict(checks=checks,passed=True,synthetic_only=True,
        scientific_performance_claim=False,technical_pass_manager_calls=2))
    return checks

def stage_c():
    attach();a=json.loads((OUT/'STAGE_A.json').read_text());b=json.loads((OUT/'STAGE_B.json').read_text())
    assert a['passed'] and b['passed'] and json.loads((OUT/'SYNTHETIC_TESTS.json').read_text())['passed']
    prior=totals();original=saved()['records'];changed=a['status']=='REVISED_BOUND' or b['status']=='REVISED_BOUND'
    mappings=[]
    for row in prior['rows']:
        mappings.append(dict(m=row['m'],q=row['q'],r=row['r'],bits=row['bits'],
            q1_special_case=row['q']==1,positive_G_supports_reference_lower_independently=True,
            exact_price_and_shot_records='unchanged parent DEPTH_18ROUND.json rows/axes',
            F_inverse_D_support='STAGE_A actual compiled inverse local spans',
            Bprime_lower_support='STAGE_B retained non-X D/frame chains'))
    if not changed:
        assert all(a['inverse_span_upper'][k]<=a['old_positive_spans'][k] for k in a['inverse_span_upper'])
        assert all(d==original['Bprime'][k]['system_min_progress'] for k,d in b['delta_map'].items())
        result=dict(status='MAINTAINED',new_all18_recomputations=0,old_depth_totals=prior['depth_totals'],
            old_comparison=prior['comparison'],mapping=mappings,
            qualification='Only the two reviewed applicability conditions reinforced; fixed-model limitations remain',
            existing_CX_RZ_shots_errors_changed=False,passed=True)
        save_json(OUT/'STAGE_C.json',result);return result
    # Conditional repair only. Reuse old finite probabilities, shots and
    # rational lane penalty; substitute changed local bounds, no optimizer.
    accum={arm:{end:F() for end in ('lower','upper')} for arm in ('Bprime','reference')};rows=[]
    inv=a['inverse_span_upper'];pos=a['old_positive_spans'];new=b['delta_map']
    orig={k:v['system_min_progress'] for k,v in original['Bprime'].items()}
    def lane(q,values):
        half=[values[f'block:{k}:half'] for k in range(4)]
        if q==1:return sum(half)
        n=q//2;return 2*n*sum(half)+(n-1)*(values['block:0:full']-2*half[0])
    def bd(q,values):
        return values['left_all']+values['right_all']+(q-1)*(values['left_no0']+values['right_no0']+values['boundary_control'])
    for row in prior['rows']:
        q=row['q'];r=row['r'];axes={}
        for axis,e in row['axes'].items():
            p=e['depth_per_shot'];tail=p['tail_Bprime'];p2=F(tail['p2']);apps=r*(1+2*p2)
            low_new=sum((new['basis:'+bid]+new['basis_inverse:'+bid])*(F(pb)+(apps-1)*F(pb)*(1-F(pb)))
                for bid,pb in tail['basis_probabilities'].items())
            lo=F(p['Bprime']['lower'])+bd(q,new)-bd(q,orig)+q*(low_new-F(tail['lower']))
            ru=F(p['reference']['upper'])+max(0,lane(q,inv)-lane(q,pos))
            bounds={'Bprime':{'lower':lo,'upper':F(p['Bprime']['upper'])},
                'reference':{'lower':F(p['reference']['lower']),'upper':ru}}
            for arm in bounds:
                for end in bounds[arm]:accum[arm][end]+=e['shots'][arm]*bounds[arm][end]
            axes[axis]=dict(shots=e['shots'],bounds={arm:{end:str(v) for end,v in ends.items()} for arm,ends in bounds.items()})
        rows.append(dict(m=row['m'],q=q,r=r,bits=row['bits'],axes=axes))
    import math
    def encode(v):return dict(exact_fraction=str(v),lower=math.nextafter(float(v),-math.inf),upper=math.nextafter(float(v),math.inf))
    margin=accum['Bprime']['lower']-accum['reference']['upper'];percent=100*margin/accum['Bprime']['lower']
    result=dict(status='REVISED_BOUND' if margin>0 else 'UNDETERMINED',new_all18_recomputations=1,
        depth_totals={arm:{end:encode(v) for end,v in ends.items()} for arm,ends in accum.items()},
        comparison=dict(reference_upper_below_Bprime_lower=margin>0,absolute_saving_lower=encode(margin),relative_saving_percent_lower=encode(percent)),
        rows=rows,mapping=mappings,existing_CX_RZ_shots_errors_changed=False,passed=True)
    save_json(OUT/'STAGE_C.json',result);return result

def finish_stop():
    a=json.loads((OUT/'STAGE_A.json').read_text());b=json.loads((OUT/'STAGE_B.json').read_text());c=json.loads((OUT/'STAGE_C.json').read_text())
    save_json(OUT/'STOP.json',dict(status='MANDATORY_STOP',reason='Authorized V5 reinforcement Stage A/B and conditional C complete',
        stage_A=a['status'],stage_B=b['status'],stage_C=c['status'],
        old_SOURCE_results_STOPs_preserved=True,next_owner='GPT/user: scientific judgment; no automatic optimization or synthesis'))
