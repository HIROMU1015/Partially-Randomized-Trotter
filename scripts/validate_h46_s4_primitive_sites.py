"""One-shot negative-stage primitive validation, maximum 66 producer calls.

No circuit builder, gate counter, trajectory, signal, SCF/DF/state generation,
Qiskit, or compiler is invoked. Frozen codec descriptions are read as metadata.
The original implementations remain unmodified.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = "b7385211e789ce9e9ab9515f3d5779fbd117bfb6"
SOURCE = "536a3671c0ec2939ff3886e40467db26a27ec65a"
PROVIDER = "217c971dd537b1cc7fbc13742bc0f6c263ff7f45"
CONFIG = {
    "H4": dict(prefix=12, n=8, q=1, preparation="05d54088dfdb8651a080105a6293d80834e4e4299d7abc788e8a10a1acaab829",
               bytes=1450047, sha256="f9618a6fda0c28bccfc415f95079ad137e846de12fbe2acaf74456d8369a2c48"),
    "H6": dict(prefix=19, n=12, q=8, preparation="d58d03efdb3f71f3e7de13decbad8acf6bf9b0bd806c984c6f18b5093a58f17e",
               bytes=5670237, sha256="ad52f9f6a3150981d7be00191d8b3a9f9af992c37c9102cba6d94fcffc8e3010"),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def checked_input(system, path):
    expected = CONFIG[system]
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("FROZEN_INPUT_PATH")
    raw = path.read_bytes()
    if len(raw)!=expected['bytes'] or sha(raw)!=expected['sha256']:
        raise ValueError("FROZEN_INPUT_BYTES_SHA")
    return json.loads(raw)


def selected_blocks(system, archive, metadata):
    expected = CONFIG[system]
    if system=="H4":
        archive = dict(schema="h6_frozen_df_preparations_v1", binding={}, codec=archive,
                       codec_digest=metadata.digest(archive), export_receipt={})
    graph = metadata.FrozenMetadataGraph(archive)
    fields = graph.fields(archive['codec']['roots'][f"faithful:p{expected['prefix']}"],"DFPartialS2Preparation")
    if (graph.scalar(fields['preparation_hash'])!=expected['preparation'] or
        graph.scalar(fields['num_system_qubits'])!=expected['n'] or
        graph.scalar(fields['deterministic_fragment_indices'])!=list(range(expected['prefix'])) or
        graph.scalar(fields['randomized_block_indices'])!=[]):
        raise ValueError("FULL_PREPARATION_BINDING")
    blocks = []
    for index, ref in enumerate(graph.sequence(fields['deterministic_blocks'])):
        node = graph.node(ref)
        one = node['type']=="DFDeterministicOneBodySpec"
        if not one and node['type']!="DFDeterministicFragmentSpec":
            raise ValueError("BLOCK_TYPE")
        b = node['fields']
        diagonal = graph.scalar(b['diagonal_eigenvalues'] if one else b['diagonal_eta'])
        lam = None if one else graph.scalar(b['lam'])
        if len(diagonal)!=expected['n'] or any(not math.isfinite(x) for x in diagonal):
            raise ValueError("DIAGONAL_DIMENSION")
        blocks.append(dict(index=index,one=one,basis_id=graph.scalar(b['basis_id']),
                           fragment=None if one else graph.scalar(b['original_fragment_index']),
                           node_sha256=ref['ref'], diagonal=diagonal, lam=lam))
    if (len(blocks)>expected['prefix']+1 or sum(b['one'] for b in blocks)>1 or
        [b['fragment'] for b in blocks if not b['one']]!=list(range(expected['prefix']))):
        raise ValueError("BLOCK_INVENTORY")
    return blocks


def primitive_receipt(pp,n):
    if not all(math.isfinite(float(x)) for x in [pp.global_phase,*[a for _,a in pp.rz],*[a for _,_,a in pp.rzz]]):
        raise ValueError('NONFINITE_PRIMITIVE')
    return dict(rz={str(q):float(angle).hex() for q,angle in pp.rz},
                zero_rz_qubits=[q for q in range(n) if q not in {p for p,_ in pp.rz}],
                nonzero_rzz_pairs=[[p,q] for p,q,angle in pp.rzz if angle!=0.],
                rzz_sha256=sha(canonical([[p,q,float(angle).hex()] for p,q,angle in pp.rzz])),
                phase_hex=float(pp.global_phase).hex(), phase_nonzero=bool(pp.global_phase!=0.))


def evaluate_system(system,blocks,f,reserve):
    c = CONFIG[system]
    mt = next(s.time for s in f.m.pf_schedule(1,.8,c['q'],"4th","symmetric_directional") if s.time<0.)
    nt = next(s.time for s in f.schedule.deterministic_control_plan(1,.8,c['q'],"4th") if s.time<0.)
    records = []
    for block in blocks:
        row = {k:v for k,v in block.items() if k not in ('diagonal','lam')}
        row['diagonal_input_sha256'] = sha(canonical([float(x).hex() for x in block['diagonal']]))
        row['lambda_hex'] = None if block['one'] else float(block['lam']).hex()
        for role, t in (("model",mt),("native",nt)):
            reserve(dict(system=system,block_index=block['index'],role=role,time_hex=t.hex()))
            data = f.ops.one_body_diagonal_primitives(f.ops.np.asarray(block['diagonal']),t) if block['one'] else \
                f.ops.df_squared_diagonal_primitives(f.ops.np.asarray(block['diagonal']),block['lam'],t)
            row[role] = primitive_receipt(data,c['n'])
        mm,nn = row['model'],row['native']
        row['cardinality_delta_native_minus_model'] = dict(
            rz=len(nn['rz'])-len(mm['rz']), rzz=len(nn['nonzero_rzz_pairs'])-len(mm['nonzero_rzz_pairs']),
            phase=int(nn['phase_nonzero'])-int(mm['phase_nonzero']))
        row['rz_presence_changed_qubits'] = sorted(int(q) for q in set(mm['rz'])^set(nn['rz']))
        records.append(row)
    net = {key:sum(r['cardinality_delta_native_minus_model'][key] for r in records) for key in ('rz','rzz','phase')}
    # Local difference reconciliation only. No complete circuit/theory count.
    da,db,dg = (net[key]*c['q'] for key in ('rz','rzz','phase'))
    delta = dict(raw=4*da+6*db+dg,rz=2*da+2*db+dg,cx=2*da+6*db,sx=0,x=0)
    return dict(system=system,model_negative_time_hex=mt.hex(),native_negative_time_hex=nt.hex(),
                blocks=records,net_per_outer_step=net,inferred_full_saved_delta=delta,
                q_repetitions_reused=c['q'],axes_reused=2,real_aligned_full_counter_evaluation=False)


def verify(plan):
    if (plan.get('approved') is not True or plan.get('runtime_authorization') is not True or
        plan.get('max_primitive_evaluations')!=66 or plan.get('automatic_retry') is not False):
        raise ValueError("SCOPE_NOT_AUTHORIZED")
    actual = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if actual!=plan['SOURCE']:
        raise ValueError("SOURCE_BINDING")
    for name,expected in plan['own_source_hashes'].items():
        if sha((ROOT/name).read_bytes())!=expected:
            raise ValueError("OWN_SOURCE_CHANGED")
    if (list(sys.version_info[:3])!=plan['environment']['python_version'] or
        importlib.metadata.version('numpy')!=plan['environment']['numpy_version']):
        raise ValueError('ENVIRONMENT_VERSION_BINDING')
    for path,expected in plan['environment']['file_sha256'].items():
        if sha(Path(path).read_bytes())!=expected:
            raise ValueError('ENVIRONMENT_FILE_BINDING')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text());verify(plan)
    out=Path(plan['output'])
    if not out.resolve().is_relative_to(Path.home()) or out.exists():
        raise ValueError("EXCLUSIVE_HOME_OUTPUT")
    out.mkdir()
    with (out/'one_shot_lock.json').open('x') as stream:
        json.dump(dict(SOURCE=plan['SOURCE'],plan_sha256=sha(args.plan.read_bytes()),grant_consumed=True),stream)
    terminal=dict(status='MANDATORY_STOP_FAILURE',first_stop=None,SOURCE=plan['SOURCE'],primitive_reservations=0,
                  compile=0,new_trajectory=0,full_wrapper_build=0,full_theoretical_recount=0,mandatory_stop=True)
    start=time.monotonic()
    try:
        audit=load_module('_site_prior_audit',ROOT/'scripts/audit_h46_s4_count_mismatch.py')
        f=audit.functions()
        expected=json.loads(audit.git_bytes(BASE,'artifacts/resource_applicability/track_a_h46_s4_count_mismatch/2026-10-18/source_closure.json'))['pinned_sources']
        for key,value in expected.items():
            if sha(f.source[key])!=value['sha256']:
                raise ValueError('ORIGINAL_SOURCE_CLOSURE')
        metadata=audit.load_definitions('_site_frozen_graph',audit.git_bytes(SOURCE,'src/trotterlib/native_df_archive_metadata.py'))
        blocks={s:selected_blocks(s,checked_input(s,plan['inputs'][s]),metadata) for s in CONFIG}
        planned=2*sum(len(v) for v in blocks.values())
        if planned>66:raise ValueError('PRIMITIVE_BUDGET')
        journal=out/'primitive_reservations.jsonl'
        def reserve(row):
            if terminal['primitive_reservations']>=66:raise ValueError('PRIMITIVE_BUDGET')
            with journal.open('a') as stream:
                stream.write(json.dumps(dict(index=terminal['primitive_reservations'],**row))+'\n')
                stream.flush();os.fsync(stream.fileno())
            terminal['primitive_reservations']+=1
        result={s:evaluate_system(s,blocks[s],f,reserve) for s in CONFIG}
        (out/'primitive_sites.json').write_bytes(json.dumps(result,indent=2,sort_keys=True).encode()+b'\n')
        for system in CONFIG:
            saved_delta=next(r['delta_native_minus_theory'] for r in audit.saved_evidence()['saved_cost_signatures'] if r['system']==system)
            if result[system]['inferred_full_saved_delta']!=saved_delta:
                raise ValueError('LOCAL_DIFFERENCES_DO_NOT_RECONCILE_SAVED_TOTALS')
        verify(plan)
        for system in CONFIG:checked_input(system,plan['inputs'][system])
        if any(k=='qiskit' or k.startswith(('qiskit.','cupy')) for k in sys.modules):
            raise ValueError('FORBIDDEN_IMPORT')
        terminal.update(status='COMPLETE_MANDATORY_STOP',planned_primitive_calls=planned,
                        inputs_and_source_verified_after=True,qiskit_imported=False)
    except Exception as exc:
        terminal['first_stop']=repr(exc)
        raise
    finally:
        terminal.update(wall_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        (out/'terminal.json').write_text(json.dumps(terminal,indent=2,sort_keys=True)+'\n')
        print(json.dumps(terminal),flush=True)


if __name__=='__main__':main()
