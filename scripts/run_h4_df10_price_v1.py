"""Fixed saved-data V1 campaign. No sampling, fit, circuits, or N/G calls.

Run develop first, then holdout with unchanged committed model/source. Outputs
are fresh files; original scientific inputs and old runs remain read-only.
"""
import argparse
import ast
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Tuple
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks.h4_df10_price_v1 import MODEL, predict

CARRIER = '155529e20a8fac39e00786c40b57830b71203c4f'
PROVIDER = 'c31ef424175c02379eafebe8fc6c7548c1d9aabe'
BASE = 'artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/'
COMPILER = '1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
PF = {8:'49709717a6c3d405c60d',16:'8e50c5560bb8f47f026f'}
PR = '14bd81a375400c9b0305'
PREPARATION = 'b99720d6189a9975a46f064ba895d2f83514671e668ce42c34a4d71e834568d7'
MODEL_FILES = ['src/trottertracks/h4_df10_price_v1.py', 'scripts/run_h4_df10_price_v1.py',
               'src/trottertracks/track_d_native_resources.py']
ORIGINAL_CONFIG = Path('/home/AbeHiromu/projects/h46-pr-exploration-execution/20261021/private/authorization_run01/H4_PR_EXPLORATION/config_APPROVED_PRIVATE.json')
PUBLIC = ROOT/'docs/research/h4_df10_price_v1_20261110'


def git(*args):
    return subprocess.check_output(['git',*args], cwd=ROOT)


def blob(ref, path):
    return git('show',ref+':'+path)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def saved(name):
    b = blob(CARRIER,BASE+name)
    return json.loads(gzip.decompress(b) if name.endswith('.gz') else b)


def reuse_functions(ref, path, names, namespace):
    """Reuse exact named function bodies; do not import historical launchers."""
    source = blob(ref,path).decode()
    nodes = [n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError('missing pinned function')
    # Avoid evaluating annotations belonging to unused chemistry/circuit APIs.
    future = ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future,*nodes],type_ignores=[]))
    exec(compile(module,ref+':'+path,'exec'),namespace)


def source_binding():
    source = git('rev-parse','HEAD').decode().strip()
    hashes = {}
    for path in MODEL_FILES:
        data = (ROOT/path).read_bytes()
        if data != blob(source,path):
            raise ValueError('uncommitted scientific model file: '+path)
        hashes[path] = digest(data)
    return source, hashes


def inputs_and_blocks():
    import numpy as np
    import qiskit
    from qiskit.circuit import ParameterExpression
    if (np.__version__,qiskit.__version__) != ('1.26.4','1.3.0'):
        raise ValueError('use saved NumPy/Qiskit runtime')
    provenance = saved('scientific_provenance.json')
    if provenance['provider_SOURCE'] != PROVIDER or provenance['compiler']['hash'] != COMPILER:
        raise ValueError('saved provider/compiler binding')
    cfg = json.loads(ORIGINAL_CONFIG.read_bytes())['systems']['H4']
    if cfg['binding'] != provenance['inputs']['H4']['binding']:
        raise ValueError('saved H4 scientific binding')
    hashes = {}
    for name, meta in cfg['inputs'].items():
        data = Path(meta['path']).read_bytes()
        if digest(data) != meta['sha256'] or len(data) != meta['bytes']:
            raise ValueError('saved input changed: '+name)
        hashes[name] = digest(data)
    receipt = json.loads(Path(cfg['inputs']['H4_tol_receipt.json']['path']).read_bytes())
    if receipt['rank'] != 10 or receipt['hamiltonian_hash'] != cfg['binding']['hamiltonian']:
        raise ValueError('saved DF rank/H')
    with np.load(cfg['inputs']['H4_tol_snapshot.npz']['path'],allow_pickle=False) as archive:
        constant = float(archive['constant'])
        matrices = [np.asarray(archive['one_body'],complex), *np.asarray(archive['g_matrices'],complex)[:7]]
        lambdas = np.real_if_close(np.asarray(archive['lambdas'],complex))
    scope = dict(np=np,math=math,ParameterExpression=ParameterExpression,
                 DiagonalEvolutionPrimitives=SimpleNamespace)
    reuse_functions(PROVIDER,'src/trotterlib/df_trotter/decompose.py',['diag_hermitian'],scope)
    reuse_functions(CARRIER,'scripts/h46_new2_native.py',['gaussian_operations'],scope)
    reuse_functions(PROVIDER,'src/trotterlib/df_trotter/ops.py',
        ['_as_real','_is_symbolic','_is_effectively_zero','_real_or_symbolic',
         'one_body_diagonal_primitives','df_squared_diagonal_primitives'],scope)
    blocks = []
    for index, matrix in enumerate(matrices):
        # Same model.hermitize and fixed eigenvalue sorting as the provider.
        matrix = .5*(matrix+matrix.conj().T)
        U, values = scope['diag_hermitian'](matrix,sort='descending_abs',assume_hermitian=True)
        operations = scope['gaussian_operations'](U)
        basis = [dict(name=g.name,qubits=list(q),parameters=[str(t) for t in g.params],
                      matrix_sha256=digest(np.ascontiguousarray(np.asarray(g.to_matrix(),dtype=np.complex128)).view(np.uint8).tobytes()))
                 for g,q in operations]
        diagonal = scope['one_body_diagonal_primitives'](values,.05) if index==0 else scope['df_squared_diagonal_primitives'](values,lambdas[index-1],.05)
        blocks.append(dict(block_id='one_body' if index==0 else 'df-fragment-'+str(index-1),basis=basis,diagonal=diagonal))
    return blocks, constant, cfg['binding'], hashes


def require_candidate(c,method,q,prefix):
    expected = dict(system='H4',method=method,formula='2nd',prefix=prefix,q=q,T=.1*q,delta=.1,
                    R_outer=4 if method=='PR' else None,K=2 if method=='PR' else None)
    if any(c.get(k)!=v for k,v in expected.items()):
        raise ValueError('candidate mismatch')


def validate_events(batches, known_bases):
    for batch in batches:
        for event in batch:
            for app in event['application_sequence']:
                if app['is_identity']:
                    continue
                ops = app['basis_change_operations']
                payload = dict(num_system_qubits=8,operations=ops,hash_policy='ordered_local_basis_operations_v1')
                value = digest(json.dumps(payload,sort_keys=True,separators=(',',':')).encode())
                if value != app['basis_hash'] or app['basis_id'] != 'df-basis-'+value:
                    raise ValueError('event basis descriptor hash')
                if value in known_bases and known_bases[value] != ops:
                    raise ValueError('basis hash reused with changed operations')
                known_bases[value] = ops


def run(stage):
    source, source_hashes = source_binding()
    previous = None
    if stage == 'holdout':
        previous = json.loads((PUBLIC/'DEVELOPMENT.json').read_bytes())
        if previous['model_source'] != source or previous['model_sha256'] != source_hashes or previous['model'] != MODEL:
            raise ValueError('holdout requires unchanged frozen model')
    import qiskit
    forbidden_calls = []
    def forbidden(*a,**kw):
        forbidden_calls.append(True)
        raise RuntimeError('QuantumCircuit construction / transpile prohibited in V1')
    with patch.object(qiskit.QuantumCircuit,'__init__',forbidden), patch.object(qiskit,'transpile',forbidden):
        blocks, constant, binding, input_hashes = inputs_and_blocks()
        rows = []
        q = 8 if stage=='develop' else 16
        pf_record = saved('reused_compiled_prices.json.gz' if q==8 else 'exploration_costs.json.gz')[PF[q]]
        require_candidate(pf_record['candidate'],'B0',q,7)
        for axis in ('cosine','sine'):
            prediction = predict(blocks,q,constant,.1*q,axis)
            actual = pf_record[axis] if q==8 else pf_record['mean_metrics'][axis]['rz_count']
            rows.append(dict(method='PF',cohort='DETERMINISTIC',candidate_id=PF[q],q=q,T=.1*q,
                             path_index=0,axis=axis,compiled_rz=int(actual),
                             residual_rz=int(actual)-prediction['structural_rz'],prediction=prediction))
        cohort,n = ('EXPLORATION',9) if stage=='develop' else ('CONFIRMATION',33)
        table = saved('exploration_costs.json.gz' if stage=='develop' else 'confirmation_costs.json.gz')[PR]
        require_candidate(table['candidate'],'PR',8,1)
        if table['cohort']!=cohort or table['n']!=n or len(table['trial_files'])!=n:
            raise ValueError('saved cohort scope')
        known_bases = {}
        private_bindings = []
        for index, filename in enumerate(table['trial_files']):
            path = Path(filename)
            raw = path.read_bytes()
            event_raw = path.with_name('events_PRIVATE.json').read_bytes()
            result, events = json.loads(raw),json.loads(event_raw)
            require_candidate(result['candidate'],'PR',8,1)
            if (result['status']!='PASS' or result['cohort']!=cohort or result['index']!=index
                    or result['preparation_hash']!=PREPARATION or events['cohort']!=cohort
                    or result['conditional'] or events['conditional']):
                raise ValueError('saved trial binding')
            validate_events(events['events'],known_bases)
            private_bindings.append(dict(index=index,result_sha256=digest(raw),events_sha256=digest(event_raw)))
            for axis in ('cosine','sine'):
                # The predictor receives no measured price or cohort label.
                prediction = predict(blocks[:2],8,result['scalar'],.8,axis,events['events'])
                actual = result['prices'][axis]['rz_count']
                rows.append(dict(method='PR',cohort=cohort,candidate_id=PR,q=8,T=.8,
                                 path_index=index,axis=axis,compiled_rz=int(actual),
                                 residual_rz=int(actual)-prediction['structural_rz'],prediction=prediction))
        summaries = []
        for method in ('PF','PR'):
            for axis in ('cosine','sine'):
                group = [r for r in rows if r['method']==method and r['axis']==axis]
                n_group = len(group)
                observed = sum(r['compiled_rz'] for r in group)
                predicted = sum(r['prediction']['structural_rz'] for r in group)
                if method=='PR' and observed/n_group != table['mean_metrics'][axis]['rz_count']:
                    raise ValueError('saved aggregate does not match original per-path prices')
                summaries.append(dict(method=method,axis=axis,n_paths=n_group,
                    observed_sum=observed,predicted_sum=predicted,mean_observed=observed/n_group,
                    mean_predicted=predicted/n_group,mean_residual=(observed-predicted)/n_group,
                    min_residual=min(r['residual_rz'] for r in group),
                    max_residual=max(r['residual_rz'] for r in group),
                    max_abs_residual=max(abs(r['residual_rz']) for r in group),
                    exact_count=sum(r['residual_rz']==0 for r in group)))
    return dict(status='FINITE_SAVED_MODEL_MATCH' if all(r['residual_rz']==0 for r in rows) else 'FINITE_SAVED_MODEL_DISCREPANCY',
        stage=stage,model=MODEL,model_source=source,model_sha256=source_hashes,
        price_carrier=CARRIER,provider_source=PROVIDER,compiler_hash=COMPILER,scientific_binding=binding,
        input_sha256=input_hashes,private_trial_bindings=private_bindings,rows=rows,summaries=summaries,
        development_sha256=None if previous is None else digest((PUBLIC/'DEVELOPMENT.json').read_bytes()),
        full_circuit_builds=len(forbidden_calls),transpile_calls=0,sampling_calls=0,fit_calls=0,
        new_signal_calls=0,new_N_calls=0,N_times_C_calls=0,bootstrap_calls=0,
        general_P0_qualified=False,all_law_expectation=None,long_q_qualified=False,total_CI=None,
        mandatory_STOP=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('develop','holdout'))
    args = parser.parse_args()
    PUBLIC.mkdir(parents=True,exist_ok=True)
    destination = PUBLIC/('DEVELOPMENT.json' if args.stage=='develop' else 'HOLDOUT.json')
    if destination.exists():
        raise SystemExit('preserve completed stage; no overwrite')
    result = run(args.stage)
    with destination.open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status','stage','model_source','summaries')},ensure_ascii=False,indent=2))
