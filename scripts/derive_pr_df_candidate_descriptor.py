"""Derive one missing DF prefix descriptor with the exact historical helper.

This mode is an explicit scientific leaf; caller fixes SOURCE/input first.
No Pauli dictionary, circuit build, signal, reference action, fit or Q3 here.
"""
import argparse
import ast
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time
from types import FunctionType, SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
CALLER='scripts/derive_pr_df_candidate_descriptor.py'
SIGNAL='fefd548ecd56f60d72031b3d385f6f5e6ee95d06'
VIEW=ROOT.parent/'d2-pr-df-ld7-signal-source-fefd-20261116'
INDEX=dict(commit='7a8bc7f132e166390c943e6b32c92cc6abe1baf4',path='artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/SOURCE_INDEX.json')
DERIVATION=dict(commit='7f4c9a171a1d4caeaeaee6c443217edec761f4c9',path='scripts/h46_new2_core.py',function='matrix_only_tail_coefficients')


def read(loc):
    value=json.loads(subprocess.check_output(['git','show',loc['commit']+':'+loc['path']],cwd=ROOT))
    for k in loc.get('selector',[]):value=value[k]
    return value


def saved_context(completion, config):
    """Use the frozen loader on identical input bytes at explicit local paths."""
    # Import by the owning controller path; the provider uses its own package.
    spec=importlib.util.spec_from_file_location('_descriptor_preparation_paths',
        ROOT/'src/trottertracks/rpe_preparation_paths.py')
    paths=importlib.util.module_from_spec(spec);spec.loader.exec_module(paths)
    index=read(INDEX);source_index=deepcopy(index);saved=index['T20_reference_files']['H4']
    relocated=False
    for entry in (saved,*index['saved_inputs']['H4']['inputs'].values()):
        old=entry['path'];entry['path']=paths.resolve(old);relocated |= old!=entry['path']
    raw=Path(saved['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=saved['sha256']:raise ValueError('Saved context descriptor bytes differ')
    original=json.loads(raw)
    for key in ('system','basis','geometry','rank','rank_policy','hamiltonian','state','snapshot'):
        if original['binding'].get(key)!=config['spec']['binding'].get(key):
            raise ValueError('Configuration/saved H binding mismatch: '+key)
    loader=completion.load_context
    if relocated:
        namespace=dict(loader.__globals__)
        namespace['p']=SimpleNamespace(SavedArtifacts=lambda root: SimpleNamespace(read=lambda locator:deepcopy(index)),
                                     same=namespace['p'].same)
        loader=FunctionType(loader.__code__,namespace)
    return loader,original,source_index


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prefix',type=int,required=True);p.add_argument('--configuration',required=True);p.add_argument('--selector',default='[]');p.add_argument('--output',type=Path,required=True);p.add_argument('--source',required=True)
    p.add_argument('--signal-view', type=Path, default=VIEW)
    p.add_argument('--max-width-endpoint', action='store_true')
    a=p.parse_args()
    if subprocess.check_output(['git','show',a.source+':'+CALLER],cwd=ROOT)!=Path(__file__).read_bytes():raise ValueError('Descriptor runner differs from fixed SOURCE')
    source_loc=dict(commit=a.source,path=CALLER);c,path=a.configuration.split(':',1);config_loc=dict(commit=c,path=path,selector=json.loads(a.selector));config=read(config_loc)
    if not (0 if a.max_width_endpoint else 1)<=a.prefix<config['spec']['binding']['rank']:raise ValueError('Nonempty head/tail prefix required')
    view=a.signal_view.resolve()
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=view,text=True).strip()!=SIGNAL:raise ValueError('Fixed signal view differs')
    sys.path[:0]=[str(view/'src'),str(view/'scripts')]
    from trottertracks import h4_pr_rmse_completion as completion
    loader,original,index=saved_context(completion,config)
    saved=index['T20_reference_files']['H4']
    a.output=a.output.resolve()
    if not a.output.is_relative_to(ROOT):raise ValueError('Descriptor output must be in the owned candidate worktree')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    marker=a.output.with_suffix('.STARTED.json');failure=a.output.with_suffix('.STOP.json')
    if a.output.exists():
        doc=json.loads(a.output.read_text())
        if doc.get('prefix')!=a.prefix or doc.get('configuration_source')!=config_loc:raise ValueError('Existing descriptor belongs to other input')
        print('REUSED_COMPLETED_DESCRIPTOR');return
    if marker.exists()or failure.exists():raise ValueError('Prior descriptor start/failure unresolved; no retry')
    marker.write_text(json.dumps(dict(source=source_loc,configuration=config_loc,prefix=a.prefix,started=time.time(),automatic_retry=False))+'\n')
    start=time.perf_counter()
    try:
        loaded,_=loader(ROOT,original['binding'])
        text=subprocess.check_output(['git','show',DERIVATION['commit']+':'+DERIVATION['path']],cwd=ROOT)
        node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name==DERIVATION['function'])
        namespace=dict(math=math)
        exec(compile(ast.Module(body=[node],type_ignores=[]),DERIVATION['commit']+':'+DERIVATION['function'],'exec'),namespace)
        scalar,lam=namespace[DERIVATION['function']](loaded.ham,a.prefix)
        if not all(math.isfinite(v)for v in (scalar,lam))or lam<=0:raise ValueError('Finite positive DF tail norm required')
        doc=dict(schema='PR_DF_TAIL_DESCRIPTOR_V1',prefix=a.prefix,descriptor=dict(scalar=scalar,lambda_R=lam),
            binding=deepcopy(config['spec']['binding']),configuration_source=config_loc,derivation_source=DERIVATION,
            execution_source=source_loc,saved_index=INDEX,input_source=index['saved_inputs']['H4']['inputs'],
            original_context=dict(**saved,index=INDEX),resources=dict(wall_seconds=time.perf_counter()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            one_body='SEPARATE_DETERMINISTIC_HEAD',head_fragments=list(range(a.prefix)),tail_fragments=list(range(a.prefix,loaded.ham.n_blocks)),
            coefficient_rule='ORIGINAL_DESCENDING_ABS_DF_EIGENVALUES_I_Z_ZZ_MATH_FSUM',scientific_calls=1,signal_calls=0,compile=0,Q3=0)
        with a.output.open('x')as f:json.dump(doc,f,indent=2,allow_nan=False);f.write('\n')
        print('DF_DESCRIPTOR_COMPLETE',a.prefix,lam,scalar,flush=True)
    except BaseException:
        import traceback
        failure.write_text(json.dumps(dict(source=source_loc,prefix=a.prefix,error=traceback.format_exc(),automatic_retry=False))+'\n');raise


if __name__=='__main__':main()
