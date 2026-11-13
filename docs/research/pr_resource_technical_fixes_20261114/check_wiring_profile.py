"""Real-loader positive/negative wiring checks in one isolated SOURCE process."""
import argparse
import ast
from copy import deepcopy
import importlib.util
import json
import linecache
from pathlib import Path
import subprocess
import sys
from types import FunctionType

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--series',choices=('A','B','C','D'),required=True)
parser.add_argument('--source',required=True)
args=parser.parse_args();root=args.root.resolve();sys.dont_write_bytecode=True
sys.path[:0]=[str(root/'src'),str(root/'scripts')]
implementation=Path(__file__).resolve().parents[3]/'src/trottertracks/pr_resource_wiring.py'
spec=importlib.util.spec_from_file_location('trottertracks._wiring_validation',implementation)
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
blocked=[];original_codes=[]
if args.series=='A':
    from trottertracks import pr_rpe_signal_provider as A
    from trottertracks import h4_pr_rmse_completion as completion
    original_codes=[f.__code__ for f in (A.evaluate,A.make_context,A.load_h4_context,A.SignalBatch.run,completion.load_context)]
elif args.series=='B':
    from trottertracks import h4_s0_v2_intake as B
    original_codes=[B.B.cost.__code__,B.p.run_missing.__code__,B.I.A.evaluate.__code__]
elif args.series=='D':
    from trottertracks import candidate_price_adapter as D
    original_codes=[D.NativeRuntime.context.__code__,D.NativeRuntime.sample.__code__,D.Adapter.acquire.__code__]

def scientific_guard(frame,event,arg):
    if event!='call':return
    code=frame.f_code
    dynamic_context=code.co_name=='context' and (code.co_filename.endswith(':context_S_only') or
        'r_price_phase1_science.py' in code.co_filename)
    kernel=code.co_name in ('spectrum','evolve_batch','sample_requests','compose','make_rte_config') and (
        'h46_' in code.co_filename or 'trotterlib/rte.py' in code.co_filename)
    if code in original_codes or dynamic_context or kernel:
        blocked.append(code.co_filename+':'+code.co_name)
        raise AssertionError('Scientific boundary crossed: '+blocked[-1])

negative=[]
def rejects(label,errors,fn):
    try:fn()
    except errors as error:negative.append(dict(check=label,error_type=type(error).__name__))
    else:raise AssertionError('Did not reject '+label)

sys.setprofile(scientific_guard)
try:
    result=w.preflight(root,args.series,args.source)
    if args.series=='A':
        old_path=list(sys.path);old_module=sys.modules.pop('h46_new2_core',None)
        sys.path[:]=[p for p in sys.path if Path(p).resolve()!=root/'scripts']
        try:rejects('missing actual scripts import path',ValueError,lambda:w.preflight(root,'A',args.source))
        finally:
            sys.path[:]=old_path
            if old_module is not None:sys.modules['h46_new2_core']=old_module
    elif args.series=='B':
        rejects('foreign SOURCE object',(ValueError,subprocess.CalledProcessError),lambda:w.preflight(root,'B','0'*40))
    elif args.series=='C':
        from trottertracks import s0a_structure_context as C
        from trottertracks import s0a_saved_structure as saved
        from trottertracks import h4_pr_physical_closure as closure
        from trottertracks import h4_structural_kernel as kernel
        fn=w.context_definition(C.load_context,root,saved.saved(root,'PLAN.json'))
        scope=dict(fn.__globals__);scope.pop('ast')
        missing=FunctionType(fn.__code__,scope)
        rejects('missing ast in actual fixed context scope',ValueError,lambda:w.check_globals(missing))
        def obsolete_descriptor(tail, time, steps):raise AssertionError('Must never execute')
        rejects('descriptor API mismatch',TypeError,lambda:w.bind_calls(closure.make_law,'make_rte_config',obsolete_descriptor))
        # The actual historical failure was seed=None (valid keyword, invalid
        # scalar type). Load that exact definition, then bind/check only.
        import trotterlib.rte as rte
        origin='c5e2e5046f674277778b224329301c04a52eec8f:src/trottertracks/h4_pr_physical_closure.py'
        data=subprocess.check_output(['git','show',origin],cwd=root,text=True)
        node=next(n for n in ast.parse(data).body if isinstance(n,ast.FunctionDef) and n.name=='diagnostic')
        scope={};exec(compile(ast.Module(body=[node],type_ignores=[]),origin,'exec'),scope)
        linecache.cache[origin]=(len(data),None,data.splitlines(True),origin)
        rejects('historical seed=None with actual integer validator',TypeError,lambda:w.bind_calls(
            scope['diagnostic'],'make_rte_config',rte.make_rte_config,
            scalar_checks={'seed':lambda value:rte.require_integer_count(value,name='seed')}))
        original=kernel.B_CONFIGURATION_SOURCE;kernel.B_CONFIGURATION_SOURCE=kernel.B_LATEST
        try:rejects('wrong configuration commit',(subprocess.CalledProcessError,KeyError,ValueError),lambda:kernel.check_inputs(root))
        finally:kernel.B_CONFIGURATION_SOURCE=original
    else:
        from trottertracks import p1_efficient_zz as P
        from trottertracks import ld12_price_requests as L
        raw=P.make_plan(root,'ARTIFICIAL_WIRING_NEGATIVE')
        # P1's original runner supplies the commit as a string, while newer
        # adapters use {'commit': SHA}. Both must bind to the same SOURCE.
        w.preflight(root,'D',args.source,plan=P.execution_plan(raw,args.source,root))
        rejects('missing Backend plan source',ValueError,lambda:w.check_backend_plan(root,raw))
        request=L.request_for(root,'PR-LD2',12,execution_source=dict(commit=args.source))
        bad=deepcopy(request);pc=bad['physical_context']
        pc['expected_scalar']=pc['input_source']['scalar_binding']['tail_identity']['value']
        bad['context_key']=D._key(pc);bad.pop('request_id');bad['request_id']=D._key(bad)
        # Hashes are valid: rejection must come from the scalar's meaning.
        D.check_request(bad)
        rejects('tail-only scalar rehashed as total',ValueError,lambda:w.check_price_request(root,bad))
finally:sys.setprofile(None)
assert not blocked
print(json.dumps(dict(positive=result,negative=negative,scientific_boundary_calls=blocked),indent=2))
