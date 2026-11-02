"""Select one new F3 artificial test file; no repository discovery/plugins."""
import ast, hashlib, importlib.util, json, os, resource, sys, time, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
DEST=Path(sys.argv[1]).resolve(); DEST.parent.mkdir(parents=True, exist_ok=True)
TEST=ROOT/'tests/tracks/algorithm_codesign/test_g6_f3_preflight_angle.py'
tag=DEST.name; DEST=DEST.parent
if any((DEST/(tag+s)).exists()for s in ('.log','.json')):raise SystemExit('refuse to overwrite test record')
paths=sorted((ROOT/'src/trottertracks/algorithm_codesign').glob('g6_f3_*.py'))+[TEST]
identity={str(p.relative_to(ROOT)):{'bytes':len(p.read_bytes()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}for p in paths}
for p in paths:ast.parse(p.read_bytes(),filename=str(p))
spec=importlib.util.spec_from_file_location('focused_g6_f3',TEST)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
suite=unittest.defaultTestLoader.loadTestsFromModule(module)
class Result(unittest.TextTestResult):
 def __init__(self,*a,**kw):super().__init__(*a,**kw);self.records=[]
 def addSuccess(self,t):super().addSuccess(t);self.records.append({'test':t.id(),'status':'PASS'})
 def addFailure(self,t,e):super().addFailure(t,e);self.records.append({'test':t.id(),'status':'FAIL'})
 def addError(self,t,e):super().addError(t,e);self.records.append({'test':t.id(),'status':'ERROR'})
 def addSkip(self,t,r):super().addSkip(t,r);self.records.append({'test':t.id(),'status':'SKIP','reason':r})
 def addSubTest(self,t,st,e):
  super().addSubTest(t,st,e)
  if e:self.records.append({'test':t.id(),'subtest':str(st),'status':'SUBTEST_FAILURE'})
start=time.monotonic()
with(DEST/(tag+'.log')).open('w')as log:
 result=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=Result).run(suite)
usage=resource.getrusage(resource.RUSAGE_SELF)
loaded={name:str(Path(mod.__file__).relative_to(ROOT))for name,mod in sys.modules.items()if getattr(mod,'__file__',None)and str(mod.__file__).startswith(str(ROOT))}
forbidden=[name for name in sys.modules if name.split('.')[0]in {'numpy','scipy','pyscf','qiskit','numba','torch','pytest'}]
d={'schema':'g6_f3_focused_artificial_tests_v1','run':tag,'python':sys.version,'selection':str(TEST.relative_to(ROOT)),'source_identity_at_execution':identity,'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skips':len(result.skipped),'successful':result.wasSuccessful(),'records':result.records,'elapsed_seconds_administrative':time.monotonic()-start,'process_cpu_seconds_administrative':usage.ru_utime+usage.ru_stime,'maxrss_KiB_administrative':usage.ru_maxrss,'project_modules_imported':loaded,'forbidden_packages_imported':forbidden,'old_frozen_pure_rational_modules_used_on_small_artificial_inputs':True,'new_scientific_calculations':0,'saved_scientific_data_reanalysis':0,'backend_sampling':0,'circuit_generation_compile_synthesis':0,'precision_selector_refinement_retry_implemented':False,'IID_performance_confidence_or_general_proof_claim':False}
(DEST/(tag+'.json')).write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps({k:d[k]for k in ('run','tests_run','failures','errors','skips','successful','elapsed_seconds_administrative','forbidden_packages_imported')}))
raise SystemExit(0 if result.wasSuccessful()and not forbidden else 1)
