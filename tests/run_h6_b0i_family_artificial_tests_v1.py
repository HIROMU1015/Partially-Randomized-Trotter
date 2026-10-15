"""Bounded stdlib-only tests; caller supplies a new exclusive home directory."""
import sys, os, resource, time, signal, json, unittest, importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if len(sys.argv)!=2: raise SystemExit('usage: PYTHON -P -B THIS_SCRIPT NEW_HOME_EVIDENCE_DIRECTORY')
E=Path(sys.argv[1]).resolve()
if not E.is_relative_to(Path('/home/AbeHiromu')): raise SystemExit('HOME_ONLY')
E.mkdir(parents=True,exist_ok=False)
TMP=E/'test-tmp';TMP.mkdir()
resource.setrlimit(resource.RLIMIT_AS,(2*2**30,2*2**30))
resource.setrlimit(resource.RLIMIT_CPU,(180,182))
resource.setrlimit(resource.RLIMIT_FSIZE,(16*2**20,16*2**20))
signal.alarm(180)
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','QISKIT_NUM_PROCS'): os.environ[k]='1'
os.environ['QISKIT_PARALLEL']='FALSE';os.environ['TMPDIR']=str(TMP)
for key in ('TMPDIR','MPLCONFIGDIR','XDG_CACHE_HOME'): os.environ[key]=str(TMP)
forbidden=('numpy','scipy','qiskit','pyscf','cupy','trotterlib','trottertracks')
counts={'science_imports_blocked':0,'science_files_blocked':0,'process_creation_blocked':0,'out_of_fixture_write_blocked':0}
def audit(event,args):
 if event=='import' and str(args[0]).split('.')[0] in forbidden:
  counts['science_imports_blocked']+=1;raise RuntimeError('SCIENCE_IMPORT_FORBIDDEN:'+str(args[0]))
 if event in ('subprocess.Popen','os.fork','os.forkpty','os.exec','os.posix_spawn'):
  counts['process_creation_blocked']+=1;raise RuntimeError('PROCESS_CREATION_FORBIDDEN')
 if event=='open' and isinstance(args[0],(str,bytes)):
  path=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2]
  if path.suffix=='.npz' or ('h6-native-b0i-pilot6-run04-v3' in str(path)):
   counts['science_files_blocked']+=1;raise RuntimeError('REAL_DATA_FORBIDDEN')
  write=(isinstance(mode,str) and any(x in mode for x in ('w','a','x','+'))) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
  if write and not path.is_relative_to(TMP):
   counts['out_of_fixture_write_blocked']+=1;raise RuntimeError('WRITE_OUTSIDE_FIXTURE:'+str(path))
os.chdir(TMP)
sys.addaudithook(audit)
start=time.monotonic()
spec=importlib.util.spec_from_file_location('full_family_tests',ROOT/'tests/test_h6_b0i_full_family_v1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(m))
report=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful(),wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,audit=counts,real_science_operations=0,real_children=0,python=sys.version,executable=sys.executable,one_process=True,internal_threads=1)
# The harness writes its report only after removing its process-local audit hook by exiting;
# stdout is captured by the parent command into the predeclared private evidence directory.
print('RESULT_JSON:'+json.dumps(report,sort_keys=True))
sys.exit(0 if result.wasSuccessful() and report['peak_RSS_bytes']<=512*2**20 else 1)
