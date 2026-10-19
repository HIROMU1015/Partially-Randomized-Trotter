"""Fixed-Git bytes + AST-only science audit. Never invokes the scientific loader."""
from pathlib import Path
from hashlib import sha256
import ast,json,subprocess,os
from trottertracks.algorithm_codesign.g6_w2c_b3_final_live_v1 import contract as K,guard as G,activation
from trottertracks.algorithm_codesign.g6_w2c_b3_live_preflight_v1 import spec as S

PREVIOUS_SOURCE='2405f5e2e5fb0dc5add39b102e3ccae3d56df0a4'
UNIFIED='7eff8a33bef4ee1f29698ab10c6da46988dc8db5'
UNIFIED_PATH='docs/research/qc_relative_form_finite_step_audit_20261017/inputs/research_project_instructions_unified_2026-10-17.md'
REFERENCES=dict(base=K.BASE,science=S.SCIENCE,closed_preflight='6c6d3376e7ff206219f5e58355f1e1080fafba04',
    exact_adapter='80b7bf61fe0a735d4031bb719c4b3e788611c3ba',independent=S.INDEPENDENT,
    global_bundle='62967269e611b61e73b70abe20e998d9c7eaaf50',api_binding='82896d53fa97235fa848d6c0833eb193af8b99d6',
    api_measured='78c571005fe766c61b32dc867e006531099ff608',previous_measured=PREVIOUS_SOURCE,unified=UNIFIED)


def audit(root):
    root=Path(root).resolve();G.install()
    def git(*args):return subprocess.check_output(['git',*args],cwd=root)
    reference_objects={k:git('rev-parse',v+'^{commit}').decode().strip() for k,v in REFERENCES.items()}
    G.require(reference_objects==REFERENCES,'FIXED_REFERENCE_OBJECTS')
    previous=json.loads(git('show',K.BASE+':docs/research/track_b_g6_w2c_b3_live_correctness_release_preparation_20261020/source_inventory.json'))
    previous_rows=[]
    for path,h in {**previous['source_pins'],**previous['input_pins']}.items():
        local=sha256((root/path).read_bytes()).hexdigest()
        original=sha256(git('show',PREVIOUS_SOURCE+':'+path)).hexdigest()
        published=sha256(git('show',K.BASE+':'+path)).hexdigest()
        G.require(h==local==original==published,'PREVIOUS_FIXED_BYTES:'+path)
        previous_rows.append(dict(path=path,sha256=h,measured_public_current_equal=True))
    core=[]
    for name in sorted(G.CORE):
        path=S.SOURCE+name+'.py';raw=(root/path).read_bytes();old=git('show',S.SCIENCE+':'+path)
        G.require(raw==old,'FIXED_SCIENCE_BYTES:'+name);ast.parse(raw)
        core.append(dict(path=path,sha256=sha256(raw).hexdigest(),AST_only=True))
    prior=ast.parse((root/S.SOURCE/'g6_w2c_b3_live_release_v1/execution.py').read_bytes())
    new=ast.parse((root/K.PKG/'execution.py').read_bytes())
    old_pair=next(n for n in prior.body if isinstance(n,ast.FunctionDef) and n.name=='run_pair')
    new_pair=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='run_pair')
    G.require(ast.dump(old_pair)==ast.dump(new_pair),'PAIR_F01_AST_CHANGED')
    common=git('show',UNIFIED+':'+UNIFIED_PATH)
    G.require(b'8.4' in common,'UNIFIED_SECTION_8_4')
    G.require(activation.APPROVED_ACTIVATION_ID is None,'NO_ACTIVE_EXECUTION_ANCHOR')
    c=K.candidate(root)
    result=dict(schema='b3_final_static_preparation_audit_v1',PASS=True,references=REFERENCES,
        fixed_git_objects=reference_objects,previous_pins=previous_rows,core_science=core,
        resolver_binding=c['final_runtime_binding'],old_regression_SOURCE=PREVIOUS_SOURCE,
        regression_tests=['test_g6_w2c_b3_parallel_f01','test_g6_w2c_b3_storage_v2.Storage'],
        pair_F01_AST_identical_to_previous=True,regression_note='Old 66-check dependency bytes unchanged; new regression SOURCE recorded by acceptance runner.',
        unified=dict(commit=UNIFIED,path=UNIFIED_PATH,sha256=sha256(common).hexdigest(),section_8_4_applied=True),
        available=dict(affinity_cpus=len(os.sched_getaffinity(0)),
            memory=Path('/proc/meminfo').read_text().splitlines()[:3],
            pressure={k:Path('/proc/pressure/'+k).read_text() for k in ('cpu','memory','io')},
            output_disk_available_bytes=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize),
        guard=G.observation(),scientific_loader_calls=0,scientific_sampling=0,scientific_bits=0,
        current_execution_permission=None,mandatory_STOP=True)
    return result
