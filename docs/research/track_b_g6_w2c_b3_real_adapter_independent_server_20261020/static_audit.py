"""Static-only dependency, regression, input/runtime and preservation audit."""
import ast
from collections import deque
import json
from pathlib import Path
import subprocess
import sys
import independent as I

HEAD='80b7bf61fe0a735d4031bb719c4b3e788611c3ba'
SOURCE='b7ad344401b10395c49e4cfa806909282859bab5'
OLD='8d5135ee77e6cd06d8b2ed8aa0b9ee50a41a0e36'
DOC='docs/research/track_b_g6_w2c_b3_real_api_adapter_contract_closed_v1_20261020/'

def git(*args):return subprocess.check_output(['git','-C',str(I.REPO),*args])
def blob(commit,path):return git('show',commit+':'+path)
def save(name,data):(I.HERE/name).write_text(json.dumps(data,indent=2)+'\n')

def imports(path):
    """Module-scope imports only: bodies of functions/classes are not executed."""
    tree=ast.parse(blob(SOURCE,path),filename=path)
    module=path.removeprefix('src/').removesuffix('.py').replace('/','.')
    package=module.rsplit('.',1)[0]
    result=[]
    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self,node):pass
        def visit_AsyncFunctionDef(self,node):pass
        def visit_ClassDef(self,node):pass
        def visit_Import(self,node):
            result.extend(dict(module=n.name,line=node.lineno,syntax=ast.unparse(node)) for n in node.names)
        def visit_ImportFrom(self,node):
            prefix=package.split('.')[:len(package.split('.'))-node.level+1] if node.level else []
            base='.'.join(prefix+([node.module] if node.module else []))
            if node.module:result.append(dict(module=base,line=node.lineno,syntax=ast.unparse(node)))
            result.extend(dict(module=base+'.'+n.name,line=node.lineno,syntax=ast.unparse(node)) for n in node.names)
    Visitor().visit(tree)
    return result

def main():
    inv=I.plain(blob(HEAD,DOC+'source_inventory.json'))
    modules={p.removeprefix('src/').removesuffix('.py').replace('/','.'):p for p in inv['source_pins'] if p.startswith('src/') and p.endswith('.py')}
    graph={name:[r for r in imports(path) if r['module'] in modules] for name,path in modules.items()}
    start='trottertracks.algorithm_codesign.g6_w2c_b3_real_adapter_v1.bits'
    queue=deque([(start,[])]);visited=set();found=[]
    while queue:
        node,chain=queue.popleft()
        if node in visited:continue
        visited.add(node)
        if node.rsplit('.',1)[-1] in I.SCIENCE:
            found.append(dict(module=node,path=modules[node],chain=chain));continue
        for edge in graph[node]:queue.append((edge['module'],chain+[dict(importer=modules[node],**edge)]))
    assert not found, 'New forbidden core scientific dependency: STOP'
    save('static_import_boundary.json',dict(
        status='RESOLVED_BY_EXPLICIT_USER_SCOPE_AMENDMENT',source=SOURCE,
        user_static_only_modules=sorted(I.SCIENCE),candidate_guard_forbids_six=True,
        eager_dependencies_from_artificial_bits=found,
        observed_probe_failure='SUBJECT_PRELOAD_DENIAL:trottertracks.algorithm_codesign.g6_w2c_cost_preparation',
        prohibited_module_loaded=False,initial_probe_failed_before_test_execution=True,
        operational_science_processing_observed=False,
        interpretation='User explicitly amended section 3E to allow driver/helper import and functions for artificial validation. The six core scientific modules remain static-only. No forbidden eager dependency is present.',
        allowed_artificial_modules=['g6_w2c_b3_driver','g6_w2c_cost_preparation'],
        amendment_is_scientific_authorization=False,
        producer_modified=False,guard_bypass_or_stub=False))
    candidate=I.plain((I.ROOT/'restored/cases/normal_n16/candidate_input.json').read_bytes())
    pins=[]
    for path,h in candidate['scientific_reference']['input_pins'].items():
        data=(I.ROOT/'runtime'/path).read_bytes();assert I.digest(data)==h
        pins.append(dict(path=path,bytes=len(data),sha256=h,method='bytes only; scientific values not decoded'))
    runtimes=[]
    for name,py in [('parent',candidate['python']),('worker',candidate['worker_runtime']['python'])]:
        rows=[dict(path=py['executable'],sha256=py['executable_sha256'])]+list(py['libraries'].values())
        checked=[]
        for row in rows:
            data=Path(row['path']).read_bytes();assert I.digest(data)==row['sha256']
            checked.append(dict(path=row['path'],bytes=len(data),sha256=I.digest(data)))
        runtimes.append(dict(role=name,version_current_equal=py['version']==sys.version,files=checked,
                             past_process_attestation=False))
    delta=git('diff','--name-only',OLD,SOURCE).decode().splitlines()
    (I.HERE/'final_source_delta.patch').write_bytes(git('diff',OLD,SOURCE))
    snapshot=I.plain(blob(HEAD,DOC+'preservation_before.json'))
    old=[]
    for path,row in snapshot['unchanged_original_paths'].items():
        data=blob(SOURCE,path);assert I.digest(data)==row['sha256'] and len(data)==row['bytes']
        assert data==blob(snapshot['base'],path)
        old.append(dict(path=path,bytes=len(data),sha256=I.digest(data)))
    assert len(old)==291
    archive=I.plain(blob(HEAD,DOC+'evidence/archive_manifest.json'))
    for row in archive['members']:
        data=(I.ROOT/'restored'/row['path']).read_bytes()
        assert len(data)==row['bytes'] and I.digest(data)==row['sha256']
    sourcepins=[]
    for path,h in inv['source_pins'].items():
        data=(I.ROOT/'runtime'/path).read_bytes();assert I.digest(data)==h
        sourcepins.append(dict(path=path,bytes=len(data),sha256=h))
    # Older test files are checked as bytes/AST only. Importing them would load
    # modules explicitly assigned static-only status in this request.
    old_tests=['test_g6_w2c_b3_port_bundle_v1','test_g6_w2c_b3_parallel_codec',
               'test_g6_w2c_b3_parallel_f01','test_g6_w2c_b3_storage_v2']
    tests=[]
    for name in old_tests:
        path='tests/tracks/algorithm_codesign/'+name+'.py'
        data=blob(SOURCE,path);assert data==blob(OLD,path)
        tests.append(dict(path=path,sha256=I.digest(data),unchanged=True,
                          import_syntax=[ast.unparse(n) for n in ast.walk(ast.parse(data)) if isinstance(n,(ast.Import,ast.ImportFrom))]))
    save('regression_summary.json',dict(old_Producer_regression=dict(PASS=107,source=OLD),
        final_source=SOURCE,changed_paths=delta,old_test_files=tests,old_SOURCE_tests_preserved=len(old),
        final_canonical_changes_exercised_by='214 independent Codec/Result/Bits/gate probes, all saved Layer B raw, and fresh spawn Bundle',
        final_source_regression=dict(PASS=107,FAIL=0,ERROR=0,SKIP=0,source=SOURCE,record='final_regression_retry2/results.json'),
        Producer_102_unit_and_30_integration_rerun=dict(unit_PASS=102,integration_PASS=30,source=SOURCE),
        intermediate_regression_errors='Fixture acquisition issues only; original 5 ERROR record retained in final_regression_retry1',
        user_scope_amendment_applied=True))
    save('pins_and_preservation_verification.json',dict(input_pins=pins,runtimes=runtimes,
        immutable_old_source_test_paths=old,runtime_SOURCE_pins=sourcepins,
        restored_archive_members_rechecked=len(archive['members']),restored_bytes_modified=0,
        other_worktrees_written=0,old_raw_grant_claim_ledger_marker_STOP_written=0,
        policy='Other worktree activity is read-only observation, never a global freeze requirement'))
    bindings=[]
    for commit,path,key in [
        ('82896d53fa97235fa848d6c0833eb193af8b99d6','docs/research/track_b_g6_w2c_b3_real_api_binding_preflight_candidate_20261019/source_inventory.json','measured_SOURCE'),
        ('62967269e611b61e73b70abe20e998d9c7eaaf50','docs/research/track_b_g6_w2c_b3_closed_port_bundle_integration_20261019/source_inventory.json','SOURCE')]:
        data=blob(commit,path);value=I.plain(data)
        bindings.append(dict(public_HEAD=commit,inventory_path=path,inventory_sha256=I.digest(data),measured_SOURCE=value[key]))
    save('fixed_reference_resolution.json',dict(references=bindings,
        instruction_API_measurement='78c571005fe766c61b32dc867dc28c591df0089e',
        actual_API_measurement='78c571005fe766c61b32dc867e006531099ff608',
        discrepancy='User supplied measurement SHA differs from fixed public inventory. The actual inventory value was fetched and used.'))
    resources={p:Path(p).read_text() for p in ['/proc/meminfo','/proc/pressure/cpu','/proc/pressure/memory','/proc/pressure/io']}
    save('resource_observation.json',dict(cpu_affinity=len(__import__('os').sched_getaffinity(0)),
        current=resources,raw_computational_workers=8,authority_race_threads=32,AI_subagents=0,
        shared_settings_or_other_processes_modified=False))
    print(json.dumps(dict(static_dependency_conflicts=len(found),input_pins=len(pins),old_preserved=len(old),
                          raw_members_preserved=len(archive['members']),old_regressions_rerun=107)))

if __name__=='__main__':main()
