"""Seal technical adapter and verify saved evidence only; no science grant."""
from pathlib import Path
import csv,hashlib,io,json,subprocess,sys,datetime
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip()
assert source=='0880f4af98b125d62599e970db89fc33634d1b61'
code=['src/trotterlib/h4_transfer_rules.py','src/trotterlib/h4_cl_binding.py','scripts/verify_h4_cl_transfer.py','tests/test_h4_cl_transfer.py','src/trotterlib/h4_certificate_gates.py','src/trotterlib/h4_evidence_binding.py','tests/test_h4_certificate_gates.py','tests/test_h4_evidence_binding.py','tests/test_h4_local_enclosure.py']
for path in code:assert hashlib.sha256(subprocess.check_output(['git','show',source+':'+path],cwd=ROOT)).hexdigest()==sha(ROOT/path),path
old=json.loads((ROOT/'artifacts/non_df_h4_local_diagnostic/2026-10-23/run01/seal.json').read_text())
for path,digest in old['runtime_files'].items():assert sha(path)==digest,path
assert sha(sys.executable)==old['python_executable_sha256']
manifest=json.loads((HERE/'evidence_manifest.json').read_text());base=json.loads((ROOT/manifest['base_manifest_path']).read_text())
inputs={str(ROOT/manifest['base_manifest_path']):manifest['base_manifest_sha256'],str(HERE/'evidence_manifest.json'):sha(HERE/'evidence_manifest.json')}
for ref in list(base['files'].values())+list(manifest['files'].values()):inputs[str(ROOT/ref['path'])]=ref['sha256']
for name in ('L_local_csv','L_basis_csv','L_support_csv'):
 p=ROOT/manifest['files'][name]['path']
 for row in csv.DictReader(io.StringIO(p.read_text())):inputs[str(ROOT/row['result_path'])]=row['result_SHA256']
runtime_source=json.loads((HERE/'literal_runtime_source_map.json').read_text())
for ref in runtime_source['runtime_sources']:inputs[ref['path']]=ref['SHA256']
for path,digest in inputs.items():assert sha(path)==digest,path
record=dict(kind='CL_TECHNICAL_SOURCE_NOT_SCIENCE_EXECUTION_GRANT',SOURCE=source,base_commit='de94417e7e1a0b984b93b81aba673b5ef94fdf98',C_scientific_SOURCE=old['C_scientific_SOURCE'],L_scientific_SOURCE=old['SOURCE'],original_scientific_SOURCE=old['original_scientific_SOURCE'],code_files={str(ROOT/p):sha(ROOT/p) for p in code},code_git_files={p:sha(ROOT/p) for p in code},runtime_files=old['runtime_files'],input_files=inputs,python_executable_sha256=sha(sys.executable),python_runtime=old['python_runtime'],scientific_execution_authorized=False,execution_authorized=False,grant_valid=False,dispatch_enabled=False,formal_certificate_accepted=False,N_G_authorized=False,scientific_grant_issued=False,scope='Static SHA/scope joins and formal identities only; real H4 numbers copied without arithmetic')
with (HERE/'SOURCE.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
command=[sys.executable,'-B',str(ROOT/'scripts/verify_h4_cl_transfer.py'),'--manifest',str(HERE/'evidence_manifest.json'),'--output',str(HERE/'check03')]
start=datetime.datetime.now(datetime.timezone.utc).isoformat();r=subprocess.run(command,capture_output=True,text=True,cwd=ROOT)
with (HERE/'sealed_cli_stdout.log').open('x') as f:f.write(r.stdout)
with (HERE/'sealed_cli_stderr.log').open('x') as f:f.write(r.stderr)
assert r.returncode==0,r.stderr
for group in ('code_files','runtime_files','input_files'):
 for path,digest in record[group].items():assert sha(path)==digest,path
verification=dict(status='PASS_SEALED_STATIC_CLI',SOURCE=source,seal_SHA256=sha(HERE/'SOURCE.json'),command=command,started_UTC=start,finished_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=r.returncode,code_files=len(record['code_files']),runtime_files=len(record['runtime_files']),input_files=len(inputs),new_H4_numeric_evaluations=0,new_scientific_grants=0,formal_certificate_accepted=False,N_G_authorized=False)
with (HERE/'seal_verification.json').open('x') as f:json.dump(verification,f,indent=2);f.write('\n')
print(json.dumps(verification))
