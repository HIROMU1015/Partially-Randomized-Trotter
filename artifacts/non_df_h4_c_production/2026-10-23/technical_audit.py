"""Known-path preservation, GitHub byte identity, static NPY header and C seal."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import zipfile
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
BASE='b0f94185f93cd5614e8f183613a2a9313a5b9d4f'
OLD=Path('/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/non-df-h4-execution-foundation-20261023')
BC=Path('/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/non-df-h4-precision-bc-20261023')
DATA='artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data'
CODE=['src/trotterlib/h4_ball_evidence.py','src/trotterlib/h4_certificate_control.py','src/trotterlib/h4_legacy_export.py','scripts/run_h4_numerical_evidence.py','tests/test_h4_ball_evidence.py','tests/test_h4_certificate_control.py','tests/check_h4_c_legacy_bridge.py','tests/run_h4_c_technical_suite.py',str((HERE/'technical_audit.py').relative_to(ROOT)),str((HERE/'runtime_probe.py').relative_to(ROOT))]
CODE+=list(json.loads((ROOT/'artifacts/non_df_h4_precision_bc/2026-10-23/B_run01/SOURCE.json').read_text())['files'])
CODE+=['tests/check_h4_precision_c_backend.py']
CODE=list(dict.fromkeys(CODE))
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,sort_keys=True);f.write('\n')
def preservation():
    previous=read(BC/'artifacts/non_df_h4_precision_bc/2026-10-23/preservation_before.json')
    paths={}
    for name,digest in previous['source_files'].items():assert sha(OLD/name)==digest,name;paths[str(OLD/name)]=digest
    for name,entry in previous['evidence_and_claim_files'].items():assert sha(OLD/name)==entry['sha256'],name;paths[str(OLD/name)]=entry['sha256']
    for name,digest in previous['prior_investigation_files'].items():
        p=Path(previous['prior_investigation_root'])/name;assert sha(p)==digest,name;paths[str(p)]=digest
    inv=read(BC/'artifacts/non_df_h4_precision_bc/2026-10-23/result_inventory.json')
    for name,digest in inv['files'].items():assert sha(BC/name)==digest,name;paths[str(BC/name)]=digest
    paths[str(BC/'artifacts/non_df_h4_precision_bc/2026-10-23/result_inventory.json')]=sha(BC/'artifacts/non_df_h4_precision_bc/2026-10-23/result_inventory.json')
    return dict(old_SOURCE_files=738,old_result_claim_files=1147,old_investigation_files=17,BC_inventory_files=99,files=paths)
def github_check():
    record=read(HERE/'github_retrieval.json');verified=[]
    commit=json.loads(record['commit']['structuredContent']['content']);assert commit['sha']==BASE
    assert subprocess.check_output(['git','rev-parse',BASE+'^{tree}'],cwd=ROOT).decode().strip()==commit['tree']['sha']
    for result in record['results']:
        assert result['status']=='fulfilled';x=result['value'];f=x['result']['structuredContent'];assert not x['result']['isError']
        data=subprocess.check_output(['git','show',BASE+':'+x['path']],cwd=ROOT)
        blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest();assert blob==f['sha']
        assert f['content'].replace('\r\n','\n')==data.decode().replace('\r\n','\n')
        verified.append(dict(path=x['path'],git_blob=blob,canonical_sha256=hashlib.sha256(data).hexdigest()))
    return dict(commit=BASE,files=verified,status='PASS_FIXED_GITHUB_BLOBS')
def schema():
    headers={}
    with zipfile.ZipFile(OLD/DATA/'snapshot.npz') as z:
        for member in z.infolist():
            with z.open(member) as f:
                assert f.read(6)==b'\x93NUMPY';major,minor=f.read(2);n=struct.unpack('<H' if major==1 else '<I',f.read(2 if major==1 else 4))[0]
                header=ast.literal_eval(f.read(n).decode());headers[member.filename]=header
    expected={'state.npy':(256,),'df_one_body.npy':(8,8),'df_lambdas.npy':(10,),'df_g_matrices.npy':(10,8,8),'non_DF_matrix.npy':(256,256),'DF_matrix.npy':(256,256)}
    for name,shape in expected.items():assert headers[name]['shape']==shape,(name,headers[name])
    return dict(scope='NPY headers only; no array payload read or numerical operation',headers=headers,expected_shapes_verified=True,real_H4_numerical_calls=0)
def inputs():
    old=read(BC/'artifacts/non_df_h4_precision_bc/2026-10-23/C_execution_contract_DRAFT.json')
    files={str(OLD/name):digest for name,digest in old['input_files_sha256'].items()}
    for name in ['C_precision_method_map.json','B_run01/diagnostics.json','B_run01/contract.json','B_run01/grant.json','B_run01/CONSUMED.json','B_run01/SOURCE.json']:
        p=BC/'artifacts/non_df_h4_precision_bc/2026-10-23'/name;files[str(p)]=sha(p)
    for p,digest in files.items():assert sha(p)==digest,p
    return dict(files=files,science_INPUT_SOURCE='0c329b30a81accb35c212be8defa66fee0dd0f12',B_SOURCE='2bd212288e8530e519c4a8135257df37f7608662',B_results_commit=BASE)
def seal(source):
    runtime=read(HERE/'runtime_manifest.json');legacy=read(HERE/'legacy_runtime_manifest.json')
    scientific=read(OLD/'artifacts/non_df_h4_pilot/2026-10-23/run01/SOURCE.json')
    code={str(ROOT/n):sha(ROOT/n) for n in CODE}
    for name in CODE:
        assert subprocess.check_output(['git','show',source+':'+name],cwd=ROOT)==(ROOT/name).read_bytes(),name
    record=dict(kind='C_PRODUCTION_TECHNICAL_SOURCE_NOT_EXECUTION_GRANT',SOURCE=source,
        code_files=code,code_git_files={name:sha(ROOT/name) for name in CODE},runtime_files=runtime['files'],
        python_executable_sha256=sha(runtime['executable']),legacy_python_executable_sha256=sha(legacy['executable']),
        legacy_source_files={str(OLD/n):digest for n,digest in scientific['files'].items()},legacy_runtime_files=legacy['files'],input_files=read(HERE/'input_manifest.json')['files'],
        execution_authorized=False,grant_valid=False,dispatch_enabled=False,science_grant_issued=False,
        historical_scientific_SOURCE='0c329b30a81accb35c212be8defa66fee0dd0f12',input_manifest_sha256=sha(HERE/'input_manifest.json'),
        routing={k:read(HERE/'C_execution_contract.json')[k] for k in ['legacy_root','legacy_python','input_directory','method_map_path','saved_results']})
    for group in ['code_files','runtime_files','legacy_source_files','legacy_runtime_files','input_files']:
        for name,digest in record[group].items():assert sha(name)==digest,(group,name)
    save(HERE/'SOURCE.json',record);return record
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',action='store_true');p.add_argument('--seal');p.add_argument('--after',action='store_true');a=p.parse_args()
    if a.baseline:
        save(HERE/'preservation_before.json',preservation());save(HERE/'github_verification.json',github_check());save(HERE/'input_schema_static.json',schema());save(HERE/'input_manifest.json',inputs());print('PASS_PRESERVATION_GITHUB_STATIC_SCHEMA')
    elif a.seal:
        r=seal(a.seal);print(json.dumps({k:len(r[k]) for k in ['code_files','runtime_files','legacy_source_files','legacy_runtime_files','input_files']}))
    elif a.after:
        after=preservation();assert after==read(HERE/'preservation_before.json');save(HERE/'preservation_after.json',after);print('PASS_PRESERVATION_UNCHANGED')
