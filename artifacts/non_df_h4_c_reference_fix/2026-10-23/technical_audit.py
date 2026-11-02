"""Reference-fix technical sealing and known-path byte audits; no H4 evaluation."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
OLD=ROOT.parent/'non-df-h4-c-production-20261023'
PRIOR=OLD/'artifacts/non_df_h4_c_production/2026-10-23'
OLD_SOURCE='49c18ccc35f14041df45bfeff455ba5e67543182'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,sort_keys=True);f.write('\n')
def control():
    s=importlib.util.spec_from_file_location('reference_fix_control',ROOT/'src/trotterlib/h4_certificate_control.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def preservation():
    before=read(HERE/'preservation_before.json')
    for name,digest in before['files'].items():assert sha(name)==digest,name
    return dict(status='PASS_BYTE_IDENTICAL',files_verified=len(before['files']),
                before_manifest_sha256=sha(HERE/'preservation_before.json'),
                old_SOURCE_held=OLD_SOURCE,old_original_native_verified=False,
                original_supervisor_cause='UNKNOWN',old_N_G_changed=False,
                scope=before['scope'])

def seal(source):
    old=read(PRIOR/'SOURCE.json');runtime=read(HERE/'runtime_manifest.json')
    # Reuse the unchanged execution graph; replace audit/probe and add regressions.
    names=[n for n in old['code_git_files'] if not n.startswith('artifacts/non_df_h4_c_production/')]
    names += [str(p.relative_to(ROOT)) for p in (HERE/'technical_audit.py',HERE/'runtime_probe.py',HERE/'check_reference_regressions.py',HERE/'check_legacy_connection.py')]
    names=list(dict.fromkeys(names));code={n:sha(ROOT/n) for n in names}
    for name in names:
        data=subprocess.check_output(['git','show',source+':'+name],cwd=ROOT)
        assert hashlib.sha256(data).hexdigest()==code[name],name
    legacy=read(HERE/'legacy_runtime_manifest.json')
    record=dict(old, SOURCE=source, code_git_files=code,
                code_files={str(ROOT/n):digest for n,digest in code.items()},
                runtime_files=runtime['files'],python_executable_sha256=sha(runtime['executable']),
                legacy_runtime_files=legacy['files'],input_files=read(HERE/'input_manifest.json')['files'],
                input_manifest_sha256=sha(HERE/'input_manifest.json'),
                supersedes_implementation_SOURCE=OLD_SOURCE,old_SOURCE_hold_remains=True,
                correction='Apply reference exponential to encoded state before contraction; reject non-column-vector signal inputs',
                real_H4_validated_with_corrected_SOURCE=False,
                execution_authorized=False,grant_valid=False,dispatch_enabled=False,science_grant_issued=False)
    ctl=control();ctl.verify_seal(record)
    assert read(HERE/'input_manifest.json')==read(PRIOR/'input_manifest.json')
    assert read(HERE/'legacy_runtime_manifest.json')==read(PRIOR/'legacy_runtime_manifest.json')
    save(HERE/'SOURCE.json',record)
    contract=read(PRIOR/'C_execution_contract.json')
    contract.update(C_production_implementation_SOURCE=source,technical_seal_sha256=sha(HERE/'SOURCE.json'),
        output_namespace=str(ROOT/'artifacts/non_df_h4_c_evidence/NEXT_EXPLICIT_APPROVAL/reference_fix_run01'),
        status='REFERENCE_FIXED_TECHNICALLY_SEALED_AWAITING_NEW_EXPLICIT_SCIENCE_APPROVAL',
        supersedes_implementation_SOURCE=OLD_SOURCE)
    ctl.validate_contract(contract);ctl.validate_routing(contract,record)
    assert not any(contract[k] for k in ('execution_authorized','grant_valid','dispatch_enabled','consumed'))
    assert contract['grant_id'] is None and contract['approval_file'] is None
    assert not Path(contract['output_namespace']).exists()
    save(HERE/'C_execution_contract.json',contract)
    return dict(status='PASS_TECHNICAL_SEAL_NO_EXECUTION_GRANT',SOURCE=source,
                counts={k:len(record[k]) for k in ('code_files','runtime_files','legacy_source_files','legacy_runtime_files','input_files')},
                contract_SHA256=sha(HERE/'C_execution_contract.json'),seal_SHA256=sha(HERE/'SOURCE.json'))

def closure():
    ctl=control();s=read(HERE/'SOURCE.json');c=read(HERE/'C_execution_contract.json')
    ctl.verify_seal(s);ctl.validate_contract(c);ctl.validate_routing(c,s)
    assert sha(HERE/'SOURCE.json')==c['technical_seal_sha256']
    assert c['C_production_implementation_SOURCE']==s['SOURCE']
    assert not any(c[k] for k in ('execution_authorized','grant_valid','dispatch_enabled','consumed'))
    assert not Path(c['output_namespace']).exists()
    return dict(status='PASS_SEAL_AND_INACTIVE_CONTRACT',SOURCE=s['SOURCE'],preservation=preservation(),
                real_H4_calls=0,real_H4_B_axes_repeated=0,science_grants_issued=0,
                shot_calculations=0,G_accounting=0,push=False)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seal');p.add_argument('--closure',action='store_true');a=p.parse_args()
    if a.seal:out=seal(a.seal);save(HERE/'seal_audit.json',out)
    elif a.closure:out=closure();save(HERE/'closure_audit.json',out)
    else:out=preservation();save(HERE/'preservation_after.json',out)
    print(json.dumps(out,ensure_ascii=False))

if __name__=='__main__':main()
