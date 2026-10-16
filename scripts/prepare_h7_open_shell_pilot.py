"""Byte/metadata-only closure/profile/readiness bundle; never issues a grant."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.h7_open_shell.contract import draft,call_limits,digest,BASE


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def write(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    source=subprocess.check_output(['git','rev-parse',a.source+'^{commit}'],cwd=ROOT).decode().strip()
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()!=source:raise ValueError('SOURCE_NOT_CURRENT_HEAD')
    out=Path(a.output).resolve()
    if not out.is_relative_to('/home/AbeHiromu'):raise ValueError('HOME_ONLY')
    out.mkdir(parents=True,exist_ok=False)
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',source,'src','scripts','tests/test_h7_open_shell.py'],cwd=ROOT).decode().splitlines()
    # Conservative superset: includes all library/provider sources, even if a
    # dynamically imported native dependency is not visible to a static walker.
    paths=[x for x in paths if x.endswith('.py') and (x.startswith('src/') or x.startswith('scripts/run_h7_') or x.startswith('scripts/prepare_h7_') or x=='tests/test_h7_open_shell.py')]
    sources={}
    for name in paths:
        raw=subprocess.check_output(['git','show',source+':'+name],cwd=ROOT)
        if hashlib.sha256(raw).hexdigest()!=sha(ROOT/name):raise ValueError('SOURCE_DIRTY:'+name)
        sources[name]=dict(blob=subprocess.check_output(['git','rev-parse',source+':'+name],cwd=ROOT).decode().strip(),bytes=len(raw),sha256=sha(ROOT/name))
    write(out/'source_closure_v1.json',dict(SOURCE=source,base=BASE,files=sources,superset=True))
    packages=('qiskit','numpy','scipy','numba','openfermion','pyscf','openfermionpyscf')
    versions={name:importlib.metadata.version(name) for name in packages}
    expected=dict(qiskit='1.3.0',numpy='1.26.4',scipy='1.14.1',numba='0.60.0',openfermion='1.6.1',pyscf='2.7.0')
    if any(versions[k]!=v for k,v in expected.items()) or sys.version_info[:2]!=(3,11):raise ValueError('PINNED_ENVIRONMENT')
    prefix=Path(sys.prefix).resolve()
    if prefix==Path(sys.base_prefix).resolve():raise ValueError('PRIVATE_VENV_REQUIRED_FOR_PROFILE')
    files={}
    for path in sorted(prefix.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':files[str(path)]=sha(path)
    files[str(Path(sys.executable).resolve())]=sha(Path(sys.executable).resolve())
    profile=dict(python=sys.version,executable=sys.executable,prefix=str(prefix),versions=versions,files=files,
        compiler_hash=draft()['compiler_hash'],old_compiler_output_bit_identity_verified=False,
        environment_modified=False)
    write(out/'private_environment_profile_v1.json',profile)
    write(out/'environment_profile_summary_v1.json',dict(python=sys.version,executable=sys.executable,versions=versions,
        file_count=len(files),profile_digest=digest(profile),compiler_hash=draft()['compiler_hash'],
        old_compiler_output_bit_identity_verified=False,environment_modified=False,scientific_calls=0))
    write(out/'execution_contract_draft_v1.json',draft())
    write(out/'binding_draft_v1.json',dict(SOURCE=source,source_hashes={k:v['sha256'] for k,v in sources.items()},
        profile_digest=digest(profile),profile_path=str(out/'private_environment_profile_v1.json'),
        contract_digest=digest(draft()),target_digest=digest(draft()['target']),
        input_digest=None,rank=None,carry_digest=None,output=None,worker_cpus=[],coordinator_cpu=None,
        approved=False,runtime_authorization=False,grant=None,sealed=False,absolute_launch_command=None,
        run_id=None,one_shot_consumed=False,new_scientific_calls=0))
    write(out/'rank_dependent_call_plan_examples_v1.json',dict(actual_H7_rank=None,examples_are_not_rank_predictions=True,
        examples={str(rank):call_limits(rank) for rank in (1,9,17,25)},
        occurrence_definition='one RTE event per micro-step; sampling_call is one request for two events',
        actual_rank_registry_and_coverage_frozen_after_authorized_input=True))
    print(json.dumps(dict(status='PREPARED_NOT_AUTHORIZED',SOURCE=source,source_files=len(sources),profile_files=len(files),
        profile_digest=digest(profile),output=str(out),science=0)))


if __name__=='__main__':main()
