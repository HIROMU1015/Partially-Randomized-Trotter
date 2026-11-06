"""Server preparation for a fresh run; no restart of the stopped local run.

The inherited contract and SOURCE files remain historical originals. This
module records the limited execution differences and current scientific SOURCE.
User permission is provided in chat, never manufactured or authenticated here.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

REFERENCE = 'a7face80e10fd74e1a49dc530409db5b7f677aaa'
BASE_CONTRACT = 'docs/research/research_c_integrated_contract_unapproved_20261105.json'
BASE_SOURCE = 'artifacts/research_c_integrated/2026-11-05/execution_preflight/SOURCE.json'
FAMILY = 'artifacts/research_c_integrated/2026-11-06'
TECHNICAL = FAMILY + '/server_technical'
CONTRACT = TECHNICAL + '/contract.json'
SOURCE = TECHNICAL + '/SOURCE.json'
ADDED_SOURCE = (
    'src/trottertracks/c_full_unitary.py',
    'src/trottertracks/c_integrated_server.py',
    'src/trottertracks/c1_h6_state_action.py',
    'scripts/prepare_research_c_integrated_server.py',
    'scripts/benchmark_research_c_full_unitary.py',
    'tests/management/test_research_c_full_unitary.py',
    'tests/management/test_research_c_integrated_server.py',
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n')


def resources(root):
    affinity = sorted(os.sched_getaffinity(0))
    memory = next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()
                  if x.startswith('MemAvailable:'))
    groups = []
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        if not line.startswith('0::'): continue
        path = Path('/sys/fs/cgroup')/line[3:].lstrip('/')
        while path.is_relative_to('/sys/fs/cgroup'):
            values = {key:(path/key).read_text().strip() for key in
                ('cpu.max','memory.max','memory.current','pids.max') if (path/key).is_file()}
            groups.append(dict(path=str(path), **values))
            if path == Path('/sys/fs/cgroup'): break
            path = path.parent
    return dict(host=os.uname().nodename, affinity=affinity,
        available_memory_bytes=memory, available_disk_bytes=shutil.disk_usage(root).free,
        cgroup_hierarchy=groups, loadavg=os.getloadavg(),
        schedulers={name:shutil.which(name) for name in ('squeue','qstat','bjobs')},
        disk_quota_tool=shutil.which('quota'),
        quota_qualification='cgroup limits observed; absent quota command does not prove no disk quota')


def effective_contract(root, base):
    c = deepcopy(base)
    c.update(date_JST='2026-11-06', status='SERVER_PREPARED_NOT_EXECUTED',
        source_code_commit=None, technical_source=SOURCE,
        fresh_outputs=[FAMILY+'/development/run01', FAMILY+'/transfer/run01'])
    c['environment']['fit_python'] = str(root/'.venv-fit/bin/python')
    c['environment']['evaluation_python'] = str(root/'.venv-eval/bin/python')
    c['server_delta'] = dict(reference_commit=REFERENCE,
        base_contract=BASE_CONTRACT, base_contract_sha256=sha(root/BASE_CONTRACT),
        old_SOURCE_and_outputs_modified=False, retry=0, resume=0,
        missing_post_stop_results='Treat as unavailable per user instruction; fresh development trajectories required',
        evaluator='full_identity_local_matrix_aer_cpu_double_v1',
        retained_checks='all512 raw/compiled plus independently simulated actual inverses, unitarity and X/Y',
        reduced_samples=False, reduced_input_space=False, tolerances_changed=False,
        scientific_scope='same registered development/transfer grid, selection32/final64, same-JU D; fresh lineage',
        source_changes='full operator evaluation, immutable event cache, server paths and native timing only',
        scientific_execution='Only after explicit user permission for this new SOURCE and contract')
    return c


def prepare(root):
    root = Path(root).resolve()
    c = effective_contract(root, read(root/BASE_CONTRACT))
    if any((root/p).exists() for p in c['fresh_outputs']) or (root/FAMILY/'INTEGRATED_STOP.json').exists():
        raise ValueError('FRESH_SERVER_RUN_REQUIRED_NO_RETRY_RESUME')
    write(root/CONTRACT, c)
    inherited = read(root/BASE_SOURCE)['files']
    paths = set(inherited) | set(ADDED_SOURCE) | {BASE_CONTRACT, BASE_SOURCE, CONTRACT}
    files = {p:sha(root/p) for p in sorted(paths)}
    changed = {p:dict(before=h, after=files[p]) for p,h in inherited.items() if files[p] != h}
    write(root/SOURCE, dict(schema='C_INTEGRATED_SERVER_SOURCE_V1',
        reference_commit=REFERENCE, base_SOURCE=BASE_SOURCE,
        base_SOURCE_sha256=sha(root/BASE_SOURCE), files=files, changed_inherited_files=changed,
        scientific_execution=0))
    return preflight(root)


def preflight(root):
    root = Path(root).resolve()
    base, c, source = (read(root/p) for p in (BASE_CONTRACT, CONTRACT, SOURCE))
    if c != effective_contract(root, base):
        raise ValueError('UNREGISTERED_SERVER_CONTRACT_DIFFERENCE')
    if (source['reference_commit'] != REFERENCE
            or source['base_SOURCE_sha256'] != sha(root/BASE_SOURCE)):
        raise ValueError('SERVER_BASE_SOURCE_IDENTITY')
    for p,digest in source['files'].items():
        path = (root/p).resolve()
        if not path.is_relative_to(root) or sha(path) != digest:
            raise ValueError('SERVER_SOURCE_OR_INPUT_DRIFT:'+p)
    for p,digest in base['saved_development_inputs'].items():
        if sha(root/p) != digest: raise ValueError('DEVELOPMENT_INPUT_DRIFT:'+p)
    if any((root/p).exists() for p in c['fresh_outputs']) or (root/FAMILY/'INTEGRATED_STOP.json').exists():
        raise ValueError('FRESH_SERVER_RUN_REQUIRED_NO_RETRY_RESUME')
    environments = {}
    for role in ('fit','evaluation'):
        pins = base['environment'][role+'_packages']
        python = c['environment'][role+'_python']
        code = ('import importlib.metadata as m,json,sys;print(json.dumps(dict('
                'python=sys.version,packages={k:m.version(k) for k in '+repr(pins)+'})))')
        value = json.loads(subprocess.check_output([python,'-B','-c',code],text=True))
        if value['packages'] != pins: raise ValueError('PACKAGE_PINS:'+role)
        environments[role] = value
    expected = base['environment']['ffsim_sources']
    code = ('import hashlib,importlib.metadata as m,json;from pathlib import Path;d=m.distribution("ffsim");'
            'print(json.dumps({k:hashlib.sha256(Path(d.locate_file("ffsim/"+k)).read_bytes()).hexdigest() for k in '
            +repr(expected)+'}))')
    actual = json.loads(subprocess.check_output([c['environment']['fit_python'],'-B','-c',code],text=True))
    if actual != expected: raise ValueError('FFSIM_SOURCE_DIFFERENCE')
    code = ('import sys;sys.path.insert(0,'+repr(str(root/'src'))+');'
            'from trottertracks.c_j_u_precision import require_extended;require_extended();print("EXTENDED_PASS")')
    if subprocess.check_output([c['environment']['fit_python'],'-B','-c',code],text=True).strip() != 'EXTENDED_PASS':
        raise ValueError('EXTENDED_ACCUMULATION_UNAVAILABLE')
    observation = resources(root)
    return dict(status='SERVER_TECHNICAL_PREFLIGHT_PASS',scientific_execution=0,
        SOURCE_sha256=sha(root/SOURCE),contract_sha256=sha(root/CONTRACT),
        changed_inherited_files=source['changed_inherited_files'],environments=environments,
        resources=observation, maximum_scientific_units=c['maximum_calls']['total'],
        next_action='explicit permission for fresh scientific execution of new SOURCE/contract')
