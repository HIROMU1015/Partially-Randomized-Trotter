"""Parallel v3 technical boundary. No science release or authority provisioning.

The atomic ledger/approval engine is the unchanged R1 implementation. This new
validator binds the entire technical manifest and configuration. Production is
deliberately closed in this unadopted SOURCE; adoption requires a new release.
"""
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import sys

from . import g6_w2c_b3_admission as A
from . import g6_w2c_b3_parallel_driver as D

SCHEMA = 'b3_parallel_admission_v3'
PREFIX = 'src/trottertracks/algorithm_codesign/'
ROLES = {name: PREFIX+'g6_w2c_b3_parallel_'+name+'.py'
         for name in ('driver', 'observer', 'scheduler', 'admission', 'host')}
ROLES['serialization'] = PREFIX+'g6_w2c_b3_exact_serialization.py'
API = 'execute_approved_v3'
SCHEMA_DIR = 'docs/research/track_b_g6_w2c_b3_parallel_fused_r1_technical_integration_20261019/output_schema/'
SCHEMA_PATHS = [SCHEMA_DIR+n+'.schema.json' for n in ('contract','envelope','f01','pair','receipt','result','runtime')] + [SCHEMA_DIR+'protocol.json']
require = A.require


def python_identity():
    executable=Path(sys.executable).resolve()
    libraries={}
    for name in ('json','fractions','multiprocessing','multiprocessing.connection',
                 'sqlite3','_sqlite3','_json','_hashlib'):
        module=importlib.import_module(name)
        path=Path(module.__file__).resolve()
        libraries[name]=dict(path=str(path),sha256=A.digest(A.read_regular(path)))
    return dict(executable=str(executable), executable_sha256=A.digest(A.read_regular(executable)),
                version=sys.version, cache_tag=sys.implementation.cache_tag, libraries=libraries)


def manifest(root, pins):
    """No imports of science. The pins include frozen science bytes only."""
    return dict(schema='b3_parallel_runtime_v3', runtime_root=str(Path(root).resolve()),
                python=python_identity(), source_roles=ROLES.copy(), source_pins=dict(pins),
                schema_pins={p:A.digest((Path(root)/p).read_bytes())for p in SCHEMA_PATHS},
                entrypoint=dict(path=ROLES['host'], api=API, sha256=pins[ROLES['host']]),
                import_locations={n: {'module': 'trottertracks.algorithm_codesign.'+Path(p).stem,
                                      'path': p, 'sha256': pins[p]} for n,p in ROLES.items()})


def check_runtime(root, runtime):
    require(runtime['schema'] == 'b3_parallel_runtime_v3' and runtime['python'] == python_identity(),
            'RUNTIME_PYTHON_MISMATCH')
    require(runtime['source_roles'] == ROLES and set(runtime['import_locations']) == set(ROLES),
            'RUNTIME_ROLE_COVERAGE')
    runtime_root = Path(runtime['runtime_root'])
    require(runtime_root.is_absolute() and runtime_root.resolve() == runtime_root, 'RUNTIME_ROOT')
    for name, relative in ROLES.items():
        item = runtime['import_locations'][name]
        require(item == dict(module='trottertracks.algorithm_codesign.'+Path(relative).stem,
                             path=relative, sha256=runtime['source_pins'][relative]), 'IMPORT_MANIFEST_MISMATCH')
        module = importlib.import_module(item['module'])
        actual = Path(module.__file__).resolve()
        require(actual == A.exact_path(runtime_root, relative), 'RUNTIME_IMPORT_LOCATION:'+name)
        require(A.digest(A.read_regular(actual)) == item['sha256'], 'RUNTIME_SOURCE_SHA:'+name)
    # Also verify every loaded old management dependency covered by source pins.
    for relative, pin in runtime['source_pins'].items():
        if not relative.endswith('.py'): continue
        name = 'trottertracks.algorithm_codesign.'+Path(relative).stem
        module = sys.modules.get(name)
        if module is not None:
            actual = Path(module.__file__).resolve()
            require(actual == A.exact_path(runtime_root, relative) and
                    A.digest(A.read_regular(actual)) == pin, 'LOADED_DEPENDENCY_MISMATCH:'+name)
    require(set(runtime['schema_pins']) == set(SCHEMA_PATHS), 'SCHEMA_PIN_COVERAGE')
    for path,pin in runtime['schema_pins'].items():
        require(A.digest(A.read_regular(A.exact_path(runtime_root,path))) == pin and
                A.digest(A.read_regular(A.exact_path(Path(root),path))) == pin, 'SCHEMA_BYTES_MISMATCH')
    fixture=runtime.get('artificial_fixture')
    if fixture is not None:
        actual=A.exact_path(runtime_root,fixture['path'])
        require(A.digest(A.read_regular(actual)) == fixture['sha256'], 'ARTIFICIAL_RUNTIME_SOURCE_SHA')
    D.check_pins(root, runtime['source_pins'])


def validate(contract, root, artificial):
    require(contract.get('schema') == SCHEMA, 'PARALLEL_V3_CONTRACT_REQUIRED')
    require(contract.get('contract_grants_execution') is False and
            contract.get('content_adopted') is False and contract.get('run_ready') is False and
            contract.get('science_release') is None and contract.get('scientific_adoption') is None,
            'UNADOPTED_TECHNICAL_CANDIDATE_ONLY')
    require(contract.get('retry') == 0 and type(contract['retry']) is int and
            contract.get('resume') is False and contract.get('mandatory_STOP') is True, 'ONE_SHOT_REQUIRED')
    # This check follows the independent credential check in _check, before any
    # reservation, output creation, module loading or scientific calculation.
    require(artificial, 'TECHNICAL_CANDIDATE_NOT_SCIENCE_RELEASE')
    r1 = contract['r1_reference']
    old = A.validate_r1(r1, root, True)
    require(contract['scope'] == r1['scope'] and contract['claim_id'] == r1['claim_id'], 'GLOBAL_SCOPE_BINDING')
    for group in ('source_pins', 'input_pins'):
        require(type(contract[group]) is dict and contract[group], 'PINS_REQUIRED')
        for path,pin in contract[group].items():
            require(type(pin) is str and A.HEX.fullmatch(pin) and
                    A.digest(A.read_regular(A.exact_path(root,path))) == pin, group.upper()+'_MISMATCH:'+path)
        require(all(contract[group].get(p) == v for p,v in r1[group].items()), 'R1_PINS_CHANGED')
    require(set(contract['source_pins']) == set(r1['source_pins']) | set(ROLES.values()), 'SOURCE_ROLE_COVERAGE')
    expected = deepcopy(r1['candidate']); expected['source_pins'] = contract['source_pins']
    require(contract['candidate'] == expected, 'R1_SCIENTIFIC_POLICY_CHANGED')
    runtime = contract['runtime_manifest']
    require(runtime['source_pins'] == contract['source_pins'] and
            runtime['entrypoint'] == dict(path=ROLES['host'], api=API, sha256=contract['source_pins'][ROLES['host']]),
            'HOST_MANIFEST_BINDING')
    require(runtime.get('artificial_fixture') == dict(path='tests/tracks/algorithm_codesign/b3_parallel_integrated_fixture.py',
               sha256=contract['input_pins'].get('artificial-fixture.py')), 'ARTIFICIAL_FIXTURE_MANIFEST')
    require(contract['schema_pins'] == runtime['schema_pins'] and
            all(contract['input_pins'].get(p) == h for p,h in contract['schema_pins'].items()), 'SCHEMA_CONTRACT_BINDING')
    check_runtime(root, runtime)
    from .g6_w2c_b3_parallel_scheduler import Settings
    settings = Settings(**contract['settings']); settings.check()
    require(contract['measurement_condition'] == ('sequential_pairs' if settings.mode == 'S' else 'concurrent_pairs'),
            'COST_CONDITION_BINDING')
    require(contract['output_schema'] == 'b3_parallel_result_v3' and
            contract['observer'] == 'cached_all_call_return' and contract['transport'] == 'coalesced_reserve_enter',
            'TECHNICAL_SEMANTICS_BINDING')
    return dict(old, formal_digest=D.identity(contract), driver_contract_digest=D.identity(contract['candidate']),
                source_pins_digest=D.identity(contract['source_pins']), input_pins_digest=D.identity(contract['input_pins']))


class _ParallelAdmission(A._Admission):
    def _check(self, driver_digest):
        require(os.getpid() == self.owner_pid, 'FORK_OWNER_MISMATCH')
        require(type(self.authority) is A.PinnedApproval, 'INDEPENDENT_SCIENCE_APPROVAL_REQUIRED')
        binding = validate(self.contract, self.root, self.artificial)
        require(driver_digest == binding['driver_contract_digest'], 'DRIVER_CONTRACT_DIGEST')
        anchor = self.authority.anchor
        require(anchor.ledger_directory == str(self.ledger.directory) and
                anchor.ledger_store_id == self.ledger.store_id, 'TRUSTED_LEDGER_IDENTITY')
        proof = self.authority.verify(binding, self.artificial)
        return dict(binding, authorization=proof, ledger_directory=str(self.ledger.directory),
                    ledger_store_id=self.ledger.store_id), proof


class ProductionAdmission(_ParallelAdmission):
    def admit_once(self, digest):
        return self._admit(digest)


class ArtificialAdmission(_ParallelAdmission):
    artificial = True

    def admit_artificial(self, digest):
        return self._admit(digest)
