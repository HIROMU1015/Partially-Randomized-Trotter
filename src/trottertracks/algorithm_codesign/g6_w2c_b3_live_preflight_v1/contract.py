"""Exact static preflight; fake-only authority never accepts a real grant/store."""
from pathlib import Path
from hashlib import sha256
import re
import subprocess
import sys
from . import spec as S
from .guard import install, require, loaded
from ..g6_w2c_b3_real_adapter_v1 import contract as OLD
from ..g6_w2c_b3_api_binding_v1.static import runtime
from .. import g6_w2c_b3_parallel_admission as A


def candidate(root):
    install()
    root = Path(root).resolve()
    old = OLD.candidate(root)
    pins = dict(old['source_pins'])
    paths = list((root/S.PKG).glob('*.py')) + list((root/S.DOC).glob('*.py'))
    paths += [root/'tests/tracks/algorithm_codesign/test_g6_w2c_b3_live_preflight_v1.py']
    pins.update({str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest() for p in paths})
    source_id = S.digest(pins)
    worker = A.manifest(str(root), pins)
    worker['artificial_fixture'] = dict(path=S.PKG+'artificial.py', sha256=pins[S.PKG+'artificial.py'])
    return dict(schema='b3_live_preflight_closed_candidate_v1', **S.CLOSED,
        runtime_root=str(root), fixed_base=S.BASE, scientific_reference_commit=S.SCIENCE,
        independent_acceptance_commit=S.INDEPENDENT, previous_artifact='024d8812e97c5d541391533680fdbb3aee4cf6d8e51f020d3302a00767af42cb',
        source_pins=dict(sorted(pins.items())), input_pins=old['scientific_reference']['input_pins'],
        source_release_candidate=source_id, input=S.INPUT, precision=S.PRECISION, angle=S.ANGLE,
        api_signatures=S.signatures(root), routes=[S.route(a,l) for a in S.ARMS for l in ('A','B')],
        scope=S.scope(), python=runtime(), worker_runtime=worker,
        authority_requirements=authority_requirements(), monitoring=monitoring(),
        science_loader='ABSENT_REQUIRES_SEPARATE_VERSIONED_RELEASE')


def verify_source_commit(c, commit):
    """Second sealing stage: compare frozen SOURCE Git objects to exact pins.

    Documentation-only commits can follow without changing the measured SOURCE.
    Unrelated worktree HEAD/status are never execution prerequisites.
    """
    require(type(commit) is str and re.fullmatch('[0-9a-f]{40}',commit), 'FULL_SOURCE_COMMIT')
    root=c['runtime_root']
    require(subprocess.check_output(['git','cat-file','-t',commit],cwd=root,text=True).strip()=='commit','SOURCE_COMMIT_TYPE')
    for path,h in {**c['source_pins'],**c['input_pins']}.items():
        blob=subprocess.check_output(['git','show',commit+':'+path],cwd=root)
        require(sha256(blob).hexdigest()==h,'GIT_SOURCE_PIN:'+path)
    return dict(commit=commit,source_release_candidate=c['source_release_candidate'],
                checked=len(c['source_pins'])+len(c['input_pins']),PASS=True)


def authority_requirements():
    return dict(schema='b3_correctness_authority_requirements_candidate_v1', real_authority=False,
        required_bindings=['source_release_candidate','input','precision','angle','python','worker_runtime',
            'seed','scope','global_claim','budgets','storage_destination','monitoring','STOP','retry','resume'],
        external_NOT_ESTABLISHED=['owner','credential_key_custody','revocation','atomic_claim',
            'commit_ACK_unknown_procedure','power_loss_durability','worker_lineage_containment',
            'receipt_caller_durability'], real_store_access_implemented=False)


def monitoring():
    return dict(total_cpu='parent CPU + all joined worker CPU exactly once',
        wall='global monotonic outer wall, never sum of pair walls',
        categories=['setup','online_sample','bits','coefficient_child_law','audit','angle','requery',
            'codec','IPC','observer','storage','RSS_HWM','IO','STOP_drain_close'],
        partitions='coarse and fine each reconcile separately with outer; residual explicit; never add together',
        real_cost='NOT_ESTABLISHED', pressure='observe; restore monitoring after transient host contention',
        failure='unknown never PASS; actual OOM, worker failure, corruption => STOP',
        CPU_RAM_wall_caps=None)


def manifest(c, scope, destination, *, artificial_faults=()):
    require(type(scope) is str and scope.startswith('ARTIFICIAL_ONLY/CORRECTNESS/'), 'FAKE_SCOPE_ONLY')
    require(all(type(f) is dict and f.get('stage') in ('generator','angle','requery','codec','bits','audit','evidence','worker_exit')
                and f.get('mode') in ('S','P1') for f in artificial_faults), 'ARTIFICIAL_FAULT_ONLY')
    return dict(schema='b3_live_preflight_artificial_manifest_v1', **S.CLOSED,
        candidate_binding=S.digest(c), source_release_candidate=c['source_release_candidate'],
        global_scope=scope, global_claim=S.digest(['FAKE_ONLY',scope,S.digest(c)]),
        run_id=S.digest(['FAKE_CORRECTNESS_RUN',scope,S.digest(c)]),
        seed=dict(artificial='ARTIFICIAL_REAL_ADAPTER_V1', proposed_science=S.SEED,
                  mode_in_seed=False, domain='namespace/arm/axis', IID_claim=False),
        budgets=S.budgets(), planned=512, tasks=S.tasks(), modes=['S','P1'],
        storage_destination=str(Path(destination).resolve()), retention='ALL_TRIAL_FULL_DOCUMENTS',
        fake_authority=True, real_grant=None, worker_runtime_binding=S.digest(c['worker_runtime']),
        monitoring_binding=S.digest(c['monitoring']), artificial_faults=list(artificial_faults))


def preflight(c, m, *, destination):
    install()
    require(S.encoded(c) == S.encoded(candidate(c['runtime_root'])), 'STATIC_CANDIDATE_BINDING')
    require(S.encoded(m) == S.encoded(manifest(c,m['global_scope'],destination,
        artificial_faults=m['artificial_faults'])), 'EXACT_MANIFEST_BINDING')
    require(not loaded(), 'CORE_ALREADY_LOADED')
    require(Path(destination).resolve() == Path(m['storage_destination']), 'STORAGE_BINDING')
    for path,h in c['input_pins'].items():
        require(sha256((Path(c['runtime_root'])/path).read_bytes()).hexdigest() == h, 'INPUT_PIN:'+path)
    for actual, expected in OLD.OLD.pins_loaded(c):
        require(actual == expected, 'LOADED_SOURCE_LOCATION')
    return S.digest(m)


def production_entry(*args, **kwargs):
    raise RuntimeError('NO_SCIENTIFIC_LOADER_IN_LIVE_PREFLIGHT_V1')
