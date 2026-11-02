"""Isolated tol-only pilot identities and authorization; stdlib imports only."""
from pathlib import Path
from types import FunctionType, SimpleNamespace
import copy
import h46_m5_contract as legacy

ROOT = Path(__file__).resolve().parents[1]
require, read, sha, save = legacy.require, legacy.read, legacy.sha, legacy.save
POLICY = 'M5_TOL_ONLY_DF_EXPLICIT_GAUSSIAN_DIRECTIONAL_SCALAR_ONCE_V1'
LANES = tuple(s + '_TOL_PILOT_' + k for s in ('H4', 'H6') for k in ('DET', 'PR'))
COMPILER, COMPILER_HASH = legacy.COMPILER, legacy.COMPILER_HASH
TARGET_PATH = 'references/h46_m5_tol_pilot_inputs_v1.json'
TARGETS = read(ROOT / TARGET_PATH)['systems']
SOURCE_PATHS = legacy.SOURCE_PATHS + (
    'scripts/h46_m5_tol_contract.py', 'scripts/h46_m5_tol_native.py',
    'scripts/h46_m5_tol_saved_input.py', 'scripts/launch_h46_m5_tol_pilot.py',
    'scripts/prepare_h46_m5_tol_pilot.py', TARGET_PATH,
)


def rebind(module, overrides):
    """Reuse function code with private globals; never patch the original module."""
    namespace = dict(module.__dict__)
    namespace.update(overrides)
    for name, value in module.__dict__.items():
        if isinstance(value, FunctionType) and value.__module__ == module.__name__:
            fn = FunctionType(value.__code__, namespace, value.__name__, value.__defaults__, value.__closure__)
            fn.__kwdefaults__ = value.__kwdefaults__
            namespace[name] = fn
    # Explicit adapters must override the original functions after cloning.
    namespace.update(overrides)
    return namespace


def candidate(system, mode, label, prefix, q, K=None, policy=None, value=None):
    row = legacy.core.candidate(system, mode, label, prefix, q, K, policy, value)
    row.pop('id')
    row['input_series'] = TARGETS[system]['binding']['series']
    row['id'] = system + '_TOL_' + mode + '_' + legacy.core.digest(row)[:16]
    return row


_systems = copy.deepcopy(legacy.core.SYSTEMS)
for _s, _t in TARGETS.items():
    _systems[_s]['rank'] = _t['binding']['rank']
    _systems[_s]['anchor'] = _t['anchor']
_core = rebind(legacy.core, {'SYSTEMS': _systems, 'candidate': candidate})
core = SimpleNamespace(**_core)


def scope(lane):
    require(lane in LANES, 'FIXED_TOL_FOUR_LANES')
    system, kind = lane[:2], lane.rsplit('_', 1)[1]
    groups = core.pilot(system)['deterministic' if kind == 'DET' else 'PR']
    return dict(lane=lane, system=system, kind=kind,
        input_series=TARGETS[system]['binding']['series'],
        cohort='M5_TOL_ONLY_TECHNICAL_PILOT_V1', groups=groups,
        tasks=[dict(id=g['id']+'__'+str(i), group=g, sample=i)
               for g in groups for i in range(1 if kind == 'DET' else 2)],
        compile_axes=18 if kind == 'DET' else 12,
        reference_jobs=1 if kind == 'DET' else 0, T=.8,
        control='symmetric_directional', native_policy=POLICY,
        compiler=COMPILER, compiler_hash=COMPILER_HASH, numeric_tolerance=1e-9,
        coefficient_optimization=False, SCF_DF_state_generation=False,
        shot_accounting=False, ranking=False, independent_confirmation=False,
        retry=False, phase3_authorized=False, mandatory_STOP=True)


def seed_value(source, lane, task, outer, piece):
    return int(core.digest(['M5_TOL_ONLY_TECHNICAL_PILOT_V1', source, lane,
                            task, outer, piece])[:16], 16)


_legacy = rebind(legacy, dict(ROOT=ROOT, core=core, POLICY=POLICY, LANES=LANES,
    SOURCE_PATHS=SOURCE_PATHS, scope=scope, seed_value=seed_value))
registry = _legacy['registry']
reserve = _legacy['reserve']
consumed = _legacy['consumed']


def validate_inputs(config):
    """Check receipts/hashes without restoring arrays or building preparation."""
    system = config['system']
    require(config['binding'] == TARGETS[system]['binding'], 'FIXED_TOL_SERIES_BINDING')
    expected = {'H4_tol_snapshot.npz', 'H4_tol_receipt.json'} if system == 'H4' else {
        'H6_snapshot.npz', 'H6_snapshot_receipt.json', 'H6_df_receipt.json'}
    require(set(config['inputs']) == expected, 'TOL_SAVED_INPUT_ROLES')
    snapshot = config['inputs']['H4_tol_snapshot.npz' if system == 'H4' else 'H6_snapshot.npz']
    require(snapshot['sha256'] == config['binding']['snapshot'], 'TOL_SNAPSHOT_IDENTITY')
    if system == 'H4':
        receipt = read(config['inputs']['H4_tol_receipt.json']['path'])
        md = receipt['hamiltonian_metadata']
        require(receipt['status'] == 'COMPLETE_MANDATORY_STOP'
                and receipt['sha256'] == snapshot['sha256']
                and receipt['hamiltonian_hash'] == config['binding']['hamiltonian']
                and receipt['state_hash'] == config['binding']['state']
                and receipt['rank'] == config['binding']['rank'], 'H4_F_RECEIPT_IDENTITY')
        require(md['input_policy'] == 'TOL_ONLY_NO_CONFIG_FALLBACK'
                and md['df_tol_requested'] == 1e-8
                and md['final_rank_supplied'] is False
                and md['coefficient_order'] == 'decomposer_generation_order'
                and md['subsequent_coefficient_cutoff'] == 0, 'H4_TOL_POLICY')


_verify_original = _legacy['verify_seals']


def verify_seals(config):
    # Includes fixed Git bytes, every used input/provider/runtime byte, and seeds.
    sealed = {str(Path(e['path']).resolve()): e for e in config['sealed_files']}
    require(len(sealed) == len(config['sealed_files']), 'UNIQUE_SEALED_PATHS')
    for entry in config['inputs'].values():
        expected = sealed.get(str(Path(entry['path']).resolve()))
        require(expected is not None and expected['sha256'] == entry['sha256']
                and expected['bytes'] == entry['bytes'], 'USED_INPUT_PATH_MUST_BE_SEALED')
    used = [config['registry_path'], config['excluded_seed_path']]
    used += [ROOT/p for p in SOURCE_PATHS]
    require(all(str(Path(p).resolve()) in sealed for p in used), 'USED_SOURCE_SEED_PATH_MUST_BE_SEALED')
    require(config['runtime']['provider_root'] == config['provider_root'], 'PROVIDER_PATH_BINDING')
    _verify_original(config)
    validate_inputs(config)


_legacy['verify_seals'] = verify_seals
authorize = _legacy['authorize']
