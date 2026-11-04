"""Default is bytes/schema/literal checks. --run is separately authorized H4 science.

The switch is not proof of permission. No grants, approval text or auth mechanism.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIOR = ROOT/'docs/research/rpe_categorical_circuit_connection_20261104'
sys.path.insert(0,str(HERE))
from integration_audit import require, Unknown
from connection_pipeline import STATUS


def load_definition(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def git(*args):
    env = {**os.environ,'GIT_OPTIONAL_LOCKS':'0','GIT_NO_LAZY_FETCH':'1'}
    return subprocess.check_output(['git',*args],cwd=ROOT,env=env)


def verify_bound(item, *, current=False):
    raw = git('show',item['commit']+':'+item['path'])
    blob = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    require(blob==item['blob'] and hashlib.sha256(raw).hexdigest()==item['sha256'],
            'fixed commit/blob/SHA256 mismatch: '+item['path'])
    if current:
        require((ROOT/item['path']).read_bytes()==raw,'current SOURCE/input changed: '+item['path'])
    return raw


def validate_tuple_join(rows, expected):
    require(len(rows)==len(expected)==18,'all18 tuple coverage')
    for row,old in zip(rows,expected,strict=True):
        require(all(row[k]==old[k] for k in ('m','q','r','K'))
                and row['q']==1<<row['m'] and row['K']==2,'round/r/K join mismatch')


def static_inputs():
    """No H deserialization, primitive/angle evaluation, interval arithmetic or constructor."""
    spec = json.loads((HERE/'SOURCE_INPUTS.json').read_text())
    require(spec['precision_bits']==96 and spec['passes']==1,'fixed precision/pass scope')
    previous = json.loads(verify_bound(spec['parent_bindings'],current=True))
    for entry in previous['production_sources']:
        verify_bound(entry,current=True)
    for entry in spec['dependencies']:
        verify_bound(entry,current=True)
    for entry in previous['technical_helpers']:
        require(hashlib.sha256((ROOT/entry['path']).read_bytes()).hexdigest()==entry['sha256'],
                'prior helper SOURCE changed')
    old_round = load_definition('_connection_fixed_round_runner',
        ROOT/'docs/research/rpe_18round_ideal_enclosure_preparation_20261104/run_round_enclosures.py')
    manifest,rows,raw = old_round.metadata()
    validate_tuple_join(rows,previous['rounds'])
    evidence = {e['path'].rsplit('/',1)[-1]+':'+e['commit']:verify_bound(e) for e in previous['evidence']}
    cat = json.loads(evidence['RESULT.json:'+previous['categorical_result_commit']])
    local = json.loads(evidence['RESULT.json:'+previous['base_commit']])
    params = json.loads(evidence['parameters.json:'+previous['base_commit']])
    export,m17 = json.loads(raw['EXPORT']),json.loads(raw['M17RESULT'])
    require(len(set(export['ordered_sector']))==70
            and all(type(s) is int and 0<=s<256 and s.bit_count()==4
                    for s in export['ordered_sector']),'little-endian N4 sector schema')
    require(cat['scientific_status']=='UNDETERMINED' and cat['RPE_tolerance_certified'] is False
            and cat['U_exec_upper'] is None and cat['U_virtual_IID_upper'] is None,
            'saved categorical scope/status changed')
    require(local['source_commit']==previous['controlled_SOURCE']
            and local['scientific_status']=='UNDETERMINED'
            and local['RPE_tolerance_certified'] is False,'local evidence SOURCE/status')
    require(spec['support_policy']=='support_for_source_basis_run_le_1'
            and spec['training_fingerprint']=='9dcdc4a28866dc223afb0ea212df645a3ed4a28e5c6dad6ddfc0ec1fcb28ba9e',
            'support policy/training fingerprint')
    fixed_adapter = load_definition('_connection_fixed_probe_adapter',
        ROOT/'docs/research/rpe_controlled_hadamard_runner_v2_20261104/probe_adapter.py')
    require(fixed_adapter.TRAINING==spec['training_fingerprint'],'fixed training SOURCE')
    theta,axes = {},{}
    for join in previous['six_r_literal_joins']:
        r = join['r']
        p = next(x for x in params if x['r']==r)
        group = next(x for x in cat['groups'] if x['r']==r)
        angles = [p['order_angles_hex'][str(n)] for n in (0,2)]
        require(angles==join['theta64_hex']==group['law']['theta64_hex']
                and p['signal_tau_hex']==join['signal_tau_hex'],'saved theta/tau literal join')
        require(group['law']['order_law']==join['order_law'],'saved uniform53 order law mismatch')
        for row in rows:
            if row['r']==r:
                require(float(row['signal_tau64']).hex()==p['signal_tau_hex'],'signal tau differs; no model fallback')
        theta[r] = angles
    for row in rows:
        m = row['m']
        old = ({**m17['original_fixed_literals'],'axes':m17['enclosures'][0]['axes']}
               if m==17 else next(x for g in cat['groups'] for x in g['rounds'] if x['m']==m))
        require(all(str(old[k])==str(row[k]) for k in ('m','q','r','K')),'saved categorical tuple join')
        require(set(old['axes'])=={'cosine','sine'},'axis coverage')
        for value in old['axes'].values():
            require(value['categorical_decision']=='SUFFICIENT_CONDITION_CONFIRMED'
                    and type(value['U_cat_upper']) is str
                    and set(value['interval'])==set(value['shot_half_width'])=={'lower','upper'},
                    'saved axis schema/decision')
        axes[m] = old['axes']  # Values are copied, never re-evaluated in this mode.
    require(len(export['components'])==218 and len(export['bases'])==11
            and len(local['support_proofs'])<=218,'component/basis coverage')
    # Relevant library SOURCE is compared without instantiating parameterized gates.
    import qiskit
    package = Path(qiskit.__file__).parent
    require(importlib.metadata.version('qiskit')=='1.3.0','Qiskit1.3.0 required')
    for entry in spec['qiskit_sources']:
        require(hashlib.sha256((package/entry['path']).read_bytes()).hexdigest()==entry['sha256'],
                'Qiskit gate SOURCE changed: '+entry['path'])
    return dict(mode='FIXED_H4_SCIENCE',manifest=manifest,rows=rows,raw=raw,export=export,
                theta_by_r=theta,saved_axes=axes,support_proofs=local['support_proofs'],
                support_policy=spec['support_policy'],training_fingerprint=spec['training_fingerprint'],
                library_evidence=True)


def describe():
    bundle = static_inputs()
    return {**STATUS,'status':'PREPARED_NOT_SCIENTIFICALLY_EXECUTED','precision_bits':96,
            'rounds':len(bundle['rows']),'axes':36,'components':218,'full_bases':11,
            'new_H4_scientific_calls':0,'default_mode':'bytes_schema_literals_only',
            'scientific_entry':'--run --output <fresh result directory>'}


def frozen_source():
    sha = git('rev-parse','HEAD').decode().strip()
    paths = [*HERE.glob('*.py'),HERE/'SOURCE_INPUTS.json',HERE/'SCIENTIFIC_EXECUTION_PLAN.md']
    for path in paths:
        require(path.read_bytes()==git('show',sha+':'+path.relative_to(ROOT).as_posix()),
                'freeze new scientific SOURCE before --run')
    versions = {k:importlib.metadata.version(k) for k in ('numpy','qiskit','python-flint')}
    require(versions=={'numpy':'1.26.4','qiskit':'1.3.0','python-flint':'0.9.0'}
            and sys.version_info[:2]==(3,11),'fixed numerical runtime')
    import flint
    require(flint.__FLINT_VERSION__=='3.6.0' and sys.float_info.radix==2
            and sys.float_info.mant_dig==53 and ctypes.CDLL(None).fegetround()==0,
            'FLINT/binary64/rounding mode')
    return dict(commit=sha,versions=versions,FLINT=flint.__FLINT_VERSION__)


def h4_context(bundle,record):
    """Only execute() behind --run calls this. No state load/norm or mean evaluation."""
    sys.path.insert(0,str(ROOT/'src'))
    from trotterlib.rte_connected_cluster_cost_validation import load_connected_cluster_hamiltonian_snapshot
    from trotterlib.df_partial_randomized_pf import df_hamiltonian_hash, split_df_hamiltonian_by_ld
    from trotterlib.df_partial_s2 import prepare_df_partial_s2
    from trotterlib.research_direction_sequence_policy import register_support_restricted_bases
    old = load_definition('_connection_fixed_probe_runner',
        ROOT/'docs/research/rpe_controlled_hadamard_runner_v2_20261104/run_local_probes.py')
    counts = record['counts']
    counts['hamiltonian_loads'] = 1
    h = load_connected_cluster_hamiltonian_snapshot(ROOT/bundle['manifest']['files']['H']['path'])
    require(df_hamiltonian_hash(h)==old.H_HASH and h.n_qubits==8 and h.n_blocks==12,'Hamiltonian identity')
    counts['preparations'] = 1
    prep = prepare_df_partial_s2(h,split_df_hamiltonian_by_ld(h,3),
        identity_policy='extract_identity_phase',coefficient_atol=1e-12)
    old.verify_preparation(prep,bundle['export'])
    counts['support_registrations'] = 1
    support,proofs = register_support_restricted_bases(h,prep)
    # No numerical residual is used as a certificate. IDs/hashes/counts fix the selected circuits.
    record['runtime_support_proofs'] = proofs
    from integration_audit import finite_hex
    for row in bundle['rows']:
        require(finite_hex(prep.exact_rte_lambda_r*(float(row['delta'])/row['r']))==
                float(row['signal_tau64']).hex(),'runtime signal tau mismatch')
    return dict(preparation=prep,support=support,export=bundle['export'],
                library_evidence=bundle['library_evidence'])


def resource_context(number_tasks):
    """Observe current resources; no fixed CPU/RAM/worker cap or shared setting change."""
    def cpu():
        values = [int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:]]
        return sum(values),values[3]+values[4]
    first = cpu()
    time.sleep(.2)
    second = cpu()
    idle = (second[1]-first[1])/max(1,second[0]-first[0])
    affinity = len(os.sched_getaffinity(0))
    memory = {line.split(':')[0]:int(line.split()[1])*1024
              for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line}
    process_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    require(memory['MemAvailable']>=process_rss,'insufficient available memory for one worker')
    # Each worker has this interpreter/import footprint plus tiny immutable 8x8 tasks.
    workers = min(number_tasks,max(1,math.ceil(affinity*idle)),
                  max(1,memory['MemAvailable']//max(1,process_rss)))
    return dict(workers=workers,affinity_cpus=affinity,observed_idle_fraction=idle,
                available_ram_bytes=memory['MemAvailable'],parent_peak_rss_bytes=process_rss,
                worker_FLINT_threads=1,precision_bits=96,
                policy='independent cached 8x8 matrices; current idle CPU and available memory',
                GPU='not selected for tiny analytic Acb matrices')


def run_science(output):
    bundle = static_inputs()
    source = frozen_source()
    output = Path(output).resolve()
    namespace = ROOT/'artifacts/rpe_controlled_connection_certificate'
    require(output.is_relative_to(namespace) and output!=namespace,'fresh certificate namespace required')
    resources = resource_context(11+len(bundle['support_proofs']))
    bundle['resources'] = resources
    old = load_definition('_connection_worker_environment',
        ROOT/'docs/research/rpe_controlled_hadamard_runner_v2_20261104/run_local_probes.py')
    from connection_pipeline import execute
    with old.worker_environment():
        return execute(output,bundle,h4_context,workers=resources['workers'],source=source)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',help='separately authorized fixed H4 evaluation')
    parser.add_argument('--output',type=Path,help='fresh directory under artifacts/rpe_controlled_connection_certificate')
    args = parser.parse_args(argv)
    require(args.run==bool(args.output),'--output is required only with --run')
    result = run_science(args.output) if args.run else describe()
    print(json.dumps(result if not args.run else {k:result[k] for k in
        (*STATUS,'status','connection_decision','counts')},ensure_ascii=False,indent=2))
    return 0 if not args.run or result['connection_decision'].startswith('CONDITIONAL_') else 2


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (OSError,KeyError,ValueError,subprocess.CalledProcessError) as exc:
        print('UNKNOWN/STOP: '+str(exc),file=sys.stderr)
        raise SystemExit(2)
