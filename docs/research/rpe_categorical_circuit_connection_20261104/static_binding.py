"""Read bytes/literals only; never evaluate H4 operators, laws, means or bounds."""
from pathlib import Path
import hashlib
import inspect
import json
import subprocess

ROOT = Path(__file__).resolve().parents[3]
BASE = '82b2df6af80eb923302d35e369c4572ed9fa0a9f'
CAT = 'd2e7a7db683e116243eae061c0e4c87865054e9e'
CAT_SOURCE = 'f4905fff592351d33ba07b927892fb9532c99711'
C_SOURCE = '7279f700a58921209b851c8d9c363d38ff2f1d98'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def data(commit, path):
    return git('show', f'{commit}:{path}')


def binding(commit, path):
    raw = data(commit, path)
    return dict(commit=commit, path=path,
                blob=git('rev-parse', f'{commit}:{path}').decode().strip(),
                sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def collect():
    fixed_path = 'docs/research/rpe_18round_ideal_enclosure_preparation_20261104/fixed_inputs.json'
    cat_path = 'artifacts/rpe_18round_ideal_enclosure/2026-11-04/run01/RESULT.json'
    parameters_path = 'artifacts/rpe_controlled_hadamard_local_probes/2026-11-04/run02/parameters.json'
    fixed = json.loads(data(CAT_SOURCE, fixed_path))
    cat = json.loads(data(CAT, cat_path))
    parameters = json.loads(data(BASE, parameters_path))
    export_info = fixed['files']['EXPORT']
    export = json.loads(data(export_info['commit'], export_info['path']))
    m17_info = fixed['files']['M17RESULT']
    m17 = json.loads(data(m17_info['commit'],m17_info['path']))
    sources = json.loads((ROOT/'docs/research/rpe_controlled_hadamard_preparation_20261104/SOURCE_BINDINGS.json').read_text())
    production = []
    for entry in sources['production_sources']:
        path = entry['path']
        if path.startswith('src/'):
            record = binding(BASE,path)
            assert record['sha256'] == entry['sha256']
            assert data(C_SOURCE,path) == data(CAT_SOURCE,path) == data(BASE,path)
            assert (ROOT/path).read_bytes() == data(BASE,path)
            production.append(record)
    joins = []
    for p in sorted(parameters, key=lambda x:x['r']):
        group = next(g for g in cat['groups'] if g['r'] == p['r'])
        angles = [p['order_angles_hex'][str(n)] for n in (0,2)]
        assert angles == group['law']['theta64_hex']
        if 'orders' in group['law']:
            assert group['law']['orders'] == [0,2]
        rows = [row for row in fixed['rounds'] if row['r'] == p['r']]
        assert all(float(row['signal_tau64']).hex() == p['signal_tau_hex'] for row in rows)
        joins.append(dict(r=p['r'], parent_m=p['parent_m'], rounds=[row['m'] for row in rows],
                          signal_tau_hex=p['signal_tau_hex'], theta64_hex=angles,
                          theta_literal_equal=True, order_model='K2:(0,2)',
                          categorical_law_kind=group['law']['kind'],
                          order_law=group['law']['order_law']))
    rounds = []
    for row in fixed['rounds']:
        old = ({**m17['original_fixed_literals'], 'axes':m17['enclosures'][-1]['axes']}
               if row['m'] == 17 else
               next(v for g in cat['groups'] for v in g['rounds'] if v['m']==row['m']))
        assert all(str(old[k]) == str(row[k]) for k in ('m','q','r','K'))
        rounds.append({**{k:row[k] for k in ('m','q','r','K')},
                       'fixed_tuple_selector': f"/rounds/{row['m']}",
                       'categorical_axis_decisions': {a:v['categorical_decision'] for a,v in old['axes'].items()}})
    assert len(rounds) == 18 and sum(len(v['categorical_axis_decisions']) for v in rounds) == 36
    # Structural/literal check only: no trigonometric/matrix/H4 evaluation.
    from connection_adapter import gaussian_descriptor_check
    for spec in export['bases'].values():
        gaussian_descriptor_check(spec['operations'],export['n_qubits'])
    from qiskit.circuit.library import RZGate, RZZGate, XXPlusYYGate
    import qiskit.circuit.add_control as add_control
    library = []
    for path in sorted({inspect.getfile(x) for x in (RZGate,RZZGate,XXPlusYYGate)} | {add_control.__file__}):
        raw = Path(path).read_bytes()
        library.append(dict(path=path, sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
    relevant = [binding(CAT_SOURCE,fixed_path),binding(CAT,cat_path),binding(BASE,parameters_path),
                binding(BASE,'artifacts/rpe_controlled_hadamard_local_probes/2026-11-04/run02/RESULT.json'),
                binding(BASE,'docs/research/rpe_controlled_hadamard_runner_v2_20261104/source_inputs.json')]
    helpers = []
    for path in [*sorted(Path(__file__).parent.glob('*.py')),
                 ROOT/'docs/research/rpe_m17_rigorous_mean_enclosure_preparation_20261103/ball_adapter.py',
                 ROOT/'docs/research/rpe_m17_rigorous_mean_enclosure_preparation_20261103/mean_adapter.py']:
        raw = path.read_bytes()
        helpers.append(dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
    return dict(kind='STATIC_BYTES_AND_LITERAL_JOIN_NOT_SCIENTIFIC_REANALYSIS',
                base_commit=BASE,categorical_result_commit=CAT,categorical_SOURCE=CAT_SOURCE,
                controlled_SOURCE=C_SOURCE, evidence=relevant, fixed_inputs=fixed['files'],
                production_sources=production,technical_helpers=helpers,qiskit_gate_source=library,
                six_r_literal_joins=joins,rounds=rounds,
                saved_full_basis_descriptor_check=dict(bases=len(export['bases']),components=len(export['components']),
                    allowed_gates=['rz','adjacent_xx_plus_yy'],all_qualified=True,
                    scope='names/wires/finite parameter metadata only; support descriptors not available'),
                classification=dict(arbitrary_trajectory_algebra='ALGEBRAICALLY_ESTABLISHED',
                    builder_structure='SOURCE_CONSISTENT', existing_12_probes='NUMERICALLY_TESTED',
                    all18_production_circuit_transfer='UNRESOLVED'),
                new_H4_scientific_calls=0,RPE_tolerance_certified=False,
                scientific_status='UNDETERMINED',U_virtual_IID_upper=None,U_exec_upper=None)


if __name__ == '__main__':
    path=Path(__file__).with_name('FIXED_BINDINGS.json')
    path.write_text(json.dumps(collect(),ensure_ascii=False,indent=2)+'\n')
    print('Static byte/literal join saved; H4 scientific evaluations=0')
