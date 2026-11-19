"""Bind fixed H6 Full PF signals and existing prices for the normal optimizer preview."""
import argparse
from copy import deepcopy
from fractions import Fraction
import gzip
import json
from pathlib import Path

from trottertracks import baseline_signal_pipeline as B
from trottertracks import pf_optimizer_connection as PF
from trottertracks import rpe_method_optimizer as O
from trottertracks import rpe_scaled_shots as S

ROOT = Path(__file__).resolve().parents[1]
COVERAGE = 'docs/research/h6_h8_rpe_readiness_20261119/COVERAGE.json'
SAVED = '3de5bb32e726127937e2989c37498c2646f9f579'
H4_PROBLEM = 'configs/rpe_method_optimizer/h4_df10_problem.json'
BASE = '97ec984197f1f01c401d1b10422ee44e1d7ebbcb'
PRICE_SOURCE = '73dcbe23678bfc50d5654f09e600d4cfbe75e697'
PRICE_INPUTS = 'docs/research/h6_df19_price_v2_20261110/PILOT_INPUTS.json'
SHORT_RUN = 'docs/research/h6_h8_rpe_readiness_20261119/short_prices_run01/'


def read(commit, path, selector=()):
    data = B.blob(ROOT, commit, path)
    value = json.loads(gzip.decompress(data) if path.endswith('.gz') else data)
    for key in selector:
        value = value[key]
    return value


def write(output, **files):
    output.mkdir(parents=True)
    for name, value in files.items():
        with (output / (name + '.json')).open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')


def short_price_refs(commit, coverage, inputs):
    """Bind the finite price receipt to the same Full19 snapshot and three cases."""
    receipt = read(commit, SHORT_RUN + 'INPUTS.json')
    results = read(commit, SHORT_RUN + 'AXIS_PRICES.json')
    early = {r['m']: r for r in coverage['rows'] if r['m'] in (0, 1, 2)}
    if (receipt['source'] != results['source'] or receipt['maximum_compile'] != 6
            or receipt['compiler'] != inputs['compiler_hash']
            or receipt['input_sha256'] != inputs['input_sha256']
            or receipt['provider'] != inputs['provider_source']
            or len(results['rows']) != len(early)
            or not 0 <= results['compile_count'] <= 6):
        raise ValueError('Short-price receipt does not match the fixed execution')
    cases = dict(receipt['cases'])
    refs = {}
    for i, row in enumerate(results['rows']):
        m = row['m']
        archive = early.get(m)
        candidate = row['candidate']
        if (archive is None or m in refs or candidate != cases.get(m)
                or candidate['id'] != archive['saved_round_candidate_id']
                or candidate['q'] != archive['q_PF']
                or candidate['T'] != float(archive['T'])
                or candidate['delta'] != float(archive['actual_delta_pf'])
                or row['preparation_hash'] != inputs['preparation_hash']['19']
                or set(row['axes']) != {'X', 'Y'}):
            raise ValueError('Short-price case does not match the archived physical round')
        refs[m] = {}
        for axis, result in row['axes'].items():
            if result['compiler'] != receipt['compiler'] or not result.get('fingerprint'):
                raise ValueError('Short-price axis lacks its fixed compiler or fingerprint')
            refs[m][axis] = dict(commit=commit, path=SHORT_RUN + 'AXIS_PRICES.json',
                                 selector=['rows', i, 'axes', axis, 'metrics', 'rz_count'])
    return refs


def build(output, private_config, short_prices_commit=None):
    coverage = read(SAVED, COVERAGE)['H6']
    old = json.loads(private_config.read_text())['systems']['H6']
    inputs = read(PRICE_SOURCE, PRICE_INPUTS)
    binding = deepcopy(inputs['scientific_binding'])
    if old['binding'] != binding or any(binding[k] != v for k, v in coverage['scientific_binding'].items()):
        raise ValueError('H6 saved physical bindings differ')
    core = read(BASE, H4_PROBLEM)
    core['problem'].update(system='H6', hamiltonian_data=dict(
        snapshot=old['inputs']['H6_snapshot.npz'], receipt=old['inputs']['H6_snapshot_receipt.json'],
        expected=dict(fragment_count=19, n_qubits=12, sector_dimension=400)))
    facts = O.hamiltonian_facts(core)
    if any(facts[k] != binding[v] for k, v in (('snapshot_sha256','snapshot'),
            ('state_hash','state'), ('hamiltonian_hash','hamiltonian'))):
        raise ValueError('H6 runtime receipt disagrees with the saved binding')
    target_ref = coverage['target_value_locator']
    core['target'].update(energy_ha=read(**target_ref), source=target_ref)
    core['accuracy']['target'] = 'SAVED_DF19_SOLVER_ENERGY_CONDITIONAL_PROXY'
    binding.update(source=dict(commit=PRICE_SOURCE, path=PRICE_INPUTS),
                   decomposition=dict(snapshot=binding['snapshot'], rank=19, completion='saved_weighted_Hermitian_completion'))
    uncertainty = read(**coverage['saved_round_ids'])['configuration']
    base = dict(binding=binding, h=O.enc(Fraction(.1)), profile=dict(compiler=binding['compiler'],
       control=binding['control'], wrapper='full_Hadamard_with_measurement', P=0,
       scope='single_hadamard_interrogation_without_state_preparation'),
       rmse_policy=dict(family='PR_PAPER_RMSE', id='H6_DF19_SCALED_RPE_Q3_PREVIEW',
           source=dict(commit=SAVED, path=COVERAGE), scientific_status='PROPOSED_NOT_ADOPTED',
           target_rmse_ha=O.enc(Fraction(S.EPSILON)), error_contract=dict(
             energy_prior_ha=uncertainty['energy_prior_ha'], decoder=uncertainty['decoder'],
             state_assumption='saved_numerical_state_conditional_ground')))
    api = B.published_api(str(ROOT))
    config = B.configure(ROOT, api.D.full_pf_spec(base, formula=coverage['formula'],
                          delta_pf=O.enc(Fraction(1.6)), rank=19, M=12), api=api)
    cid = config['id']
    physical = PF.physical_ids(ROOT, config)
    short = short_price_refs(short_prices_commit, coverage, inputs) if short_prices_commit else {}
    means, axes = [], []
    for row, archive in zip(config['rounds'], coverage['rows'], strict=True):
        m = row['m']
        old_id = archive['saved_round_candidate_id']
        if (m != archive['m'] or physical[m] != old_id or row['q_PF'] != archive['q_PF']
                or float(api.B.a.exact_scalar(row['T'])).hex() != float(archive['T']).hex()):
            raise ValueError('Canonical H6 round differs from the archived candidate')
        raw_ref = archive['raw_signal']
        raw = read(**raw_ref)
        if set(raw) != {'real','imag'} or any(not -1 <= v <= 1 for v in raw.values()):
            raise ValueError('Raw mean is not a finite physical X/Y input')
        means.append(dict(m=m, identity=row['identity'], law_binding=row['law_binding'],
            physical_candidate=dict(id=old_id), raw=raw, source=raw_ref,
            status='SIGNAL_AVAILABLE_NOT_RMSE_QUALIFIED'))
        for axis, source in short.get(m, archive['axis_price_source']).items():
            if source is None:
                continue
            value = read(**source)
            value = value['rz_count'] if isinstance(value, dict) and 'rz_count' in value else value
            cost = O.frac(value) if not isinstance(value, float) else Fraction(value)
            if cost.denominator != 1 or cost <= 0:
                raise ValueError('Expected a positive deterministic compiled integer')
            axes.append(dict(m=m, axis=axis, physical_id=old_id, configuration_id=cid, unit_price_RZ=int(cost),
                kind='DIRECT_COMPILED_DETERMINISTIC', price_evidence=source))
    def path(name):
        return str(output / (name + '.json'))
    registry = dict(schema=PF.SCHEMA, configurations={cid:dict(O.jsonable(PF.configuration_metadata(config)),
        configuration=path('CONFIGURATION'), signal_packet=path('SIGNALS'), prices=path('PARTIAL_PRICES'))})
    core['evidence']=dict(saved_catalog_inputs=path('CATALOG_INPUTS'), pf_candidate_index=path('CANDIDATE_INDEX'))
    core['methods']={'FULL_PF':dict(core['methods']['FULL_PF'], formulas=[coverage['formula']],
                                  p_values=None, configuration_ids=[cid])}
    core['execution_routes']={}
    core.pop('price_rules', None)
    comparison = dict(binding=binding, accuracy=dict(core['accuracy'], MSE_upper=O.enc(O.tau(core))),
                      target=core['target'], M=12)
    round_ = dict(round_id='H6_FULL_PF_SAVED_READINESS_V1', methods=['FULL_PF'],
        science_approved=False, initial='all', max_new_q3_per_method={'FULL_PF':0},
        pf_evidence_registry=path('REGISTRY'), search_overrides=dict(rule=S.RULE,
            scaled_shots=S.settings(dict(M=[12], grid=dict(min_k=0,max_k=0,divisor=4), compare_paper=False))))
    write(output, CATALOG_INPUTS=dict(comparison=comparison, sources=[]), CANDIDATE_INDEX=dict(candidates=[]),
          CONFIGURATION=config, SIGNALS=dict(configuration_id=cid, means=means),
          PARTIAL_PRICES=dict(coverage='PARTIAL_DIRECT_PRICE_EVIDENCE', axes=axes), REGISTRY=registry,
          PROBLEM=core, ROUND=round_)
    return cid


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--saved-input-config', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--short-prices-commit', help='fixed commit containing the additional six axis receipts')
    args=ap.parse_args()
    print(build(args.output.resolve(), args.saved_input_config, args.short_prices_commit))
