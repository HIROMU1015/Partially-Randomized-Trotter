"""Source-bound copies of published H4 PR diagnostics for B's statistics layer.

No numerical reanalysis, probability generation, N solver, operator action,
normalizer, decoder or qualification of the physical error model is performed.
"""
from copy import deepcopy

from . import pr_rpe_rmse_s0_v2 as V
p = V.p

RELEASE = '169d3316dd008eb98dd9be56fe4ecefc5c1a7d60'
SCIENCE = 'c76ffe5dbb2a8dd1a05f1a85038f2d48b959f73e'
B = '3216d27a69ac8532d2a22e78f5f94491d202d238'
DIR = 'docs/research/h4_pr_s1_s2_rmse_results_20261113/'
CONFIG = dict(commit=B, path='docs/research/h4_s0_v2_integration_20261113/run01/CONFIGURATION_V2.json')
N_PACKET = dict(commit=RELEASE, path=DIR+'run01/N_HANDOFF.json')
M1 = '3e035c0d61386010f59ad7ea792bc7a5cefd18ac'
M1_SCIENCE = 'ac873e7fcc42e937084fa153476cbd87565702b4'
M1_PATH = 'docs/research/h4_pr_m1_rmse_results_20261113/run01/result.json'
TRUTH = dict(commit='89c91c692e1d35de94d2e7afbe890aabe6d80b7e',
    path='docs/research/h4_df10_rpe_batch_results_20261110/run01/result.json', selector=['records', 5, 'diagnostics'])
CONFIG_ID = '587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911'


def same(left, right, label):
    if not p.same(left, right):
        raise ValueError(label)


def load(root):
    """Read immutable public JSON, with each original diagnostic locator intact."""
    store = p.SavedArtifacts(root)
    config = store.read(CONFIG)
    V.b_metadata(root).validate(config)  # B owns canonical identity validation.
    packet = store.read(N_PACKET)
    s0 = store.read(dict(commit=V.S0, path=V.S0_DIR+'result.json'))
    m1 = store.read(dict(commit=M1, path=M1_PATH))
    batch = store.read(dict(commit=RELEASE, path=DIR+'run01/result.json'))
    same(s0['scientific_source'], V.S0_SCIENCE, 'S0 scientific SOURCE')
    same(m1['science_source'], M1_SCIENCE, 'm1 scientific SOURCE')
    same(batch['source'], SCIENCE, 'S1/S2 scientific SOURCE')
    entries = []
    for i, (d, model) in enumerate(zip(s0['diagnostics'], s0['model_records'])):
        entries.append(dict(m=model['identity']['m'], diagnostics=d, model=model,
            scientific_source=V.S0_SCIENCE,
            source=dict(commit=V.S0, path=V.S0_DIR+'result.json', selector=['diagnostics', i])))
    entries.append(dict(m=1, diagnostics=m1['diagnostics'], model=m1['model_record'],
        scientific_source=M1_SCIENCE, source=dict(commit=M1, path=M1_PATH, selector=['diagnostics'])))
    models = {v['identity']['m']: v for v in batch['model_records']}
    for i, record in enumerate(batch['diagnostics']):
        entries.append(dict(m=record['m'], diagnostics=record['diagnostics'], model=models[record['m']],
            reference_context=record['reference'], mean_source=record['mean_source'], scientific_source=SCIENCE,
            source=dict(commit=RELEASE, path=DIR+'run01/result.json', selector=['diagnostics', i, 'diagnostics'])))
    source_index = store.read(dict(commit=RELEASE, path=DIR+'SOURCE_INDEX.json'))
    truth = store.read(TRUTH)['records'][5]['diagnostics']
    return config, packet, entries, truth, source_index, store.sources


def assemble(config, packet, entries, truth, source_index):
    """Validate field associations, then copy; never derive a new physical value."""
    same(config['id'], CONFIG_ID, 'Fixed B configuration')
    same(packet['configuration_id'], config['id'], 'N configuration')
    same(packet['policy'], config['spec']['rmse_policy'], 'RMSE policy')
    if packet['status'] != 'UNQUALIFIED' or packet['rmse_evidence']['rmse_bound_ha'] is not None:
        raise ValueError('Published global qualification must remain unresolved')
    if len(config['rounds']) != 13 or len(packet['rounds']) != 13 or len(entries) != 13:
        raise ValueError('All 13 physical rounds required')
    by_m = {e['m']: e for e in entries}
    same(sorted(by_m), list(range(13)), 'Missing or duplicate diagnostic round')
    rows = []
    for m, (r, n) in enumerate(zip(config['rounds'], packet['rounds'])):
        e = by_m[m]; d = e['diagnostics']; model = e['model']; c = r['physical_candidate']
        same([r['m'], n['m'], model['identity']['m']], [m, m, m], 'Round order')
        same(n['identity'], r['identity'], 'Whole schedule identity / M / kappa / binding')
        same(r['identity']['configuration']['spec'], config['spec'], 'Canonical configuration spec')
        same(r['identity']['configuration']['configuration_id'], config['id'], 'Canonical configuration ID')
        same(n['law_binding'], r['law_binding'], 'Law or SOURCE mismatch')
        same(n['source_candidate_id'], c['id'], 'Physical ID')
        same(d['candidate'], c, 'Diagnostic physical candidate')
        same(model['identity']['candidate'], c, 'Model physical candidate')
        for key in ('structure', 'q_RPE', 'q_PF'):
            same(model['identity'][key], r['identity'][key], 'A/B '+key)
        for key in ('binding', 'profile', 'contract'):
            same(model['identity'][key], config['spec']['A_input']['design'][key], 'A '+key)
        same(model['request_id'], n['source_request_id'], 'A request identity')
        same(model['policy_id'], 'PR_RPE_PAPER_V2_EMPIRICAL_RMSE_V1', 'Policy separation')
        same(model['signal'], d['signal'], 'Mean / normalization record association')
        same([model['nominal_N_X'], model['nominal_N_Y']], [n['nominal_N_X'], n['nominal_N_Y']], 'Stored nominal N')
        same(e['scientific_source'], n['diagnostic_source']['commit'], 'Original diagnostic SOURCE')
        expected_carrier = V.S0 if m in (0, 6) else M1 if m == 1 else RELEASE
        same(e['source']['commit'], expected_carrier, 'Original published diagnostic carrier')
        if d['numerical_gates_passed'] is not True or d['rigorous_error_bound'] is not False:
            raise ValueError('Saved gates or qualification changed')
        if any(n[k] is not None for k in ('N_X', 'N_Y')) or any(model[k] is not None for k in ('sufficient_N_X', 'sufficient_N_Y')):
            raise ValueError('No sufficient N is available')
        for field in ('reference', 'raw', 'corrected', 'exact_tail', 'total_signed_bias', 'outer_PF_or_discard_bias', 'finite_RTE_bias'):
            if set(d['signal'][field]) != {'real', 'imag'}:
                raise ValueError('Missing complex signal or bias field')
        # Read required saved fields; do not re-evaluate their gates or formulae.
        for key in ('normalization', 'reference_pair', 'bias_decomposition_residuals', 'allowance',
                    'observed', 'conditional_error_disk', 'conditional_geometry', 'actual_implementation'):
            if not isinstance(d[key], dict):
                raise ValueError('Missing diagnostic evidence: '+key)
        same(d['signal']['log_B'], d['normalization']['log_B'], 'Normalizer association')
        if any(d['actual_implementation'][k] is not None for k in ('raw_error', 'law_error', 'ground_identification_error', 'RMSE_bound_ha')):
            raise ValueError('Unknown actual / ground errors must stay null')
        if m in (3, 4):
            norm = d['norm_evidence']
            if norm['kind'] != 'SOURCE_GATE_LIMIT_NOT_OBSERVED' or norm['observed_error'] is not None:
                raise ValueError('m3/m4 observed norm remains missing')
            if d['signal']['allowance_evidence'].get('exact_tail_norm_error') is not None:
                raise ValueError('m3/m4 observed norm must not be filled')
        if e.get('reference_context'):
            same(e['reference_context']['T'], c['T'], 'Reference time')
            same(e['reference_context']['binding'], config['spec']['A_input']['design']['binding'], 'Reference binding')
            same(e['reference_context']['evidence'], d['reference_pair'], 'Reference pair association')
        rows.append(dict(m=m, physical_candidate=deepcopy(c), structure=deepcopy(r['structure']),
            canonical_candidate_id=r['candidate']['id'], source_request_id=n['source_request_id'],
            identity_source=dict(**CONFIG, selector=['rounds', m, 'identity']), law_binding=deepcopy(r['law_binding']),
            scientific_source=e['scientific_source'], diagnostics_source=deepcopy(e['source']),
            mean_provenance=deepcopy(model['provenance']['finite_mean']),
            reference_context=deepcopy(e.get('reference_context')),
            diagnostics=deepcopy(d), nominal_N_X=n['nominal_N_X'], nominal_N_Y=n['nominal_N_Y'],
            nominal_source=dict(**N_PACKET, selector=['rounds', m]),
            sufficient_N_X=None, sufficient_N_Y=None, energy_bias_ha=None, energy_RMSE_bound_ha=None,
            qualification='SAVED_LOCAL_EMPIRICAL_ONLY', scientific_values_recomputed=False))
    design = config['spec']['A_input']['design']
    same(truth['candidate'], config['rounds'][1]['physical_candidate'], 'State evidence carrier candidate')
    same(truth['ideal_numerical_pilot']['binding'], design['binding'], 'State / Hamiltonian binding')
    same(truth['ground_state_certified'], False, 'Ground identification remains conditional')
    return dict(schema='A_H4_PR_PHYSICAL_INPUTS_V1', status='COPIED_EVIDENCE_GLOBAL_UNQUALIFIED',
        configuration_id=config['id'], configuration_source=deepcopy(CONFIG), N_packet_source=deepcopy(N_PACKET),
        binding=deepcopy(design['binding']), profile=deepcopy(design['profile']),
        conditions=deepcopy(config['spec']['A_input']), policy=deepcopy(packet['policy']),
        measurement_contract=dict(**deepcopy(p.MEASUREMENT), outcome_plus_one='ancilla bit 0',
            probability_plus_one=dict(X='(1 + Re(raw))/2', Y='(1 + Im(raw))/2'),
            probabilities_evaluated=False, raw_is_unscaled_mean=True, corrected_relation='corrected = B * raw (ideal mean, numerical closure saved)',
            normalization='B is the positive whole-interrogation finite-K normalizer; do not apply attenuation twice',
            sampling_assumption='fresh independent ideal-law trajectory and Born outcome per shot; axes and rounds independent',
            actual_sampler_qualified=False),
        target=dict(energy_target='ground energy of saved DF-H in sector (2,2)',
            reference_signal='saved-state <psi|exp(-i H_DF T)|psi>, not automatically exp(-i E_ground T)',
            truth_proxy_source=deepcopy(TRUTH), reference_solver_energy_ha=truth['reference_solver_energy_ha'],
            state_residual_ha=truth['state_residual_ha'], ground_state_certified=False,
            truth_proxy_adopted_for_new_simulation=False, energy_shift=0,
            reference_energy_used_for_prior_or_branch=False), rounds=rows,
        historical_annotations=deepcopy(source_index['technical_annotations']),
        numerical_allowances_certified=False, common_effective_H_qualified=False,
        sufficient_N_qualified=False, global_RMSE_bound=None, new_science_calls=0,
        saved_values_reanalyzed=False, mandatory_stop=True)


def build(root):
    config, packet, entries, truth, source_index, sources = load(root)
    result = assemble(config, packet, entries, truth, source_index)
    result['read_sources'] = sources
    return result
