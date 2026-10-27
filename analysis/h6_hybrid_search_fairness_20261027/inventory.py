#!/usr/bin/env python3
"""Fixed Git evidence -> literal H6 coverage. No scientific kernels or ranking.

Only metadata counts and integer event counts q*r are derived. Saved scientific
numbers stay strings; no float/Fraction arithmetic, RNG, sampling or compilation.
"""
import csv
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import subprocess
from collections import Counter
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = 'b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b'
OLD = '8fd1ceb283309a7e515ff4a533a253260bdd0f78'
PREP = '25d248a1f2ed88de62b0e621c2d1cea8532804c4'
EXT = '10cb3e5cca7834ebf75301bd91bd093ca51d4a6b'
P = 'artifacts/pauli_h2_h6_comparison/2026-10-25/'
I = 'artifacts/pauli_hybrid_rte_preparation/2026-10-25/'
EPS = ('0.05', '0.01', '0.005', '0.001', '0.0005')
# Reuse the existing non-scientific inventory's CSV/null conventions.
spec = importlib.util.spec_from_file_location('saved_inventory', ROOT / 'scripts/inventory_pauli_hybrid_rte.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)


@lru_cache(None)
def blob(ref, path):
    return subprocess.check_output(['git', 'show', ref + ':' + path], cwd=ROOT)


def obj(ref, path):
    raw = blob(ref, path)
    if path.endswith('.gz'):
        raw = gzip.decompress(raw)
    return json.loads(raw, parse_float=str, parse_int=str)


def csvrows(ref, path):
    return list(csv.DictReader(io.StringIO(blob(ref, path).decode())))


def scalar(value):
    return '' if value is None else str(value)


def one(rows, **selector):
    found = [r for r in rows if all(r[k] == v for k, v in selector.items())]
    assert len(found) == 1, selector
    return found[0]


def assemble():
    inventory = csvrows(PREP, I + 'candidate_evidence_inventory.csv')
    saved = [r for r in inventory if r['molecule'] == 'H6' and r['group'] == 'Hybrid']
    catalog = obj(OLD, P + 'batch_catalog.json')
    registered = {}
    for batch in catalog:
        if batch['molecule'] != '6':
            continue
        for c in batch['candidates']:
            cfg = c['config']
            if cfg['method'] == 'PR' and 0 < int(cfg['L_D']) < 918:
                assert c['candidate_id'] not in registered
                registered[c['candidate_id']] = (cfg, batch)
    assert set(registered) == {r['candidate_id'] for r in saved}
    canonical = obj(OLD, P + 'H6/candidate_results.json.gz')
    confirmed = obj(PREP, I + 'evidence/H6/confirmation01/results.json')
    confirm_plan = obj(PREP, I + 'evidence/H6/confirmation01/plan.json')
    archives = obj(OLD, P + 'data_archives.json')
    archive = next(a for a in archives if a['molecule'] == '6')
    archived_ids = {m.group(0) for path in archive['members']
                    for m in [re.search(r'PR_p\d+_q\d+_r\d+_K\d+', path)] if m}
    archive_hybrid = {cid for cid in archived_ids if not cid.startswith('PR_p0000_')}
    assert archive_hybrid == set(registered)
    assert {int(re.search(r'_r(\d+)', cid).group(1)) for cid in archived_ids} == {2, 4, 8}
    out = []
    for r in saved:
        cfg, batch = registered[r['candidate_id']]
        for k in ('L_D', 'q', 'r', 'K'):
            assert r[k] == cfg[k]
        originals = canonical if r['cohort'] == 'exploration' else confirmed
        x = one(originals, candidate_id=r['candidate_id'], epsilon_sig=r['epsilon_sig'], cohort=r['cohort'])
        for k in ('G_exact', 'G_SE', 'N_cos', 'N_sin', 'eligibility'):
            assert r[k] == scalar(x[k]), (r['candidate_id'], k)
        st = x['price_statistics']
        assert r['price_n'] == st['n']
        assert r['B_exact'] == x['target']['B_saved_exact']
        for axis, short in (('cosine', 'cos'), ('sine', 'sin')):
            assert r['C_' + short + '_exact'] == st['mean_exact'][axis]
            assert r['C_' + short + '_SE'] == st['SE'][axis]
            for src, dst in (('signed_bias_exact', 'bias_'), ('u_enc_exact', 'u_')):
                assert r[dst + short + '_exact'] == x['target']['axes'][axis][src]
        ref = OLD if r['cohort'] == 'exploration' else PREP
        source_path = P + 'H6/candidate_results.json.gz' if ref == OLD else I + 'evidence/H6/confirmation01/results.json'
        source = batch['SOURCE']['commit'] if ref == OLD else confirm_plan['SOURCE']['commit']
        case_member = r['detail_source'].removeprefix(P)
        target_member = case_member + '/target_refined.json'
        assert target_member in archive['members']
        plan_member = batch['raw_plan_archive_member']
        assert plan_member in archive['members']
        rec = dict(molecule='H6', prefix=r['L_D'], q=r['q'], r=r['r'], K=r['K'],
                   s=str(int(r['q']) * int(r['r'])), candidate_id=r['candidate_id'],
                   registered='YES_HISTORICAL', signal_evaluated='YES_SAVED',
                   epsilon_sig=r['epsilon_sig'], eligibility_saved=r['eligibility'],
                   compiled_price_acquired='YES_SAVED',
                   independently_confirmed='YES' if r['existing_confirmation_n'] else 'NO_SAVED_CONFIRMATION',
                   cohort=r['cohort'], n=r['price_n'],
                   B_exact=r['B_exact'], bias_cos_exact=r['bias_cos_exact'], bias_sin_exact=r['bias_sin_exact'],
                   u_cos_exact=r['u_cos_exact'], u_sin_exact=r['u_sin_exact'],
                   N_cos=r['N_cos'], N_sin=r['N_sin'], C_cos_exact=r['C_cos_exact'], C_sin_exact=r['C_sin_exact'],
                   C_cos_SE=r['C_cos_SE'], C_sin_SE=r['C_sin_SE'], G_exact=r['G_exact'], G_SE=r['G_SE'],
                   G_availability=r['G_availability'], price_SOURCE_commit=source,
                   price_SOURCE_locator=(P + 'data_archives.json#' + plan_member) if ref == OLD else I + 'evidence/H6/confirmation01/plan.json',
                   target_SOURCE_locator=P + 'data_archives.json#H6/precision_refinement_*/plan.json; target.error_model',
                   target_evidence=P + 'data_archives.json#' + target_member,
                   result_commit=ref, result_file=source_path,
                   result_url='https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/' + ref + '/' + source_path,
                   inventory_commit=PREP, inventory_file=I + 'candidate_evidence_inventory.csv',
                   comparison_scope='saved conditional rho=0; empirical price; no new ranking')
        out.append(rec)
    # Proposals are not a new scientific registration or active execution plan.
    for prefix, q, r, K in ((78, 8, 16, 2), (78, 8, 32, 2), (78, 8, 64, 2), (684, 1, 16, 4)):
        cid = f'PR_p{prefix:04d}_q{q}_r{r}_K{K}'
        assert cid not in registered and cid not in archived_ids
        for eps in EPS:
            row = {k: '' for k in out[0]}
            row.update(molecule='H6', prefix=str(prefix), q=str(q), r=str(r), K=str(K), s=str(q*r),
                       candidate_id=cid, registered='NO_PROPOSAL_ONLY', signal_evaluated='NO_NOT_EXECUTED',
                       epsilon_sig=eps, eligibility_saved='NOT_EVALUATED', compiled_price_acquired='NO_NOT_ACQUIRED',
                       independently_confirmed='NO_NOT_ACQUIRED', cohort='NOT_CREATED',
                       G_availability='NOT_ACQUIRED_NOT_INELIGIBLE', comparison_scope='DISABLED_PROPOSAL_REQUIRES_SCIENTIFIC_PERMISSION')
            out.append(row)
    assert len(out) == 510 and len(saved) == 490 and len(registered) == 94
    assert len({(r['candidate_id'], r['epsilon_sig'], r['cohort']) for r in out}) == len(out)
    return out


def main():
    rows = assemble()
    prior.csvsave(OUT / 'H6_HYBRID_COVERAGE.csv', rows)
    counts = {}
    for eps in EPS:
        rr = [r for r in rows if r['epsilon_sig'] == eps and r['cohort'] == 'exploration']
        counts[eps] = dict(Counter(r['eligibility_saved'] for r in rr))
    print(json.dumps(dict(status='PASS_LITERAL_COVERAGE', historical_candidates=94,
                         historical_rows=490, proposal_only_candidates=4, rows=len(rows),
                         saved_status_counts=counts, scientific_arithmetic=False,
                         new_sampling_compile_state_actions=0), ensure_ascii=False))


if __name__ == '__main__':
    main()
