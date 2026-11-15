"""Generate PR configurations (L_D fixed) with a changed step width, kappa or formula: metadata only.

The schedule (q_m, R_m, signed pieces, per-piece event allocation, law binding) comes from each
method's fixed mapper, loaded unchanged from a read-only route view:
  prdf_a_fixed_grid  D2's B release 437ba4ac A-fixed-grid adapter (``run_pr_df_ld7_v0.signal_modules``)
  pauli_native       A's P1 B API (B ce0517de native ``generate`` with the shared paper mapper)
No signal, price, compile or Q3.  For each item the base (saved) configuration is regenerated first;
the unchanged request must reproduce it exactly.

    python run_rpe_optimizer_config_generator.py --request REQ.json --output OUT.json
"""
import argparse
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)
SUPPORTED = {'prdf_a_fixed_grid': ['2nd', '4th(m5_best)'], 'pauli_native': ['2nd', '4th(m5_best)']}


def git_json(view, loc):
    value = json.loads(subprocess.check_output(['git', 'show', loc['commit'] + ':' + loc['path']], cwd=view))
    for key in loc.get('selector', []):
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def enc(x):
    x = Fraction(x)
    return dict(numerator=x.numerator, denominator=x.denominator)


def dec(v):
    return Fraction(v['numerator'], v['denominator']) if isinstance(v, dict) else Fraction(v)


def generator(kind, view):
    sys.path[:0] = [str(view / 'scripts'), str(view / 'src')]
    if kind == 'prdf_a_fixed_grid':
        import run_pr_df_ld7_v0 as S
        b = S.signal_modules()[3]
        def build(base, ch):
            inputs = deepcopy(base['spec']['A_input'])
            if 'delta_over_h' in ch:  # requested PF step width in units of the fixed RPE base time h
                inputs['design']['delta'] = float(Fraction(inputs['design']['base_time']) * Fraction(ch['delta_over_h']))
            if 'kappa_multiplier' in ch:
                inputs['kappa_paper'] = str(Fraction(inputs['kappa_paper']) * Fraction(ch['kappa_multiplier']))
            if 'formula' in ch:
                inputs['design']['formula'] = ch['formula']
            config = b.generate(b.a_spec(inputs, 'PR'));b.validate(config)
            return config
        return b, build
    if kind == 'pauli_native':
        from trottertracks import df_head_pauli_tail_ld3 as A
        b = A.p1.b_api(view)
        def build(base, ch):
            spec = deepcopy(base['spec'])
            if 'delta_over_h' in ch:  # requested PF step width in units of the fixed RPE base time h
                spec['delta_pf'] = enc(dec(spec['h']) * Fraction(ch['delta_over_h']))
            if 'kappa_multiplier' in ch:
                spec['kappa']['value'] = enc(dec(spec['kappa']['value']) * Fraction(ch['kappa_multiplier']))
            if 'formula' in ch:
                spec['formula'] = ch['formula']
            config = b.generate(spec);b.validate(config)
            return config
        return b, build
    raise ValueError('Unknown generator kind: ' + kind)


def structure_checks(config):
    """Signed pieces and per-piece event allocation must be consistent in every round."""
    rows, ok = [], True
    for r in config['rounds']:
        st = r['structure'];w = st.get('signed_weights') or [];alloc = st.get('allocation');R = r.get('R_outer');q = r.get('q_PF')
        good = (bool(w) and abs(sum(w) - 1.0) < 1e-12 and (alloc is None or (len(alloc) == len(w) and sum(alloc) == R))
                and st.get('event_count') == (None if R is None else q * R) and (R is None or R >= len(w)))
        ok &= good
        rows.append(dict(m=r['m'], q=q, R=R, pieces=len(w), negative_pieces=sum(1 for x in w if x < 0), allocation=alloc,
                         event_count=st.get('event_count'), ok=good))
    return ok, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', type=Path, required=True);p.add_argument('--output', type=Path, required=True)
    a = p.parse_args();req = json.loads(a.request.read_text());view = Path(req['view'])
    b, build = generator(req['kind'], view);out = [];reproduced = {}
    for item in req['items']:
        base = git_json(view, item['base']);ch = item.get('changes', {});bkey = json.dumps(item['base'], sort_keys=True)
        row = dict(item_id=item['item_id'], base=item['base'], changes=ch, kind=req['kind'], supported_formulas=SUPPORTED[req['kind']])
        try:
            if ch.get('formula') and ch['formula'] not in SUPPORTED[req['kind']]:
                row.update(status='UNSUPPORTED_FORMULA');out.append(row);continue
            if bkey not in reproduced:  # the unchanged request must reproduce the saved configuration (once per base)
                reproduced[bkey] = build(base, {}) == base
            row['base_reproduced'] = reproduced[bkey]
            if not row['base_reproduced']:
                row.update(status='BASE_NOT_REPRODUCED');out.append(row);continue
            config = base if not ch else build(base, ch);ok, rows = structure_checks(config)
            row.update(status='GENERATED' if ok else 'STRUCTURE_CHECK_FAILED', configuration=config, structure=rows,
                       configuration_id=config['id'], identical_to_base=config == base)
        except Exception as exc:
            row.update(status='GENERATION_REJECTED', reason=type(exc).__name__ + ': ' + str(exc))
        out.append(row)
    a.output.write_text(json.dumps(dict(kind=req['kind'], view_commit=req.get('view_commit'), items=out), ensure_ascii=False) + '\n')
    print(json.dumps({r['item_id']: r['status'] for r in out}))


if __name__ == '__main__':
    main()
