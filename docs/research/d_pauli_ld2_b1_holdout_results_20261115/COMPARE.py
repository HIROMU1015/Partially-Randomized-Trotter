"""Exact descriptive comparison only; consumes admitted receipts, never fits/qualifies."""
from fractions import Fraction
import argparse
import json
from pathlib import Path
import subprocess

FROZEN = {'commit': 'b74e4e536392c2bbe94cd371436615fc56c0564f',
          'path': 'docs/research/d_pauli_ld2_b1_cal_results_20261115/c2_freeze/run01/FROZEN_MODELS.json'}
UNITS = ((5, 'X'), (5, 'Y'), (6, 'X'), (6, 'Y'))
VERDICT = 'NOT_ASSESSED_NO_ADOPTED_TOLERANCE'

def encode(x):
    return {'numerator': x.numerator, 'denominator': x.denominator}

def rational(x):
    if (set(x) != {'numerator', 'denominator'} or
        type(x['numerator']) is not int or type(x['denominator']) is not int or x['denominator'] <= 0):
        raise ValueError('Exact integer rational required')
    return Fraction(x['numerator'], x['denominator'])

def compare(predictions, units):
    pmap = {(p['m'], p['axis']): (i, p) for i, p in enumerate(predictions)}
    if len(pmap) != 4 or len(predictions) != 4 or set(pmap) != set(UNITS):
        raise ValueError('Exactly four distinct (m, axis) frozen predictions required')
    umap = {(u['m'], u['axis']): u for u in units}
    if len(umap) != len(units) or not set(umap) <= set(UNITS):
        raise ValueError('Duplicate or foreign observation unit')
    rows = []
    for m, axis in UNITS:
        pi, p = pmap[m, axis]; expected = list(range(2 if m == 5 else 1))
        u = umap.get((m, axis), {}); paths = u.get('paths', [])
        reasons = list(u.get('reasons', []))
        if sorted(x['index'] for x in paths) != expected:
            reasons.append('INCOMPLETE_OR_DUPLICATE_FIXED_PATH_SET')
        for x in paths:
            if x.get('valid') is not True:
                reasons.append(x.get('reason') or 'RECEIPT_NOT_ADMITTED')
            if type(x.get('compiled_RZ')) is not int or x['compiled_RZ'] < 0:
                reasons.append('MISSING_OR_INVALID_INTEGER_PRICE')
        row = dict(m=m, axis=axis, n_paths=len(expected), prediction_source=dict(FROZEN, selector=['predictions', pi]),
            P=encode(rational(p['predicted_RZ'])), O=None, signed_error=None, absolute_error=None,
            signed_relative_error=None, absolute_relative_error=None, valid=not reasons,
            reasons=sorted(set(reasons)), paths=paths, model_verdict=VERDICT)
        if not reasons:
            observed = sum(Fraction(x['compiled_RZ']) for x in paths) / len(expected)
            delta = rational(p['predicted_RZ']) - observed
            row.update(O=encode(observed), signed_error=encode(delta), absolute_error=encode(abs(delta)))
            if observed:
                row.update(signed_relative_error=encode(delta / observed), absolute_relative_error=encode(abs(delta) / observed))
            else:
                row['reasons'] = ['ZERO_OBSERVED_PRICE']
        rows.append(row)
    status = ('INCOMPLETE_OR_INVALID' if not all(r['valid'] for r in rows) else
              'DESCRIPTIVE_COMPLETE_RELATIVE_UNDEFINED' if any(r['O']['numerator'] == 0 for r in rows) else
              'DESCRIPTIVE_COMPLETE')
    return dict(campaign=status, model_verdict=VERDICT, tolerance=None, rows=rows,
                model_qualified=False, law_expectation=False, fit_run=False, E_C=None, CI=None, N=None, G=None)

def self_test():
    from copy import deepcopy
    ps = [dict(m=m, axis=a, predicted_RZ=encode(Fraction(12 if m == 5 else 3))) for m,a in UNITS]
    us = [dict(m=m, axis=a, paths=[dict(index=i,valid=True,compiled_RZ=z) for i,z in enumerate([8,12] if m==5 else [4])]) for m,a in UNITS]
    a=compare(ps,us)
    assert a['campaign']=='DESCRIPTIVE_COMPLETE'
    assert a['rows'][0]['O']==encode(Fraction(10)) and a['rows'][0]['signed_error']==encode(Fraction(2))
    assert a['rows'][0]['signed_relative_error']==encode(Fraction(1,5))
    assert a['rows'][2]['signed_error']==encode(Fraction(-1)) and a['rows'][2]['absolute_relative_error']==encode(Fraction(1,4))
    assert compare(list(reversed(ps)),list(reversed(us)))['rows'][0]['P']==encode(Fraction(12))
    zero=deepcopy(us);zero[2]['paths'][0]['compiled_RZ']=0
    z=compare(ps,zero);assert z['campaign']=='DESCRIPTIVE_COMPLETE_RELATIVE_UNDEFINED'
    assert z['rows'][2]['signed_relative_error'] is None and z['rows'][2]['reasons']==['ZERO_OBSERVED_PRICE']
    zero_p=deepcopy(ps);zero_p[2]['predicted_RZ']=encode(Fraction(0))
    assert compare(zero_p,zero)['rows'][2]['absolute_relative_error'] is None
    missing=deepcopy(us);missing[0]['paths'].pop()
    x=compare(ps,missing);assert x['campaign']=='INCOMPLETE_OR_INVALID' and x['rows'][0]['O'] is None
    failed=deepcopy(us);failed[1]['paths'][0].update(valid=False,reason='GATE_FAILED')
    assert compare(ps,failed)['rows'][1]['signed_error'] is None
    assert all(r['model_verdict']==VERDICT for r in a['rows'])
    print('PASS: exact arithmetic/sign/observed denominator; keyed axes; zero and 0/0; missing path; failed gate; no model verdict')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true');parser.add_argument('--admission');parser.add_argument('--output')
    args=parser.parse_args()
    if args.self_test:self_test()
    else:
        frozen=json.loads(subprocess.check_output(['git','show',FROZEN['commit']+':'+FROZEN['path']]))
        data=json.loads(Path(args.admission).read_text())
        result=compare(frozen['predictions'],data['units'])
        result['admission_source']=args.admission
        with Path(args.output).open('x') as out:json.dump(result,out,indent=2);out.write('\n')
