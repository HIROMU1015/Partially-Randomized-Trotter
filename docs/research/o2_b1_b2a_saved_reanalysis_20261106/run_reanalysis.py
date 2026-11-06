"""Fixed saved-value B2-a/B1 analysis. No sampling, circuit imports or shot rules.

Run only within the scientific scope authorized by the user on 2026-11-06.
Input root holds the public GitHub bytes listed in INPUT_INDEX.json.
"""
import argparse
import csv
from functools import lru_cache
from fractions import Fraction
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import tarfile

import analysis_kernels as A

HERE = Path(__file__).resolve().parent
L = '155529e20a8fac39e00786c40b57830b71203c4f'
D = 'ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed'
LP = 'artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/'
DP = 'artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/'


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def pointer(value, path):
    for token in path.strip('/').split('/') if path.strip('/') else []:
        if token == 'rows' and isinstance(value, list):
            continue
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


class Analysis:
    def __init__(self, inputs):
        self.inputs = inputs
        self.checked = 0
        for r in json.loads((HERE/'INPUT_INDEX.json').read_text())['references']:
            A.verified_bytes(inputs/r['commit']/r['path'], r['sha256'])
            self.checked += 1
        specs = json.loads((HERE/'SOURCE_SPECS.json').read_text())
        def source(name):
            s = specs[name]
            return inputs/s['commit']/s['path'], s['sha256']
        A.verified_bytes(*source('allocation'))
        self.rte = A.standalone_module(*source('rte'), 'o2_fixed_rte')
        self.plan = A.standalone_module(*source('plan'), 'o2_fixed_plan')
        self.tau = A.selected_functions(*source('canonical_tau'),
                   ['require','canonical_tau'], {'math': math})['canonical_tau']
        scope = dict(math=math, Fraction=Fraction, statistics=statistics)
        self.paired = A.selected_functions(*source('paired'),
                      ['require','close','rational','frac','moments','paired'], scope)['paired']
        self.cost_for = A.selected_functions(*source('cost_for'),
                        ['fraction','cost_for'], scope)['cost_for']
        self.price_cache = {}
        self.rz_checks = []
        self.stops = []

    @lru_cache(None)
    def load(self, commit, path):
        data = (self.inputs/commit/path).read_bytes()
        if path.endswith('.gz'):
            data = gzip.decompress(data)
        if path.endswith('.csv'):
            return list(csv.DictReader(data.decode().splitlines()))
        return json.loads(data, parse_float=str)

    def resolve(self, ref):
        return pointer(self.load(ref['commit'],ref['path']),ref.get('pointer',''))

    def b2a(self):
        targets = list(csv.DictReader((HERE/'B2A_FIXED_CANDIDATES.csv').read_text().splitlines()))
        ordered = sorted(enumerate(targets), key=lambda p: (p[1]['R_outer']!='32' or p[1]['K']!='4', p[0]))
        results = {}
        stopped = set()
        for index, t in ordered:
            group = (t['family'], t['system'])
            row = dict(t)
            if group in stopped:
                row.update(exact_status='UNRESOLVED',approximation_status='UNRESOLVED',issue='INPUT_GROUP_STOPPED_AFTER_MISMATCH')
                results[index] = row
                continue
            try:
                signal = self.load(t['result_commit'],t['result_path'])[t['candidate_id']]
                candidate = signal['candidate']
                for key in ('system','formula','T','q','R_outer','K','prefix'):
                    assert str(candidate[key]) == str(t[key]), ('CANDIDATE_BINDING',key)
                lam = float(pointer(self.load(t['lambda_commit'],t['lambda_path']),t['lambda_pointer']))
                assert math.isfinite(lam) and lam >= 0
                weights = self.plan.weights(t['formula'])
                counts = self.plan.allocation(weights,int(t['R_outer']))
                v = A.normalization_terms(weights,counts,T=float(t['T']),q=int(t['q']),K=int(t['K']),lambda_R=lam,finite_distribution=self.rte.finite_rte_distribution,canonical_tau=self.tau)
                saved = float(pointer(self.load(t['result_commit'],t['result_path']),t['log_B_pointer']))
                exact, leading, continuous = v['exact'], v['integer_leading'], v['continuous']
                assert all(math.isfinite(z) for z in (saved,exact,leading,continuous))
                error = exact-saved
                eta = max(1e-10,256*2**-52*v['events_per_trajectory']*max(1,abs(exact),abs(leading),abs(continuous)))
                bnd = 5/8*v['fourth_order_scale']
                status = 'EXACT_REPRODUCED' if abs(error)<=1e-10 else 'INCONSISTENT'
                approx = 'APPROXIMATION_CONSISTENT'
                if exact<=eta or leading<=eta:
                    approx = 'UNRESOLVED'
                elif v['max_abs_tau']>1:
                    approx = 'APPROXIMATION_LIMITED'
                elif not (-eta <= leading-exact <= bnd+eta):
                    approx = 'INCONSISTENT'
                if leading-continuous < -eta:
                    approx = 'INCONSISTENT'
                row.update(lambda_R=lam,DF_rank=10 if t['system']=='H4' else 19,geometry='linear_1Angstrom',basis='STO-3G',rank_policy='tol_only_1e-8',stage_weights=json.dumps(weights),integer_allocation=json.dumps(counts),minimum_one_stages=sum(r==1 for r in counts),saved_log_B=saved,exact_log_B=exact,integer_leading_log_B=leading,continuous_log_B=continuous,exact_minus_saved=error,leading_minus_exact=leading-exact,leading_minus_continuous=leading-continuous,continuous_minus_exact=continuous-exact,max_abs_tau=v['max_abs_tau'],fourth_order_scale=v['fourth_order_scale'],small_tau_residual_bound=bnd if v['max_abs_tau']<=1 else '',roundoff_guard=eta,exact_status=status,approximation_status=approx)
                for key,numer in [('leading_relative_error',leading-exact),('allocation_relative_effect',leading-continuous),('continuous_relative_error',continuous-exact)]:
                    row[key] = numer/exact if exact>eta else ''
                if status=='INCONSISTENT' or approx=='INCONSISTENT':
                    stopped.add(group)
                    self.stops.append(dict(task='B2A',group=group,candidate=t['candidate_id'],reason='FIXED_CRITERION_VIOLATION'))
            except (AssertionError,KeyError,ValueError,OverflowError) as e:
                row.update(exact_status='UNRESOLVED',approximation_status='UNRESOLVED',issue=str(e))
                stopped.add(group)
                self.stops.append(dict(task='B2A',group=group,candidate=t['candidate_id'],reason=type(e).__name__+':'+str(e)))
            results[index] = row
        output = [results[i] for i in range(len(targets))]
        write_csv(HERE/'B2A_NORMALIZATION_RESULTS.csv',output)
        return output

    def archive_prices(self, ref):
        key = (ref['commit'],ref['path'])
        if key in self.price_cache:
            return self.price_cache[key]
        # Read JSON only for the already frozen candidate/cohort set.
        frozen = json.loads((HERE/'B1_FIXED_COMPARISONS.json').read_text())['comparisons']
        wanted = {(a['candidate_id'],a['cohort']) for c in frozen for a in c['arms'].values() if a.get('price_source',{}).get('commit')==ref['commit'] and a.get('price_source',{}).get('path')==ref['path']}
        data = b''.join((self.inputs/ref['commit']/p).read_bytes() for p in ref['path'].split(' + '))
        groups = {}
        with tarfile.open(fileobj=io.BytesIO(data),mode='r:xz') as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith('/result.json') or '/prices/' not in member.name:
                    continue
                if not any('/'+cid+'/prices/'+cohort+'/' in member.name for cid,cohort in wanted):
                    continue
                record = json.load(archive.extractfile(member))
                k = (record['candidate_id'],record['cohort'])
                assert k in wanted
                assert record['state_preparation_excluded'] is True
                assert str(record['qiskit_version']).startswith('1.3')
                idx = int(record['index'])
                entries = groups.setdefault(k,{})
                assert idx not in entries, ('DUPLICATE_PRICE_INDEX',k,idx)
                entries[idx] = record['axes']
        self.price_cache[key] = groups
        return groups

    def raw_axes(self, arm):
        ref = arm['price_source']
        cid, cohort = arm['candidate_id'],arm['cohort']
        if '.tar.xz.part' in ref['path']:
            indexed = self.archive_prices(ref)[cid,cohort]
            assert set(indexed)==set(range(arm['matched_raw_record_count']))
            return [indexed[i] for i in sorted(indexed)]
        data = self.load(ref['commit'],ref['path'])
        if ref['path'].endswith('trajectory_prices.csv'):
            rows = [r for r in data if r['candidate_id']==cid and r['cohort']==cohort]
            assert len(rows)==arm['matched_raw_record_count']
            assert {int(r['index']) for r in rows}==set(range(len(rows)))
            return [{a:{'rz_count':int(r['RZ_'+suffix]),'cx_count':int(r['CX_'+suffix])} for a,suffix in [('cosine','cos'),('sine','sin')]} for r in rows]
        rows = [r for r in data['trials'] if r['candidate_id']==cid and r['cohort']==cohort]
        assert rows, ('MISSING_H8_PRICE',cid,cohort)
        return [r['prices'] for r in rows]

    def expected_rz(self, comp, label, arm, source):
        # Preflight reused the N pointer for RZ. Confirmation RZ belongs to
        # its confirmation record, while the saved N is unchanged.
        if comp['study']=='DF_B3_CONFIRMATION':
            matches = [r for r in self.load(D,DP+'independent_comparisons.json') if r['B3_id']==comp['arms']['B3']['candidate_id'] and r['family']==comp['baseline_family'] and str(r['epsilon'])==str(comp['epsilon'])]
            assert len(matches)==1
            return matches[0]['B3_G' if label=='B3' else 'baseline_G'],False
        if comp['study']=='H8_STAGE_C':
            return source['mean_G_exact'],True
        value = source.get('G_exact')
        if value not in (None,''):
            return value,True
        assert source.get('G') not in (None,'')
        return source['G'],False

    def arm(self, comp, label, arm):
        source = self.resolve(arm['saved_N_source'])
        nc,ns = int(arm['N_cos']),int(arm['N_sin'])
        assert nc>=0 and ns>=0
        if 'N_cos' in source:
            assert int(source['N_cos'])==nc and int(source['N_sin'])==ns
        else:
            assert int(source['fixed_N_cos'])==nc and int(source['fixed_N_sin'])==ns
        saved_cid = source.get('id',source.get('candidate_id',source.get('candidate',{}).get('id')))
        if saved_cid is not None:
            assert saved_cid==arm['candidate_id']
        expected, exact_available = self.expected_rz(comp,label,arm,source)
        raw = 'RAW' in arm['CX_availability']
        if raw:
            axes = self.raw_axes(arm)
            assert all(type(v[a][m]) is int and v[a][m]>=0 for v in axes for a in ('cosine','sine') for m in ('rz_count','cx_count'))
            values = {}
            for metric in ('rz_count','cx_count'):
                pairs = [(v['cosine'][metric],v['sine'][metric]) for v in axes]
                result = A.paired_metric_cost(pairs,nc,ns,paired=self.paired,cost_for=self.cost_for,cohort=arm['cohort'])
                st = self.paired(pairs)
                values[metric] = dict(G=Fraction(result['G_exact']),SE=result['G_SE'],n=len(axes),C_cos=Fraction(st['cosine']['mean_exact']['numerator'],st['cosine']['mean_exact']['denominator']),C_sin=Fraction(st['sine']['mean_exact']['numerator'],st['sine']['mean_exact']['denominator']))
            rz,cx = values['rz_count'],values['cx_count']
            uncertainty = 'DESCRIPTIVE_PAIRED_SAMPLE_SE' if len(axes)>1 else 'DETERMINISTIC_NO_MC_SE'
        else:
            price = self.resolve(arm['price_source'])
            mean = price['mean_metrics']
            for axis in ('cosine','sine'):
                assert mean[axis]['cx_count'] not in (None,'')
            cc,cs = Fraction(mean['cosine']['cx_count']),Fraction(mean['sine']['cx_count'])
            cx = dict(G=nc*cc+ns*cs,SE=None,n=price.get('n',source.get('n')),C_cos=cc,C_sin=cs)
            if 'unit_statistics' in price:
                method = source.get('method','PR')
                if method=='PR':
                    z = self.cost_for(dict(candidate=dict(method='PR'),N_cos=nc,N_sin=ns),price)
                    rz = dict(G=Fraction(z['G_exact']),SE=z['G_SE'])
                else:
                    st=price['unit_statistics'];ccr=st['cosine']['mean_exact'];csr=st['sine']['mean_exact']
                    rz=dict(G=nc*Fraction(ccr['numerator'],ccr['denominator'])+ns*Fraction(csr['numerator'],csr['denominator']),SE=None)
            else:
                rz=dict(G=nc*Fraction(mean['cosine']['rz_count'])+ns*Fraction(mean['sine']['rz_count']),SE=None)
            uncertainty = 'UNCERTAINTY_UNRESOLVED_MEAN_ONLY'
        expected_f = Fraction(expected)
        diff = rz['G']-expected_f
        # Rounded mean records are checked against their recorded precision;
        # exact raw and sufficient-statistic values are checked exactly.
        exact_check = exact_available and (raw or (not raw and 'unit_statistics' in self.resolve(arm['price_source'])))
        cond = arm.get('conditions',{})
        events = int(cond.get('q') or 1)*int(cond.get('R_outer') or cond.get('r') or 1)
        allowance = max(1e-10,256*2**-52*events*max(1,abs(float(rz['G'])),abs(float(expected_f))))
        passed = diff==0 if exact_check else abs(float(diff))<=allowance
        check = dict(comparison_id=comp['comparison_id'],arm=label,candidate_id=arm['candidate_id'],cohort=arm['cohort'],RZ_check='EXACT_EQUAL' if exact_check and passed else ('FLOAT_PRECISION_MATCH' if passed else 'INCONSISTENT'),absolute_error=float(abs(diff)),allowance=0 if exact_check else allowance)
        self.rz_checks.append(check)
        if not passed:
            raise ValueError('SAVED_RZ_REPRODUCTION_FAILED')
        return dict(id=arm['candidate_id'],cohort=arm['cohort'],N_cos=nc,N_sin=ns,RZ=rz['G'],CX=cx['G'],CX_SE=cx['SE'],n=cx['n'],C_cos=cx['C_cos'],C_sin=cx['C_sin'],uncertainty=uncertainty,rz_check=check['RZ_check'])

    def b1(self):
        frozen = json.loads((HERE/'B1_FIXED_COMPARISONS.json').read_text())
        rows = []
        for comp_index, comp in enumerate(frozen['comparisons']):
            row = {k:comp[k] for k in ('study','comparison_id','system','T')}
            row.update(epsilon=comp.get('epsilon',''),baseline_family=comp.get('baseline_family',''),price_policy='SAME_SAVED_COMPILED_FULL_WRAPPER_P0',N_policy='ORIGINAL_SAVED_N_FIXED',DIRECT_MODEL='DIRECT_SAVED_COMPILED',selection=comp['selection'])
            try:
                labels = list(comp['arms'])
                vals = [self.arm(comp,label,comp['arms'][label]) for label in labels]
                a,b = vals
                for label,v in zip(('left','right'),vals):
                    for key in ('id','cohort','N_cos','N_sin','RZ','CX','CX_SE','n','C_cos','C_sin','uncertainty','rz_check'):
                        row[label+'_'+key] = str(v[key]) if isinstance(v[key],Fraction) else v[key]
                dr,dc = a['RZ']-b['RZ'],a['CX']-b['CX']
                row.update(left_label=labels[0],right_label=labels[1],RZ_left_over_right=str(a['RZ']/b['RZ']) if b['RZ'] else '',CX_left_over_right=str(a['CX']/b['CX']) if b['CX'] else '',RZ_left_lower=dr<0,CX_left_lower=dc<0,CX_left_minus_right=str(dc),direction_status='POINT_DIFFERENCE_SMALL' if dr==0 or dc==0 else ('DIRECTION_PRESERVED' if (dr<0)==(dc<0) else 'DIRECTION_REVERSED'))
                se_a,se_b = a['CX_SE'],b['CX_SE']
                # These fixed price cohorts were independently acquired in the
                # original studies; this remains a conditional IID assumption.
                deterministic_b = b['uncertainty']=='DETERMINISTIC_NO_MC_SE'
                if se_a is not None and (se_b is not None or deterministic_b):
                    se = math.hypot(se_a,se_b or 0)
                    lo,hi = float(dc)-1.96*se,float(dc)+1.96*se
                    row.update(CX_difference_SE=se,descriptive95_low=lo,descriptive95_high=hi,uncertainty_status='DESCRIPTIVE_INTERVAL_SEPARATES_ZERO' if lo>0 or hi<0 else 'UNCERTAINTY_UNRESOLVED',population_tail_guarantee=False)
                else:
                    row.update(uncertainty_status='UNCERTAINTY_UNRESOLVED',population_tail_guarantee=False)
                row['source_refs'] = f'B1_FIXED_COMPARISONS.json#/comparisons/{comp_index}'
            except (AssertionError,KeyError,ValueError,TypeError) as e:
                row.update(direction_status='DATA_UNAVAILABLE',uncertainty_status='UNCERTAINTY_UNRESOLVED',issue=type(e).__name__+':'+str(e))
                self.stops.append(dict(task='B1',comparison=comp['comparison_id'],reason=row['issue']))
            rows.append(row)
        for missing in frozen['unavailable']:
            rows.append(dict(study=missing['study'],comparison_id=missing['study'],direction_status='DATA_UNAVAILABLE',uncertainty_status='UNCERTAINTY_UNRESOLVED',issue=missing['reason']))
        write_csv(HERE/'B1_CX_SENSITIVITY.csv',rows)
        # Reuse already accounted CX values: no new P3 ratio/ranking/depth cost.
        quote = frozen['existing_metric_quotes'][0]['source']
        p3 = self.load(quote['commit'],quote['path'])
        fields = ('arm','epsilon_sig','status','N_cosine','N_sine','N_total','RZ_G_exact','RZ_price_SE','CX_G_exact','CX_price_SE')
        write_csv(HERE/'P3_EXISTING_CX_QUOTED.csv',[dict({k:r[k] for k in fields},source_commit=quote['commit'],source_path=quote['path']) for r in p3])
        return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',required=True,type=Path)
    p.add_argument('--task',choices=['b2a','b1','all'],default='all')
    args = p.parse_args()
    analysis = Analysis(args.inputs)
    summary = dict(fixed_files_verified=analysis.checked)
    if args.task in ('b2a','all'):
        rows = analysis.b2a()
        summary['B2A'] = {k:sum(r.get('exact_status')==k for r in rows) for k in ('EXACT_REPRODUCED','INCONSISTENT','UNRESOLVED')}
    if args.task in ('b1','all'):
        rows = analysis.b1()
        summary['B1'] = {k:sum(r.get('direction_status')==k for r in rows) for k in ('DIRECTION_PRESERVED','DIRECTION_REVERSED','POINT_DIFFERENCE_SMALL','DATA_UNAVAILABLE')}
    summary['RZ_checks'] = analysis.rz_checks
    summary['stops'] = analysis.stops
    dump(HERE/('EXECUTION.json' if args.task=='all' else 'EXECUTION_'+args.task.upper()+'.json'),summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('RZ_checks','stops')},ensure_ascii=False))
    print('RZ_checks',len(analysis.rz_checks),'STOPS',len(analysis.stops))


if __name__=='__main__':
    main()
