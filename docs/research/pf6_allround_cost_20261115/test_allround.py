"""Artificial checks: join, 156-axis split, integer totals, ties, order changes, rejections.

C prices here are fictional integers; B rows/K are the fixed published inputs.
"""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_pf6_allround',HERE/'allround.py')
M=importlib.util.module_from_spec(spec);sys.modules[spec.name]=M;spec.loader.exec_module(M)
B=M.document(M.B8,M.B8P+'INPUTS.json');S=M.document(M.B8,M.B8P+'PARTIAL_SUMS.json');R=M.document(M.REC_FULL,M.RECP)
SRC=M.ref('f'*40,'ARTIFICIAL_C_TABLE.json')


def table(price=lambda r:1000):
    rows=[]
    for r in R['rows']:
        x={k:r[k]for k in ('configuration_id','physical_id','route','formula','width_over_h','m','axis','q','T','model_group')}
        v=price(r)
        x.update(compiled_RZ=v,exact_RZ=dict(numerator=v,denominator=1)if isinstance(v,int)else v,
                 mean=dict(rz_count=v),classification='MODEL_PREDICTION',direct_observation=False,
                 conditional_use_adopted=True,evidence_kind='FIXED_COMPILER_CLOSURE_DERIVATION_CONDITIONAL',
                 model_group_id=x.pop('model_group'))
        rows.append(x)
    return dict(rows=rows)


def run(doc,b=B,s=S):
    rows,issues=M.join(b,s,SRC,doc,R)
    return rows,issues


class AllRound(unittest.TestCase):
    def test_156_axes_and_integer_totals(self):
        rows,issues=run(table())
        self.assertEqual(issues,[]);self.assertEqual(len(rows),156)
        self.assertEqual(sum(r['price_class']==M.DIRECT for r in rows),96)
        self.assertEqual(sum(r['price_class']==M.CLOSURE for r in rows),60)
        out,order,flips,gap=M.totals(rows,[c['label']for c in B['candidates']],S)
        U=1000*2*(31+27+19+15+11)
        for r in out:
            self.assertEqual((r['K'],r['U'],r['total']),(M.EXPECTED_K[r['label']],U,M.EXPECTED_K[r['label']]+U))
            self.assertEqual((r['direct_axes'],r['closure_axes']),(16,10))
        self.assertEqual([r['label']for r in order][0],'B0I_P7_S2_16h_over_7');self.assertEqual(flips,[])
        self.assertEqual(gap['gap'],107719628-81621086)

    def test_X_and_Y_each_multiplied_and_rank_change_detected(self):
        # Make T-M5 expensive on Y only at m12: X/Y are separate terms.
        def price(r):return 10**7 if (r['configuration_id'].startswith('424eecbf')and r['axis']=='Y'and r['m']==12)else 0
        rows,_=run(table(price));out,order,flips,_=M.totals(rows,[c['label']for c in B['candidates']],S)
        tm5=next(r for r in out if r['label']=='B0I_P7_M5_16h')
        self.assertEqual(tm5['U'],11*10**7);self.assertEqual(tm5['U_rounds'][-1],dict(m=12,X=0,Y=11*10**7,total=11*10**7))
        names={(f['known_cheaper'],f['known_more_expensive'])for f in flips}
        self.assertIn(('B0I_P7_M5_16h','FULL_PF_Y4_16h_over_3'),names)
        self.assertIn(('B0I_P7_M5_16h','FULL_PF_M5_16h'),names)

    def test_equal_totals_share_rank(self):
        gap=M.EXPECTED_K['FULL_PF_Y4_16h_over_3']-M.EXPECTED_K['B0I_P7_M5_16h']  # 6,255,978
        # Exact tie: T-M5 pays the known gap through integer m7/m8 X prices (31a+27b=gap).
        a,b=next((a,b)for a in range(0,gap//31+1)for b in [ (gap-31*a)//27 ] if 31*a+27*b==gap and b>=0)
        def price2(r):
            if r['configuration_id'].startswith('424eecbf')and r['axis']=='X':
                return a if r['m']==7 else(b if r['m']==8 else 0)
            return 0
        rows,_=run(table(price2));out,order,flips,_=M.totals(rows,[c['label']for c in B['candidates']],S)
        tm5=next(r for r in out if r['label']=='B0I_P7_M5_16h');fy4=next(r for r in out if r['label']=='FULL_PF_Y4_16h_over_3')
        self.assertEqual(tm5['total'],fy4['total']);self.assertEqual(tm5['rank'],fy4['rank'])
        self.assertIn('TIED',[f['result']for f in flips])

    def test_missing_duplicate_mismatch_and_direct_label_rejected(self):
        d=table();d['rows'].pop(0);_,issues=run(d)
        self.assertTrue(any('missing'in i['reason']for i in issues))
        d=table();d['rows'].append(deepcopy(d['rows'][0]));_,issues=run(d)
        self.assertTrue(any('duplicate'in i['reason']for i in issues))
        d=table();d['rows'][0]['physical_id']='0'*20;_,issues=run(d)
        self.assertTrue(any('missing'in i['reason']for i in issues)and any('not matched'in i['reason']for i in issues))
        d=table();d['rows'][1]['formula']='4th';_,issues=run(d)
        self.assertTrue(any('formula'in i['reason']for i in issues))
        d=table();d['rows'][2]['classification']=M.DIRECT;_,issues=run(d)
        self.assertTrue(any('direct'in i['reason']for i in issues))
        d=table();d['rows'][3]['compiled_RZ']=1.5;_,issues=run(d)
        self.assertTrue(any('integer'in i['reason']for i in issues))
        d=table();d['rows'][4]['compiled_RZ']+=1;_,issues=run(d)  # disagreeing price fields
        self.assertTrue(any('price field'in i['reason']for i in issues))
        d=table();d['rows'][5]['conditional_use_adopted']=False;_,issues=run(d)
        self.assertTrue(any('adopted'in i['reason']for i in issues))
        d=table();d['rows'][6]['direct_observation']=True;_,issues=run(d)
        self.assertTrue(any('non-direct'in i['reason']for i in issues))
        d=table();d['rows'][7]['canonical_id']='RMSE:0';_,issues=run(d)
        self.assertTrue(any('canonical'in i['reason']for i in issues))

    def test_known_subtotal_guard(self):
        s=deepcopy(S);s['rows'][0]['K_i']+=1
        with self.assertRaisesRegex(ValueError,'B K differs'):run(table(),s=s)
        rows,_=run(table());rows=deepcopy(rows)
        next(r for r in rows if r['price_class']==M.DIRECT)['C']+=1
        with self.assertRaisesRegex(ValueError,'Recomputed K'):M.totals(rows,[c['label']for c in B['candidates']],S)

    def test_large_integers_stay_exact(self):
        rows,_=run(table(lambda r:10**30+1));out,_,_,_=M.totals(rows,[c['label']for c in B['candidates']],S)
        self.assertEqual(out[0]['U'],(10**30+1)*2*(31+27+19+15+11))


if __name__=='__main__':
    unittest.main(verbosity=2)
