"""Artificial receipts/words; real fixed metadata only, never actual S/C fit."""
from copy import deepcopy
import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('p7_receipt_helper',ROOT/'scripts/prepare_p7_price_intake.py')
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
M=P.M
from trottertracks.r_price_phase1_science import functions

FIX=dict(fixture='ARTIFICIAL_P7_PRICE_ONLY')


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t,cls.f=M.d_callers(ROOT)
        cls.plan=next(p for p in cls.t.plans(ROOT) if p['formula']=='4th(m5_best)')
        cls.old=M.document(ROOT,dict(commit=P.OLD_ADMISSION,path=P.OLD_DIR+'/CELL_PLAN.json'))

    def packet(self):
        t=M.d_callers(ROOT)[0];t.price_bindings.__globals__['COHORT']='EXPLORATION'
        cell=self.plan['cells'][0]
        trial=dict(scalar=.125,prices=dict(cosine=dict(rz_count=111),sine=dict(rz_count=222)))
        a,b,c=t.price_bindings(ROOT,self.plan['plan'],cell,trial,price_source=FIX,
            evidence_commit='a'*40,evidence_path='ARTIFICIAL_P7_ASSERTION.json')
        return dict(plan=self.plan['plan'],prices=a,sides=b,proofs=c)

    def test_revision_changes_only_two_actions(self):
        old=deepcopy(self.old);new,changed=P.revise_plan(old)
        self.assertEqual(old,self.old);self.assertEqual(len(changed),2);self.assertEqual(len(new),36)
        for i,(a,b) in enumerate(zip(old,new)):
            diff={k for k in a if a[k]!=b[k]}
            self.assertEqual(diff,{'action'} if i in changed else set())
        self.assertEqual(sum(r['action']!='NEW_COMPILE' for r in new),4)
        self.assertEqual(sum(r['role']=='CALIBRATION' and r['action']=='NEW_COMPILE' for r in new),20)
        self.assertEqual(sum(r['role']=='HOLDOUT' and r['action']=='NEW_COMPILE' for r in new),12)

    def test_revision_refuses_missing_axis_or_changed_role(self):
        for which in ('missing','role'):
            rows=deepcopy(self.old);i=next(i for i,r in enumerate(rows) if r['physical_candidate']['id']==P.PID)
            if which=='missing':rows.pop(i)
            else:rows[i]['role']='HOLDOUT'
            with self.subTest(which=which),self.assertRaises(ValueError):P.revise_plan(rows)

    def test_p7_guard_and_B_price_kind(self):
        packet=self.packet()
        with patch.object(M.G,'_document',return_value=packet['proofs']):r=M.probe(ROOT,packet,'a'*40)
        self.assertEqual(r['tiers'],dict(X='DETERMINISTIC_DIRECT',Y='DETERMINISTIC_DIRECT'))
        self.assertEqual(len(r['guard']['checked']),2);self.assertIsNone(r['G'])
        for side in packet['sides']:
            self.assertEqual(side['implementation']['scalar_quantity'],'HAMILTONIAN_CONSTANT_PLUS_B0I_TAIL_IDENTITY')
            self.assertEqual(side['implementation']['rte_rotation_lowering'],M.G.NO_RTE)

    def test_p7_rejects_constant_only_policy_wrong_scalar_or_missing_axis(self):
        for change in ('quantity','scalar','axis'):
            packet=self.packet();side=packet['sides'][0]
            if change=='quantity':side['implementation']['scalar_quantity']='HAMILTONIAN_CONSTANT'
            elif change=='scalar':side['implementation']['resolved_scalar']=.25
            else:side['implementation']['measurement']['measurement_axis']='Y'
            packet['proofs'][side['implementation_evidence']['selector'][0]]['implementation']=deepcopy(side['implementation'])
            with patch.object(M.G,'_document',return_value=packet['proofs']),self.subTest(change=change),self.assertRaises(ValueError):
                M.probe(ROOT,packet,'a'*40)

    def test_saved_wrapper_axis_and_scalar_are_separate_from_gate_count(self):
        def item(name,q,c=(),params=()):return SimpleNamespace(operation=SimpleNamespace(name=name,params=params),qubits=q,clbits=c)
        def qc(axis):
            data=[item('h',(8,)),item('x',(0,)),item('p',(8,),params=(-.25,))]
            if axis=='Y':data.append(item('sdg',(8,)))
            data += [item('h',(8,)),item('measure',(8,),(0,))]
            return SimpleNamespace(num_qubits=9,num_clbits=1,data=data,find_bit=lambda b:SimpleNamespace(index=b))
        for axis in ('X','Y'):
            word=qc(axis);self.assertEqual(P.wrapper_axis(word,axis,.125,2)['axis'],axis)
            for mode in ('other_axis','scalar','twice','measurement'):
                x=qc(axis)
                if mode=='other_axis':arg='Y' if axis=='X' else 'X'
                else:arg=axis
                if mode=='scalar':x.data[2].operation.params=(-.5,)
                if mode=='twice':x.data.insert(1,item('p',(8,),params=(-.25,)))
                if mode=='measurement':x.data[-1].clbits=(1,)
                with self.subTest(axis=axis,mode=mode),self.assertRaises(ValueError):P.wrapper_axis(x,arg,.125,2)

    def test_D_stage_selection_needs_updated_action_metadata(self):
        rows,_=P.revise_plan(self.old)
        scope=dict(deepcopy=deepcopy,json=json,STAGES=dict(CALIBRATION=(4,5),HOLDOUT=(6,)),NEW='NEW_COMPILE',
            CELL_PLAN=FIX,read_pinned=lambda root,ref:json.dumps(rows).encode())
        functions(ROOT,P.STAGED_D,'scripts/pf_price_stages.py',('select','new_cells'),scope)
        templates=self.t.plans(ROOT)
        with self.assertRaisesRegex(ValueError,'CELL_PLAN'):scope['select'](ROOT,templates,'TRUNCATED_PF','CALIBRATION')
        for p in templates:
            for c in p['cells']:
                if c['row']['id']==P.PID:c['action']='SAVED_RECEIPT_FOR_C_ACCEPTANCE'
        selected=scope['select'](ROOT,templates,'TRUNCATED_PF','CALIBRATION')
        self.assertEqual(len(scope['new_cells'](selected)),5) # 10 new X/Y wrappers

    def test_existing_PR_coverage_reused_without_numeric_values(self):
        out=P.pr_coverage(ROOT);self.assertEqual(len(out['coverage']),52)
        self.assertEqual(sum(r['structural_expectation_available'] for r in out['coverage']),26)
        self.assertEqual(sum(r['finite_cohort_available'] for r in out['coverage']),4)
        self.assertFalse(any(r['qualified_compiled_expectation_available'] or r['level2_qualified'] for r in out['coverage']))
        self.assertFalse(out['price_means_recomputed']);self.assertFalse(out['fit_or_prediction'])
        self.assertTrue(all(r['model_prediction'] is None and r['mean_C'] is None for r in out['coverage']))
        self.assertTrue(all(r['finite_cohort'][0]['role']=='DEVELOPMENT' for r in out['coverage'] if r['finite_cohort']))
        for r in out['coverage']:
            if r['finite_cohort']:
                self.assertEqual([(v['source'],v['n_paths']) for v in r['finite_cohort']],[('P2',4),('P1',2)])

    def test_C1_roles_and_domains_unchanged(self):
        rows=P.c1_catalog(ROOT);self.assertEqual(len(rows),8)
        self.assertTrue(all(not r['active_LD1_LD2_schedule_match'] and not r['qualified_for_candidate'] for r in rows))
        x=[r for r in rows if r['axis']=='X']
        self.assertEqual(sorted((r['physical_candidate']['q'],r['physical_candidate']['R_outer'],r['n_paths']) for r in x),
            sorted([(8,4,9),(16,4,33),(8,8,8),(16,8,8)]))
        self.assertEqual({r['cell_id']:r['role'] for r in x},dict(H4_q8_R4='CALIBRATION',H4_q16_R4='CALIBRATION',
            H4_q8_R8='CALIBRATION',H4_q16_R8='VALIDATION'))
        self.assertTrue(all('C' not in r and 'S' not in r for r in rows))


if __name__=='__main__':unittest.main(verbosity=2)
