"""Artificial fixtures/callbacks only; no scientific source import or H2 input."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'tests'))
import h2_pr_diagonal_a as p
import h2_pr_diagonal_a_backend as dormant
import h2_pr_identity_recovery as r
from test_h2_pr_equivalence_contract import fixture
BINDING={k:hashlib.sha256(k.encode()).hexdigest() for k in
         ('snapshot','snapshot_receipt','hamiltonian','state','preparation','common_operations')}
SOURCES={k:hashlib.sha256(k.encode()).hexdigest() for k in ('prepare','one_body','df_fragment')}


def receipt(plan):
    rows=[]
    for piece,argument in p.TIMES:
        for block in p.BLOCKS:
            vals=[.125*(k+1) for k in range(4)] if block=='one_body' else [.5]*4
            rows.append(dict(piece_time_hex=piece,primitive_argument_hex=argument,block=block,
                global_phase_hex=float(.25 if block=='one_body' else .375).hex(),
                rz=[dict(wires=[k],angle_hex=float(x).hex()) for k,x in enumerate(vals)],
                rzz=[] if block=='one_body' else [dict(wires=list(ij),angle_hex=float(.25).hex()) for ij in
                    ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))]))
    return dict(schema='H2_INDEPENDENT_DIAGONAL_PRIMITIVES_V1',binding=deepcopy(plan['binding']),
      producer_sources=deepcopy(plan['producer_sources']),input_dependencies=[dict(role=k,sha256=plan['binding'][k])
      for k in ('snapshot','snapshot_receipt')],origin='COMMON_HAMILTONIAN_PREPARATION_EXISTING_PRIMITIVES',
      input_stage_closed=True,rows=rows)


class FixtureMixin:
    def setUp(self):
        self.plan=p.draft(BINDING,SOURCES);self.packet=receipt(self.plan)
    def compare(self,packet=None,circuit=None,common=None,identity=None,pieces=None):
        a,b,c=fixture();c['preparation_sha256']=BINDING['preparation'];c['operations_sha256']=BINDING['common_operations']
        task=p.alternative.anchors()[0]['task']
        i=dict(task=task['id'],candidate=task['candidate']['id'],sample=task['sample'],axis='cosine',
               cohort='EXPLORATION',preparation=BINDING['preparation'])
        # One-piece artificial circuit; fix its independent synthetic schedule.
        with patch.object(p,'expected_piece_times',return_value=('0x1.999999999999ap-3',)):
            return p.compare_saved_axis(packet or self.packet,self.plan,circuit or a,b,common or c,'cosine',.25,
                                    pieces or ['0x1.999999999999ap-3'],identity or i)


class ReceiptTests(FixtureMixin,unittest.TestCase):
    def test_independent_artificial_receipt_and_axis(self):
        self.assertEqual(len(p.validate_independent_receipt(self.packet,self.plan)),6)
        x=self.compare();self.assertFalse(x['scientific_GO']);self.assertFalse(x['B_resolved'])

    def test_all_time_mappings_single_half(self):
        for piece,argument in p.TIMES:
            p.time_pair(piece,argument);self.assertEqual(float.fromhex(piece)/2,float.fromhex(argument))

    def test_unhalved_primitive_time_refused(self):
        for piece,argument in p.TIMES:
            with self.assertRaises(r.EvidenceError):p.time_pair(piece,piece)

    def test_twice_halved_primitive_time_refused(self):
        for piece,argument in p.TIMES:
            with self.assertRaises(r.EvidenceError):p.time_pair(piece,(float.fromhex(argument)/2).hex())

    def test_noncanonical_or_nonfinite_hex_refused(self):
        for value in ('0.1','nan','inf','-inf','0x1.000p+0'):
            with self.subTest(value=value):
                with self.assertRaises(r.EvidenceError):p.canonical_hex(value)

    def test_each_binding_field_mismatch_refused(self):
        for k in BINDING:
            v=deepcopy(self.packet);v['binding'][k]='f'*64
            with self.subTest(k=k):
                with self.assertRaises(r.EvidenceError):p.validate_independent_receipt(v,self.plan)

    def test_producer_SOURCE_mismatch_refused(self):
        v=deepcopy(self.packet);v['producer_sources']['one_body']='f'*64
        with self.assertRaises(r.EvidenceError):p.validate_independent_receipt(v,self.plan)

    def test_QPY_event_cost_dependency_contamination_refused(self):
        for role in ('QPY','archived_wrapper','compiled_raw','event','cost_receipt','expected_angle'):
            v=deepcopy(self.packet);v['input_dependencies'].append(dict(role=role,sha256='a'*64))
            with self.subTest(role=role):
                with self.assertRaises(r.EvidenceError):p.validate_independent_receipt(v,self.plan)

    def test_independence_label_alone_not_sufficient(self):
        v=deepcopy(self.packet);v['input_dependencies']=[]
        with self.assertRaises(r.EvidenceError):p.validate_independent_receipt(v,self.plan)

    def test_QPY_origin_duplicate_values_refused(self):
        v=deepcopy(self.packet);v['origin']='SAVED_QPY_ANGLES'
        with self.assertRaises(r.EvidenceError):p.validate_independent_receipt(v,self.plan)

    def test_unclosed_independent_stage_refused(self):
        v=deepcopy(self.packet);v['input_stage_closed']=False
        with self.assertRaises(r.EvidenceError):p.validate_independent_receipt(v,self.plan)

    def test_missing_or_extra_primitive_refused(self):
        for n in (5,7):
            v=deepcopy(self.packet);v['rows']=(v['rows']*2)[:n]
            with self.assertRaises(r.EvidenceError):p.validate_independent_receipt(v,self.plan)

    def test_reordered_time_or_block_refused(self):
        v=deepcopy(self.packet);v['rows'][0],v['rows'][1]=v['rows'][1],v['rows'][0]
        with self.assertRaises(r.EvidenceError):p.validate_independent_receipt(v,self.plan)

    def test_missing_wire_or_order_refused(self):
        v=deepcopy(self.packet);v['rows'][0]['rz'].reverse()
        with self.assertRaises(r.EvidenceError):p.validate_independent_receipt(v,self.plan)

    def test_forged_source_input_extra_fields_refused(self):
        v=deepcopy(self.packet);v['expected_QPY_parameters']={}
        with self.assertRaises(r.EvidenceError):p.validate_independent_receipt(v,self.plan)

    def test_every_primitive_row_missing_field_refused(self):
        for key in self.packet['rows'][0]:
            v=deepcopy(self.packet);del v['rows'][0][key]
            with self.subTest(key=key):
                with self.assertRaises((r.EvidenceError,KeyError)):p.validate_independent_receipt(v,self.plan)


class CompareTests(FixtureMixin,unittest.TestCase):
    def test_changed_absolute_angle_with_intact_mirror_refused(self):
        v=deepcopy(self.packet);v['rows'][-2]['rz'][0]['angle_hex']=float(.2).hex()
        with self.assertRaises(r.EvidenceError):self.compare(v)

    def test_changed_independent_phase_refused(self):
        v=deepcopy(self.packet);v['rows'][-2]['global_phase_hex']=float(.3).hex()
        with self.assertRaisesRegex(r.EvidenceError,'BLOCK_PHASE'):self.compare(v)

    def test_sign_change_refused(self):
        v=deepcopy(self.packet);v['rows'][-1]['rzz'][0]['angle_hex']=float(-.25).hex()
        with self.assertRaises(r.EvidenceError):self.compare(v)

    def test_other_cohort_refused(self):
        task=p.alternative.anchors()[0]['task'];v=dict(task=task['id'],candidate=task['candidate']['id'],
            sample=task['sample'],axis='cosine',cohort='CONFIRMATION',preparation=BINDING['preparation'])
        with self.assertRaises(r.EvidenceError):self.compare(identity=v)

    def test_other_axis_or_candidate_or_preparation_refused(self):
        task=p.alternative.anchors()[0]['task'];v=dict(task=task['id'],candidate=task['candidate']['id'],
            sample=task['sample'],axis='cosine',cohort='EXPLORATION',preparation=BINDING['preparation'])
        for key,value in (('axis','sine'),('candidate','OTHER'),('preparation','f'*64),('sample',False)):
            changed=dict(v);changed[key]=value
            with self.subTest(key=key):
                with self.assertRaises(r.EvidenceError):self.compare(identity=changed)

    def test_wrong_known_piece_time_is_not_interchangeable(self):
        with self.assertRaisesRegex(r.EvidenceError,'SIGNED_SCHEDULE'):
            self.compare(pieces=[p.TIMES[0][0]])

    def test_fixed_original_S2_Y4_schedules(self):
        anchors=p.alternative.anchors()
        s2=next(x['task']['id'] for x in anchors if x['task']['candidate']['id']=='H2O32_c7ec590a32a859bc')
        y4=next(x['task']['id'] for x in anchors if x['task']['candidate']['id']=='H2O32_8a2ecdd0294ffb74')
        self.assertEqual(p.expected_piece_times(s2),('0x1.999999999999ap-3',)*4)
        self.assertEqual(len(p.expected_piece_times(y4)),6)
        self.assertEqual(set(p.expected_piece_times(y4)),{x[0] for x in p.TIMES[:2]})

    def test_unknown_time_refused(self):
        with self.assertRaises(r.EvidenceError):self.compare(pieces=[float(.3).hex()])

    def test_missing_piece_refused(self):
        with self.assertRaises(r.EvidenceError):self.compare(pieces=[p.TIMES[0][0],p.TIMES[1][0]])

    def test_changed_QPY_wire_refused(self):
        a,b,c=fixture();a['instructions'][2]['wires']=[1]
        with self.assertRaises(r.EvidenceError):self.compare(circuit=a)

    def test_changed_common_preparation_refused(self):
        a,b,c=fixture();c.update(preparation_sha256='f'*64,operations_sha256=BINDING['common_operations'])
        with self.assertRaises(r.EvidenceError):self.compare(common=c)


class ControlTests(FixtureMixin,unittest.TestCase):
    def test_each_contract_scope_field_mutation_refused(self):
        for key in self.plan:
            v=deepcopy(self.plan);v[key]='MUTATION'
            with self.subTest(key=key):
                with self.assertRaises((r.EvidenceError,TypeError)):p.validate_draft(v)

    def test_preflight_is_not_scientific_permission(self):
        v=p.preflight(self.plan,BINDING,SOURCES)
        self.assertFalse(v['execution_ready']);self.assertEqual(v['actual_scientific_calls'],0)

    def test_actual_cli_refuses_before_input_read(self):
        x=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/h2_pr_diagonal_a.py'),'--execute','--draft','/MISSING'],capture_output=True,text=True)
        self.assertNotEqual(x.returncode,0);self.assertIn('NOT_APPROVED',x.stderr);self.assertNotIn('FileNotFoundError',x.stderr)

    def test_dormant_real_backend_refuses_even_fake_activation(self):
        with self.assertRaises(r.EvidenceError):dormant.PlannedExistingBackend(object(),{})

    def test_actual_domain_injected_backend_refused_before_bootstrap(self):
        v=SimpleNamespace(domain='SAVED_H2',bootstrap=lambda:1/0)
        with self.assertRaises(r.EvidenceError):p.synthetic_workflow(self.plan,v,[None]*134)

    def test_budget_overrun_never_invokes_callback(self):
        ledger=p.Ledger();ledger.call('bootstrap',lambda:None)
        with self.assertRaises(r.EvidenceError):ledger.call('bootstrap',lambda:1/0)
        with self.assertRaises(r.EvidenceError):ledger.call('compile',lambda:1/0)

    def test_existing_functions_receive_primitive_argument_without_halving(self):
        # Build only a fake adapter object; every scientific dependency is injected.
        obj=object.__new__(dormant.PlannedExistingBackend);seen=[]
        obj.np=SimpleNamespace(asarray=lambda x:x)
        obj.ops=SimpleNamespace(one_body_diagonal_primitives=lambda a,t:seen.append(('one_body',t)),
            df_squared_diagonal_primitives=lambda a,l,t:seen.append(('df_fragment',t)))
        prep=SimpleNamespace(deterministic_blocks=[SimpleNamespace(diagonal_eigenvalues=[1,2,3,4]),
                         SimpleNamespace(diagonal_eta=[1,2,3,4],lam=.3)])
        for piece,argument in p.TIMES:
            for block in p.BLOCKS:obj.primitive(prep,block,argument)
        self.assertEqual(seen,[(b,float.fromhex(t)) for x,t in p.TIMES for b in p.BLOCKS])

    def test_exact_mock_trace_six_primitives_before_134_QPY_compares(self):
        events=[];plan=self.plan;packet=self.packet
        class Fake:
            domain='ARTIFICIAL'
            def bootstrap(self):events.append('bootstrap')
            def setup(self):events.append('setup');return object()
            def prepare(self,model):events.append('prepare');return object()
            def primitive(self,prep,block,argument):
                events.append('primitive');x=next(x for x in packet['rows'] if x['primitive_argument_hex']==argument and x['block']==block)
                return SimpleNamespace(global_phase=float.fromhex(x['global_phase_hex']),
                    rz=tuple((v['wires'][0],float.fromhex(v['angle_hex'])) for v in x['rz']),
                    rzz=tuple((*v['wires'],float.fromhex(v['angle_hex'])) for v in x['rzz']))
            def input_dependencies(self):return packet['input_dependencies']
            def compare(self,job,raw,sha):
                self_outer.assertEqual(events[:9],['bootstrap','setup','prepare']+['primitive']*6)
                self_outer.assertEqual(hashlib.sha256(raw).hexdigest(),sha);events.append('compare')
        self_outer=self;x=p.synthetic_workflow(plan,Fake(),[None]*134)
        self.assertEqual(x['used'],p.BUDGET);self.assertEqual(events.count('compare'),134);self.assertFalse(x['scientific_GO'])

    def test_no_scientific_imports(self):
        self.assertFalse(any(k.split('.')[0] in ('numpy','qiskit','scipy','trotterlib','trottertracks') for k in sys.modules))


if __name__=='__main__':unittest.main()
