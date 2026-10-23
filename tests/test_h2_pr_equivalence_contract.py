"""Artificial dictionaries/byte fixtures only. No molecular input or science."""
from copy import deepcopy
from itertools import combinations
import json
from pathlib import Path
import subprocess
import sys
import unittest
import tempfile
import hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import h2_pr_stream_conformance as stream
import h2_pr_equivalence_contract as policy
import h2_pr_identity_recovery as r
import h2_pr_runtime_relevance as runtime


def gate(name,wires,values=(),clbits=()):
    controls=1 if name in ('CXGate','CZGate','CRZGate') else 0
    return dict(name=name,wires=list(wires),params=[dict(kind='float',hex=float(v).hex()) for v in values],
                clbits=list(clbits),label='',controls=controls,ctrl_state=controls)


def fixture(axis='cosine'):
    """Hand-written artificial one-piece pattern; no Qiskit/physical lowerer."""
    gs=[[dict(gate='p',params=[v.hex()],wires=[w])] for v,w in
        ((.125,0),(.25,1),(.375,2),(.5,3),(.625,0),(.75,1))]
    receipt=dict(operation_stream=[dict(rotation_sha256=str(i)*64,operations=g) for i,g in enumerate(gs)])
    def app(index,role,support,group=2,identity=False,sign=1):
        return dict(application_index=index,role=role,is_identity=identity,coefficient_sign=sign,
          diagonal_pauli_support=support,basis_id='ARTIFICIAL'+str(group),basis_hash=str(group)*64,
          component_id='ARTIFICIAL_COMPONENT_'+str(index),basis_change_operations=[] if identity else
          [dict(name='p',parameters=[str(float.fromhex(gs[group][0]['params'][0]))],qubits=gs[group][0]['wires'])])
    # Two adjacent equal-basis rotations, an identity flush, a switched basis,
    # and a negative product with an event phase. These exercise source ordering.
    events=[[dict(phase=dict(real=1,imag=0),rotation_angle=.125,application_sequence=[app(0,'rotation',[0])]),
             dict(phase=dict(real=1,imag=0),rotation_angle=.25,application_sequence=[app(0,'rotation',[1])]),
             dict(phase=dict(real=1,imag=0),rotation_angle=.125,application_sequence=[app(0,'rotation',[],identity=True)]),
             dict(phase=dict(real=-1,imag=0),rotation_angle=.375,application_sequence=[
                 app(0,'product',[2],group=3,sign=-1),app(1,'rotation',[3],group=3)])]]
    items=[gate('HGate',[4]),gate('PhaseGate',[0],[-.125])]
    items += [gate('RZGate',[k],[.125*(k+1)]) for k in range(4)]
    items += [gate('PhaseGate',[0],[.125]),gate('PhaseGate',[1],[-.25])]
    pairs=list(combinations(range(4),2))
    items += [gate('RZZGate',pair,[.25]) for pair in pairs]
    items += [gate('RZGate',[k],[.5]) for k in range(4)]
    items += [gate('PhaseGate',[1],[.25]),gate('PhaseGate',[2],[-.375]),
              gate('CRZGate',[4,0],[.25]),gate('CRZGate',[4,1],[.5]),
              gate('PhaseGate',[2],[.375]),gate('PhaseGate',[3],[-.5]),
              gate('CZGate',[4,2]),gate('CRZGate',[4,3],[.75]),gate('PhaseGate',[3],[.5]),
              gate('PhaseGate',[4],[(3.141592653589793-.125)+3.141592653589793]),
              gate('PhaseGate',[1],[-.25])]
    for left,right in pairs:
        items += [gate('CXGate',[left,right]),gate('CXGate',[4,right]),gate('RZGate',[right],[-.25]),
                  gate('CXGate',[4,right]),gate('CXGate',[left,right])]
    for k in range(4):items += [gate('CXGate',[4,k]),gate('RZGate',[k],[-.5]),gate('CXGate',[4,k])]
    items += [gate('RZGate',[4],[.75]),gate('PhaseGate',[1],[.25]),gate('PhaseGate',[0],[-.125])]
    for k in range(4):items += [gate('CXGate',[4,k]),gate('RZGate',[k],[-.125*(k+1)]),gate('CXGate',[4,k])]
    items += [gate('RZGate',[4],[.5]),gate('PhaseGate',[0],[.125]),gate('PhaseGate',[4],[-.8*.25])]
    if axis=='sine':items += [gate('SdgGate',[4])]
    items += [gate('HGate',[4]),gate('Measure',[4],clbits=[0])]
    circuit=dict(nq=5,nc=1,phase=dict(kind='float',hex=float(.625).hex()),custom={},instructions=items)
    return circuit,events,receipt


class StreamTests(unittest.TestCase):
    def run_fixture(self,c=None,e=None,p=None,axis='cosine'):
        a,b,d=fixture(axis);return stream.recognize(c or a,e or b,p or d,axis,.25)

    def test_complete_artificial_pattern(self):
        x=self.run_fixture();self.assertEqual(x['conditions']['complete_instruction_coverage'],'VERIFIED')
        self.assertEqual(x['conditions']['absolute_deterministic_diagonal_vs_common_preparation'],'UNRESOLVED')
        self.assertTrue(x['global_phase_relation_exact']);self.assertEqual(x['identity_flushes'],1)
        self.assertEqual(x['reuse_boundaries'],2)
        self.assertFalse(x['scientific_operator_equivalence'])

    def test_sine_full_wrapper(self):self.run_fixture(axis='sine')

    def test_wrong_inverse_sign_rejected(self):
        c,e,p=fixture();c['instructions'][1]['params'][0]['hex']=float(.125).hex()
        with self.assertRaises(r.EvidenceError):self.run_fixture(c,e,p)

    def test_every_instruction_omission_is_detected(self):
        c,e,p=fixture()
        for index in range(len(c['instructions'])):
            with self.subTest(index=index):
                changed=deepcopy(c);changed['instructions'].pop(index)
                with self.assertRaises((r.EvidenceError,IndexError)):self.run_fixture(changed,e,p)

    def test_every_gate_wire_change_is_detected(self):
        c,e,p=fixture()
        for index in range(len(c['instructions'])):
            with self.subTest(index=index):
                changed=deepcopy(c);changed['instructions'][index]['wires'][0]=(changed['instructions'][index]['wires'][0]+1)%5
                with self.assertRaises(r.EvidenceError):self.run_fixture(changed,e,p)

    def test_extra_basis_at_reuse_boundary_rejected(self):
        c,e,p=fixture();index=next(i for i,x in enumerate(c['instructions']) if x['name']=='CRZGate')
        c['instructions'].insert(index+1,gate('PhaseGate',[2],[.375]))
        with self.assertRaises(r.EvidenceError):self.run_fixture(c,e,p)

    def test_event_order_change_rejected(self):
        c,e,p=fixture();e[0][0],e[0][1]=e[0][1],e[0][0]
        with self.assertRaises(r.EvidenceError):self.run_fixture(c,e,p)

    def test_event_phase_change_rejected(self):
        c,e,p=fixture();e[0][-1]['phase']['real']=1
        with self.assertRaises(r.EvidenceError):self.run_fixture(c,e,p)

    def test_product_sign_changes_relative_phase(self):
        c,e,p=fixture();e[0][-1]['application_sequence'][0]['coefficient_sign']=1
        with self.assertRaises(r.EvidenceError):self.run_fixture(c,e,p)

    def test_identity_angle_changes_relative_phase(self):
        c,e,p=fixture();e[0][2]['rotation_angle']=.25
        with self.assertRaises(r.EvidenceError):self.run_fixture(c,e,p)

    def test_directional_mirror_angle_change_rejected(self):
        c,e,p=fixture();index=next(i for i,x in enumerate(c['instructions']) if x['name']=='CXGate')
        c['instructions'][index+2]['params'][0]['hex']=float(.25).hex()
        with self.assertRaises(r.EvidenceError):self.run_fixture(c,e,p)

    def test_unknown_global_relation_is_unresolved(self):
        c,e,p=fixture();c['phase']['hex']=float(.5).hex()
        x=self.run_fixture(c,e,p);self.assertEqual(x['conditions']['serialized_global_phase_relation'],'UNRESOLVED')

    def test_scalar_duplication_rejected(self):
        c,e,p=fixture();c['instructions'].insert(-2,gate('PhaseGate',[4],[-.8*.25]))
        with self.assertRaises(r.EvidenceError):self.run_fixture(c,e,p)

    def test_control_state_metadata_rejected(self):
        c,e,p=fixture();next(i for i in c['instructions'] if i['name']=='CRZGate')['ctrl_state']=0
        with self.assertRaises(r.EvidenceError):self.run_fixture(c,e,p)

    def test_measurement_metadata_rejected(self):
        c,e,p=fixture();c['instructions'][-1]['controls']=1
        with self.assertRaises(r.EvidenceError):self.run_fixture(c,e,p)

    def test_unknown_gaussian_gate_rejected(self):
        c,e,p=fixture();p['operation_stream'][0]['operations'][0]['gate']='unknown'
        with self.assertRaises(r.EvidenceError):self.run_fixture(c,e,p)

    def test_unregistered_event_basis_rejected(self):
        c,e,p=fixture();e[0][0]['application_sequence'][0]['basis_change_operations'][0]['parameters']=['.9']
        with self.assertRaises(r.EvidenceError):self.run_fixture(c,e,p)

    def test_wrong_initial_H_parameter_rejected(self):
        c,e,p=fixture();c['instructions'][0]['params']=[dict(kind='float',hex=0.0.hex())]
        with self.assertRaises(r.EvidenceError):self.run_fixture(c,e,p)

    def test_mirrored_diagonal_change_never_proves_absolute_binding(self):
        c,e,p=fixture()
        c['instructions'][2]['params'][0]['hex']=float(.2).hex()
        for x in reversed(c['instructions']):
            if x['name']=='RZGate' and x['wires']==[0]:
                x['params'][0]['hex']=float(-.2).hex();break
        result=self.run_fixture(c,e,p)
        self.assertEqual(result['conditions']['absolute_deterministic_diagonal_vs_common_preparation'],'UNRESOLVED')
        self.assertFalse(result['scientific_operator_equivalence'])

    def test_nonfinite_parameter_rejected(self):
        with self.assertRaises(r.EvidenceError):stream.number(dict(kind='float',hex='nan'))


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.draft=policy.draft();cls.rows=[]
        for anchor in policy.anchors():
            task=anchor['task']
            for axis in ('cosine','sine'):
                cls.rows.append(dict(task=task['id'],candidate=task['candidate']['id'],sample=task['sample'],axis=axis,
                  cohort='EXPLORATION',result_sha256=anchor['acceptance']['saved']['result_sha256'],
                  correctness=dict(scope='SINGLE_SAVED_INPUT_STATE_CT01',strictly_below=1e-7,PASS=True,
                                   phase_alignment=False,receipt_sha256=policy.original_correctness_sha(anchor,axis)),
                  conditions={name:dict(status='NOT_APPLICABLE' if name in policy.OUTSIDE else 'VERIFIED',
                     evidence_sha256='b'*64,scope='ARTIFICIAL_METADATA_ONLY') for name in policy.A+policy.B+policy.OUTSIDE}))

    def test_complete_artificial_packet_never_scientific_GO(self):
        x=policy.preflight(self.draft,self.rows)
        self.assertEqual(x['technical_axis_counts']['A']['VERIFIED'],134)
        self.assertEqual(x['technical_axis_counts']['B']['VERIFIED'],134)
        self.assertFalse(x['execution_ready']);self.assertFalse(x['scientific_GO'])
        self.assertEqual(x['strict_reuse_accepted_axes'],0)

    def test_A_does_not_imply_B(self):
        rows=deepcopy(self.rows)
        for row in rows:row['conditions'][policy.B[-1]]['status']='UNRESOLVED'
        x=policy.preflight(self.draft,rows)
        self.assertEqual(x['technical_axis_counts']['A']['VERIFIED'],134)
        self.assertEqual(x['technical_axis_counts']['B']['UNRESOLVED'],134)

    def test_mismatch_distinct_from_unknown(self):
        rows=deepcopy(self.rows);rows[0]['conditions'][policy.A[0]]['status']='MISMATCH'
        x=policy.preflight(self.draft,rows);self.assertEqual(x['technical_axis_counts']['A']['MISMATCH'],1)

    def test_each_required_condition_cannot_be_omitted(self):
        for name in policy.A+policy.B:
            rows=deepcopy(self.rows);del rows[0]['conditions'][name]
            with self.subTest(name=name):
                with self.assertRaises(r.EvidenceError):policy.preflight(self.draft,rows)

    def test_runtime_evidence_unknown_blocks_B(self):
        rows=deepcopy(self.rows);rows[0]['conditions'][policy.B[-1]]['status']='UNRESOLVED'
        self.assertEqual(policy.preflight(self.draft,rows)['technical_axis_counts']['B']['UNRESOLVED'],1)

    def test_unknown_status_rejected(self):
        rows=deepcopy(self.rows);rows[0]['conditions'][policy.A[0]]['status']='LIKELY_OK'
        with self.assertRaises(r.EvidenceError):policy.preflight(self.draft,rows)

    def test_required_condition_cannot_be_not_applicable(self):
        rows=deepcopy(self.rows);rows[0]['conditions'][policy.A[0]]['status']='NOT_APPLICABLE'
        with self.assertRaises(r.EvidenceError):policy.preflight(self.draft,rows)

    def test_evidence_SHA_invalid_rejected(self):
        rows=deepcopy(self.rows);rows[0]['conditions'][policy.A[0]]['evidence_sha256']='bad'
        with self.assertRaises(Exception):policy.preflight(self.draft,rows)

    def test_other_cohort_rejected(self):
        rows=deepcopy(self.rows);rows[0]['cohort']='CONFIRMATION'
        with self.assertRaises(r.EvidenceError):policy.preflight(self.draft,rows)

    def test_other_candidate_rejected(self):
        rows=deepcopy(self.rows);rows[0]['candidate']='OTHER'
        with self.assertRaises(r.EvidenceError):policy.preflight(self.draft,rows)

    def test_different_original_receipt_rejected(self):
        rows=deepcopy(self.rows);rows[0]['result_sha256']='c'*64
        with self.assertRaises(r.EvidenceError):policy.preflight(self.draft,rows)

    def test_duplicate_axis_rejected(self):
        rows=deepcopy(self.rows);rows[-1]=deepcopy(rows[0])
        with self.assertRaises(r.EvidenceError):policy.preflight(self.draft,rows)

    def test_missing_axis_rejected(self):
        with self.assertRaises(r.EvidenceError):policy.preflight(self.draft,self.rows[:-1])

    def test_bool_sample_rejected(self):
        rows=deepcopy(self.rows);rows[0]['sample']=False
        with self.assertRaises(r.EvidenceError):policy.preflight(self.draft,rows)

    def test_correctness_operator_scope_promotion_rejected(self):
        rows=deepcopy(self.rows);rows[0]['correctness']['scope']='ALL_STATES_OPERATOR'
        with self.assertRaises(r.EvidenceError):policy.preflight(self.draft,rows)

    def test_correctness_threshold_relaxation_rejected(self):
        rows=deepcopy(self.rows);rows[0]['correctness']['strictly_below']=1e-6
        with self.assertRaises(r.EvidenceError):policy.preflight(self.draft,rows)

    def test_correctness_phase_fit_rejected(self):
        rows=deepcopy(self.rows);rows[0]['correctness']['phase_alignment']=True
        with self.assertRaises(r.EvidenceError):policy.preflight(self.draft,rows)

    def test_outside_scope_cannot_be_claimed_verified(self):
        rows=deepcopy(self.rows);rows[0]['conditions'][policy.OUTSIDE[0]]['status']='VERIFIED'
        with self.assertRaises(r.EvidenceError):policy.preflight(self.draft,rows)

    def test_every_policy_field_mutation_rejected(self):
        for name in self.draft:
            d=deepcopy(self.draft);d[name]='ARTIFICIAL_MUTATION'
            with self.subTest(field=name):
                with self.assertRaises(r.EvidenceError):policy.validate_draft(d)

    def test_scientific_execute_refused_before_input_read(self):
        result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/h2_pr_equivalence_contract.py'),
                               '--execute','--draft','/ARTIFICIAL_MISSING'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0);self.assertIn('NO_SCIENTIFIC_EXECUTION_ENTRY',result.stderr)
        self.assertNotIn('FileNotFoundError',result.stderr)

    def test_correctness_receipt_SHA_mismatch_rejected(self):
        rows=deepcopy(self.rows);rows[0]['correctness']['receipt_sha256']='f'*64
        with self.assertRaises(r.EvidenceError):policy.preflight(self.draft,rows)

    def test_no_scientific_imports(self):
        self.assertFalse(any(x.split('.')[0] in ('numpy','qiskit','scipy','trotterlib','trottertracks') for x in sys.modules))


class ByteAndDependencyTests(unittest.TestCase):
    def test_original_SHA_match(self):
        with tempfile.TemporaryDirectory(prefix='h2_alt_artificial_') as directory:
            p=Path(directory)/'artificial';p.write_bytes(b'ARTIFICIAL_BYTES')
            x=dict(bytes=len(p.read_bytes()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            self.assertEqual(r.verified_bytes(p,x),b'ARTIFICIAL_BYTES')

    def test_original_SHA_mutation_rejected(self):
        with tempfile.TemporaryDirectory(prefix='h2_alt_artificial_') as directory:
            p=Path(directory)/'artificial';p.write_bytes(b'ARTIFICIAL_BYTES')
            x=dict(bytes=len(p.read_bytes()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            p.write_bytes(b'ARTIFICIAL_XYTES')
            with self.assertRaises(r.EvidenceError):r.verified_bytes(p,x)

    def test_ELF_direct_dependencies_are_data_not_load_observation(self):
        text='0x01 (NEEDED) Shared library: [libm.so.6]\n0x01 (NEEDED) Shared library: [liblzma.so.5]'
        self.assertEqual(runtime.needed_libraries(text),['libm.so.6','liblzma.so.5'])
        self.assertEqual(runtime.needed_libraries('artificial no dynamic dependencies'),[])

    def test_static_import_inspection_does_not_import_library(self):
        with tempfile.TemporaryDirectory(prefix='h2_alt_artificial_') as directory:
            p=Path(directory)/'artificial.py';p.write_text('import qiskit\nfrom numpy import array\n')
            result=runtime.source_record(p,'ARTIFICIAL',('import',))
            self.assertEqual(result['imports'],['numpy','qiskit'])
            self.assertNotIn('qiskit',sys.modules);self.assertNotIn('numpy',sys.modules)



if __name__=='__main__':unittest.main()
