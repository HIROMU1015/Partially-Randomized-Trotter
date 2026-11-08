"""Only invented saved-schema fixtures. Never load real N1/state/law."""
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction as F
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from qiskit import QuantumCircuit,ClassicalRegister
from qiskit.circuit.library import RZGate,XXPlusYYGate,CSwapGate,XGate
from qiskit.quantum_info import Operator,Statevector

from trottertracks.algorithm_codesign import vacuum_h4_adapter as a
from trottertracks.algorithm_codesign import vacuum_h4_numeric as num
from trottertracks.algorithm_codesign import vacuum_h4_execution as exe
from test_a1_engine_connection import EngineConnectionTests as ExistingFixture

EVIDENCE=[]


def fixture():
    f=ExistingFixture
    prep=deepcopy(f.saved);config=deepcopy(f.config_record);policy=deepcopy(f.policy)
    labels=sorted(c['component_id'] for c in prep['components'])
    assert len(labels)>=3
    law=dict(uniform_grid_size=2**53,orders=[0,2],order_masses=[dict(exact='3/4'),dict(exact='1/4')],
        application_label_order='products_then_rotation',
        Q_definition='Q(order)*product(Q(label_position)); independent uniform-word model',
        label_masses=[dict(label=x,exact=str((F(1,4),F(1,4),F(1,2))[i]) if i<3 else '0')
                      for i,x in enumerate(labels)])
    source=dict(source_commit='invented',scope=dict(system_qubits=2,m=0,q=1,r=1,K=2,
        delta=.02,coefficient_atol=0.,model='invented two-mode fixture; no molecular inputs'))
    return dict(zip(('SOURCE.json','CONFIG.json','LAW.json','POLICY.json','PREPARATION.json'),
                    (source,config,law,policy,prep)))


def context(records):return a.SavedPR(records,expected_hashes=a.record_hashes(records))


def body(c):
    n=(c.num_qubits-1)//2 if c.num_qubits==5 else c.num_qubits-1
    return num.clean_body(c,logical_ancilla=c.num_qubits-1,
        reset_wires=tuple(range(n,2*n)) if c.num_qubits==5 else ())


def initial(psi,width,target):
    out=np.zeros(1<<width,complex)
    for i,z in enumerate(psi):
        out[sum(((i>>k)&1)<<q for k,q in enumerate(target))]=z
    return out


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):ExistingFixture.setUpClass() # reuse fixture only, no old tests

    def test_state_free_restore_reuses_fixed_selected_builder(self):
        r=fixture();c=context(r)
        self.assertFalse(hasattr(c,'raw_state'))
        old=ExistingFixture().make_engine()
        for p in a.select_paths(r):
            for axis in ('X','Y'):
                x=c.selected_circuit(p['order'],p['labels'],axis)
                y=old.selected_circuit(p['order'],p['labels'],axis)
                self.assertEqual(a.old.canonical([(i.operation.name,i.operation.params,
                    [x.find_bit(q).index for q in i.qubits]) for i in x.data]),
                    a.old.canonical([(i.operation.name,i.operation.params,
                    [y.find_bit(q).index for q in i.qubits]) for i in y.data]))
        for name in ('probability','qualify','uniform_inventory'):
            with self.assertRaises(ValueError):getattr(c,name)()

    def test_path_rule_is_positive_cost_blind_and_order_fixed(self):
        r=fixture();p=a.select_paths(r)
        labels=sorted(x['label'] for x in r['LAW.json']['label_masses'] if F(x['exact'])>0)
        self.assertEqual(p[0]['labels'],(labels[0],));self.assertEqual(p[1]['labels'],tuple(labels))
        self.assertEqual([x['order'] for x in p],[0,2])
        r['LAW.json']['label_masses'].reverse()
        self.assertEqual(a.select_paths(r),p)
        EVIDENCE.append(dict(case='cost_blind_rule',paths=p,old_measurements_used=False))

    def test_identity_source_policy_and_saved_matrix_tampering_stop(self):
        r=fixture();hashes=a.record_hashes(r);r['SOURCE.json']['scope']['q']=2
        with self.assertRaisesRegex(ValueError,'record identity'):a.SavedPR(r,expected_hashes=hashes)
        r=fixture();r['POLICY.json']['declaration']['source_commit']='changed'
        with self.assertRaisesRegex(ValueError,'policy'):context(r)
        r=fixture();rows=next(iter(r['PREPARATION.json']['basis_operations'].values()))
        rows[0]['matrix_sha256']='bad'
        with self.assertRaises(ValueError):context(r)

    def test_traced_natural_split_and_raw_scalar_order(self):
        r=fixture();c=context(r)
        for p in a.select_paths(r):
            raw,L,R,split=c.body_with_boundaries(p['order'],p['labels'])
            self.assertEqual(split['unit_count'],2*len(c.blocks)+p['order']+1)
            self.assertEqual(split['right_units'],split['unit_count']//2)
            full=Operator(R.circuit()).data
            full=Operator(L.circuit()).data@full*np.exp(1j*float(L.scalar+R.scalar))
            # Fixed controlled builder evaluated only on its control1 block.
            control=Operator(raw).data[4:,4:]
            self.assertLess(np.max(abs(full-control)),8e-12)
            self.assertGreater(np.linalg.norm(Operator(L.circuit()).data@Operator(R.circuit()).data
                -Operator(R.circuit()).data@Operator(L.circuit()).data),1e-8)
            for axis in ('X','Y'):
                exe._same_evolution(raw,c.selected_circuit(p['order'],p['labels'],axis),axis)
            EVIDENCE.append(dict(case='saved_schema_split',path=p,split=split,
                scalar_terms=[*R.scalar_terms,*L.scalar_terms]))

    def test_XY_probability_phase_reference_and_permuted_wires(self):
        r=fixture();c=context(r);psi=np.array([1,2j,3,-4j])/np.sqrt(30)
        for p in a.select_paths(r):
            raw,L,R,_=c.body_with_boundaries(p['order'],p['labels'])
            z=np.vdot(psi,Operator(raw).data[4:,4:]@psi)
            for axis in ('X','Y'):
                for mapping in ((0,1,2,3,4),(4,3,2,1,0)):
                    ref,ledger=a.build_reference(L,R,axis,mapping=mapping)
                    clean=num.clean_body(ref,logical_ancilla=mapping[-1],reset_wires=mapping[2:4])
                    vec=Statevector(initial(psi,5,mapping[:2])).evolve(clean).data
                    prob=sum(abs(x)**2 for i,x in enumerate(vec) if not ((i>>mapping[-1])&1))
                    expected=z.real if axis=='X' else z.imag
                    self.assertLess(abs(2*prob-1-expected),8e-12)
                    leakage=sum(abs(x)**2 for i,x in enumerate(vec) if any((i>>q)&1 for q in mapping[2:4]))
                    self.assertLess(leakage,8e-12)
                    self.assertEqual(F(ledger['correction_exact']),
                        F(ledger['L_scalar_free_vacuum'])+F(ledger['R_scalar_free_vacuum'])+L.scalar+R.scalar)

    def test_eight_system_descriptor_mapping_without_state_enumeration(self):
        r=a.Lane(8,(('rz',((.2).hex(),),(7,)),('xx_plus_yy',((.3).hex(),(.4).hex()),(0,7))),((.1).hex(),))
        l=a.Lane(8,(('rzz',((-.3).hex(),),(0,7)),),((-.2).hex(),))
        q,ledger=a.build_reference(l,r,'Y')
        self.assertEqual(q.num_qubits,17);self.assertEqual(q.count_ops()['cswap'],16)
        self.assertEqual(q.count_ops()['reset'],8);self.assertEqual(q.count_ops()['measure'],1)
        self.assertEqual(q.find_bit(q.data[-1].qubits[0]).index,16)
        self.assertEqual(float(q.global_phase),0.)
        EVIDENCE.append(dict(case='invented_8_system_descriptor',qubits=17,
            statevector_columns_evaluated=0,ledger=ledger))

    def test_unsupported_gate_control_and_invalid_mapping_fail_closed(self):
        q=QuantumCircuit(3);q.h(0)
        with self.assertRaisesRegex(ValueError,'unsupported'):a.decontrol(q,2,0,1)
        q=QuantumCircuit(3);q.append(RZGate(.3).control(1,ctrl_state=0),[2,0])
        with self.assertRaisesRegex(ValueError,'control'):a.decontrol(q,2,0,1)
        lane=a.Lane(2,(),())
        with self.assertRaisesRegex(ValueError,'permutation'):a.build_reference(lane,lane,'X',mapping=[0]*5)

    def test_correction_rounding_is_exactly_accounted_without_modulo(self):
        lane=a.Lane(2,(),((.1).hex(),(.2).hex(),(1000.).hex()))
        _,ledger=a.build_reference(lane,a.Lane(2,(),()),'X')
        exact=F(.1)+F(.2)+1000
        self.assertEqual(F(ledger['correction_rounding_upper']),abs(F(float(exact))-exact))
        self.assertEqual(float.fromhex(ledger['correction_hex']),float(exact))
        self.assertGreater(F(ledger['correction_rounding_upper']),0)

    def test_new_primitive_exact_matrices_and_old_interval_reuse(self):
        for g in (CSwapGate(),XGate()):
            self.assertEqual(num.certificate(g)['error_upper'],0)
        self.assertEqual(num.certificate(RZGate(.3))['error_upper'],
                         a.old.certify_gate(RZGate(.3))['error_upper'])
        q=QuantumCircuit(1);q.h(0)
        _,b=num.simulate_isometry(q,1)
        self.assertGreater(F(b['frobenius_simulation_error_upper']),0)

    def test_roundoff_encloses_exact_fraction_operator_contraction(self):
        q=QuantumCircuit(2);q.rz(.37,0);q.append(XXPlusYYGate(.23,-.4),[1,0]);q.rzz(-.71,0,1)
        actual,proof=num.simulate_isometry(q,2);maximum=F(0)
        for j in range(4):
            v=[(F(int(i==j)),F(0)) for i in range(4)]
            for item in q.data:
                cert=num.certificate(item.operation)
                v=a.old.apply_exact(v,cert['matrix'],[q.find_bit(t).index for t in item.qubits])
            for i,(x,y) in enumerate(v):
                maximum=max(maximum,abs(F(actual[i,j].real)-x),abs(F(actual[i,j].imag)-y))
        self.assertLessEqual(maximum,F(proof['frobenius_simulation_error_upper']))
        EVIDENCE.append(dict(case='explicit_IEEE_vs_exact_Fraction',max_component=str(maximum),
                            proof=proof))

    def test_phase_error_cannot_be_hidden_by_fitting(self):
        q=QuantumCircuit(2);q.h(1);q.crz(.3,1,0);q.h(1)
        b=ClassicalRegister(1);q.add_register(b);q.measure(1,b)
        changed=q.copy();changed.global_phase=.1
        proof=num.qualify_compilation(q,changed,1)
        self.assertGreater(F(proof['projected_operator_error_upper']),F(9,100))
        self.assertFalse(proof['phase_fitting'])

    def test_reset_readout_dynamic_and_layout_guards(self):
        q=QuantumCircuit(3,1);q.reset(1);q.h(2);q.measure(2,0)
        num.clean_body(q,logical_ancilla=2,reset_wires=(1,))
        with self.assertRaisesRegex(ValueError,'reset'):num.clean_body(q,logical_ancilla=2)
        q.x(2)
        with self.assertRaisesRegex(ValueError,'post-readout'):num.clean_body(q,logical_ancilla=2,reset_wires=(1,))
        with self.assertRaisesRegex(ValueError,'permutation'):
            num.clean_body(q,logical_ancilla=2,permutation=[0,0,2],reset_wires=(1,))

    def test_profile_is_level2_and_old_profile_unchanged(self):
        exe.Profile().validate()
        self.assertEqual(exe.CompilerProfile().optimization_level,1)
        with self.assertRaises(ValueError):exe.Profile(optimization_level=1).validate()

    def test_existing_STOP_output_collision_prevents_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'stopped';out.mkdir();(out/'STOP.json').write_text('preserve')
            with patch.object(exe.saved,'load_fixed_inputs',side_effect=AssertionError('must not load')):
                with self.assertRaises(FileExistsError):
                    exe.run(temp,out,expected_source_commit='0'*40)
            self.assertEqual((out/'STOP.json').read_text(),'preserve')

    def test_adapter_has_no_state_RNG_or_science_loader(self):
        tree=ast.parse(Path(a.__file__).read_text())
        attrs={x.attr for x in ast.walk(tree) if isinstance(x,ast.Attribute)}
        names={x.id for x in ast.walk(tree) if isinstance(x,ast.Name)}
        self.assertFalse({'Statevector','Operator','eigh','random','default_rng','load_fixed_inputs'}&(attrs|names))
        self.assertFalse(hasattr(a.SavedPR,'raw_state'))


if __name__=='__main__':unittest.main()
