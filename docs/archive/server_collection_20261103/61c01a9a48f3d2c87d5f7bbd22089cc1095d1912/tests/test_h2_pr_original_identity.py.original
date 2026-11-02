"""Synthetic byte fixtures only; never import science or open private inputs."""
import copy
import json
from pathlib import Path
import struct
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import h2_pr_saved_qpy as q
import h2_pr_identity_recovery as r
import audit_h2_pr_original_identity as audit


def instruction(name,wires=(),params=(),clbits=()):
    n=name.encode();raw=struct.pack('!HHHIIBHqII',len(n),0,len(params),len(wires),len(clbits),0,0,0,0,0)+n
    for i in wires:raw+=struct.pack('!cI',b'q',i)
    for i in clbits:raw+=struct.pack('!cI',b'c',i)
    for p in params:raw+=struct.pack('!cQ',b'f',8)+struct.pack('<d',p)
    return raw


def circuit(instructions=(),phase=.375,layout=False,nq=5,nc=1):
    meta=b'{}';name=b'artificial'
    return (struct.pack('!HcHIIQIQI',len(name),b'f',8,nq,nc,len(meta),0,len(instructions),0)+name+
        struct.pack('!d',phase)+meta+struct.pack('!Q',0)+b''.join(instructions)+struct.pack('!H',0)+
        struct.pack('!?iiiIi',layout,-1,-1,-1,0,-1))


def qpy(raw):return struct.pack('!6sBBBBQc',b'QISKIT',13,1,3,0,1,b'e')+b'q'+raw


def item(name,wires,params=(),clbits=()):
    return dict(name=name,wires=list(wires),params=[dict(kind='float',hex=float(x).hex()) for x in params],
                clbits=list(clbits),label='',controls=0,ctrl_state=0)


def wrapper(axis='cosine',scalar=.25):
    items=[item('HGate',[4]),item('CRZGate',[4,2],[.4]),item('PhaseGate',[4],[-.8*scalar])]
    if axis=='sine':items.append(item('SdgGate',[4]))
    items.extend([item('HGate',[4]),item('Measure',[4],clbits=[0])])
    return dict(nq=5,nc=1,phase=dict(kind='float',hex=0.0.hex()),custom={},instructions=items)


class Parser(unittest.TestCase):
    def test_numeric_header_and_instruction_byte_orders_differ(self):
        c=q.read_qpy(qpy(circuit([instruction('PhaseGate',[4],[.125])])))
        self.assertEqual(c['phase']['hex'],float(.375).hex())
        self.assertEqual(c['instructions'][0]['params'][0]['hex'],float(.125).hex())

    def test_truncated_stream_rejected(self):
        raw=qpy(circuit([instruction('PhaseGate',[4],[.125])]))
        for length in (1,19,40,len(raw)-1):
            with self.subTest(length=length):
                with self.assertRaises(r.EvidenceError):q.read_qpy(raw[:length])

    def test_unrecognized_QPY_version_rejected(self):
        raw=bytearray(qpy(circuit()));raw[6]=12
        with self.assertRaisesRegex(r.EvidenceError,'FIXED_QPY'):q.read_qpy(bytes(raw))

    def test_extra_file_bytes_rejected(self):
        with self.assertRaisesRegex(r.EvidenceError,'TRAILING_BYTES'):q.read_qpy(qpy(circuit())+b'X')

    def test_wrong_wire_rejected(self):
        with self.assertRaisesRegex(r.EvidenceError,'QUBIT_WIRE'):q.read_qpy(qpy(circuit([instruction('XGate',[5])])))

    def test_layout_not_silently_ignored(self):
        with self.assertRaisesRegex(r.EvidenceError,'LAYOUT_UNSUPPORTED'):q.read_qpy(qpy(circuit(layout=True)))

    def test_nonfinite_instruction_parameter_rejected(self):
        with self.assertRaisesRegex(r.EvidenceError,'NONFINITE_FLOAT'):q.read_qpy(qpy(circuit([instruction('RZGate',[0],[float('nan')])])))

    def test_symbolic_values_remain_opaque_never_evaluated(self):
        value=q.Reader(b'').value(b'e',b'ARTIFICIAL_NONEXECUTABLE_SYMBOLIC_PAYLOAD')
        self.assertEqual(value['kind'],'opaque:e');self.assertIn('sha256',value)

    def test_base_gate_header_declares_qubits_but_has_no_wire_bytes(self):
        n=b'RZZGate';raw=struct.pack('!HHHIIBHqII',len(n),0,1,2,0,0,0,0,0,0)+n+struct.pack('!cQ',b'f',8)+struct.pack('<d',.2)
        parsed=q.Reader(raw).instruction(None,None)
        self.assertEqual(parsed['params'][0]['hex'],float(.2).hex());self.assertEqual(parsed['wires'],[])


class MetadataJoins(unittest.TestCase):
    def event(self,angle=.2):
        return [[dict(rotation_angle=angle,application_sequence=[dict(is_identity=False,role='rotation',
            diagonal_pauli_support=[2],coefficient_sign=1,basis_change_operations=[])])]]

    def test_event_central_order_sign_and_support(self):
        self.assertEqual(q.event_central_join(wrapper(),self.event())['central_applications'],1)

    def test_event_negative_coefficient_sign_is_used(self):
        e=self.event(-.2);e[0][0]['application_sequence'][0]['coefficient_sign']=-1
        q.event_central_join(wrapper(),e)

    def test_changed_event_angle_rejected(self):
        with self.assertRaisesRegex(r.EvidenceError,'CENTRAL_ORDER'):q.event_central_join(wrapper(),self.event(.3))

    def test_changed_event_support_rejected(self):
        e=self.event();e[0][0]['application_sequence'][0]['diagonal_pauli_support']=[1]
        with self.assertRaisesRegex(r.EvidenceError,'CENTRAL_ORDER'):q.event_central_join(wrapper(),e)

    def controlled_rzz(self):
        c=wrapper();base=item('RZZGate',[],[.4])
        body=dict(nq=3,nc=0,phase=dict(kind='float',hex=0.0.hex()),custom={},
            instructions=[item('CCXGate',[0,1,2]),item('CRZGate',[0,2],[.4]),item('CCXGate',[0,1,2])])
        c['custom']={'artificial_uuid':dict(controls=1,ctrl_state=1,base_operation=base,body=body)}
        c['instructions'][1]=item('artificial_uuid',[4,1,2],[.4])
        e=self.event();e[0][0]['application_sequence'][0]['diagonal_pauli_support']=[1,2]
        return c,e

    def test_saved_controlled_RZZ_body_and_base_match_event(self):
        c,e=self.controlled_rzz()
        self.assertEqual(q.event_central_join(c,e)['central_applications'],1)

    def test_wrong_controlled_RZZ_definition_rejected(self):
        c,e=self.controlled_rzz();c['custom']['artificial_uuid']['body']['instructions'][1]['wires']=[0,1]
        with self.assertRaisesRegex(r.EvidenceError,'CONTROLLED_RZZ_BODY'):q.event_central_join(c,e)

    def test_wrong_controlled_RZZ_base_parameter_rejected(self):
        c,e=self.controlled_rzz();c['custom']['artificial_uuid']['base_operation']['params'][0]['hex']=float(.5).hex()
        with self.assertRaisesRegex(r.EvidenceError,'CONTROLLED_RZZ_BODY'):q.event_central_join(c,e)

    def test_structural_digest_ignores_custom_uuid_and_preserves_angle(self):
        a,_=self.controlled_rzz();b=copy.deepcopy(a)
        b['custom']['different_uuid']=b['custom'].pop('artificial_uuid');b['instructions'][1]['name']='different_uuid'
        self.assertEqual(q.structural_digest(a),q.structural_digest(b))
        b['instructions'][1]['params'][0]['hex']=float(.5).hex()
        self.assertNotEqual(q.structural_digest(a),q.structural_digest(b))

    def test_wrong_control_wire_rejected(self):
        c=wrapper();c['instructions'][1]['wires']=[3,2]
        with self.assertRaisesRegex(r.EvidenceError,'CENTRAL_ORDER'):q.event_central_join(c,self.event())

    def test_wrapper_scalar_and_sine_scope(self):
        a=audit.wrapper_fields(wrapper(),'cosine',.25);b=audit.wrapper_fields(wrapper('sine'),'sine',.25)
        self.assertEqual(a['evolution_structure_sha256'],b['evolution_structure_sha256'])

    def test_wrong_scalar_rejected(self):
        with self.assertRaisesRegex(r.EvidenceError,'SCALAR_ONCE'):audit.wrapper_fields(wrapper(),'cosine',.5)

    def test_wrong_measurement_wire_rejected(self):
        c=wrapper();c['instructions'][-1]['wires']=[3]
        with self.assertRaisesRegex(r.EvidenceError,'WRAPPER_WIRES'):audit.wrapper_fields(c,'cosine',.25)

    def test_missing_sine_phase_rejected(self):
        with self.assertRaisesRegex(r.EvidenceError,'SINE_WRAPPER'):audit.wrapper_fields(wrapper(),'sine',.25)

    def test_Gaussian_fields_require_exact_hex_and_wire_sequence(self):
        c=wrapper();c['instructions'].insert(1,item('PhaseGate',[0],[.125]))
        receipt=dict(operation_stream=[dict(rotation_sha256='a'*64,operations=[dict(gate='p',params=[float(.125).hex()],wires=[0])])])
        self.assertEqual(q.gaussian_matches(c,receipt)[0]['forward_exact_occurrences'],1)
        c['instructions'][1]['wires']=[1]
        self.assertEqual(q.gaussian_matches(c,receipt)[0]['forward_exact_occurrences'],0)

    def test_event_decimal_repr_converts_to_exact_hex(self):
        e=self.event();e[0][0]['application_sequence'][0]['basis_change_operations']=[dict(name='p',parameters=['0.125'],qubits=[0])]
        prep=dict(operation_stream=[dict(operations=[dict(gate='p',params=[float(.125).hex()],wires=[0])])])
        self.assertEqual(q.event_basis_join(e,prep)['applications'],1)

    def test_event_basis_change_mismatch_rejected(self):
        e=self.event();e[0][0]['application_sequence'][0]['basis_change_operations']=[dict(name='p',parameters=['0.125'],qubits=[0])]
        prep=dict(operation_stream=[dict(operations=[dict(gate='p',params=[float(.25).hex()],wires=[0])])])
        with self.assertRaisesRegex(r.EvidenceError,'BASIS_VS_COMMON'):q.event_basis_join(e,prep)

    def test_data_read_imports_no_scientific_library(self):
        self.assertFalse(any(k.split('.')[0] in ('numpy','qiskit','scipy','trotterlib','trottertracks') for k in sys.modules))


if __name__=='__main__':unittest.main()
