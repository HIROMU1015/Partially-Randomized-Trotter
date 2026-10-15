"""Fabricated frozen descriptions only. No native object construction."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_archive_metadata', ROOT/'src/trotterlib/native_df_archive_metadata.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    nodes = {}
    def add(record):
        key=m.digest(record); nodes[key]=record
        return {'ref':key}
    qubits=add(dict(type='tuple',items=[1,0]))
    params=add(dict(type='tuple',items=['-0.0','0.5']))
    op=dict(name='xx_plus_yy',qubits=[1,0],parameters=['-0.0','0.5'],matrix_sha256='a'*64)
    desc=add(dict(type='BasisChangeOperation',fields=dict(name=op['name'],qubits=qubits,
        parameters=params,matrix_sha256=op['matrix_sha256'])))
    ops=add(dict(type='tuple',items=[desc]))
    gate=add(dict(type='gate',name='xx_plus_yy',parameters=[{'float_hex':'-0x0.0p+0'},
        {'float_hex':'0x1.0000000000000p-1'}],label=None,matrix_sha256='a'*64))
    pair=add(dict(type='tuple',items=[gate,qubits]))
    runtime=add(dict(type='tuple',items=[pair]))
    root=add(dict(type='DFPartialS2Preparation',fields=dict(metadata=ops,runtime=runtime)))
    codec=dict(roots={'fabricated':root},nodes=nodes)
    archive=dict(schema='h6_frozen_df_preparations_v1',binding={},codec=codec,
                 codec_digest=m.digest(codec),export_receipt={})
    bh=m.digest(dict(num_system_qubits=2,operations=[op],hash_policy='ordered_local_basis_operations_v1'))
    return archive,ops,runtime,bh


class ArchiveMetadataTests(unittest.TestCase):
    def setUp(self):self.a,self.ops,self.runtime,self.bh=fixture()

    def test_descriptions_and_runtime_match_without_gate_creation(self):
        g=m.FrozenMetadataGraph(self.a)
        ops=g.basis_description(self.ops,n=2,basis_id='synthetic',basis_hash=self.bh)
        self.assertEqual(ops[0]['qubits'],[1,0])
        self.assertTrue(g.compare_runtime_descriptions(self.runtime,ops))

    def test_signed_zero_preserved(self):
        g=m.FrozenMetadataGraph(self.a)
        self.assertEqual(g.scalar({'float_hex':'-0x0.0p+0'}).hex(),'-0x0.0p+0')

    def test_corrupted_codec_digest_rejected(self):
        self.a['codec_digest']='f'*64
        with self.assertRaises(ValueError):m.FrozenMetadataGraph(self.a)

    def test_node_bytes_changed_under_same_hash_rejected(self):
        key=next(iter(self.a['codec']['nodes']));self.a['codec']['nodes'][key]['items'].reverse()
        self.a['codec_digest']=m.digest(self.a['codec'])
        with self.assertRaises(ValueError):m.FrozenMetadataGraph(self.a)

    def test_missing_reference_rejected(self):
        self.a['codec']['roots']['fabricated']={'ref':'f'*64};self.a['codec_digest']=m.digest(self.a['codec'])
        with self.assertRaises(ValueError):m.FrozenMetadataGraph(self.a)

    def test_opaque_gate_rejected_even_with_new_node_hash(self):
        gate=dict(type='gate',name='unitary',parameters=[],label=None,matrix_sha256='a'*64)
        self.a['codec']['nodes'][m.digest(gate)]=gate;self.a['codec_digest']=m.digest(self.a['codec'])
        with self.assertRaises(ValueError):m.FrozenMetadataGraph(self.a)

    def test_arrays_are_never_decoded(self):
        node=dict(type='ndarray',dtype='complex128',data='not-an-array')
        self.a['codec']['nodes'][m.digest(node)]=node;self.a['codec_digest']=m.digest(self.a['codec'])
        with self.assertRaises(ValueError):m.FrozenMetadataGraph(self.a)

    def test_noncanonical_or_nonfinite_hex_rejected(self):
        g=m.FrozenMetadataGraph(self.a)
        for value in ['nan','inf','0x1p-1']:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):g.scalar({'float_hex':value})

    def test_wrong_basis_hash_or_qubit_order_rejected(self):
        g=m.FrozenMetadataGraph(self.a)
        with self.assertRaises(ValueError):g.basis_description(self.ops,n=2,basis_id='synthetic',basis_hash='f'*64)
        described=g.basis_description(self.ops,n=2,basis_id='synthetic',basis_hash=self.bh)
        described[0]['qubits'].reverse()
        with self.assertRaises(ValueError):g.compare_runtime_descriptions(self.runtime,described)

    def test_wrong_runtime_parameter_or_matrix_claim_rejected(self):
        g=m.FrozenMetadataGraph(self.a)
        described=g.basis_description(self.ops,n=2,basis_id='synthetic',basis_hash=self.bh)
        for key,value in [('parameters',['0.0','0.5']),('matrix_sha256','b'*64)]:
            d=copy.deepcopy(described);d[0][key]=value
            with self.assertRaises(ValueError):g.compare_runtime_descriptions(self.runtime,d)

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaises(ValueError):json.loads('{"a":1,"a":2}',object_pairs_hook=m._pairs)

    def test_scalar_reader_refuses_live_object_metadata(self):
        g=m.FrozenMetadataGraph(self.a)
        with self.assertRaises(ValueError):g.scalar(self.a['codec']['roots']['fabricated'])


if __name__ == '__main__':unittest.main()
