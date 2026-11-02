"""Metadata/cache regression only: no circuits, QPY, transpiler or science.

The proxy uses the installed Qiskit metadata descriptor but never constructs a
QuantumCircuit. QPY and transpile are mocked before the production helper runs.
"""
from pathlib import Path
import copy
import json
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))

from qiskit import QuantumCircuit
from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
from trottertracks.pr_dictionary_validation.gates import compiled_wrapper


class MetadataProxy:
    # Exercise the real pinned API restriction without building a circuit.
    metadata=QuantumCircuit.metadata

    def __init__(self,payload,metadata=None,name='original'):
        self.payload=copy.deepcopy(payload)
        self.metadata=copy.deepcopy({} if metadata is None else metadata)
        self.name=name
        self.layout=None

    def copy(self):
        return MetadataProxy(self.payload,self.metadata,self.name)


def fake_qpy_dump(proxy,stream):
    """Opaque strings only; no quantum IR serialization or numerical angles."""
    stream.write(json.dumps(dict(payload=proxy.payload,metadata=proxy.metadata,
                               name=proxy.name),sort_keys=True).encode())


class CompileMetadataTests(unittest.TestCase):
    def setUp(self):
        self.dump_patch=patch('qiskit.qpy.dump',side_effect=fake_qpy_dump)
        self.transpile_patch=patch('qiskit.transpile',side_effect=lambda proxy,**kw:proxy.copy())
        self.lower_patch=patch('qiskit.transpiler.PassManager.run',side_effect=lambda proxy:proxy.copy())
        self.dump=self.dump_patch.start();self.transpile=self.transpile_patch.start()
        self.lower_patch.start()
        self.addCleanup(self.dump_patch.stop);self.addCleanup(self.transpile_patch.stop)
        self.addCleanup(self.lower_patch.stop)
        self.identity=dict(input='mock_input',source='mock_source',environment='mock_env')
        self.profile=CompilerProfile()

    def test_metadata_normalization_preserves_original_and_reuses_cache(self):
        original=MetadataProxy(dict(parameter='opaque_A',global_phase='opaque_phase_A'),
                               metadata={'description':'first'},name='first_name')
        with self.assertRaises(TypeError):
            original.copy().metadata=None
        cache={}
        first=compiled_wrapper(original,self.profile,cache=cache,identity_context=self.identity)
        self.assertEqual(original.metadata,{'description':'first'})
        self.assertEqual(original.name,'first_name')
        self.assertEqual(original.payload,first.payload)
        serialized=self.dump.call_args.args[0]
        self.assertEqual(serialized.metadata,{})
        self.assertEqual(serialized.name,'full_ir')
        alias=MetadataProxy(original.payload,metadata={'description':'second'},name='second_name')
        second=compiled_wrapper(alias,self.profile,cache=cache,identity_context=self.identity)
        self.assertEqual(first.payload,second.payload)
        self.assertIsNot(first,second)
        self.assertEqual(self.transpile.call_count,1)  # mocked; actual calls=0
        self.assertEqual(len(cache),1)

    def test_payload_phase_and_bound_identity_are_not_aliased(self):
        original=MetadataProxy(dict(parameter='opaque_A',global_phase='opaque_phase_A'))
        cache={}
        compiled_wrapper(original,self.profile,cache=cache,identity_context=self.identity)
        for field,value in [('parameter','opaque_B'),('global_phase','opaque_phase_B')]:
            changed=original.copy();changed.payload[field]=value
            compiled_wrapper(changed,self.profile,cache=cache,identity_context=self.identity)
        compiled_wrapper(original,self.profile,cache=cache,
                         identity_context=dict(self.identity,source='other_mock_source'))
        self.assertEqual(len(cache),4)
        self.assertEqual(self.transpile.call_count,4)  # mocked; actual calls=0


if __name__=='__main__':
    unittest.main()
