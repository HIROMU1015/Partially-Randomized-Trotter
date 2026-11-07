"""Saved string schema and explicit revised SOURCE lock; no H4 science."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_cs_v0 import fixture, policy
import test_cs_v0_execution as previous
from trottertracks.algorithm_codesign import cs_v0 as core, cs_v0_execution as execution
from trottertracks.algorithm_codesign.same_position_residual import Angle


def basis_fixture(strings=True):
    p,l,u=fixture()
    basis=p['basis_operations']['basis']
    basis.append(dict(name='xx_plus_yy',qubits=[0,1],parameters=[.1,-1.5707963267948966],matrix_sha256='mock'))
    central=u['inventory']['deterministic_half_step_records'][0]['half_step_gates'][1:-1]
    forward=[];backward=[]
    for row in basis:
        params=[float(x) for x in row['parameters']]
        forward.append(dict(key=[row['name'],[x.hex() for x in params]],qubits=row['qubits']))
        params[0]=-params[0]
        backward.append(dict(key=[row['name'],[x.hex() for x in params]],qubits=row['qubits']))
        if strings:row['parameters']=[str(x) for x in row['parameters']]
    u['inventory']['deterministic_half_step_records'][0]['half_step_gates']=list(reversed(backward))+central+forward
    return p,l,u


class StringSchemaTests(unittest.TestCase):
    def test_saved_strings_and_numeric_fixture_bind_identically(self):
        fn=lambda c:Angle(rad='1/7')
        a=core.bind_slots(*basis_fixture(True),fn,n=2,policy=policy())
        b=core.bind_slots(*basis_fixture(False),fn,n=2,policy=policy())
        self.assertEqual(a,b)
        self.assertEqual(a['sites'][0].site_id,'D/forward/one/central0/factor0')

    def test_xx_plus_yy_inverse_negates_theta_only_and_retains_beta(self):
        p,l,u=basis_fixture();first=u['inventory']['deterministic_half_step_records'][0]['half_step_gates'][0]
        self.assertEqual(first['key'][1],[(-.1).hex(),(-1.5707963267948966).hex()])
        core.bind_slots(p,l,u,lambda _:Angle(),n=2,policy=policy())
        first['key'][1][1]=(1.5707963267948966).hex()
        with self.assertRaises(ValueError):
            core.bind_slots(p,l,u,lambda _:self.fail('bad phase/basis reached resolver'),n=2,policy=policy())

    def test_unsupported_parameter_string_stops_without_resolver(self):
        p,l,u=basis_fixture();p['basis_operations']['basis'][-1]['parameters'][0]='unsupported-symbol'
        with self.assertRaises(ValueError):
            core.bind_slots(p,l,u,lambda _:self.fail('invalid string reached resolver'),n=2,policy=policy())


class RetryBoundaryTests(unittest.TestCase):
    def test_new_output_preserves_old_STOP_and_no_second_execution(self):
        helper=previous.ActivationTests(methodName='test_complete_call_order_dedup_and_durable_outputs')
        helper.setUp();self.addCleanup(helper.doCleanups)
        old=helper.root/'artifacts/cs_v0_science/2026-11-08/campaign_run01/STOP.json'
        old.parent.mkdir(parents=True);old.write_text('{"status":"old STOP"}\n');raw=old.read_bytes()
        out=execution.run(helper.root,'f'*40)
        self.assertEqual(out.name,'campaign_run02');self.assertEqual(old.read_bytes(),raw)
        self.assertEqual(json.loads((out/'SOURCE.json').read_text())['technical_reexecution_after_schema_fix'],True)
        calls=helper.backend.call_count
        with self.assertRaises(FileExistsError):execution.run(helper.root,'f'*40)
        self.assertEqual(helper.backend.call_count,calls);self.assertEqual(old.read_bytes(),raw)

    def test_complete_explicit_SOURCE_rebinding_and_mismatch_STOP(self):
        script=Path(__file__).resolve().parents[3]/'scripts/tracks/algorithm_codesign/cs_v0_preflight.py'
        spec=importlib.util.spec_from_file_location('revised_static_preflight',script)
        static=importlib.util.module_from_spec(spec);spec.loader.exec_module(static)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);d=root/execution.OLD;d.mkdir(parents=True)
            (root/'core.py').write_text('revised\n');(root/'fixed.py').write_text('fixed\n')
            sha=lambda b:hashlib.sha256(b).hexdigest()
            legacy={'core.py':sha(b'old\n'),'fixed.py':sha(b'fixed\n')}
            (d/'SOURCE_AND_CHECKS.json').write_text(json.dumps(dict(fixed_inputs=[],execution_source_sha256=legacy)))
            (d/'NATIVE_PROFILE.json').write_text(json.dumps(dict(tool=dict(runtime_executable_sha256=sha(Path(sys.executable).resolve().read_bytes()),tool_identity_sha256='0'*64))))
            revised=dict(legacy,**{'core.py':sha(b'revised\n')})
            with patch('trottertracks.algorithm_codesign.synthesis_placement.wrapper_launch.verify_runtime',return_value={}):
                with self.assertRaises(ValueError):static.inspect(root)
                self.assertEqual(static.inspect(root,source_sha256=revised)['science_calls'],0)
                for wrong in ({'core.py':revised['core.py']},dict(revised,extra='0'*64)):
                    with self.assertRaises(ValueError):static.inspect(root,source_sha256=wrong)
                (root/'fixed.py').write_text('tampered\n')
                with self.assertRaises(ValueError):static.inspect(root,source_sha256=revised)


if __name__=='__main__':unittest.main()
