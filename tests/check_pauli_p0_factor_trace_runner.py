"""Exercise the actual scientific caller with entirely artificial inputs."""
from copy import deepcopy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from openfermion import QubitOperator

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_pauli_p0_factor_trace as R
from trottertracks import pauli_p0_factor_trace as F


class Caller(unittest.TestCase):
    def run_artificial(self,base,*,match=True,source=None):
        source=source or R.subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        # Saved tensor order has mode 0 as MSB after the caller's bit reversal.
        z=np.diag([1.]*128+[-1.]*128);tail=np.eye(256)+.25*z
        archive=io.BytesIO();np.savez_compressed(archive,head=np.eye(256),tail=tail,Pauli_nonidentity=.25*z)
        default=QubitOperator(((0,'Z'),),.25)
        shadow=QubitOperator();shadow.terms=dict(default.terms);shadow.terms[()]=1.
        bridge=QubitOperator();bridge.terms=dict(default.terms);bridge.terms[()]=.5
        raw=R.raw_coefficients(default)
        def read(path):
            ld=int(path.split('/')[0][2:])
            if path.endswith('JW_COEFFICIENTS.json'):return deepcopy(raw)
            if path.endswith('FIXED_INPUTS.json'):return dict(binding=dict(system='ARTIFICIAL'))
            return dict(prefixes=[dict(prefix=ld,pauli_identity=0.,saved_DF_descriptor=dict(scalar=1.),
                identity_metrics=dict(Pauli_tail_full_fro=16.,Pauli_tail_saved_scalar_fro=0.))])
        def blob(root,commit,path):
            if path in R.CODE:return (ROOT/path).read_bytes()
            return archive.getvalue()
        facts=dict(rows=[dict(L_D=p)for p in (9,8,7)])
        result=dict(evidence=dict(ARTIFICIAL=True),default_jw=default,shadow_jw=shadow,bridge_jw=bridge)
        if not match:result['default_jw']=QubitOperator(((0,'Z'),),.5)
        with patch.object(R,'preflight',return_value=facts),patch.object(R,'read',side_effect=read), \
             patch.object(R.P0,'blob',side_effect=blob), \
             patch.object(R.P0,'load_saved',return_value=(SimpleNamespace(n_qubits=8),None,None,None,dict(artificial=True))), \
             patch.object(F,'trace',return_value=result),patch('sys.stdout',new=io.StringIO()):
            code=R.execute(source,base)
        return code

    def test_real_caller_writes_all_three_mock_traces_and_keeps_original_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'fresh';code=self.run_artificial(out)
            self.assertEqual(code,0,(out/'RESULT.json').read_text())
            got=json.loads((out/'RESULT.json').read_text())
            self.assertEqual([r['L_D']for r in got['factor_traces']],[9,8,7])
            self.assertEqual(got['scope']['split_traces'],3)
            self.assertEqual(got['scope']['mean_calls'],0)
            self.assertTrue(got['original_STOPs_preserved'])
            self.assertFalse(got['production_dictionary_adopted'])
            for row in got['factor_traces']:
                self.assertFalse(row['default']['within_original_absolute_gate'])
                self.assertTrue(row['shadow']['within_original_absolute_gate'])
            with self.assertRaisesRegex(ValueError,'FRESH'):self.run_artificial(out)

    def test_default_mismatch_stops_with_raw_trace_and_no_remaining_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'fresh';self.assertEqual(self.run_artificial(out,match=False),2)
            got=json.loads((out/'RESULT.json').read_text())
            self.assertEqual(got['status'],'STOP_DIAGNOSTIC')
            self.assertTrue((out/'LD9/COEFFICIENT_TRACE.json.gz').exists())
            self.assertTrue((out/'LD9/DIAGNOSTIC_JW.json').exists())
            self.assertFalse((out/'LD8/COEFFICIENT_TRACE.json.gz').exists())
            self.assertEqual(got['scope']['split_traces'],0)


if __name__=='__main__':unittest.main(verbosity=2)
