"""Activation delta only: invented metadata, mock backend, fresh temp outputs.

The old 18/9 tests are NOT executed. Only their pure tool fixture is reused.
No saved H4 input, real synthesis, state, RNG, circuit, price acquisition.
"""
from contextlib import ExitStack
from fractions import Fraction as F
import importlib.util
import io
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from test_cs_v0 import tool
from trottertracks.algorithm_codesign import cs_v0_execution as e
from trottertracks.algorithm_codesign.same_position_residual import Angle, RotationSite

SOURCE = 'f'*40


def invented_binding():
    sites = tuple(RotationSite(role+'/invented/factor0', 'I'*8+'Z',
                  Angle(pi=F(sign, 2), rad=F(sign, 10**8)), role+'_DIAGONAL')
                  for role, sign in (('D', 1), ('R', -1)))
    return dict(sites=sites, fixed_control_partners=tuple(RotationSite(
        s.site_id.replace('factor0', 'factor1'), 'Z'+'I'*7+'Z', s.angle.scaled(-1), 'FIXED') for s in sites),
        fixed_D_boundaries=[], fixed_selected_R_basis='invented', policy_identity='invented')


class ActivationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        d=self.root/e.OLD;d.mkdir(parents=True)
        t=tool();t['interval_dps']=100;t['synthesizer_options']['dps']=100
        (d/'NATIVE_PROFILE.json').write_text(json.dumps(dict(tool=t)))
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.preflight=self.stack.enter_context(patch.object(e,'preflight',return_value=dict(invented=True)))
        self.bind=self.stack.enter_context(patch.object(e,'bind_fixed',side_effect=lambda r:invented_binding()))
        def backend(expr,*args):
            return 'WWWWWWWS' if expr.kind=='linear' and expr.a else ''
        self.backend=self.stack.enter_context(patch.object(e.native,'backend',side_effect=backend))
        self.stack.enter_context(patch.object(e,'resources',return_value=dict(invented=True)))

    @property
    def out(self):return self.root/e.OUTPUT

    def read(self,path):return json.loads((self.out/path).read_text())

    def test_complete_call_order_dedup_and_durable_outputs(self):
        old=signal.getsignal(signal.SIGTERM);self.assertEqual(e.run(self.root,SOURCE),self.out)
        self.assertEqual(signal.getsignal(signal.SIGTERM),old)
        self.assertEqual(self.preflight.call_count,1);self.assertEqual(self.bind.call_count,1)
        counts=self.read('CALL_COUNTS.json');kinds=[c['kind'] for c in counts['calls']]
        self.assertEqual(kinds.count('binding'),1);self.assertEqual(kinds.count('local_comparison'),2)
        self.assertEqual(kinds.count('residual_table'),2);self.assertEqual(kinds.count('positive_key'),4)
        self.assertEqual(kinds.count('local_price'),2);self.assertEqual(self.backend.call_count,3)
        self.assertEqual(counts['backend'],dict(started=3,returned=3))
        self.assertTrue(all(c['state']=='RETURN' for c in counts['calls']))
        self.assertEqual(self.read('STOP.json')['status'],'MANDATORY_STOP_COMPLETE')
        self.assertEqual(self.read('RESULT.json')['full_price'],'UNKNOWN')
        self.assertEqual(len(list((self.out/'cache').glob('*.raw.json'))),4)
        journal=[json.loads(p.read_text()) for p in sorted((self.out/'journal').glob('*.json'))]
        for i,item in enumerate(journal):
            if item['event']=='BACKEND_START':
                self.assertTrue(any(v['event']=='CALL_START' and v.get('identity')==item['key'] for v in journal[:i]))
        self.assertEqual(len(self.read('spec0_finite.json')['events']),2)
        # R uses actual adjoint of shared positive rows, not new acquisition.
        self.assertTrue(any(g[0]=='w' for g in self.read('spec1_price.json')['S0']['native']['time_gates']))

    def test_preflight_failure_stops_before_science_and_keeps_source(self):
        self.preflight.side_effect=ValueError('mock profile mismatch')
        with self.assertRaises(ValueError):e.run(self.root,SOURCE)
        self.bind.assert_not_called();self.backend.assert_not_called()
        self.assertEqual(self.read('STOP.json')['stage'],'A')
        self.assertEqual(self.read('CALL_COUNTS.json')['calls'],[])
        self.assertEqual(self.read('RESULT.json')['missing_native_price'],'UNKNOWN')

    def test_first_scope_failure_never_selects_alternative(self):
        b=invented_binding();b['fixed_control_partners']=()
        self.bind.side_effect=None;self.bind.return_value=b
        with patch.object(e,'local_comparison',side_effect=AssertionError('bad scope reached comparison')):
            with self.assertRaises(ValueError):e.run(self.root,SOURCE)
        self.assertEqual(self.bind.call_count,1);self.backend.assert_not_called()

    def test_finite_cap_stops_before_any_acquisition(self):
        original=e.core.residual_table
        def too_large(r):return dict(original(r),finite_error_upper=F(1,10**10))
        with patch.object(e.core,'residual_table',side_effect=too_large):
            with self.assertRaises(ValueError):e.run(self.root,SOURCE)
        self.backend.assert_not_called();self.assertEqual(self.read('CALL_COUNTS.json')['qualified_keys'],0)

    def test_guard_failure_preserves_raw_sequence_and_forbids_repeat(self):
        self.backend.side_effect=lambda *args:'T'
        with self.assertRaises(ValueError):e.run(self.root,SOURCE)
        self.assertEqual(self.backend.call_count,1)
        raw=list((self.out/'cache').glob('*.raw.json'));self.assertEqual(len(raw),1)
        self.assertEqual(json.loads(raw[0].read_text())['sequence'],'T')
        self.assertEqual(self.read('CALL_COUNTS.json')['qualified_keys'],0)
        with self.assertRaises(FileExistsError):e.run(self.root,SOURCE)
        self.assertEqual(self.backend.call_count,1)

    def test_save_failure_keeps_raw_and_STOP_without_reacquisition(self):
        real=e.write;failed=False
        def broken(path,value):
            nonlocal failed
            if not failed and path.parent.name=='cache' and not path.name.endswith('.raw.json'):
                failed=True;raise OSError('invented durable save failure')
            return real(path,value)
        with patch.object(e,'write',side_effect=broken):
            with self.assertRaises(OSError):e.run(self.root,SOURCE)
        self.assertEqual(self.backend.call_count,1)
        self.assertEqual(len(list((self.out/'cache').glob('*.raw.json'))),1)
        self.assertIn('OSError',self.read('STOP.json')['reason'])
        self.assertEqual(self.read('CALL_COUNTS.json')['qualified_keys'],0)

    def test_local_error_cap_stops_before_price(self):
        with patch.object(e.core,'local_bounds',return_value=dict(operator_mean_error_upper=F(1))),patch.object(e,'price_group',side_effect=AssertionError('cap violation reached price')):
            with self.assertRaises(ValueError):e.run(self.root,SOURCE)
        self.assertTrue((self.out/'spec0_errors.json').exists())
        self.assertFalse((self.out/'spec0_price.json').exists())

    def test_interruption_preserves_prefix_and_restores_handler(self):
        original=self.backend.side_effect
        def interrupted(*args):
            if self.backend.call_count==2:raise InterruptedError('invented interruption')
            return original(*args)
        self.backend.side_effect=interrupted;old=signal.getsignal(signal.SIGTERM)
        with self.assertRaises(InterruptedError):e.run(self.root,SOURCE)
        self.assertEqual(signal.getsignal(signal.SIGTERM),old)
        counts=self.read('CALL_COUNTS.json')
        self.assertEqual(counts['backend'],dict(started=2,returned=1));self.assertEqual(counts['qualified_keys'],2)
        self.assertEqual(self.read('RESULT.json')['missing_native_price'],'UNKNOWN')
        self.assertEqual(counts['retries'],0)

    def test_default_CLI_has_no_science_side_effect(self):
        script=Path(__file__).resolve().parents[3]/'scripts/tracks/algorithm_codesign/run_cs_v0.py'
        spec=importlib.util.spec_from_file_location('cs_v0_activation_cli',script)
        cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        with patch.object(e,'run',side_effect=AssertionError('default activated science')),patch('sys.stdout',new_callable=io.StringIO):
            self.assertEqual(cli.main([]),0)
        self.assertFalse(self.out.exists())


class PreflightBindingTests(unittest.TestCase):
    def test_new_source_remote_and_old_preflight_connection_on_synthetic_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);d=root/e.DOC;d.mkdir(parents=True)
            script=root/'scripts/tracks/algorithm_codesign/cs_v0_preflight.py';script.parent.mkdir(parents=True)
            script.write_text('def inspect(root):\n return {"science_calls":0,"invented":True}\n')
            (root/'owned.py').write_text('pass\n')
            (d/'SOURCE.json').write_text(json.dumps(dict(activation_source_sha256={'owned.py':e.digest(root/'owned.py')})))
            old=root/e.OLD;old.mkdir(parents=True)
            (old/'SOURCE_AND_CHECKS.json').write_text(json.dumps(dict(fixed_inputs=[dict(invented=True)])))
            def git(argv,**kwargs):
                if argv[1:]==['rev-parse','HEAD']:return SOURCE+'\n'
                if argv[1]=='ls-remote':return SOURCE+'\trefs/heads/mock\n'
                if argv[1]=='rev-parse':return 'e'*40+'\n'
                return ''
            with patch.object(e.subprocess,'check_output',side_effect=git),patch.object(e,'resources',return_value=dict(memory=dict(MemAvailable='100000 kB'),observed_idle_core_equivalents='10')):
                self.assertEqual(e.preflight(root,SOURCE)['old_preflight']['science_calls'],0)
                (root/'owned.py').write_text('changed\n')
                with self.assertRaises(ValueError):e.preflight(root,SOURCE)


if __name__=='__main__':unittest.main()
