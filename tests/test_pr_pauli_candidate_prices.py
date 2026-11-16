"""Owned Pauli price-coordinator fixtures only; no scientific child executes."""
from contextlib import ExitStack, nullcontext
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('_pauli_candidate_prices',ROOT/'scripts/run_pr_pauli_candidate_prices.py')
W=importlib.util.module_from_spec(s);s.loader.exec_module(W)


class PriceWrapperTests(unittest.TestCase):
    def fixture(self,d,multi=False,reuse=False):
        workspace=Path(d)/'cal';workspace.mkdir()
        for name in W.A_FILES+W.AUDIT_FILES:
            p=workspace/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'FIXED')
        config=dict(id='ARTIFICIAL_LD4',spec={'L_D':4})
        quote=dict(configuration_id=config['id'],policy='MODEL',complete=False,rows=[])
        registry=Path(d)/'REGISTRY.json';W.OPS.write(registry,dict(entries=[dict(name='PAULI_LD5',fixture='UNCHANGED')]))
        request=dict(workspace=str(workspace),directory='results/candidate',configuration=config,
            configuration_source={'commit':'SAVED','path':'CONFIG.json'},prepared_item={'configuration':config},price_quote=quote,
            private_root=str(Path(d)/'private'),registry_file=str(registry),d2_view='ARTIFICIAL_FIXED')
        calls=[];state=dict(counter=0,head='CODE');base=workspace/request['directory'];calbase=base/'calibration'
        implementation={'fixture':'EXPLICIT_LD4_RECIPE'}
        def git(w,*args):
            if args==('branch','--show-current'):return 'OWNED'
            if args==('rev-parse','HEAD'):return state['head']
            raise AssertionError(args)
        def commit(w,paths,message):
            state['counter']+=1;state['head']='COMMIT'+str(state['counter']);return state['head']
        def command(w,cmd,log,marker=None):
            calls.append(list(cmd));Path(log).write_text('ARTIFICIAL')
            if 'scripts/prepare_pauli_calibration.py'in cmd:
                spec=dict(prefix=4,base=str(Path(request['directory'])/'calibration'),private=str(Path(d)/'private'/'calibration'),cell={'R':1})
                requestfile=base/'CALIBRATION_00.json';W.OPS.write(requestfile,{'driver_spec':spec})
                group=dict(status='REUSE_SAVED_CALIBRATION'if reuse else'CALIBRATION_REQUEST_READY_NOT_EXECUTED',
                    saved_calibration_source={'commit':'SAVED','path':'BINDING.json'}if reuse else None,
                    driver_request={'driver_spec':spec},driver_request_file=str(requestfile))
                groups=[group,deepcopy(group)]if multi else[group]
                W.OPS.write(base/'CAL_PREPARATION.json',dict(status='CALIBRATION_GROUPS_READY_NOT_EXECUTED'if multi else group['status'],
                    calibration_requests=groups,needs=[],implementation=implementation))
            elif '--freeze'in cmd:W.OPS.write(calbase/'BINDING.json',{'fixture':True})
            elif '--execute'in cmd:
                W.OPS.write(calbase/'run01/RESULT.json',dict(status='FINITE_LD4_CAL_PATHS_COMPLETE',compiles=4,summaries=[{},{}],
                    source={'commit':'EXECUTION_CODE'},binding={'commit':cmd[cmd.index('--execute')+1]}))
            elif '--analyze'in cmd:
                W.OPS.write(calbase/'PRICE_26AXIS.json',dict(status='PRICE_RULE_SUPPORTED_MODEL_REFERENCE'))
                W.OPS.write(calbase/'audit_run01/AUDIT_RESULT.json',{'rule_supported':True})
            return 0
        quoted=[]
        def requote(req,cfg,imp,out):
            quoted.append(imp);return dict(quote,complete=True)
        patches=[patch.object(W.OPS,'owned_checkout_lock',lambda w:nullcontext()),
            patch.object(W.subprocess,'check_output',return_value=b'FIXED'),patch.object(W.OPS,'git',side_effect=git),
            patch.object(W.OPS,'commit',side_effect=commit),patch.object(W.OPS,'publish',return_value='OWNED'),
            patch.object(W,'command',side_effect=command),patch.object(W,'requote',side_effect=requote)]
        return request,calls,patches,quoted

    def test_full_chain_register_requote_and_completed_resume(self):
        with tempfile.TemporaryDirectory()as d:
            request,calls,patches,quoted=self.fixture(d)
            with ExitStack()as stack:
                for p in patches:stack.enter_context(p)
                first=W.run(request,execute=True);n=len(calls);second=W.run(request,execute=True)
            self.assertEqual(first['status'],'PRICES_READY');self.assertEqual(second['status'],'PRICES_READY')
            self.assertEqual(len(calls),n);self.assertEqual(first['new_compiles'],4)
            self.assertEqual(second['new_compiles_this_invocation'],0)
            self.assertEqual(quoted,[{'fixture':'EXPLICIT_LD4_RECIPE'}])
            registry=W.OPS.read(request['registry_file'])
            self.assertEqual(registry['entries'][0],dict(name='PAULI_LD5',fixture='UNCHANGED'))
            self.assertEqual(registry['entries'][1]['name'],'PAULI_LD4')
            self.assertNotEqual(first['binding_commit'],first['raw_commit'])
            self.assertTrue(any('--input-preflight'in c for c in calls))
            self.assertTrue(any('--analyze'in c for c in calls))

    def test_multistep_held_before_compile(self):
        with tempfile.TemporaryDirectory()as d:
            request,calls,patches,quoted=self.fixture(d,multi=True)
            with ExitStack()as stack:
                for p in patches:stack.enter_context(p)
                result=W.run(request,execute=True)
            self.assertEqual(result['status'],'NEEDS_PRICE_RULE_MULTISTEP_REGISTRATION')
            self.assertEqual(len(calls),1);self.assertEqual(result['new_compiles'],0)

    def test_saved_calibration_preserved_without_new_execution(self):
        with tempfile.TemporaryDirectory()as d:
            request,calls,patches,quoted=self.fixture(d,reuse=True)
            with ExitStack()as stack:
                for p in patches:stack.enter_context(p)
                result=W.run(request,execute=True)
            self.assertEqual(result['status'],'REUSE_SAVED_CALIBRATION_NEEDS_MODEL_LOCATOR')
            self.assertEqual(len(calls),1);self.assertEqual(result['new_compiles'],0)


if __name__=='__main__':unittest.main(verbosity=2)
