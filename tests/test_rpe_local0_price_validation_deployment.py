"""Deployment-only checks: old local0 bootstrap and the next fixed child SOURCE.

The bootstrap is the preserved deployed client file. Preflight and Popen are
replaced only in an isolated synthetic process; no normal search is launched.
The separate strict-SOURCE check documents the current metadata-HEAD boundary.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
OLD_CLIENT = Path('/home/AbeHiromu/projects/rpe-local0-search/absolute-kappa-9ad0260777a4/local0-handoff/tools/local0-search/remote.py')
OLD_ROOT = Path('/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-local0-kappa-source-9ad0260777a4')


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(directory):
    root = directory / 'new-source'; (root / 'scripts').mkdir(parents=True)
    (root / 'src').mkdir()
    (root / 'scripts' / 'run_rpe_local0_search.py').write_text("raise AssertionError('No real child may launch')\n")
    (root / 'scripts' / 'run_rpe_method_optimizer.py').write_text("raise AssertionError('No optimizer may launch')\n")
    problem = dict(problem=dict(system='ARTIFICIAL_DEPLOYMENT_ONLY',hamiltonian_data=dict(expected=dict(fragment_count=10))),
        accuracy=dict(epsilon_E_decimal='1'),target=dict(energy_ha=0),rpe=dict(M=12),
        methods=dict(PR_PAULI=dict(family='PR',L_D_range=dict(min=1,max_offset_from_fragment_count=-1),caps=[2]*13,generation={})),
        search=dict(decrement_steps=[1],large_step_headroom='9/10',seed_rules=[],max_steps_per_method=2,max_new_q3_per_method=0),
        execution_routes={})
    spec=dict(round_id='ARTIFICIAL_DEPLOYMENT',methods=['PR_PAULI'],science_approved=False,
        max_new_q3_per_method={'PR_PAULI':0},initial='all',executor=dict(prior_run_dirs=[]))
    dump(directory/'PROBLEM.json',problem);dump(directory/'ROUND.default.json',spec)
    config=dict(code_root=str(root),code_source='ARTIFICIAL_NEW_FIXED_SOURCE',computation_source='ARTIFICIAL_SCIENCE_SOURCE',
        problem=str(directory/'PROBLEM.json'),round=str(directory/'ROUND.default.json'),entry='method',
        state_directory=str(root/'docs/research/local0_search_state'),python=sys.executable)
    config['config_path']=str(directory/'SERVICE.json');dump(config['config_path'],config)
    return root,problem,spec,config


class Local0PriceValidationDeployment(unittest.TestCase):
    def test_preserved_old_bootstrap_uses_new_current_service_for_next_child(self):
        if not OLD_CLIENT.is_file() or not (OLD_ROOT/'src/trottertracks/rpe_local0_search.py').is_file():
            self.skipTest('Preserved local0 deployment is not available on this host')
        with tempfile.TemporaryDirectory()as temporary:
            directory=Path(temporary);root,_,_,config=fixture(directory)
            pointer=directory/'CURRENT_SERVICE.json';dump(pointer,dict(service_config=config['config_path']))
            old=directory/'old-state';old.mkdir();(old/'STOP').write_text('PRESERVED_OLD_STOP')
            before=(old/'STOP').read_bytes()
            request=dict(scope=dict(methods=['PR_PAULI'],per_method=dict(PR_PAULI=dict(L_D=dict(min=2,max=2))),shots='auto',
                stop=dict(max_new_q3_per_method=0)))
            payload=dict(old_root=str(OLD_ROOT),old_bootstrap=str(OLD_CLIENT),pointer=str(pointer),request=request,
                         expected_root=str(root),expected_source=config['code_source'])
            harness=r'''
import base64,contextlib,io,json,pathlib,subprocess,sys
from unittest.mock import patch
v=json.loads(sys.argv[1]);old=pathlib.Path(v['old_root']);sys.path[:0]=[str(old/'src'),str(old/'scripts')]
from trottertracks import rpe_local0_search as api
assert pathlib.Path(api.__file__).resolve()==(old/'src/trottertracks/rpe_local0_search.py').resolve()
commands=[];preflights=[]
def preflight(self,run,problem,spec):
    preflights.append(dict(run=str(run),code_source=self.config['code_source']))
    return dict(status='PASS',scientific_calls=0,ARTIFICIAL=True)
class Child:
    pid=123456789

def child(command,**kwargs):
    commands.append(command)
    assert command[2]==str(pathlib.Path(v['expected_root'])/'scripts/run_rpe_local0_search.py')
    return Child()
config=dict(code_root=str(old),server_config=v['pointer'],action='start',instruction='ARTIFICIAL deployment test only',
    request_base64=base64.b64encode(json.dumps(v['request']).encode()).decode())
buffer=io.StringIO()
with patch.object(api.Gateway,'normal_preflight',preflight),patch.object(subprocess,'Popen',side_effect=child),contextlib.redirect_stdout(buffer):
    exec(compile(pathlib.Path(v['old_bootstrap']).read_text(),v['old_bootstrap'],'exec'),{'CONFIG':config})
result=json.loads(buffer.getvalue())['result'];run=pathlib.Path(result['run_directory'])
assert pathlib.Path(v['expected_root'])in run.parents
assert preflights==[dict(run=str(run),code_source=v['expected_source'])]
assert len(commands)==1 and commands[0][3]=='--config'
assert json.loads((run/'APPLIED_SCOPE.json').read_text())['code_source']==v['expected_source']
assert result['status']=='STARTED' and result['started'] is True
print(json.dumps(dict(imported_gateway=str(api.__file__),next_child=commands[0],preflight=preflights[0],real_launches=0)))
'''
            result=subprocess.run([sys.executable,'-B','-c',harness,json.dumps(payload)],check=True,
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            observed=json.loads(result.stdout)
            self.assertEqual(observed['real_launches'],0)
            self.assertEqual(observed['preflight']['code_source'],config['code_source'])
            self.assertEqual(Path(observed['next_child'][2]),root/'scripts/run_rpe_local0_search.py')
            self.assertEqual(before,(old/'STOP').read_bytes())
            self.assertFalse((old/'CURRENT_TARGET.json').exists())

    def test_new_normal_cli_preview_creates_no_run_or_scientific_job(self):
        with tempfile.TemporaryDirectory()as temporary:
            directory=Path(temporary);root,_,_,config=fixture(directory)
            request=directory/'PREVIEW.json'
            dump(request,dict(action='preview',scope=dict(methods=['PR_PAULI'],
                per_method=dict(PR_PAULI=dict(L_D=dict(min=2,max=2))),shots='auto',
                stop=dict(max_new_q3_per_method=0))))
            command=[sys.executable,'-B',str(ROOT/'scripts/run_rpe_local0_search.py'),
                '--config',config['config_path'],'--request',str(request)]
            completed=subprocess.run(command,check=True,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            result=json.loads(completed.stdout)
            self.assertEqual(result['status'],'PREVIEW')
            self.assertIs(result['started'],False)
            state=Path(config['state_directory'])
            self.assertTrue(state.is_relative_to(root))
            self.assertFalse((state/'runs').exists())
            self.assertFalse((state/'CURRENT_TARGET.json').exists())
            self.assertFalse(list(state.glob('**/REQUEST.json')))

    def test_metadata_publication_allows_same_run_but_scientific_changes_do_not(self):
        spec=importlib.util.spec_from_file_location('_local0_deployment_fixed_guard',ROOT/'src/trottertracks/rpe_local0_search.py')
        api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
        with tempfile.TemporaryDirectory()as temporary:
            directory=Path(temporary);root,problem,round_spec,config=fixture(directory)
            def git(*args):return subprocess.check_output(['git',*args],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
            git('init');git('config','user.name','Synthetic Fixture');git('config','user.email','fixture@example.invalid')
            git('add','src','scripts');git('commit','-qm','fixed synthetic code')
            config['code_source']=git('rev-parse','HEAD')
            run=root/'docs/research/local0_search_state/runs/ARTIFICIAL';run.mkdir(parents=True)
            applied=dict(entry='method',scope={},signature='ARTIFICIAL',problem_sha256=api.digest(problem),round_requested_sha256=api.digest(round_spec))
            dump(run/'APPLIED_SCOPE.json',applied);dump(run/'PROBLEM.json',problem);dump(run/'ROUND.json',round_spec)
            dump(run/'DUPLICATE_PREFLIGHT.json',dict(status='PASS'));dump(run/'AUTHORIZATION.json',dict(scope_signature='ARTIFICIAL',kind='EXPLICIT_USER_START'))
            (root/'docs/metadata.txt').write_text('Synthetic data publication only')
            git('add','docs/metadata.txt');git('commit','-qm','metadata-only publication')
            self.assertNotEqual(git('rev-parse','HEAD'),config['code_source'])
            class Child:
                pid=123456789
                def wait(self):return 0
            checked=api.fixed_implementation(config)
            with patch.object(api,'fixed_implementation',return_value=checked),patch.object(api.subprocess,'Popen',return_value=Child())as child:
                api.execute_run(config,run,'ARTIFICIAL_METADATA_RESUME')
                child.assert_called_once()
            self.assertEqual(api.load(run/'NORMAL_DRIVER.json')['implementation']['source'],config['code_source'])
            original=(root/'scripts/run_rpe_method_optimizer.py').read_text()
            (root/'scripts/run_rpe_method_optimizer.py').write_text(original+'# changed scientific code\n')
            with self.assertRaisesRegex(api.RequestError,'fixed implementation/settings changed'):
                api.fixed_implementation(config)
            git('add','scripts');git('commit','-qm','changed scientific code')
            with self.assertRaisesRegex(api.RequestError,'fixed implementation/settings changed'):
                api.fixed_implementation(config)
            git('restore','--source',config['code_source'],'--','scripts')
            (root/'src/extra.py').write_text('# untracked imported code\n')
            with self.assertRaisesRegex(api.RequestError,'fixed implementation/settings changed'):
                api.fixed_implementation(config)

    def test_same_run_can_select_only_a_bound_operational_schedule_entry(self):
        from trottertracks import rpe_local0_search as api
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root, problem, round_spec, config = fixture(directory)
            run = directory/'search-artificial'
            run.mkdir()
            applied = dict(run_id=run.name, entry='schedule', scope={}, signature='ARTIFICIAL',
                           problem_sha256=api.digest(problem), round_requested_sha256=api.digest(round_spec))
            dump(run/'APPLIED_SCOPE.json', applied)
            dump(run/'PROBLEM.json', problem)
            dump(run/'ROUND.json', round_spec)
            dump(run/'DUPLICATE_PREFLIGHT.json', dict(status='PASS'))
            dump(run/'AUTHORIZATION.json', dict(scope_signature='ARTIFICIAL', kind='EXPLICIT_USER_START'))
            deployment = directory/'new-operations'
            driver = deployment/'scripts/run_rpe_schedule_search.py'
            driver.parent.mkdir(parents=True)
            driver.write_text('# artificial scheduler replacement\n')
            selection = dict(run_id=run.name, base_source=config['code_source'], source='FIXED_SCHEDULER',
                             root=str(deployment), driver_sha256=api.hashlib.sha256(driver.read_bytes()).hexdigest())
            child = Mock(pid=1234)
            child.wait.return_value = 0
            config.pop('publication', None)
            with patch.object(api, 'fixed_implementation', return_value={'source': config['code_source']}) as check, \
                    patch.object(api.subprocess, 'Popen', return_value=child) as popen:
                api.execute_run(config, run, 'ARTIFICIAL_RESUME', schedule_resume=selection)
                command = popen.call_args.args[0]
                self.assertEqual(command[4:6], [str(root), str(driver)])
                self.assertEqual(popen.call_args.kwargs['cwd'], str(root))
                self.assertEqual(api.load(run/'NORMAL_DRIVER.json')['implementation']['schedule_source'], 'FIXED_SCHEDULER')
                check.assert_called_once_with(config)
                driver.write_text('# changed after selection\n')
                with self.assertRaisesRegex(api.RequestError, 'differs from its selected deployment'):
                    api.execute_run(config, run, 'ARTIFICIAL_CHANGED', schedule_resume=selection)
                popen.assert_called_once()


subprocess_run=subprocess.run
if __name__=='__main__':unittest.main()
