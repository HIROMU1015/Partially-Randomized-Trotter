"""Technical checks for the in-allocation search and chunked result import."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

from trottertracks import rpe_pbs
from trottertracks.rpe_pbs_offline import OfflineSearch, prepare_plan
from trottertracks.rpe_q3_worker import atomic, read

ROOT = Path(__file__).resolve().parents[1]


def test_offline_parts_advance_and_import_only_completed_results(tmp_path, monkeypatch):
    revision = 'a'*40
    spec = {'science_approved': True, 'methods':['PR_PAULI', 'PR_DF'],
            'round_id':'fixed', 'max_new_q3_per_method':{'PR_PAULI':'UNLIMITED', 'PR_DF':'UNLIMITED'},
            'search_overrides':{'requested_widths':{'PR_PAULI':[1,2]}}, 'executor':{}}
    run = tmp_path/'central'
    run.mkdir()
    for name, value in (('ROUND.json', spec), ('PROBLEM.json', {}), ('APPLIED_SCOPE.json', {'run_id':'search-test'})):
        atomic(run/name, value)
    settings = {'code_root':str(ROOT), 'interpreter':sys.executable,
                'source':revision, 'memory_bytes_per_task':2*rpe_pbs.GIB}
    atomic(run/'ROUND.json', dict(spec, max_new_q3_per_method={'PR_PAULI':1, 'PR_DF':'UNLIMITED'}))
    assert prepare_plan(run, settings) is None
    atomic(run/'ROUND.json', spec)
    plan = prepare_plan(run, settings)
    remote = tmp_path/'remote'
    allocation = remote/'allocations'/'token'
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    commands = []

    def launch(command, **_):
        commands.append(command)
        return SimpleNamespace(poll=lambda: 0)

    profile = {'dp_workers':2, 'pbs':{'library_threads':1},
               'slot_policy':{'resource_limits':{'cpu_count':8, 'memory_bytes':12*rpe_pbs.GIB}}}
    resources = {'cpu_count':8, 'maximum_slots':8, 'memory_bytes':12*rpe_pbs.GIB}
    search = OfflineSearch(plan, profile, allocation, resources, popen=launch)
    search.validated_entry = ROOT/'scripts/run_rpe_schedule_search.py'
    assert not search.advance(disconnected=False, runnable=True, batch=8)
    assert search.advance(disconnected=True, runnable=True, batch=8)
    part = search.part
    assert read(part/'ROUND.json')['search_overrides']['requested_widths'] == {'PR_PAULI':[1]}
    assert commands[0][commands[0].index('--batch')+1] == '4'
    atomic(part/'SEARCH_TERMINATION.json', {'reason':'ARTIFICIAL_COMPLETE'})
    job = part/'jobs'/'PR_PAULI'/'saved-job'
    job.mkdir(parents=True)
    atomic(job/'REQUEST.json', {'request_id':'saved-job', 'method':'PR_PAULI', 'optimizer_source':revision})
    atomic(job/'PROCESS_RETURN.json', {'returncode':0, 'signal':None})
    atomic(job/'OUTCOME.json', {'status':'ARTIFICIAL', 'value':10**7167})
    assert search.advance(disconnected=True, runnable=True, batch=8)
    assert read(search.part/'ROUND.json')['search_overrides']['requested_widths'] == {'PR_PAULI':[2]}
    assert search.revision == 2
    assert not search.advance(disconnected=False, runnable=True, batch=8)
    assert len(commands) == 2

    spec = importlib.util.spec_from_file_location('offline_gateway_test', ROOT/'scripts/rpe_pbs_gateway.py')
    gateway = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gateway)
    connection = SimpleNamespace(call=lambda value: gateway.handle(remote, value), close=lambda: None)
    manager = object.__new__(rpe_pbs.PbsManager)
    manager.offline_plan, manager.settings, manager.directory = plan, {}, run/'distributed'/'pbs'
    manager.executor = SimpleNamespace(computation_source=revision)
    manager.log = lambda *_args, **_kwargs: None
    with patch.object(rpe_pbs, 'Gateway', return_value=connection):
        manager._collect_offline_history('token')
        manager._collect_offline_history('token')
    recovered = manager.directory/'offline-history'/'jobs'/'PR_PAULI'/'saved-job'
    assert (recovered/'OUTCOME.json').read_bytes() == (job/'OUTCOME.json').read_bytes()
    coverage = read(manager.directory/'offline-history'/'coverage'/'token.json')
    assert coverage['parts'][part.name]['termination'] == {'reason':'ARTIFICIAL_COMPLETE'}
