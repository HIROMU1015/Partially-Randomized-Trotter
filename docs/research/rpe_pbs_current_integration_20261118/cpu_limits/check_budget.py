from concurrent.futures import ThreadPoolExecutor
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile

path = Path(__file__).with_name('pbs_cpu_gateway.py')
spec = spec_from_file_location('policy', path)
policy = module_from_spec(spec)
spec.loader.exec_module(policy)
with tempfile.TemporaryDirectory() as temp:
    runs = Path(temp)
    settings = dict(runs=temp, max_cpus=100)
    jobs = {}
    created = []
    def read(path):
        return json.loads(path.read_text()) if path.exists() else None
    def ensure(root, payload):
        assert payload['action'] == 'ensure'
        allocation = payload['spec']
        path = root / 'allocations' / allocation['token'] / 'SPEC.json'
        if path.exists():
            return dict(existing=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(allocation))
        created.append(path)
        return dict(existing=False)
    gateway = SimpleNamespace(qstat=lambda: jobs, safe_id=lambda x:x, read=read, handle=ensure)
    def submit(run, token, cpus):
        return policy.handle(gateway, settings, runs/run,
            dict(action='ensure', spec=dict(token=token, resources=dict(ncpus=cpus))))
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(submit, f'run{i}', str(i)*32, 96) for i in range(2)]
        answers = [f.result() for f in futures]
    assert sum(result.get('status') == 'CPU_BUDGET_WAIT' for result in answers) == 1
    assert len(created) == 1
    path = created[0]
    token = path.parent.name
    owner = path.parents[2].name
    assert submit(owner, token, 96) == dict(existing=True)
    assert submit('extra', 'a'*32, 4) == dict(existing=False)
    assert submit('extra', 'b'*32, 1)['status'] == 'CPU_BUDGET_WAIT'
    jobs['owned1'] = dict(Job_Name='RPE'+token[:12], job_state='F',
                            Resource_List=dict(ncpus=96))
    assert submit('next', 'c'*32, 96) == dict(existing=False)
    jobs['other'] = dict(Job_Name='UNRELATED', job_state='R',
                         Resource_List=dict(ncpus=200))
    assert policy.reserved_cpus(gateway, runs, runs/'next') == 100
    try:
        submit('too-much', 'd'*32, 101)
    except ValueError:
        pass
    else:
        raise AssertionError('single request over cap allowed')
    print('PASS: concurrent 96 CPU requests, exact 96+4, waiting before qsub, replay, terminal release, unrelated jobs, oversize rejection')
