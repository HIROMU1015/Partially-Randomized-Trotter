"""Invented packets and mocked numerical handlers; scientific calls are zero."""
from io import StringIO
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import c_integrated_execution_rpc as m


def test_worker_protocol_logs_are_separate_and_first_error_stops():
    class Handler:
        def request(self, operation, **request):
            print('ARTIFICIAL handler chatter')
            if operation == 'fail':
                raise ValueError('ARTIFICIAL_FAILURE')
            return request
    output, errors = StringIO(), StringIO()
    m.worker_loop(Handler(), input_stream=StringIO(
        '{"operation":"echo","value":3}\n{"operation":"close"}\n'),
        output_stream=output, error_stream=errors)
    assert json.loads(output.getvalue()) == dict(ok=True, value=dict(value=3))
    assert 'handler chatter' in errors.getvalue()
    output = StringIO()
    with pytest.raises(ValueError, match='ARTIFICIAL_FAILURE'):
        m.worker_loop(Handler(), input_stream=StringIO(
            '{"operation":"fail"}\n{"operation":"echo"}\n'),
            output_stream=output, error_stream=errors)
    assert len(output.getvalue().splitlines()) == 1
    assert json.loads(output.getvalue())['ok'] is False


def test_rpc_process_persistent_transport_and_exclusive_config(tmp_path):
    script = tmp_path/'artificial_worker.py'
    script.write_text("import sys\nsys.path.insert(0,"+repr(str(ROOT/'src'))+")\n"+"""import argparse,json,sys,os
from trottertracks import c_integrated_execution_rpc as m
p=argparse.ArgumentParser()
p.add_argument('--root');p.add_argument('--worker');p.add_argument('--worker-config')
a=p.parse_args()
class Handler:
 def __init__(self):self.count=0
 def request(self,operation,**r):
  self.count+=1
  print('ARTIFICIAL_PYTHON_STDOUT')
  os.write(1,b'ARTIFICIAL_NATIVE_STDOUT\\n')
  return dict(count=self.count,value=r.get('value'))
m.PhysicalWorker=lambda root,out:Handler()
m.worker_main(a.root,a.worker,a.worker_config)
""")
    worker = m.RpcProcess(tmp_path, sys.executable, 'physical', 'worker', script_path=script)
    try:
        assert worker.request('echo', value=2) == dict(count=1, value=2)
        assert worker.request('echo', value=3) == dict(count=2, value=3)
        with pytest.raises(ValueError, match='CLOSE_METHOD'):
            worker.request('close')
    finally:
        worker.close()
    worker.close()
    log = worker.log_path.read_text()
    assert 'ARTIFICIAL_PYTHON_STDOUT' in log and 'ARTIFICIAL_NATIVE_STDOUT' in log
    with pytest.raises(RuntimeError, match='RPC_CLOSED'):
        worker.request('echo')
    with pytest.raises(FileExistsError):
        m.RpcProcess(tmp_path, sys.executable, 'physical', 'worker', script_path=script)


def test_physical_delegation_pickle_array_bindings_and_no_matrix_history(tmp_path):
    from trottertracks import c1_stage1_execution as io
    source = tmp_path/'invented_packet.npz'
    source.write_bytes(b'ARTIFICIAL_ONLY_NOT_A_SAVED_SCIENTIFIC_ARRAY')
    candidate = tmp_path/'invented_candidate.npz'
    candidate.write_bytes(b'ARTIFICIAL_CANDIDATE')
    values = dict(J=np.ones((1,1,1)), U=np.ones((1,1,1)))
    class FakeEvaluation:
        def __init__(self, root, config, out):
            self.out = out
            self.ctx = NS(sector=NS(basis_indices=np.array([0,1])))
            self.packet = dict(invented=True)
            self.physical = np.eye(2)
            self.cache = {}
            (out/'physical_reference.npz').write_bytes(b'ARTIFICIAL_REFERENCE')
        def assess(self, label, path):
            self.cache[label] = dict(d=values)
            (self.out/'physical').mkdir(exist_ok=True)
            return dict(candidate_sha256=io.sha(path), budget_PASS=True)
    def ground(H, out, label):
        assert label == 'physical'
        (out/'ground_physical.npz').write_bytes(b'ARTIFICIAL_GROUND')
        return None, dict(status='PASS')
    factory = Mock(side_effect=FakeEvaluation)
    g = Mock(side_effect=ground)
    worker = m.PhysicalWorker(tmp_path, tmp_path/'physical_rpc', evaluation_factory=factory,
        io_module=io, ground_function=g, stage1_module=NS(bit_reverse_indices=lambda n:np.arange(256)))
    ready = worker.request('initialize', input_rel=source.name,
        system=dict(norb=4,rank=8,ld=2,input_sha256=io.sha(source)))
    with (tmp_path/ready['common_pickle']).open('rb') as stream:
        common = pickle.load(stream)
    assert common['input_identity'] == io.sha(source)
    np.testing.assert_array_equal(common['basis'], [0,1])
    first = worker.request('assess', label='trial:one', path=candidate.name)
    second = worker.request('assess', label='trial:one', path=candidate.name)
    assert first['reference_path'] != second['reference_path']
    assert ':' not in first['physical_label']
    assert first['evaluated_array_identity']['J'] == io.array_record(values['J'])
    assert first['candidate_path'] == candidate.name and first['input_sha256'] == io.sha(source)
    assert not worker.evaluation.cache
    assert worker.request('ground', out='ground')['status'] == 'PASS'
    assert g.call_count == factory.call_count == 1
    with pytest.raises(ValueError, match='GROUND_ALREADY'):
        worker.request('ground', out='ground2')


def test_evaluation_retains_opaque_state_and_compacts_final_for_parent_comparison(tmp_path):
    common = dict(p={},ctx='INVENTED_CTX',input_identity='input',basis=[1],
                  target='INVENTED_TARGET',shells={})
    path = tmp_path/'common.pkl'
    with path.open('wb') as stream:
        pickle.dump(common, stream)
    state = dict(candidate_sha256='candidate',input_sha256='input',D_executed=False)
    def selection(*args):
        return dict(status='SELECTION_COMPLETE'), state.copy()
    def deterministic(root,runtime,s,out):
        s.update(D_executed=True,D_output=str(out))
        return dict(status='D_COMPLETE', candidate=s['candidate_sha256'])
    def final(root,runtime,s,r,out):
        assert s['D_executed']
        value = dict(report=dict(status='FINAL_COMPLETE'), rows=[dict(rz_count=1)],
            bindings=dict(ideal_exact_native=dict(W=1)),result=dict(calibration={'mean':1}),
            packet={'opaque':object()},candidate_sha256='candidate',input_sha256='input',r=r)
        return value['report'],value
    register = Mock()
    backend = NS(candidate_selection=Mock(side_effect=selection),
        candidate_d=Mock(side_effect=deterministic), final_pair=Mock(side_effect=final))
    worker = m.EvaluationWorker(tmp_path,tmp_path/'eval',backend=backend,legacy_register=register)
    worker.request('initialize', common_pickle=path.name, runtime={'frozen':True})
    register.assert_called_once_with()
    def select(key):
        return worker.request('selection', task=dict(id=key),candidate_path='candidate',
            reference_path='reference',physical_record={},out=key)
    assert select('cell1')['state_key'] == 'cell1'
    d = worker.request('D',state_key='cell1',out='D')
    select('cell2')
    worker.request('attach_D',state_key='cell2',D_result=d['result'],D_output='D',
                   candidate_sha256='candidate',input_sha256='input')
    result = worker.request('final',state_key='cell2',r=32,out='final')
    assert set(result['final']['result']) == {'calibration'}
    assert 'packet' not in result['final']
    json.dumps(result,allow_nan=False)
    assert backend.candidate_d.call_count == 1
    with pytest.raises(ValueError, match='ALREADY_PRODUCED'):
        select('cell1')
    worker.request('release',state_key='cell1')
    assert 'cell1' not in worker.states
    select('cell3')
    with pytest.raises(ValueError,match='D_BINDING'):
        worker.request('attach_D',state_key='cell3',D_result=d['result'],D_output='D',
                       candidate_sha256='wrong',input_sha256='input')
