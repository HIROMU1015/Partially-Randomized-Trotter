"""Task-local JSON RPC plumbing for the fixed integrated execution.

Only the parent launches science. Import, transport, and synthetic tests do not
load physical inputs. Existing numerical APIs own all numerical definitions.
"""
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import threading
import traceback


def _path(root, value):
    path = Path(value)
    return path if path.is_absolute() else Path(root)/path


def _relative(root, path):
    return str(Path(path).absolute().relative_to(Path(root).absolute()))


def _write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


class RpcProcess:
    """One persistent child with synchronous, serialized requests.

    Parent CLI must delegate --worker/--worker-config to worker_main().
    Config/log files are exclusive creations; no worker retry/resume is added.
    """
    def __init__(self, root, python, worker_mode, out, affinity=None, *,
                 config=None, script_path=None):
        self.root = Path(root).absolute()
        self.out = _path(self.root, out).absolute()
        self.out.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.closed = False
        payload = dict(config or {})
        payload.update(out=_relative(self.root, self.out),
                       affinity=None if affinity is None else sorted(affinity))
        self.config_path = self.out/'rpc_config.json'
        self.log_path = self.out/'rpc_worker.log'
        _write_json(self.config_path, payload)
        self.log = self.log_path.open('x', encoding='utf-8')
        script = (self.root/'scripts/run_research_c_integrated_execution.py'
                  if script_path is None else Path(script_path))
        command = [str(python), '-B', str(script), '--root', str(self.root),
                   '--worker', worker_mode, '--worker-config', str(self.config_path)]
        try:
            self.process = subprocess.Popen(command, cwd=self.root, text=True,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                bufsize=1)
        except BaseException:
            self.log.close()
            raise

    def request(self, operation, **values):
        if operation == 'close':
            raise ValueError('USE_RPC_CLOSE_METHOD')
        with self.lock:
            if self.closed:
                raise RuntimeError('RPC_CLOSED')
            try:
                self.process.stdin.write(json.dumps(dict(operation=operation, **values),
                    ensure_ascii=False, allow_nan=False)+'\n')
                self.process.stdin.flush()
                line = self.process.stdout.readline()
            except (BrokenPipeError, OSError) as error:
                raise RuntimeError('RPC_WORKER_PIPE:'+str(self.log_path)) from error
            if not line:
                raise RuntimeError('RPC_WORKER_EXIT:'+str(self.log_path))
            response = json.loads(line)
            if response.get('ok') is not True:
                raise RuntimeError('RPC_WORKER_FAILURE:'+response.get('error', 'unknown')+
                                   '; log='+str(self.log_path))
            return response['value']

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            try:
                if self.process.poll() is None:
                    self.process.stdin.write('{"operation":"close"}\n')
                    self.process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            finally:
                self.process.stdin.close()
            code = self.process.wait()
            self.process.stdout.close()
            self.log.close()
            if code:
                raise RuntimeError('RPC_WORKER_NONZERO_EXIT:'+str(code)+'; log='+str(self.log_path))

    def __enter__(self):
        return self

    def __exit__(self, kind, value, tb):
        try:
            self.close()
        except BaseException:
            if kind is None:
                raise


class PhysicalWorker:
    def __init__(self, root, out, *, evaluation_factory=None, io_module=None,
                 ground_function=None, stage1_module=None):
        self.root, self.out = Path(root).absolute(), Path(out).absolute()
        self.out.mkdir(parents=True, exist_ok=True)
        self.evaluation_factory, self.io = evaluation_factory, io_module
        self.ground_function, self.stage1 = ground_function, stage1_module
        self.evaluation = None
        self.sequence = 0
        self.ground_done = False

    def _dependencies(self):
        if self.io is None:
            from . import c1_stage1_execution
            self.io = c1_stage1_execution
        if self.stage1 is None:
            from . import c1_stage1
            self.stage1 = c1_stage1
        if self.evaluation_factory is None:
            from .c_pr_integrated_evaluation import Evaluation
            self.evaluation_factory = Evaluation

    def request(self, operation, **request):
        self._dependencies()
        if operation == 'initialize':
            if self.evaluation is not None:
                raise ValueError('PHYSICAL_ALREADY_INITIALIZED')
            system = request['system']
            if (system.get('norb'), system.get('rank'), system.get('ld')) != (4, 8, 2):
                raise ValueError('FIXED_H4_CDF8_LD2_REQUIRED')
            input_rel = request['input_rel']
            digest = self.io.sha(_path(self.root, input_rel))
            if digest != system['input_sha256']:
                raise ValueError('PHYSICAL_INPUT_IDENTITY')
            contract=dict(input_paths=dict(input=input_rel), DF_budget_Ha=.00019875)
            if request.get('reuse_common_pickle'):
                self.stage1.legacy_df_module()
                with _path(self.root,request['reuse_common_pickle']).open('rb') as stream:
                    saved=pickle.load(stream)
                if saved['input_identity']!=digest:raise ValueError('SAVED_CONTEXT_INPUT_IDENTITY')
                self.evaluation=self.evaluation_factory.from_context(self.root,contract,self.out,saved)
            else:
                self.evaluation = self.evaluation_factory(self.root,contract,self.out)
            ev = self.evaluation
            common = dict(p=ev.packet, ctx=ev.ctx, physical_reference=ev.physical,
                input_identity=digest,
                basis=self.stage1.bit_reverse_indices(8)[ev.ctx.sector.basis_indices])
            path = self.out/'common_context.pkl'
            with path.open('xb') as stream:
                pickle.dump(common, stream, protocol=5)
            self.input_identity = digest
            return dict(status='READY', input_sha256=digest,
                common_pickle=_relative(self.root, path),
                physical_reference=_relative(self.root, self.out/'physical_reference.npz'))
        if self.evaluation is None:
            raise ValueError('PHYSICAL_INITIALIZE_REQUIRED')
        ev = self.evaluation
        if operation == 'assess':
            self.sequence += 1
            requested = request['label']
            label = f'{self.sequence:07d}_'+re.sub('[^A-Za-z0-9_]', '_', str(requested))
            path = _path(self.root, request['path']).absolute()
            row = ev.assess(label, path)
            try:
                values = ev.cache[label]['d']
                row = dict(row, input_sha256=self.input_identity,
                    candidate_path=_relative(self.root, path),
                    reference_path=_relative(self.root, ev.out/'physical'/(label+'.npz')),
                    evaluated_array_identity={key:self.io.array_record(values[key]) for key in ('J','U')},
                    requested_label=requested, physical_label=label)
                self.io.save(ev.out/'physical'/(label+'_binding.json'), row)
                return row
            finally:
                # Reference arrays are preserved in the existing NPZ export.
                # Keep no per-trial matrix history in a long-lived process.
                ev.cache.pop(label, None)
        if operation == 'ground':
            if self.ground_done:
                raise ValueError('PHYSICAL_GROUND_ALREADY_PRODUCED')
            self.ground_done = True
            out = _path(self.root, request['out'])
            out.mkdir(parents=True, exist_ok=False)
            if self.ground_function is None:
                from .c_near_practical_r32 import ground
                self.ground_function = ground
            _, row = self.ground_function(ev.physical, out, 'physical')
            return dict(row, ground_path=_relative(self.root, out/'ground_physical.npz'))
        raise ValueError('PHYSICAL_OPERATION:'+operation)


def compact_final(final):
    """Only metadata needed by the existing final comparison crosses JSON."""
    return dict(report=final['report'], rows=final['rows'], bindings=final['bindings'],
                result=dict(calibration=final['result']['calibration']),
                candidate_sha256=final['candidate_sha256'],
                input_sha256=final['input_sha256'], r=final['r'])


class EvaluationWorker:
    def __init__(self, root, out, *, backend=None, legacy_register=None):
        self.root, self.out = Path(root).absolute(), Path(out).absolute()
        self.backend, self.legacy_register = backend, legacy_register
        self.common, self.runtime, self.states = None, None, {}

    def request(self, operation, **request):
        if operation == 'initialize':
            if self.common is not None:
                raise ValueError('EVALUATION_ALREADY_INITIALIZED')
            if self.legacy_register is None:
                from .c1_stage1 import legacy_df_module
                self.legacy_register = legacy_df_module
            self.legacy_register()
            with _path(self.root, request['common_pickle']).open('rb') as stream:
                common = pickle.load(stream)
            if not {'p','ctx','input_identity','basis','target','shells'} <= common.keys():
                raise ValueError('QUALIFIED_COMMON_PACKET_REQUIRED')
            self.common, self.runtime = common, request['runtime']
            if self.backend is None:
                from . import c_integrated_downstream_execution
                self.backend = c_integrated_downstream_execution
            return dict(status='READY', input_sha256=common['input_identity'])
        if self.common is None:
            raise ValueError('EVALUATION_INITIALIZE_REQUIRED')
        if operation == 'selection':
            task = request['task']
            key = request.get('state_key', task['id'])
            if key in self.states:
                raise ValueError('SELECTION_STATE_ALREADY_PRODUCED')
            report, state = self.backend.candidate_selection(self.root, self.runtime,
                _path(self.root, request['candidate_path']),
                _path(self.root, request['reference_path']), request['physical_record'],
                self.common, task, _path(self.root, request['out']))
            self.states[key] = state
            return dict(state_key=key, report=report)
        if operation == 'release':
            key = request['state_key']
            if key not in self.states:
                raise ValueError('UNKNOWN_RETAINED_STATE')
            del self.states[key]
            return dict(status='RELEASED', state_key=key)
        if operation == 'compare':
            from .c1_stage3_integrated import Calls
            out = _path(self.root, request['out'])
            out.mkdir(parents=True, exist_ok=False)
            call = Calls(out, dict(main_comparison=1), 1)
            return self.backend.compare_final(request.get('pr'), request.get('entry'),
                out/'result.json', call, primary=request['primary'])
        key = request['state_key']
        state = self.states[key]
        if operation == 'D':
            value = self.backend.candidate_d(self.root, self.runtime, state,
                _path(self.root, request['out']))
            return dict(state_key=key, result=value)
        if operation == 'attach_D':
            value = request['D_result']
            if (state['D_executed'] or request['candidate_sha256'] != state['candidate_sha256']
                    or request['input_sha256'] != state['input_sha256']
                    or value['candidate'] != state['candidate_sha256']):
                raise ValueError('SAME_CANDIDATE_INPUT_D_BINDING_REQUIRED')
            state.update(D=value, D_executed=True,
                         D_output=str(_path(self.root, request['D_output']).absolute()))
            return dict(status='SAME_CANDIDATE_D_ATTACHED', state_key=key, scientific_calls=0)
        if operation == 'final':
            report, final = self.backend.final_pair(self.root, self.runtime, state,
                request['r'], _path(self.root, request['out']))
            return dict(state_key=key, report=report, final=compact_final(final))
        raise ValueError('EVALUATION_OPERATION:'+operation)


def worker_loop(handler, *, input_stream=None, output_stream=None, error_stream=None):
    """Protocol failure stops the child; no implicit replay of a failed request."""
    incoming = sys.stdin if input_stream is None else input_stream
    outgoing = sys.stdout if output_stream is None else output_stream
    errors = sys.stderr if error_stream is None else error_stream
    with redirect_stdout(errors):
        for line in incoming:
            try:
                request = json.loads(line)
                operation = request.pop('operation')
                if operation == 'close':
                    return
                value = handler.request(operation, **request)
                response = dict(ok=True, value=value)
                encoded = json.dumps(response, ensure_ascii=False, allow_nan=False)
            except BaseException as error:
                traceback.print_exc(file=errors)
                outgoing.write(json.dumps(dict(ok=False, error=repr(error)), allow_nan=False)+'\n')
                outgoing.flush()
                raise
            outgoing.write(encoded+'\n')
            outgoing.flush()


def worker_main(root, worker_mode, config):
    """CLI delegation target; config is a decoded dict or config JSON path."""
    if not isinstance(config, dict):
        config = json.loads(Path(config).read_text())
    if config.get('affinity') is not None:
        os.sched_setaffinity(0, config['affinity'])
    out = _path(root, config['out'])
    if worker_mode == 'physical':
        worker = PhysicalWorker(root, out)
    elif worker_mode in ('evaluation', 'eval'):
        worker = EvaluationWorker(root, out)
    else:
        raise ValueError('REGISTERED_RPC_WORKER_MODE')
    # Preserve the protocol pipe before redirecting both Python and native
    # stdout to the child's log. C/Fortran output must not corrupt JSON lines.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), 'w', buffering=1)
    try:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        worker_loop(worker, output_stream=protocol)
    finally:
        protocol.close()
