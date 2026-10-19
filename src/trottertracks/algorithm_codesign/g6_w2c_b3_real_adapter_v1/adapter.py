"""Closed artificial factory. Production construction is absent."""
from copy import deepcopy
from pathlib import Path
import os
from . import codec as C, bits as Bits, fixture as X, result as R, contract as K
from ..g6_w2c_b3_api_binding_v1 import ports as PORT
from .. import g6_w2c_b3_parallel_driver as D


def bits_factory(arm,axis):
    return Bits.ArtificialBits(arm,axis)


class ArtificialFactory:
    artificial=True
    def __init__(self,runtime_manifest,context):
        self.runtime_manifest=runtime_manifest
        self.context=deepcopy(context);self.seal=K.digest(self.context)

    def __call__(self,task,settings):
        PORT.install_guard()
        C.need(type(self) is ArtificialFactory and self.artificial is True,'EXACT_ARTIFICIAL_FACTORY')
        C.need(self.seal==K.digest(self.context),'WORKER_CONTEXT_MUTATION')
        root=Path(self.runtime_manifest['runtime_root'])
        C.need(Path(__file__).resolve()==root/K.PKG/'adapter.py','ADAPTER_LOCATION')
        for path,h in self.runtime_manifest['source_pins'].items():
            C.need(C.digest_bytes((root/path).read_bytes())==h,'WORKER_SOURCE:'+path)
        C.need(self.context['runtime_binding']==K.digest(self.runtime_manifest),'WORKER_RUNTIME')
        C.need(self.context['artificial_only'] is True and self.context['endpoint_binding']['real_dispatch'] is False,
               'ARTIFICIAL_FACTORY_ONLY')
        C.need(K.encoded(task)==K.encoded(next(t for t in self.context['tasks'] if t['key']==task['key'])),
               'WORKER_TASK')
        return Endpoint(task,settings,self.context)


class Endpoint:
    def __init__(self,task,settings,context):
        self.task,self.settings,self.context=deepcopy(task),settings,deepcopy(context)
        self.stopped=False;self.last=None;self.adapted=None

    def shared_identity(self):
        return D.identity(dict(schema='ARTIFICIAL_ADAPTER_V1',task=self.task,
            endpoint=self.context['endpoint_binding'],contract=self.context['candidate_binding'],
            manifest=self.context['manifest_binding']))

    def requested(self,stage,tid=None):
        return any(x['mode']==self.task['condition'] and x.get('pair',0)==self.task['pair_index'] and
            x.get('phase',0)==self.task['phase_index']%3 and x['stage']==stage and
            (tid is None or tid.endswith('/X/'+str(x.get('trial',0)))) for x in self.context['port_faults'])

    def fault(self,stage,tid=None):
        if self.requested(stage,tid):
            if stage=='worker_exit':os._exit(88)
            raise R.AdapterFailure('ARTIFICIAL_TECHNICAL_FAILURE','ARTIFICIAL_'+stage.upper(),stage,
                                   input_id='ARTIFICIAL_INPUT_V1',details={'trial_id':tid})

    def sample(self,bits,trial_id):
        C.need(not self.stopped,'ENDPOINT_TERMINAL_STOP')
        try:
            self.fault('worker_exit',trial_id);self.fault('generator',trial_id)
            C.need(type(bits) is Bits.ArtificialBits,'EXACT_ARTIFICIAL_BITS')
            phase,pa,arm,layer,axis,index=trial_id.split('/')
            C.need((phase,pa,arm,layer)==tuple(self.task[k] for k in ('phase','pass_name','arm','layer')),
                   'TRIAL_TASK')
            ident=PORT.TrialIdentity(self.context['run_id'],self.task['condition'],arm,layer,axis,phase,pa,int(index))
            ident.global_id()
            record,consumption=X.build(bits,trial_id=trial_id,arm=arm)
            if self.requested('codec',trial_id):record=C.Document(record.document,'0'*64)
            if self.requested('evidence',trial_id):
                data=record.read();data['source_id']='MUTATED';record=C.Document.create(data)
            if self.requested('bits',trial_id):consumption['draws'][-1]['value_hex']='0x1'
            Bits.validate_trace(consumption,arm=arm,axis=axis)
            self.adapted=X.adapt(record,trial_id=trial_id,arm=arm,layer=layer)
            self.fault('audit',trial_id)
            a=self.adapted
            C.need(a['driver_kind'] in ('ZERO_R','ZERO_C','EVENT'),'IDENTITY_NOT_DRIVER_ZERO')
            dispatch=dict(schema='b3_ARTIFICIAL_adapter_dispatch_v1',artificial_only=True,
                endpoint_binding=K.digest(self.context['endpoint_binding']),candidate_binding=self.context['candidate_binding'],
                manifest_binding=self.context['manifest_binding'],runtime_binding=self.context['runtime_binding'],
                factory=self.context['endpoint_binding']['adapter'],run_id=self.context['run_id'],
                global_trial_id=ident.global_id(),global_offset=self.task['offset']+(0 if axis=='X' else self.task['n'])+int(index),
                task_key=self.task['key'],worker_pid=os.getpid(),
                event_digest=D.identity(dict(kind=a['driver_kind'],event=a['event'])),
                adapted_digest=C.fingerprint(a),consumption_digest=C.fingerprint(consumption))
            evidence=dict(artificial=True,origin='SYNTHETIC_DOCUMENTS_NOT_REAL_G7_F3',
                          consumption=consumption,adapted=a,dispatch=dispatch)
            self.last=D.Outcome(a['driver_kind'],a['event'],evidence);self.last.validate()
            return self.last
        except BaseException as exc:
            self.stopped=True
            if isinstance(exc,R.AdapterFailure):raise
            raise R.AdapterFailure('ADAPTER_VALIDATION_FAILURE',str(exc),'sample.validation',
                                   details={'exception_type':type(exc).__name__,'trial_id':trial_id}) from exc

    def angle_for(self,result):
        C.need(not self.stopped,'ENDPOINT_TERMINAL_STOP')
        try:
            C.need(result is self.last and self.task['layer']=='B','ANGLE_RESULT_IDENTITY')
            self.fault('angle');self.fault('requery')
            query,angle=X.synthetic_angle(self.adapted)
            return R.angle_binding(self.adapted,requery_input=query,requery_event=result.event,
                                   angle_record=angle,synthetic=True)
        except BaseException as exc:
            self.stopped=True
            if isinstance(exc,R.AdapterFailure):raise
            raise R.AdapterFailure('ANGLE_VALIDATION_FAILURE',str(exc),'angle.requery') from exc
