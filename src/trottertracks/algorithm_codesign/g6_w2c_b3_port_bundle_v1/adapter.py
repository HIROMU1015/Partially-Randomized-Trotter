"""Only the fixed artificial API Port is accepted; no real loader exists."""
from pathlib import Path
from copy import deepcopy
import os
from ..g6_w2c_b3_api_binding_v1 import ports as PORT
from ..g6_w2c_b3_api_binding_v1.static import need,sha,digest,encoded
from .. import g6_w2c_b3_parallel_driver as D
from . import contract as K

def bits_factory(arm,axis):
    PORT.install_guard()
    return PORT.ArtificialBits(arm,axis)

class PortFailure(RuntimeError):
    def __init__(self,stage,reason):
        super().__init__(reason);self.stage=stage;self.reason=reason
    def to_dict(self):
        return dict(schema='b3_ARTIFICIAL_port_failure_v1',category='ARTIFICIAL_PORT_TECHNICAL_FAILURE',
            stage=self.stage,reason=self.reason,mandatory_STOP=True,is_random_zero=False)

class ArtificialFactory:
    artificial=True
    def __init__(self,runtime_manifest,context):
        self.runtime_manifest=runtime_manifest
        self.context=deepcopy(context)
        self.seal=digest(self.context)

    def __call__(self,task,settings):
        PORT.install_guard()
        need(type(self)is ArtificialFactory and self.artificial is True,'EXACT_ARTIFICIAL_FACTORY')
        need(self.seal==digest(self.context),'WORKER_FACTORY_CONTEXT_MUTATION')
        root=Path(self.runtime_manifest['runtime_root'])
        need(Path(__file__).resolve()==root/K.PKG/'adapter.py','ADAPTER_LOADED_LOCATION')
        need(Path(PORT.__file__).resolve()==root/K.PORT/'ports.py','PORT_LOADED_LOCATION')
        for path,h in self.runtime_manifest['source_pins'].items():
            need(sha((root/path).read_bytes())==h,'WORKER_SOURCE_BYTES:'+path)
        need(self.context['runtime_binding']==digest(self.runtime_manifest),'WORKER_RUNTIME_BINDING')
        need(self.context['artificial_only']is True and self.context['endpoint_binding']['real_dispatch']is False,'WORKER_ARTIFICIAL_ONLY')
        need(encoded(task)==encoded(next(t for t in self.context['tasks']if t['key']==task['key'])),'WORKER_TASK_BINDING')
        return Endpoint(task,settings,self.context)

class Endpoint:
    def __init__(self,task,settings,context):
        self.task,self.context=deepcopy(task),deepcopy(context)
        self.settings=settings
        self.port=PORT.ArtificialEndpoint(task,run=context['run_id'])
        self.last=None

    def shared_identity(self):
        return D.identity(dict(port=self.port.shared_identity(),binding=self.context['endpoint_binding'],
            contract=self.context['candidate_binding'],manifest=self.context['manifest_binding'],task=self.task))

    def fault(self,stage,trial_id=None):
        selected=next((x for x in self.context['port_faults']if x['mode']==self.task['condition']
            and x.get('pair',0)==self.task['pair_index']and x.get('phase',0)==self.task['phase_index']%3
            and x['stage']==stage and(trial_id is None or trial_id.endswith('/X/'+str(x.get('trial',0))))),None)
        if selected and stage=='worker_exit':os._exit(88)
        if selected:raise PortFailure(stage,'ARTIFICIAL_PORT_'+stage.upper()+'_FAILURE')
        return False

    def sample(self,bits,trial_id):
        try:return self._sample(bits,trial_id)
        except PortFailure:raise
        except BaseException as exc:
            raise PortFailure('sample.validation',type(exc).__name__+':'+str(exc))from exc

    def _sample(self,bits,trial_id):
        if self.legacy_fault('exit',trial_id):os._exit(88)
        if self.legacy_fault('sample',trial_id):raise PortFailure('generator','ARTIFICIAL_PORT_GENERATOR_FAILURE')
        self.fault('worker_exit',trial_id);self.fault('generator',trial_id)
        result=self.port.sample(bits,trial_id)
        self.fault('audit',trial_id)
        fields=trial_id.split('/');axis=fields[-2];i=int(fields[-1])
        if self.fault_requested('invalid_bits',trial_id):
            result.evidence['consumption']['draws'][-1]['width']=159
        ident=PORT.TrialIdentity(self.context['run_id'],self.task['condition'],self.task['arm'],self.task['layer'],axis,self.task['phase'],self.task['pass_name'],i)
        PORT.validate_result(result,ident,result.evidence['consumption'])
        if result.kind=='EVENT':
            result.event['artificial_port_variant']=self.context['variant']
        if self.fault_requested('invalid_result',trial_id):
            result.event={} if result.kind=='EVENT' else None
            result.kind='INVALID'
        result.validate()
        if self.legacy_fault('after_event',trial_id)and result.kind=='EVENT':result.event['phase_i_power']=99
        dispatch=dict(schema='b3_ARTIFICIAL_global_port_dispatch_v1',artificial_only=True,
            endpoint_binding=digest(self.context['endpoint_binding']),candidate_binding=self.context['candidate_binding'],
            manifest_binding=self.context['manifest_binding'],runtime_binding=self.context['runtime_binding'],
            factory=self.context['endpoint_binding']['adapter'],variant=self.context['variant'],
            run_id=self.context['run_id'],global_trial_id=ident.global_id(),
            global_offset=self.task['offset']+(0 if axis=='X'else self.task['n'])+i,
            task_key=self.task['key'],worker_pid=os.getpid(),
            port_event_digest=D.identity(dict(kind=result.kind,event=result.event)),
            consumption_digest=D.identity(result.evidence['consumption']))
        result.evidence['dispatch']=dispatch
        self.last=result
        return result

    def fault_requested(self,stage,tid):
        return any(x['mode']==self.task['condition']and x.get('pair',0)==self.task['pair_index']
            and x.get('phase',0)==self.task['phase_index']%3 and x['stage']==stage
            and tid.endswith('/X/'+str(x.get('trial',0)))for x in self.context['port_faults'])

    def angle_for(self,result):
        if self.legacy_fault('angle'):raise PortFailure('angle','ARTIFICIAL_PORT_ANGLE_FAILURE')
        self.fault('angle');self.fault('requery')
        return self.port.angle_for(result)

    def legacy_fault(self,stage,tid=None):
        return any(pair==self.task['pair_index']and phase==self.task['phase_index']and where==stage
            and(tid is None or tid.endswith('/X/'+str(trial)))for pair,phase,where,trial in self.settings.faults)
