"""Three owned case coordinators, six preparation workers, no compile dispatch.

The driver reserves global calls before each case fork. A child consumes only
its bound ticket and returns small, byte-verified QPY descriptors. Coordinators
share their pair's first CPU; they wait while the two numerical workers run.
"""
import math
from pathlib import Path
import h8_key_compile_contract as c
import h8_key_compile_parallel as pool


class CasePermit:
    """A fork-local, case-bound view of calls already reserved by the driver."""
    def __init__(self,parent,task,directory):
        self.parent=parent;self.config=parent.config;self.grant=parent.grant
        self.output=Path(directory);self.case_id=task['id'];self.calls={k:0 for k in c.CASE_PREPARATION_CAPS}
        case_ids=[case['id'] for case in c.binding()['cases']]
        c.require(self.case_id in case_ids,'REGISTERED_PREPARATION_CASE')
        i=case_ids.index(self.case_id)
        c.require(task['worker_cpus']==parent.grant['worker_cpus'][2*i:2*i+2],
                  'REGISTERED_CASE_CPU_PAIR')
        self.ticket=c.read(task['ticket_path']);self.ticket_sha256=task['ticket_sha256']
        expected=c.home(parent.output/('preparation_ticket_'+self.case_id+'.json'))
        c.require(Path(task['ticket_path'])==expected and c.sha(expected)==self.ticket_sha256 and
            self.ticket['case_id']==self.case_id and self.ticket['SOURCE']==self.config['SOURCE'] and
            self.ticket['config_sha256']==c.digest(self.config) and
            self.ticket['grant_sha256']==c.digest(self.grant) and
            self.ticket['worker_cpus']==task['worker_cpus'] and
            self.ticket['call_caps']==c.CASE_PREPARATION_CAPS,'PREPARATION_TICKET_BINDING')
        expected_calls=[(key,i) for key,count in c.CASE_PREPARATION_CAPS.items() for i in range(1,count+1)]
        c.require([(r['call'],r['case_ordinal']) for r in self.ticket['reservations']]==expected_calls,
                  'EXACT_CASE_RESERVATIONS')
        for row in self.ticket['reservations']:self._verify_reservation(row)

    def _verify_reservation(self,row):
        path=self.parent.output/('reservation_'+row['call']+'_'+str(row['ordinal'])+'.json')
        c.require(c.sha(path)==row['sha256'] and c.read(path)==dict(call=row['call'],ordinal=row['ordinal']),
                  'PARENT_CALL_RESERVATION_CHANGED')

    def require(self,scope):self.parent.require(scope)

    def take(self,key):
        c.require(key in self.calls and self.calls[key]<c.CASE_PREPARATION_CAPS[key],
                  'CASE_CALL_CAP:'+key)
        ordinal=self.calls[key]+1
        row=next(r for r in self.ticket['reservations'] if (r['call'],r['case_ordinal'])==(key,ordinal))
        self._verify_reservation(row)
        c.save(self.output/('preparation_use_'+key+'_'+str(ordinal)+'.json'),
               dict(case_id=self.case_id,SOURCE=self.config['SOURCE'],reservation=row))
        self.calls[key]=ordinal

    def complete(self):
        c.require(self.calls==c.CASE_PREPARATION_CAPS,'INCOMPLETE_CASE_PREPARATION_CALLS')
        return dict(case_id=self.case_id,SOURCE=self.config['SOURCE'],
                    preparation_call_counts=self.calls,ticket_sha256=self.ticket_sha256)


def prepare_cases(permit,build,*,map_owned=pool.map_owned):
    """Build all registered cases concurrently; any failure prevents all compile."""
    cases=c.binding()['cases'];cpus=permit.grant['worker_cpus']
    c.require(len(cases)==3 and len(cpus)==6 and len(set(cpus))==6,'THREE_CASES_SIX_PREPARATION_CPUS')
    by={case['id']:case for case in cases}
    tasks=[dict(id=case['id'],worker_cpus=cpus[2*i:2*i+2]) for i,case in enumerate(cases)]
    def reserve(task):
        rows=[]
        for key,count in c.CASE_PREPARATION_CAPS.items():
            for i in range(1,count+1):
                permit.take(key);ordinal=permit.calls[key]
                path=permit.output/('reservation_'+key+'_'+str(ordinal)+'.json')
                rows.append(dict(call=key,case_ordinal=i,ordinal=ordinal,sha256=c.sha(path)))
        ticket=dict(case_id=task['id'],SOURCE=permit.config['SOURCE'],
            config_sha256=c.digest(permit.config),grant_sha256=c.digest(permit.grant),
            worker_cpus=task['worker_cpus'],call_caps=c.CASE_PREPARATION_CAPS,reservations=rows)
        path=permit.output/('preparation_ticket_'+task['id']+'.json');c.save(path,ticket)
        task.update(ticket_path=str(path),ticket_sha256=c.sha(path))
    def callback(task,directory):
        child=CasePermit(permit,task,directory)
        result=build(by[task['id']],child,task['worker_cpus'])
        result.update(child.complete())
        return result
    receipt=map_owned(tasks,callback,cpus=cpus[::2],limits=dict(max_workers=3,cleanup_TERM_seconds=10.),
        directory=permit.output/'parallel_preparation',deadline=math.inf,
        reserve=reserve,font_seed=permit.config.get('font_seed'))
    c.require(receipt['status']=='PASS' and receipt['all_reaped'] and not receipt['cleanup_errors'] and
        [w['task_id'] for w in receipt['workers']]==[case['id'] for case in cases],
        'ALL_CASE_PREPARATION_NATIVE_CLEANUP')
    prepared={}
    for worker,task in zip(receipt['workers'],tasks):
        path=Path(worker['directory'])/'task_result.json';row=c.read(path)
        c.require(worker['exit_code']==0 and worker['reaped']is True and c.sha(path)==worker['result_sha256'] and
            row['case_id']==task['id'] and row['SOURCE']==permit.config['SOURCE'] and
            row['ticket_sha256']==task['ticket_sha256'] and
            row['preparation_call_counts']==c.CASE_PREPARATION_CAPS,'CASE_PREPARATION_NATIVE_BINDING')
        prepared[task['id']]=row
    c.save(permit.output/'preparation_complete.json',dict(status='PASS_THREE_CASES_PREPARED',
        SOURCE=permit.config['SOURCE'],case_ids=list(prepared),compile_calls=permit.calls['compile'],
        native_receipt_sha256=c.sha(permit.output/'parallel_preparation/parallel_receipt.json')))
    return prepared
