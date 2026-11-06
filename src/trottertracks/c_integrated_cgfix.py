"""Fresh, separately permitted execution using 17 saved qualified trajectories.

No old task is resumed. One failed cell is a new trajectory; all downstream
cohorts use the new SOURCE namespace. Saved originals remain unchanged.
"""
from copy import deepcopy
import json
from pathlib import Path

from . import c_integrated_execution as execution
from . import c_integrated_server as server
from . import c1_stage1_execution as io

FAMILY='artifacts/research_c_integrated/2026-11-06-picklefix'
TECHNICAL=FAMILY+'/technical'
HANDOFF=TECHNICAL+'/saved_input_handoff.json'
CONTRACT=TECHNICAL+'/contract.json'
SOURCE=TECHNICAL+'/SOURCE.json'
PREVIOUS_CONTRACT='artifacts/research_c_integrated/2026-11-06/server_technical/contract.json'
PREVIOUS_SOURCE='artifacts/research_c_integrated/2026-11-06/server_technical/SOURCE.json'


def contract(root):
    base=io.read(Path(root)/PREVIOUS_CONTRACT);c=deepcopy(base)
    c.update(status='CGFIX_PREPARED_NOT_EXECUTED',technical_source=SOURCE,
        fresh_outputs=[FAMILY+'/development/run01',FAMILY+'/transfer/run01'])
    c['server_delta']['missing_post_stop_results']='17 successful trajectories are byte-bound saved inputs; exactly one failed trajectory needs new computation'
    c['server_delta']['source_changes']+='; capture actual solver-input gradient for initial CG jac=None; reuse saved original-H context and17 trajectories'
    c['maximum_calls']['components']['development_initialization']={'saved_context_reuse':1}
    c['maximum_calls']['components']['development_optimization']={
        k:v//18 for k,v in base['maximum_calls']['components']['development_optimization'].items()}
    c['maximum_calls']['stage_totals']['development_initialization']=1
    c['maximum_calls']['stage_totals']['development_optimization']=3213
    c['maximum_calls']['total']=sum(c['maximum_calls']['stage_totals'].values())
    assert c['maximum_calls']['total']==118188
    c['cgfix']=dict(previous_SOURCE=PREVIOUS_SOURCE,
        previous_SOURCE_sha256=server.sha(Path(root)/PREVIOUS_SOURCE),handoff=HANDOFF,
        handoff_sha256=server.sha(Path(root)/HANDOFF),
        completed_trajectories_reused=17,new_development_cells=['H4_100_C0_ENTRY_g03'],
        qualification_change='Only status3/nit0/unchanged initial point with missing raw jac may use the actual captured solver-input gradient; all independent checks and thresholds remain required',
        missing_raw_jac_marked_unavailable=True,raw_optimizer_failure_preserved=True,
        extra_VG_or_optimizer_calls=0,retry=0,resume=0,
        scientific_execution='Separate explicit permission for this new SOURCE, qualification route and fresh scope')
    c['bootstrap_predecessor']=dict(
        SOURCE='artifacts/research_c_integrated/2026-11-06-cgfix/technical/SOURCE.json',
        SOURCE_sha256='c5c48ba38b88d5c914c51f710113e65f806a900a67d7ade03fd76a29290d9608',
        STOP='artifacts/research_c_integrated/2026-11-06-cgfix/INTEGRATED_STOP.json',
        completed_scientific_stages=0,correction='register existing legacy package before unpickling context')
    c['technical_bootstrap_recovery']=dict(
        scope='Non-numerical import/path/serialization fixes before any new optimizer/VG/HVP/PF/RTE/native or target generation',
        permission_scope='The proposed permission includes such startup-only technical corrections and fresh replacement launch without another intermediate user confirmation',
        conditions='Preserve every prior SOURCE/raw/STOP; record corrected SOURCE and fresh output before replacement; never retry/resume an old run or replay completed scientific work; keep all scientific conditions, numerical engines, thresholds and sampling policies unchanged',
        scientific_failure='Not covered: numerical/correctness failures, scientific condition or qualification changes require separate permission')
    return c


def prepare(root):
    root=Path(root).resolve();c=contract(root)
    if any((root/p).exists() for p in c['fresh_outputs']):raise ValueError('FRESH_CGFIX_SCOPE_REQUIRED')
    server.write(root/CONTRACT,c)
    previous=io.read(root/PREVIOUS_SOURCE);handoff=io.read(root/HANDOFF)
    paths=set(previous['files'])|set(handoff['files'])|{PREVIOUS_SOURCE,HANDOFF,CONTRACT,
        c['bootstrap_predecessor']['SOURCE'],c['bootstrap_predecessor']['STOP'],
        'src/trottertracks/c_integrated_cgfix.py','scripts/prepare_research_c_integrated_cgfix.py',
        'tests/management/test_research_c_inner_missing_jac.py','tests/management/test_research_c_integrated_cgfix.py',
        'tests/management/test_research_c_saved_context_rpc.py'}
    files={p:server.sha(root/p) for p in sorted(paths)}
    changed={p:dict(before=h,after=files[p]) for p,h in previous['files'].items() if files[p]!=h}
    server.write(root/SOURCE,dict(schema='C_INTEGRATED_CGFIX_SOURCE_V1',files=files,
        previous_SOURCE_sha256=server.sha(root/PREVIOUS_SOURCE),changed_inherited_files=changed))
    return preflight(root)


def preflight(root):
    root=Path(root).resolve();c=io.read(root/CONTRACT);source=io.read(root/SOURCE)
    if c!=contract(root):raise ValueError('CGFIX_CONTRACT_DIFFERENCE')
    for p,digest in source['files'].items():
        value=(root/p).resolve()
        if not value.is_relative_to(root) or server.sha(value)!=digest:raise ValueError('CGFIX_SOURCE_OR_INPUT:'+p)
    saved=io.read(root/HANDOFF);stop=io.read(root/saved['original_global_STOP'])
    if not stop['failed'] or stop['retry'] or stop['resume']:raise ValueError('PRIOR_STOP_MUST_REMAIN')
    bootstrap=c['bootstrap_predecessor'];previous=io.read(root/bootstrap['STOP'])
    if (server.sha(root/bootstrap['SOURCE'])!=bootstrap['SOURCE_sha256']
            or not previous['failed'] or previous['completed_stage_units']!=0
            or previous['retry'] or previous['resume']):
        raise ValueError('PRESERVED_BOOTSTRAP_STOP_REQUIRED')
    if len(saved['completed'])!=17 or [x['job_id'] for x in saved['failed']]!=c['cgfix']['new_development_cells']:
        raise ValueError('FIXED_SAVED17_NEW1_SCOPE')
    for item in saved['completed']:
        record=io.read(root/item['result_path']);local=io.read((root/item['result_path']).parent/'STOP.json')
        if local['failed'] or record['retry'] or record['resume'] or set(record['checkpoints'])!={'16','64'}:
            raise ValueError('QUALIFIED_SAVED_TRAJECTORY_REQUIRED')
    if any((root/p).exists() for p in c['fresh_outputs']) or (root/FAMILY/'INTEGRATED_STOP.json').exists():
        raise ValueError('FRESH_CGFIX_NO_RETRY_RESUME')
    # The preserved compiler/fit environments belong to the previous job and
    # are only read here; no shared or existing package installation is changed.
    import subprocess
    environments={}
    for role in ('fit','evaluation'):
        pins=c['environment'][role+'_packages']
        code='import importlib.metadata as m,json;print(json.dumps({k:m.version(k) for k in '+repr(pins)+'}))'
        actual=json.loads(subprocess.check_output([c['environment'][role+'_python'],'-B','-c',code],text=True))
        if actual!=pins:raise ValueError('CGFIX_ENVIRONMENT:'+role)
        environments[role]=actual
    expected=c['environment']['ffsim_sources']
    code=('import hashlib,importlib.metadata as m,json;from pathlib import Path;d=m.distribution("ffsim");'
        'print(json.dumps({k:hashlib.sha256(Path(d.locate_file("ffsim/"+k)).read_bytes()).hexdigest() for k in '+repr(expected)+'}))')
    actual=json.loads(subprocess.check_output([c['environment']['fit_python'],'-B','-c',code],text=True))
    if actual!=expected:raise ValueError('CGFIX_FFSIM_SOURCE')
    code=('import sys;sys.path.insert(0,'+repr(str(root/'src'))+');'
        'from trottertracks.c_j_u_precision import require_extended;require_extended();print("EXTENDED_PASS")')
    if subprocess.check_output([c['environment']['fit_python'],'-B','-c',code],text=True).strip()!='EXTENDED_PASS':
        raise ValueError('CGFIX_EXTENDED_ACCUMULATION')
    return dict(status='CGFIX_PASSIVE_PREFLIGHT_PASS',scientific_execution=0,
        SOURCE_sha256=server.sha(root/SOURCE),contract_sha256=server.sha(root/CONTRACT),
        saved_trajectories=17,new_development_cells=c['cgfix']['new_development_cells'],
        maximum_new_logical_units=118188,environments=environments,resources=server.resources(root))


class Run(execution.Run):
    def __init__(self,root,source):
        super().__init__(root,source,contract_path=CONTRACT,family=FAMILY)

    def initialize(self,geometry,output,strengths=None):
        if geometry!='H4_100':return super().initialize(geometry,output,strengths)
        from .c_integrated_execution_rpc import RpcProcess
        saved=io.read(self.root/HANDOFF);initial=io.read(self.root/saved['initialization'])
        output=self.root/output;output.mkdir(parents=True,exist_ok=False)
        io.save(output/'contract.json',self.c);io.save(output/'SOURCE.json',io.read(self.source))
        shared=output/'initialization';shared.mkdir()
        call=execution.Calls(shared,dict(saved_context_reuse=1),1)
        physical=RpcProcess(self.root,self.c['environment']['evaluation_python'],'physical',output/'physical_service',self.cpus)
        self.services.append(physical)
        context=call('saved_context_reuse','byte-bound original-H context; no new CAR/H',lambda:
            physical.request('initialize',system=initial['system'],
                input_rel=self.c['input_roles']['input'],reuse_common_pickle=saved['common_pickle']))
        specs={}
        jobs=saved['completed']+saved['failed']
        for item in jobs:
            task=io.read(self.root/item['task_path'])
            specs.setdefault((task['start'],task['strength']),{})[task['arm']]=task['objective_spec']
        reused={item['job_id']:io.read(self.root/item['result_path']) for item in saved['completed']}
        io.save(shared/'result.json',dict(input_origin=saved['initialization'],
            common_origin=saved['common_pickle'],reused_trajectories=list(reused),new_cells=self.c['cgfix']['new_development_cells']))
        self.stage('development_initialization',sum(call.counts.values()))
        return dict(geometry=geometry,output=output,system=initial['system'],input_path=self.c['input_roles']['input'],
            input_sha256=initial['system']['input_sha256'],scale=initial['scale'],starts=initial['starts'],
            specs=specs,physical=physical,physical_context=context,strengths=None,reused_optimization=reused)
