"""Separately approved H6 continuation, retaining every original registration.

The original pilot authorizer stays intact. Its scoped task/registry provider is
replaced by an explicit, hash-bound 74/322 acceptance adapter for this entry only.
"""
from pathlib import Path
from types import FunctionType
import subprocess
import h46_pr_pilot_contract as pilot
import h46_pr_design_pilot as design

ROOT=pilot.ROOT
STAGES=pilot.STAGES
MANIFEST=pilot.MANIFEST
SIGNAL_RECORDS=pilot.SIGNAL_RECORDS
COMPILE_TOLERANCE=pilot.COMPILE_TOLERANCE
OTHER_TOLERANCE=pilot.OTHER_TOLERANCE
sha,read,save,seal,digest,require=(getattr(pilot,n) for n in ('sha','read','save','seal','digest','require'))
storage=pilot.storage
manifest=pilot.manifest
task_limits=pilot.task_limits
consumed=pilot.consumed
reserve=pilot.reserve
pipeline=pilot.pipeline
NEW_PATHS=('scripts/h6_pr_pilot_continuation_contract.py',
 'scripts/prepare_h6_pr_pilot_continuation.py','scripts/run_h6_pr_pilot_continuation.py',
 'tests/test_h6_pr_pilot_continuation.py')
SOURCE_PATHS=tuple(dict.fromkeys(pilot.SOURCE_PATHS+NEW_PATHS))
OLD_SOURCE='de5485d4b7c6de25dd9b6137c9a12bce697242c6'
OLD_PLAN='4b9dd672040b12d82efb5093c413894a32e862bfac99904524fb616ff8b23676'
ACCEPTED_RESULT_SET='23e80f820379f66466b10a8e3030d2422863f289d55a16d5b2023e05c23720e0'
MODE='H6_ORIGINAL396_ACCEPT74_CONTINUE322_V1'

def partition(full,accepted,remaining):
    ids=[t['id'] for t in full]
    require(len(full)==396 and len(set(ids))==396,'ORIGINAL396_UNIQUE')
    require(len(accepted)==74 and len(set(accepted))==74 and len(remaining)==322
      and len(set(remaining))==322 and not set(accepted)&set(remaining)
      and set(accepted+remaining)==set(ids),'EXACT74_322_DISJOINT_PARTITION')
    require(accepted==[i for i in ids if i in set(accepted)]
      and remaining==[i for i in ids if i in set(remaining)],'ORIGINAL_REGISTRATION_ORDER')
    return [t for t in full if t['id'] in set(remaining)]

def acceptance(cfg):
    require(cfg.get('continuation_mode')==MODE and cfg['stage']=='H6_DESIGN_PILOT',
      'ONLY_APPROVED_H6_CONTINUATION')
    a=read(cfg['acceptance_path'])
    require(sha(cfg['acceptance_path'])==cfg['acceptance_sha256']
      ==cfg['plan']['acceptance_sha256']==cfg['grant']['acceptance_sha256'],
      'ACCEPTANCE_HASH_BINDING')
    require(a['status']=='SEPARATELY_ACCEPTED74_ORIGINAL_STOP_RETAINED'
      and a['old_SOURCE']==OLD_SOURCE and a['old_plan_sha256']==OLD_PLAN
      and a['result_set_sha256']==ACCEPTED_RESULT_SET,'FIXED_ORIGINAL_ACCEPTANCE')
    old=read(a['old_config_path'])
    require(sha(a['old_config_path'])==a['old_config_sha256']
      and old['SOURCE']==OLD_SOURCE and old['plan_sha256']==OLD_PLAN,'ORIGINAL_CONFIG_BYTES')
    require(cfg['registry_path']==old['registry_path']
      and cfg['excluded_seeds_path']==old['excluded_seeds_path']
      and cfg['campaign_id']==old['campaign_id'],'ORIGINAL_SEED_NAMESPACE_NOT_REDRAWN')
    for row in a['old_state_files']:
        require(sha(row['path'])==row['sha256'],'ORIGINAL_STOP_MARKER_EXIT_UNCHANGED')
    result_set=[]
    for row in a['accepted_results']:
        require(sha(row['path'])==row['sha256'],'ACCEPTED_RESULT_BYTES')
        result_set.append(dict(task=row['id'],sha256=row['sha256']))
    require(digest(sorted(result_set,key=lambda r:r['task']))==ACCEPTED_RESULT_SET
      and [r['id'] for r in a['accepted_results']]==a['completed_ids'],'ACCEPTED74_RESULT_SET')
    full=design.tasks_for('H6_DESIGN_PILOT',cfg['decisions']['n_P_by_candidate'])
    require(full==old['tasks'],'ORIGINAL396_EXACT_CANDIDATES_AND_SAMPLES')
    pending=partition(full,a['completed_ids'],a['remaining_ids'])
    require(cfg['tasks']==pending,'ONLY_ORIGINAL_INCOMPLETE322_TASKS')
    return a,old,full,pending

class Scope:
    def __init__(self,cfg):self.cfg=cfg
    def __getattr__(self,name):return getattr(design,name)
    def tasks_for(self,stage,n):
        require(stage==self.cfg['stage'] and n==self.cfg['decisions']['n_P_by_candidate'],
          'UNCHANGED_ORIGINAL33_PER_GROUP')
        return acceptance(self.cfg)[3]
    def registry(self,tasks,source,campaign,cohort,excluded):
        a,old,full,pending=acceptance(self.cfg)
        require(tasks==pending and source==self.cfg['SOURCE']
          and campaign==old['campaign_id'] and cohort==design.COHORT,'CONTINUATION_REGISTRY_IDENTITY')
        # Re-derive all original seeds from their original SOURCE/namespace.
        # New SOURCE identifies the repaired execution, never a seed redraw.
        return design.registry(full,OLD_SOURCE,old['campaign_id'],cohort,excluded)

def verify(cfg):
    pilot.verify(cfg)
    acceptance(cfg)
    indexed={r['path']:r for r in cfg['sealed_files']}
    require(cfg['acceptance_path'] in indexed,'ACCEPTANCE_RECEIPT_SEALED')
    require(cfg['plan']['continuation_mode']==cfg['continuation_mode']==MODE,'MODE_PLAN_BINDING')
    for rel in NEW_PATHS:
        p=ROOT/rel
        require(str(p) in indexed and p.read_bytes()==subprocess.check_output(
          ['git','show',cfg['SOURCE']+':'+rel],cwd=ROOT),'COMMITTED_CONTINUATION_SOURCE:'+rel)
    approval=read(cfg['approval_receipt_path'])
    require(approval['acceptance_sha256']==cfg['acceptance_sha256']
      and approval['reuse74_authorized'] is True and approval['continue322_authorized'] is True
      and approval['fixed_original_registry'] is True,'EXPLICIT_REUSE_AND_CONTINUATION_APPROVAL')
    from h46_pr_pilot_repair import enabled
    require(enabled(cfg),'REPAIRED_OPTIMIZED_H6_ONLY')

def authorize(cfg):
    # Keep the original early refusal before accessing any acceptance/input.
    require(cfg.get('authorized') is True and cfg.get('grant',{}).get('valid') is True,
      'EXPLICIT_PILOT_EXECUTION_APPROVAL_REQUIRED')
    scoped=Scope(cfg)
    bound=FunctionType(pilot.authorize.__code__,dict(pilot.__dict__,design=scoped,verify=verify),
      'authorize_continuation',pilot.authorize.__defaults__)
    bound(cfg)

def bind():
    """Bind the separately named entry to unchanged native/launcher functions."""
    import sys
    import h46_pr_pilot_native as native
    import run_h46_pr_design_pilot as runner
    import audit_h46_pr_pilot_execution as audit
    native.c=runner.c=audit.c=sys.modules[__name__]
    return runner,audit
