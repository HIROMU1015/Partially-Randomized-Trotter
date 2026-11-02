"""Separate R1 source review, direct-child authorization, exclusive marker."""
import json,re,time
from pathlib import Path
from ..synthesis_placement.wrapper_launch import (
    git,sha,consume_marker,BudgetGuard as SharedBudgetGuard,serialize_result,verify_runtime,
)


class BudgetGuard(SharedBudgetGuard):
    """Shared one-process guard plus a per-key deadline; no worker/retry."""
    def __init__(self,caps):
        super().__init__(caps);self.key_start=None;self.key_cpu=None
    def begin_key(self):
        if self.key_start is not None:raise RuntimeError('nested synthesis forbidden')
        self.key_start=time.monotonic();self.key_cpu=self.usage()['cpu_seconds']
    def end_key(self):
        self.check();u=self.usage();wall=time.monotonic()-self.key_start
        cpu=u['cpu_seconds']-self.key_cpu;self.key_start=None;self.key_cpu=None
        return {'wall_seconds':wall,'cpu_seconds':cpu,'peak_RSS_KiB':u['peak_RSS_KiB']}
    def check(self,*args):
        super().check(*args)
        if self.key_start is not None:
            if time.monotonic()-self.key_start>=self.caps['per_key_wall_seconds']:
                raise TimeoutError('R1 per-key wall cap hit; no retry')
            if self.usage()['cpu_seconds']-self.key_cpu>=self.caps['per_key_cpu_seconds']:
                raise TimeoutError('R1 per-key CPU cap hit; no retry')


def validate_binding(auth,contract_hash,head,parents,changed,dirty,allowed):
    source=auth.get('source_commit')
    if not isinstance(source,str) or not re.fullmatch(r'[0-9a-f]{40}',source):
        raise PermissionError('fixed R1 source review pending')
    if (auth.get('status')!='APPROVED_FOR_ONE_R1_RUN'
        or auth.get('science_execution_authorized') is not True
        or type(auth.get('runs')) is not int or auth['runs']!=1
        or type(auth.get('retries')) is not int or auth['retries']!=0
        or auth.get('mandatory_STOP') is not True):raise PermissionError('separate R1 one-shot authorization required')
    text=auth.get('explicit_execution_instruction')
    if not isinstance(text,str) or len(text.strip())<10:raise PermissionError('explicit execution instruction missing')
    if auth.get('contract_sha256')!=contract_hash:raise PermissionError('contract identity mismatch')
    if head==source or parents!=[source] or dirty:raise PermissionError('clean direct authorization-only child required')
    if not changed or not set(changed).issubset(allowed):raise PermissionError('forbidden authorization child change')


def verify_source(root,contract):
    m=json.loads((root/contract['source_manifest_path']).read_text())
    if m.get('focused_tests_passed') is not True or m.get('static_key_inventory_passed') is not True:
        raise PermissionError('source preparation incomplete')
    for path,digest in m['critical_sha256'].items():
        if sha(root/path)!=digest:raise PermissionError('reviewed source identity mismatch: '+path)
    return m


def verify_launch(root,contract_path):
    root=Path(root);contract=json.loads(Path(contract_path).read_text())
    auth=json.loads((root/contract['authorization_path']).read_text())
    source=auth.get('source_commit')
    if not isinstance(source,str) or not re.fullmatch(r'[0-9a-f]{40}',source):
        raise PermissionError('R1 source review and separate execution authorization pending')
    head=git(root,'rev-parse','HEAD');changed=git(root,'diff','--name-only',source,head).splitlines()
    allowed={contract['authorization_path'],contract['optional_receipt_path']}
    validate_binding(auth,sha(contract_path),head,git(root,'show','-s','--format=%P',head).split(),
                     changed,bool(git(root,'status','--porcelain','--untracked-files=all')),allowed)
    if contract['authorization_path'] not in changed:raise PermissionError('authorization JSON must be in child A')
    verify_source(root,contract)
    return contract,auth,head
