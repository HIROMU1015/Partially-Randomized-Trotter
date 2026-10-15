"""Metadata-only H5 pilot scope. No grant is issued by this module."""
import hashlib
import json
from pathlib import Path

NAMESPACE='H5_NEUTRAL_ROHF_NATIVE_PILOT_V1'
BASE='681ec23c93fdd1090d325908f4b155a79115afdf'
COMPILER_HASH='1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def target():
    return dict(model='linear_H5',geometry=[('H',(0.,0.,float(i))) for i in range(5)],
        distance_angstrom=1.,basis='sto-3g',charge=0,multiplicity=2,spin=1,
        electrons=5,alpha=3,beta=2,spatial_orbitals=5,spin_orbitals=10,sector_dimension=100,
        scf='ROHF',common_spatial_MO=True,state_policy='DF_SECTOR_SOLVER',T=.8)


def registry(rank):
    if type(rank) is not int or rank<1:raise ValueError('ACTUAL_RETURNED_RANK')
    p=(rank+3)//4
    rows=[]
    for method,prefix,formula,r,K in (('B0',p,'2nd',None,None),('B0I',p,'2nd',None,None),
        ('B1',rank,'2nd',None,None),('B1',rank,'4th',None,None),('B2',p,'2nd',2,2),('B2',p,'2nd',2,4)):
        cid=f'{method}_p{prefix}_q1_{formula}'+(f'_r{r}_K{K}' if r else '')
        rows.append(dict(id=cid,method=method,prefix=prefix,formula=formula,q=1,r=r,K=K,R=r))
    return rows


def seed(source,input_digest,cell_id,replica):
    if len(source)!=40 or len(input_digest)!=64 or type(replica) is not int or not 0<=replica<8:
        raise ValueError('FRESH_SEED_BINDING')
    return int(digest([NAMESPACE,source,input_digest,cell_id,replica])[:16],16)


def tasks(rank,source,input_digest,*,extension=False,eligible=()):
    result=[]
    for c in registry(rank):
        if extension and (c['method']!='B2' or c['id'] not in eligible):continue
        for replica in (range(1,8) if extension else (0,)):
            result.append(dict(cell=c,replica=replica,phase='pilot_extension' if extension else 'pilot_initial',
                seed=seed(source,input_digest,c['id'],replica) if c['R'] else None,
                validate=replica==0,heavy=c['formula']=='4th'))
    return result


def call_limits(rank):
    from ..ax2b_h6_contract import primitive_time_schedule
    times={(i,t) for cell in registry(rank) if cell['method']!='B0I'
        for i,t in primitive_time_schedule(cell,T=.8)['unique_primitive_times']}
    return dict(scf=1,integral_build=1,df_decomposition=1,state_solve=1,solver_matvec=10000,
        reference_matvec=100,reference_matrix=2,signal=6,primitive=3*len(times),control_probe=32,
        trajectory=16,sampling_call=16,occurrence=32,compile=40,
        tail_matvec=500000,deterministic_action=500000)


def draft():
    return dict(schema='H5_NEUTRAL_OPEN_SHELL_PILOT_PREPARATION_V1',namespace=NAMESPACE,
        target=target(),actual_rank=None,prefix_rule='ceil(actual_rank/4)',registry_before_input='SYMBOLIC_SIX_CELLS',
        maximum_groups=20,maximum_compile=40,maximum_PR_trajectory=16,maximum_RTE_occurrence=32,
        first_stage_groups=6,first_stage_compile=12,extension='remaining seven trajectories for eligible PR only',
        epsilons=[.05,.01,.005,.001,.0005],compiler_hash=COMPILER_HASH,
        optimization=dict(content_checked_hash_reuse=True,paired_evolution_reuse=True,
            shared_signal_preparations=True,spin_eigensystem_and_phase_cache=True,
            parallel_group_cap=4,internal_threads=1,heavy_separate=True),
        resources_policy='Original detailed CPU/RAM/wall/storage requirements waived by user; select at launch',
        cpu_allocation=None,allowed_cpus=[],input_binding=None,profile_binding=None,output_binding=None,
        carry_policy='PRESERVE_OLD_LINEAGES_NO_RESET_NO_REFUND_UNKNOWN_RETAINED; new H5 increments separate',
        approved=False,runtime_authorization=False,grant=None,sealed=False,absolute_launch_command=None,
        automatic_retry=False,mandatory_stop=True,next_stage_authorized=False,H7_authorized=False)


class Calls:
    def __init__(self,caps):self.caps=dict(caps);self.used={k:0 for k in caps};self.recorder=None
    @property
    def limits(self):
        # The pinned bounded_solver reads this interface before invoking a
        # matvec. Keep it tied to caps after the actual-rank plan is installed.
        return self.caps
    def take(self,key,number=1):
        if key not in self.caps or type(number) is not int or number<0:raise ValueError('CALL_CATEGORY')
        if self.used[key]+number>self.caps[key]:raise RuntimeError('CALL_CAP:'+key)
        self.used[key]+=number
        if self.recorder is not None:self.recorder(key,self.used)


def verify_grant(manifest,grant,source):
    if not grant or grant.get('kind')!='H5_NEUTRAL_OPEN_SHELL_ONE_SHOT_V1':raise ValueError('NEW_H5_GRANT_REQUIRED')
    if manifest.get('SOURCE')!=source or grant.get('SOURCE')!=source or len(source)!=40:raise ValueError('SOURCE_BINDING')
    if not all(grant.get(k) is True and manifest.get(k) is True for k in ('approved','runtime_authorization')):raise ValueError('NOT_APPROVED')
    if grant.get('manifest_digest')!=digest(manifest) or grant.get('maximum_compile')!=40:raise ValueError('GRANT_SCOPE')
    for key in ('profile_digest','carry_digest','target_digest','output'):
        if grant.get(key)!=manifest.get(key) or not manifest.get(key):raise ValueError('GRANT_BINDING:'+key)
    if manifest['target_digest']!=digest(target()) or manifest.get('compiler_hash')!=COMPILER_HASH:
        raise ValueError('TARGET_OR_COMPILER')
    if manifest.get('contract_digest')!=digest(draft()):raise ValueError('PILOT_CONTRACT')
    if not grant.get('one_shot') or grant.get('automatic_retry') is not False:raise ValueError('ONE_SHOT_ONLY')
    out=Path(manifest['output']).resolve()
    if not out.is_relative_to('/home/AbeHiromu'):raise ValueError('OUTPUT_HOME')
    return out
