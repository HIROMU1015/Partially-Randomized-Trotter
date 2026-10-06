# 容量計算・quota観測の読み取り専用手順

本作業はmetadataだけの容量算術と既存filesystemの読み取り専用観測。
science sourceやproduction runnerはimport/実行しない。NumPy/OpenFermion/PySCF/CuPy等もimportしない。
配列・NPY・NPZは作成せず、shape/dtypeの整数metadataを使う。
fixed output/registry、分子snapshot、旧runtime/cache/checkpointを探索しない。
書き込みは新worktree内の軽量準備bundleだけ。

## 実行記録と再計算根拠

既存venv Pythonを-B、PYTHONNOUSERSITE/PYTHONDONTWRITEBYTECODEで使い、
BLAS/OMP/MKL/NUMEXPR/RAYONは1、QISKIT_PARALLEL=false、QISKIT_NUM_PROCS=1。
外部worker/taskset/科学runnerは起動しない。installや環境設定の永続変更はしない。
処理は1回だけ。stdout/stderrは storage-review-attempt-01.log に保持。
8項目の算術/不変性検査は成功。新test campaignは追加しない。

公開直前には保存手順をmainとして実行せず、標準library runpyでmain guardを通さず読み込み、
quota_and_filesystem getterだけを1回再照会した。科学sourceはimport/実行していない。
その結果はprepublication_capacity_observation_v1.json。容量modelの再生成は0。

以下はその一回の手順の保存用コピーで、再実行指示ではない。
手順ファイルの生成後にbyte-identicalなplan/auth/reviewコピーと文書を追加しており、
当時のexistence guardを緩めて再実行しない。
procedureはscience sourceの一部でも変更でもない。

## quota・filesystem確認

既存親directoryをread-onlyでopenし、statvfsのnonroot f_bavail/f_favailを使用。
/proc/self/mountinfoでmountを確定し、own uid/gid/groupsとread-only access確認を記録。
固定outputそのもののresolve/stat/作成はしない。

quotaコマンドは未導入。初期own Q_GETQUOTAのuser/group ESRCH、project EPERMを
quota_initial_query_v1.json に保存した。EPERMを「quotaなし」と判断しない。
その後、既存libc/kernelのquotactl_fd Q_GETFMTをuser/group/project各typeへread-only照会し、
全ESRCHを quota_status_query_v1.json と最終 filesystem_quota_observation_v1.json に保存した。
Q_GETFMTは状態照会であり、既存Linux quota_getfmtの非active戻り値ESRCHに対応する。
installed x86_64 syscall headerから__NR_quotactl_fd=443を読んで番号を固定し、
観測手順は既存platform x86_64も確認する。quota設定・on/off・syncは実施しない。
FS_IOC_FSGETXATTR=0x801c581fも既存directoryからread-only照会し、project id0/inheritなしを確認。

参照：
[Linux quota_getfmt](https://github.com/torvalds/linux/blob/v6.8/fs/quota/quota.c)、
[Linux quota.h](https://github.com/torvalds/linux/blob/v6.8/include/uapi/linux/quota.h)、
[quotactl_fd man page](https://man7.org/linux/man-pages/man2/quotactl.2.html)、
[NumPy1.26.4 serialization source](https://github.com/numpy/numpy/blob/v1.26.4/numpy/lib/npyio.py)。
sourceとinstalled保存sourceのSHAは array_and_storage_estimate_v1.json に保持。

quota3種は観測時非有効と確認済み。nullの残量fieldは不明ではなく非適用の意味。
将来quota active/照会不明になればown user/group/project残bytes/inode確認までSTOP。
probe file、大容量dummy、予約、削除/移動、共有設定変更は0。

## 保存した手順全文

procedure SHA-256: `c2e2f412f8a6da08ecb5eac4791dcdf2948eaa3c47b1c9fd93588d468cdae547`

```python
"""Integer/shape metadata only; read-only filesystem and quota status getters."""
import ast
import builtins
import ctypes
from datetime import datetime,timezone,timedelta
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys

ROOT=Path.cwd()
SCIENCE=Path('/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-geometry-resource-observer-fix-20261006')
BUNDLE='artifacts/resource_applicability/track_a_h4_input_generation_storage_review/2026-10-06'
OUTPUT='/home/AbeHiromu/projects/partially-randomized-trotter/artifacts/resource_applicability/track_a_h4_geometry_execution/track-a-h4-geometry-v2-20261006-run01'
COUNTS=dict(molecular_access=0,science_package_import=0,science_array_NPZ_NPY_generation=0,production_taskset_worker_launch=0,
    GPU=0,quota_settings_changed=0,data_delete_move_reserve=0,source_plan_auth_review_changed=0,transpile=0)


def guard():
    original=builtins.__import__
    def guarded(name,*a,**kw):
        if name.split('.')[0] in ('numpy','scipy','pyscf','openfermion','openfermionpyscf','cupy','torch','pynvml'):
            COUNTS['science_package_import']+=1;raise RuntimeError('science imports forbidden')
        return original(name,*a,**kw)
    builtins.__import__=guarded
    def check(path):
        if not isinstance(path,(str,bytes,os.PathLike)):return
        text=os.fsdecode(path)
        if text==OUTPUT or text.startswith(OUTPUT+'/') or text.endswith(('.npz','.npy','.pkl','.pickle','.sqlite','.db')) or any(
            p in text.split('/') for p in ('.runtime','runtime','cache','caches','checkpoint','checkpoints','registry','registries')):
            COUNTS['molecular_access']+=1;raise RuntimeError('scientific/output path forbidden')
    stat,lstat,resolve,vfs=os.stat,os.lstat,Path.resolve,os.statvfs
    def gs(path,*a,**kw):check(path);return stat(path,*a,**kw)
    def gl(path,*a,**kw):check(path);return lstat(path,*a,**kw)
    def gr(self,*a,**kw):check(self);return resolve(self,*a,**kw)
    def gv(path,*a,**kw):check(path);return vfs(path,*a,**kw)
    os.stat,os.lstat,Path.resolve,os.statvfs=gs,gl,gr,gv
    def audit(event,args):
        if event=='open':check(args[0])
        if event in ('subprocess.Popen','os.system','os.posix_spawn','os.fork','os.exec','os.sched_setaffinity'):
            COUNTS['production_taskset_worker_launch']+=1;raise RuntimeError('no commands/taskset/workers')
    sys.addaudithook(audit)


def save(name,value):
    with (ROOT/BUNDLE/name).open('x') as out:
        json.dump(value,out,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False);out.write('\n')


def inventory():
    # Shape from fixed H4/STO-3G source; sizes are integers, no array allocated.
    rows=[
        ('coordinates',(4,3),'float64',8,'generate_input: four coordinate triples'),
        ('hAO',(4,4),'float64 expected; complex128 ceiling',16,'generate_input: four AO hcore; conservative dtype ceiling'),
        ('overlap',(4,4),'float64 expected; complex128 ceiling',16,'generate_input: four AO overlap'),
        ('eriAO',(4,4,4,4),'float64 expected; complex128 ceiling',16,'generate_input: int2e four AO'),
        ('MOs',(4,4),'float64',8,'canonical_mos: explicit real 4x4'),
        ('MO_energies',(4,),'float64',8,'canonical_mos: four energies'),
        ('hMO',(4,4),'float64 expected; complex128 ceiling',16,'C.T @ hAO @ C'),
        ('eriMO',(4,4,4,4),'float64 expected; complex128 ceiling',16,'ao2mo.restore(1,...,4), transpose'),
        ('hspin',(8,8),'float64',8,'installed spinorb_from_spatial numpy.zeros'),
        ('twospin',(8,8,8,8),'float64',8,'installed spinorb_from_spatial numpy.zeros'),
        ('one',(8,8),'complex128',16,'hspin plus complex one_body_correction'),
        ('nuclear',(),'float64',8,'np.asarray(float energy_nuc)'),
        ('H',(256,256),'complex128',16,'one_body_operator/summed blocks, ground_state dimension gate'),
        ('blocks',(12,256,256),'complex128',16,'returned DF count12; eight-spin operators'),
        ('lambdas',(12,),'float64',8,'factorize eigh real eigenvalues, first12'),
        ('G',(12,8,8),'float64',8,'factorize canonical_squares.real, first12'),
        ('correction',(8,8),'complex128',16,'installed low_rank: zeros(...,complex)'),
        ('raw_eigenvalues',(16,),'float64',8,'full interaction eigh16'),
        ('raw_eigenvectors',(16,16),'float64 expected; complex128 ceiling',16,'full eigh16; conservative ceiling'),
        ('canonical_eigenvectors',(16,16),'float64',8,'vectors.real times real signs'),
        ('permutation',(16,),'int64',8,'64bit pinned numpy argsort'),
        ('raw_indices',(12,),'int64',8,'permutation first12'),
        ('signs',(16,),'float64',8,'np.ones(16)'),
        ('weights',(16,),'float64',8,'abs(real eigenvalues), squared real norms'),
        ('truncation_value',(),'float64',8,'real weight cumsum difference'),
        ('sector',(36,),'int64',8,'explicit np.int64 sector_indices'),
        ('sector_H',(36,36),'complex128',16,'H sector selection after dtype=np.complex128'),
        ('sector_state',(36,),'complex128',16,'explicit astype(np.complex128)'),
        ('state',(256,),'complex128',16,'explicit np.zeros(256,dtype=np.complex128)'),
        ('qiskit_state',(256,),'complex128',16,'permutation of full state'),
        ('energy',(),'float64',8,'np.asarray(real Rayleigh)'),
        ('gap',(),'float64',8,'np.asarray(real eigenvalue gap)'),
    ]
    result=[]
    for name,shape,dtype,size,basis in rows:
        result.append({'name':name,'NPZ_shape':list(shape),'elements':math.prod(shape),'dtype_basis':dtype,'itemsize_ceiling':size,
            'payload_bytes_ceiling':math.prod(shape)*size,'source_basis':basis,'actual_array_generated':False})
    source=(SCIENCE/'src/trottertracks/resource_applicability/h4_geometry/inputs.py').read_bytes()
    tree=ast.parse(source)
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    direct=[]
    for n in ast.walk(functions['generate_input']):
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='arrays' for t in n.targets):
            direct=[k.arg for k in n.value.keywords if k.arg is not None]
    returned={}
    for name in ('factorize','ground_state'):
        for n in ast.walk(functions[name]):
            if isinstance(n,ast.Return) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='dict':
                returned[name]=[k.arg for k in n.value.keywords if k.arg is not None]
    assert set(r['name'] for r in result)==set(direct+returned['factorize']+returned['ground_state'])
    assert len(result)==32
    return result


def quota_and_filesystem():
    parent=Path('/home/AbeHiromu/projects/partially-randomized-trotter/artifacts')
    st=os.statvfs(parent)
    mounts=[]
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        before,after=line.split(' - ',1);fields=before.split();suffix=after.split();mount=Path(fields[4])
        if parent.is_relative_to(mount):mounts.append((len(mount.parts),fields,suffix,line))
        if Path(OUTPUT).is_relative_to(mount) and not parent.is_relative_to(mount):
            raise RuntimeError('output would be on a different unobserved filesystem')
    _,fields,suffix,mount=max(mounts,key=lambda m:m[0])
    assert suffix[0]=='ext4' and 'rw' in fields[5].split(',') and not (st.f_flag & os.ST_RDONLY)
    header=Path('/usr/include/x86_64-linux-gnu/asm/unistd_64.h').read_text()
    nr=int(re.search(r'^#define __NR_quotactl_fd (\d+)$',header,re.M).group(1));assert os.uname().machine=='x86_64'
    libc=ctypes.CDLL(None,use_errno=True);libc.syscall.restype=ctypes.c_long
    fd=os.open(parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        meta=os.fstat(fd);buf=ctypes.create_string_buffer(28);ctypes.set_errno(0)
        rc=libc.ioctl(ctypes.c_int(fd),ctypes.c_ulong(0x801c581f),ctypes.byref(buf));e=ctypes.get_errno()
        attrs=dict(zip(('xflags','extsize','nextents','project_id','cowextsize'),struct.unpack('5I',buf.raw[:20]))) if rc==0 else None
        queries=[]
        for label,kind in [('user',0),('group',1),('project',2)]:
            result=ctypes.c_uint32();ctypes.set_errno(0)
            qrc=libc.syscall(ctypes.c_long(nr),ctypes.c_int(fd),ctypes.c_uint((0x800004<<8)|kind),ctypes.c_uint(0),ctypes.byref(result))
            qe=ctypes.get_errno()
            queries.append({'type':label,'command':'Q_GETFMT','return':qrc,'errno':qe,'error':os.strerror(qe) if qe else None,
                'format':result.value if qrc==0 else None,'interpretation':'NOT_ACTIVE_ON_THIS_FILESYSTEM' if qrc==-1 and qe==errno.ESRCH else 'ACTIVE_OR_UNCONFIRMED'})
    finally:os.close(fd)
    quota_verified=all(q['interpretation']=='NOT_ACTIVE_ON_THIS_FILESYSTEM' for q in queries)
    return {'schema_version':'h4-storage-readonly-observation-v1','observed_utc':datetime.now(timezone.utc).isoformat(),
        'observed_JST':datetime.now(timezone(timedelta(hours=9))).isoformat(),'existing_ancestor':str(parent),
        'filesystem_type':suffix[0],'filesystem_source':suffix[1],'mountpoint':fields[4],'mountinfo':mount,
        'block_size':st.f_bsize,'fragment_size':st.f_frsize,'available_bytes_nonroot':st.f_bavail*st.f_frsize,
        'free_bytes_including_reserved':st.f_bfree*st.f_frsize,'reserved_blocks_not_used_as_available':True,
        'free_inodes':st.f_ffree,'available_inodes':st.f_favail,'total_inodes':st.f_files,
        'readonly_existing_parent_write_search_access':os.access(parent,os.W_OK|os.X_OK,effective_ids=True),
        'uid':os.getuid(),'gid':os.getgid(),'groups':os.getgroups(),'ancestor_uid':meta.st_uid,'ancestor_gid':meta.st_gid,
        'ancestor_mode':oct(meta.st_mode),'project_attr_query':{'command':'FS_IOC_FSGETXATTR','return':rc,'errno':e,'attributes':attrs},
        'quota_queries':queries,'quota_user_group_project_status_verified':quota_verified,
        'quota_interpretation':'USER_GROUP_PROJECT_QUOTA_NOT_ACTIVE_AT_OBSERVATION' if quota_verified else 'UNCONFIRMED',
        'quota_remaining_bytes':None,'quota_remaining_inodes':None,'quota_remaining_null_reason':'No active quota; filesystem availability applies' if quota_verified else 'quota unknown',
        'quota_settings_changed':False,'probe_file_created':False,'output_registry_resolve_stat_create':False,'capacity_reserved':False}


def main():
    guard()
    rows=inventory();observation=quota_and_filesystem();block=observation['fragment_size'];assert block==4096
    round_block=lambda n:((n+block-1)//block)*block
    array_payload=sum(r['payload_bytes_ceiling'] for r in rows)
    assert sum(r['payload_bytes_ceiling'] for r in rows if r['name'] in ('H','blocks'))==13*2**20
    # Per member: NPY header/padding + local/central ZIP/ZIP64/name allowance.
    npz_per_input=array_payload+len(rows)*1024+256
    wall_records=72*3600+2  # one second Event.wait, plus boundary/closing allowance
    worker_logs=6
    final_files=wall_records+6+worker_logs+1+1  # NPZ, worker logs, freeze, journal
    temporary_files=8  # driver + watcher + up to six worker-I/O log publishers
    directories=3
    inode_count=final_files+temporary_files+directories
    planned_inode_requirement=260000
    # Hard protocol ceiling avoids relying solely on inferred array dtypes:
    # each generation response frame<=64MiB, so its NPZ bytes<=64MiB.
    ipc_per_input=64*2**20
    monitor_json_bytes=128
    freeze_json_bytes=128*2**10
    publications=wall_records+6+worker_logs+1
    journal_bytes=publications*128
    components={
        'six_input_NPZ_shape_dtype_bound_final_bytes':6*round_block(npz_per_input),
        'six_input_NPZ_IPC64MiB_ceiling_plus_full_temp_final_double_allowance':2*6*ipc_per_input,
        'all_72h_monitor_JSON_blocks':wall_records*round_block(monitor_json_bytes),
        'generation_freeze_JSON_temp_final_allowance':2*round_block(freeze_json_bytes),
        'six_worker_text_logs_temp_final_allowance':2*6*round_block(8192),
        'append_only_byte_budget_journal':round_block(journal_bytes),
        'conservative_inode_metadata_one_block_per_new_inode':inode_count*block,
        'directory_entries_and_index_allowance_256_bytes_each':round_block((final_files+temporary_files+2*directories)*256),
        'ancestor_directory_initial_blocks':directories*block,
        'filesystem_extent_journal_misc_margin':64*2**20,
    }
    # Shape-bound line is evidence only. The looser IPC bound replaces it.
    physical_bound=sum(value for key,value in components.items() if key!='six_input_NPZ_shape_dtype_bound_final_bytes')
    required_bytes=((physical_bound+2**30-1)//2**30)*2**30
    assert required_bytes==3*2**30 and inode_count<=planned_inode_requirement
    payload_publish_bound=6*ipc_per_input+wall_records*monitor_json_bytes+6*8192+freeze_json_bytes
    accounting_charge=2*payload_publish_bound+publications*128
    assert accounting_charge<10*2**30
    source_files=['inputs.py','execution.py','resources.py','workers.py','ledger.py']
    source_hashes={name:hashlib.sha256((SCIENCE/'src/trottertracks/resource_applicability/h4_geometry'/name).read_bytes()).hexdigest() for name in source_files}
    installed_paths=['/home/AbeHiromu/venvs/trotter-common/lib/python3.12/site-packages/numpy/lib/npyio.py',
        '/home/AbeHiromu/venvs/trotter-common/lib/python3.12/site-packages/numpy/lib/format.py',
        '/home/AbeHiromu/venvs/trotter-common/lib/python3.12/site-packages/openfermion/chem/molecular_data.py',
        '/home/AbeHiromu/venvs/trotter-common/lib/python3.12/site-packages/openfermion/circuits/low_rank.py']
    estimate={'schema_version':'h4-generation-stage-storage-estimate-v1','scope':'SIX_INPUTS_GENERATE_FREEZE_STOP_ONLY',
        'science_SOURCE_COMMIT':'9ab38665920dfb5ac0a9d038233e1f3bf5d8fe5a','science_checkout_root':str(SCIENCE),
        'source_hashes_used':source_hashes,'installed_serialization_static_hashes':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in installed_paths},
        'arrays':rows,'array_count':32,'H_and_12_blocks_payload_bytes_per_input':13*2**20,
        'all_array_payload_bytes_ceiling_per_input':array_payload,'NPZ_bytes_ceiling_per_input_from_shapes':npz_per_input,
        'serialization':'np.savez ZIP_STORED; 32 NPY members; ZIP64 local headers; no external scientific file generated',
        'serialization_allowance':{'per_NPY_and_ZIP_member_bytes':1024,'archive_tail_bytes':256,'assumption_not_measured_file':True},
        'runtime_generation_response_frame_ceiling':ipc_per_input,'NPZ_space_bound_uses_frame_ceiling':True,
        'maximum_wall_seconds':72*3600,'monitor_wait_seconds_source':1,'max_wall_records_with_two_boundary_allowance':wall_records,
        'monitor_record_bytes_allowance':monitor_json_bytes,'generation_freeze_JSON_allowance':freeze_json_bytes,
        'max_worker_log_files':6,'worker_log_bytes_source_cap':8192,'max_publications':publications,
        'budget_journal_record_bytes_source':128,'journal_bytes_ceiling':journal_bytes,
        'temporary_final_source_relation':'os.link(.pending,final) aliases same inode; no second data copy in actual write. Capacity estimate nevertheless double-charges all six full 64MiB input files and ancillary large outputs conservatively.',
        'components_bytes':components,'shape_bound_component_excluded_from_sum_as_replaced_by_IPC_bound':True,
        'physical_planning_bound_bytes':physical_bound,'required_available_bytes_rounded':required_bytes,
        'new_inode_count_bound':inode_count,'required_available_inodes_rounded':planned_inode_requirement,
        'inode_metadata_allowance':'4096 bytes per new inode is conservative planning overhead; ext4 inode tables are normally already allocated. Actual inode size was not measured and is not assumed to be 256.',
        'directory_metadata_allowance':'256 bytes per final/temporary directory entry plus directory blocks and 64MiB miscellaneous metadata margin; assumption, not measured production footprint.',
        'source_budget_cumulative_charge_ceiling':accounting_charge,'source_campaign_output_cap':10*2**30,
        'campaign_cap_not_a_full_free_space_start_requirement':True,
        'scientific_arrays_NPZ_NPY_generated':0,'monitoring_72h_executed':False,'scope_beyond_generation_approved':False}
    space_ok=observation['available_bytes_nonroot']>=required_bytes
    inode_ok=observation['available_inodes']>=planned_inode_requirement
    verified=observation['quota_user_group_project_status_verified'] and observation['readonly_existing_parent_write_search_access']
    verdict='足りる' if space_ok and inode_ok and verified else ('不足' if not space_ok or not inode_ok else '確認不能')
    decision={'schema_version':'h4-generation-stage-storage-verdict-v1','observed_utc':observation['observed_utc'],
        'verdict':verdict,'required_bytes':required_bytes,'effective_available_bytes':observation['available_bytes_nonroot'] if verified else None,
        'filesystem_available_bytes':observation['available_bytes_nonroot'],'required_inodes':planned_inode_requirement,
        'effective_available_inodes':observation['available_inodes'] if verified else None,
        'byte_margin':observation['available_bytes_nonroot']-required_bytes,'inode_margin':observation['available_inodes']-planned_inode_requirement,
        'quota_user_group_project_verified_inactive':observation['quota_user_group_project_status_verified'],
        'only_generation_stage_capacity_supported':True,'signal_compile_capacity_approved':False,
        'ten_GiB_full_campaign_free_space_condition_removed_for_this_stage':True,'original_proposal_history_unchanged':True,
        'CPU_use_approved':False,'review_approved':False,'launch_executed':False,
        'fresh_checks_before_launch':{'available_bytes_at_least':required_bytes,'available_inodes_at_least':planned_inode_requirement,
            'quota_still_inactive_or_active_quota_remaining_verified_sufficient':True},
        'status':'H4_INPUT_GENERATION_STAGE_STORAGE_CONFIRMED_AWAITING_FINAL_APPROVAL' if verdict=='足りる' else 'H4_INPUT_GENERATION_STAGE_STORAGE_'+('INSUFFICIENT' if verdict=='不足' else 'UNCONFIRMED')+'_STOP',
        'failure_minimum_resolution':None if verdict=='足りる' else 'Read-only confirmation or explicit remedy proposal required; no root change/deletion/reservation performed.',
        'guard_counts':COUNTS,'metadata_arithmetic_checks':{'count':8,'failures':0,'errors':0,'skipped':0},'mandatory_stop':True}
    save('array_and_storage_estimate_v1.json',estimate);save('filesystem_quota_observation_v1.json',observation);save('storage_verdict_v1.json',decision)
    print(json.dumps({'array_count':32,'H_and_blocks_MiB':13,'shape_NPZ_per_input_MiB':npz_per_input/2**20,
        'wall_records_max':wall_records,'publications':publications,'physical_bound_GiB':physical_bound/2**30,
        'required_GiB':required_bytes/2**30,'available_GiB':observation['available_bytes_nonroot']/2**30,
        'required_inodes':planned_inode_requirement,'available_inodes':observation['available_inodes'],
        'quota_status':observation['quota_interpretation'],'verdict':verdict,'guard_counts':COUNTS},sort_keys=True))


if __name__=='__main__':main()

```
