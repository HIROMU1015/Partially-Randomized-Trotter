"""Read-only native preparation/fresh-launch observations. No scientific loads."""
import ast
import ctypes
import errno
import fcntl
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

from .identity import require, fingerprint
from .resources import cpus, observe_memory


def private_path(value):
    p=Path(value)
    require(p.is_absolute() and p.is_relative_to('/home/AbeHiromu') and '..' not in p.parts,
            'absolute home-local path required')
    require(not any(x.is_symlink() for x in (p,*p.parents)), 'symlink path forbidden')
    return p


def streaming_sha(path):
    path=private_path(path)
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        before=os.fstat(fd)
        require(before.st_uid==os.getuid() and before.st_nlink==1,'owned regular receipt')
        require(__import__('stat').S_ISREG(before.st_mode),'regular byte receipt')
        digest=hashlib.sha256();count=0
        while data:=os.read(fd,65536):digest.update(data);count+=len(data)
        after=os.fstat(fd)
        require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
                (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns) and count==before.st_size,
                'receipt changed during streaming hash')
        return {'sha256':digest.hexdigest(),'bytes':count}
    finally:os.close(fd)


def receipt_inventory(root, expected):
    root=private_path(root);items={};missing=[]
    for name,sha in expected.items():
        require(Path(name).name==name,'receipt basename')
        p=root/name
        if not p.exists():missing.append(name);continue
        row=streaming_sha(p);require(row['sha256']==sha,'frozen byte SHA mismatch: '+name)
        items[name]=row
    return dict(root=str(root),files=items,missing=missing,complete=not missing,scientific_arrays_loaded=False)


def environment_profile(reference_names, installed_sources):
    dependencies={}
    for name in sorted(reference_names):
        d=metadata.distribution(name);raw=Path(d._path)/'RECORD'
        dependencies[name]=dict(version=d.version,normalized_RECORD_sha256=hashlib.sha256(d.read_text('RECORD').encode()).hexdigest(),
                                raw_RECORD_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
    sources={str(p):streaming_sha(p)['sha256'] for p in installed_sources}
    value=dict(python=sys.executable,python_version=sys.version,dependencies=dependencies,
               installed_sources=sources,old_compiler_output_equivalence=False)
    return {**value,'fingerprint':fingerprint('h4-newhost-environment-v2',value)}


def compiler_profile(options, inherited_defaults):
    path=Path(metadata.distribution('qiskit').locate_file('qiskit/compiler/transpiler.py'))
    tree=ast.parse(path.read_bytes())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='transpile']
    require(len(functions)==1,'compiler entry-point AST')
    function=functions[0]
    args=function.args.args
    defaults=[None]*(len(args)-len(function.args.defaults))+function.args.defaults
    def repr_default(node):
        return "<class 'inspect._empty'>" if node is None else repr(ast.literal_eval(node))
    observed={a.arg:repr_default(d) for a,d in zip(args,defaults)}
    observed.update({a.arg:repr_default(d) for a,d in zip(function.args.kwonlyargs,function.args.kw_defaults)})
    require(observed==inherited_defaults,'compiler inherited defaults changed')
    plugins=sorted([dict(group=e.group,name=e.name,value=e.value) for e in metadata.entry_points()
                    if e.group.startswith('qiskit.')],key=lambda e:(e['group'],e['name'],e['value']))
    value=dict(explicit_options=options,inherited_defaults=observed,plugins_metadata=plugins,
       qiskit_version=metadata.version('qiskit'),rustworkx_version=metadata.version('rustworkx'),
       entry_point_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),old_compiler_output_equivalence=False)
    require(options['num_processes']==1,'single compiler process')
    return {**value,'fingerprint':fingerprint('h4-newhost-compiler-v2',value)}


class DQBlock(ctypes.Structure):
    _fields_=[(n,ctypes.c_uint64) for n in ('bhardlimit','bsoftlimit','curspace','ihardlimit','isoftlimit',
                                           'curinodes','btime','itime')]+[('valid',ctypes.c_uint32)]


def quota_readonly(path):
    """Only Q_GETFMT/Q_GETQUOTA and FS_IOC_FSGETXATTR. Never quota setters.

    Native commands and units are from this host's linux/quota.h and fs.h.
    Permission/unsupported errors stay UNKNOWN; ESRCH is disabled only for a
    successful filesystem lookup through quotactl_fd.
    """
    p=private_path(path);fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    libc=ctypes.CDLL(None,use_errno=True);results=[];project=None
    try:
        buf=bytearray(28)
        try:
            fcntl.ioctl(fd,0x801c581f,buf,True);project=struct.unpack('5I8s',buf)[3]
        except OSError:pass
        ids=[('user',0,os.getuid()),*[("group",1,g) for g in sorted(set([os.getgid(),*os.getgroups()]))]]
        if project is not None:ids.append(('project',2,project))
        else:results.append(dict(role='project',status='UNKNOWN_PROJECT_ID'))
        if hasattr(libc,'quotactl_fd'):
            q=libc.quotactl_fd;q.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p];q.restype=ctypes.c_int
        elif os.uname().machine=='x86_64':
            # Verified this host's asm/unistd_64.h: __NR_quotactl_fd 443.
            # Only the two constant GET commands below reach this wrapper.
            def q(target,cmd,ident,buffer):
                return libc.syscall(ctypes.c_long(443),ctypes.c_int(target),cmd,ctypes.c_int(ident),buffer)
        else:return dict(status='UNKNOWN_NO_QUOTACTL_FD',items=results)
        for name,kind,ident in ids:
            fmt=ctypes.c_uint32();ret=q(fd,ctypes.c_int((0x800004<<8)|kind),0,ctypes.byref(fmt));error=ctypes.get_errno()
            row=dict(role=name,id=ident)
            if ret!=0:
                row.update(status='DISABLED' if error==errno.ESRCH else 'UNKNOWN',errno=error)
            else:
                block=DQBlock();ret=q(fd,ctypes.c_int((0x800007<<8)|kind),ident,ctypes.byref(block));error=ctypes.get_errno()
                if ret!=0:row.update(status='UNKNOWN',errno=error,format=fmt.value)
                else:
                    row.update(status='ACTIVE',format=fmt.value,**{n:getattr(block,n) for n,_ in block._fields_})
                    # Treat soft limits as hard for this unattended one-shot.
                    limits=[v*1024 for v in (block.bhardlimit,block.bsoftlimit) if v]
                    ilimits=[v for v in (block.ihardlimit,block.isoftlimit) if v]
                    row['available_bytes']=max(0,min(limits)-block.curspace) if limits else None
                    row['available_inodes']=max(0,min(ilimits)-block.curinodes) if ilimits else None
            results.append(row)
        known=all(x['status'] in ('DISABLED','ACTIVE') for x in results)
        return dict(status='KNOWN' if known else 'UNKNOWN',items=results,read_only_commands=['Q_GETFMT','Q_GETQUOTA','FS_IOC_FSGETXATTR'])
    finally:os.close(fd)


def cpu_sample():
    values={}
    for line in Path('/proc/stat').read_text().splitlines():
        words=line.split()
        if words and words[0].startswith('cpu') and words[0][3:].isdigit():
            numbers=list(map(int,words[1:9]));values[int(words[0][3:])]=(sum(numbers),numbers[3]+numbers[4])
    return values


def host_readonly(path, *, sample_seconds=0):
    begun=time.monotonic();memory=observe_memory()
    online=cpus(Path('/sys/devices/system/cpu/online').read_text());scheduler=set(os.sched_getaffinity(0))
    topology=[]
    first=cpu_sample()
    if sample_seconds:time.sleep(min(sample_seconds,3))
    last=cpu_sample()
    for cpu in sorted(online & scheduler):
        base=Path('/sys/devices/system/cpu')/('cpu%d'%cpu)
        package=int((base/'topology/physical_package_id').read_text());core=int((base/'topology/core_id').read_text())
        nodes=sorted(int(p.name[4:]) for p in base.glob('node[0-9]*'))
        total=last[cpu][0]-first[cpu][0];idle=last[cpu][1]-first[cpu][1]
        topology.append(dict(cpu=cpu,package=package,core=core,numa=nodes[0] if len(nodes)==1 else None,
                             busy_fraction=(total-idle)/total if total else None))
    private=private_path(path);stats=os.statvfs(private);quota=quota_readonly(private)
    memory['process_cpus']=sorted(memory['process_cpus'])
    return dict(observed_monotonic=time.monotonic(),duration_seconds=time.monotonic()-begun,
      memory=memory,online_cpus=sorted(online),scheduler_affinity=sorted(scheduler),topology=topology,
      filesystem=dict(path=str(private),available_bytes=stats.f_bavail*stats.f_frsize,
          available_inodes=stats.f_favail,block_bytes=stats.f_frsize,device=os.stat(private).st_dev),quota=quota)


def cpu_proposal(observation, workers=12):
    representatives={}
    for item in observation['topology']:
        key=(item['package'],item['core'])
        if key not in representatives:representatives[key]=item
    choices=sorted(representatives.values(),key=lambda x:(x['busy_fraction'] if x['busy_fraction'] is not None else 1,x['cpu']))
    workers=min(workers,12,max(0,len(choices)-2))
    require(workers>=1,'need worker plus separate driver/observer cores')
    selected=choices[:workers+2]
    return dict(workers=[[x['cpu']] for x in selected[:workers]],driver=[selected[-2]['cpu']],
        observer=[selected[-1]['cpu']],selected_topology=selected,permission=False,
        rationale='Distinct available physical cores with lowest passive sample load; no reservation or affinity change.')


def static_invocations(templates):
    require(len(templates)==218,'registered templates')
    random=[t for t in templates if t['method'] in ('B2','B3')]
    baseline=[t for t in templates if t['method'] in ('B0','B1')]
    logical=6*(len(random)*32*2+len(baseline)*2)
    require(len(random)==194 and len(baseline)==24 and logical==74784,'fixed logical map')
    return dict(random_templates=194,baseline_templates=24,logical_wrappers=logical,
        prior_actual_consumed_or_reserved=20,new_actual_worst_case=logical,cumulative_actual_worst_case=logical+20,
        existing_cumulative_actual_cap=74784,guaranteed_cache_savings=0,minimum_cap_amendment=20,
        guarantee_reason='Reuse is within geometry/template/axis and requires an already COMPLETE identical numerical circuit. Random K2/K4 with draw-dependent circuits admits no static >=20 saving guarantee; no actual trajectories or circuits evaluated.')


def storage_projection(block=4096, *, library_cache_bytes=0, prior_charge=None):
    """Hard per-file formats plus worst-case zero reuse; no runtime execution."""
    from .launch_binding import FILE_LIMITS,CONTROL_LOG_CAP,CARRY
    rows=[('record',74784,FILE_LIMITS['record-']),('ledger',149569,FILE_LIMITS['ledger-']),
          ('worker_log',74784,FILE_LIMITS['worker-log-']),('signal',1308,FILE_LIMITS['signal-']),
          ('map_complete',1,4096),('launch_stop',1,4096),('lock',1,0)]
    components=[dict(name=n,count=c,payload_cap=size,allocated_bytes=c*((size+block-1)//block)*block,
                     charged_bytes=c*(2*size+128)) for n,c,size in rows]
    trace=(259200+2)*8192+16384
    publications=sum(c for _,c,_ in rows)
    journal=(publications+3)*128  # carried row + two direct reservations
    directory=(publications+10)*256
    temp=14*524288
    observer_charge=2*(trace+8192)+128
    control_charge=2*(CONTROL_LOG_CAP+65536)+128
    prior_charge=CARRY['charged_bytes'] if prior_charge is None else prior_charge
    require(type(prior_charge) is int and prior_charge>=0 and type(library_cache_bytes) is int and library_cache_bytes>=0,
            'nonnegative storage carry/cache')
    cache_charge=2*library_cache_bytes+128 if library_cache_bytes else 0
    charge=prior_charge+128+observer_charge+control_charge+cache_charge+sum(r['charged_bytes'] for r in components)
    physical=sum(r['allocated_bytes'] for r in components)+trace+8192+CONTROL_LOG_CAP+65536+journal+directory+temp+64*2**20+library_cache_bytes
    required=((physical+2**30-1)//2**30)*2**30
    inodes=publications+14+16  # live exclusive temp files + control/trace/directory margin
    return dict(block_bytes=block,components=components,observer_trace_cap_bytes=trace,
       observer_charge_bytes=observer_charge,control_charge_bytes=control_charge,library_cache_charge_bytes=cache_charge,
       prior_charge_bytes=prior_charge,journal_bytes=journal,
       temporary_publishers=14,temporary_files_bytes=temp,directory_bytes=directory,
       metadata_margin_bytes=64*2**20,physical_bound_bytes=physical,required_bytes=required,
       required_inodes=((inodes+999)//1000)*1000,cumulative_charge_bound=charge,
       remaining_charge_margin_bytes=10*2**30-charge,old_inputs_control_transfer_copies='Separate byte inventory; not regenerated or charged as fresh science output.',
       format_limits_enforced=True,filesystem_quota_is_fresh_condition=True)
