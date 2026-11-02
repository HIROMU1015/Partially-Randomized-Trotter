"""Read-only quota observation, copied from frozen H4 infrastructure.

Origin: d0726e946f89f8c30a08d383fb8ff48b363629ec prelaunch_audit.py.
Only the two GET commands and read-only FS_IOC_FSGETXATTR are used.
No scientific policy or H4 runtime dependency is imported.
"""
import ctypes
import errno
import fcntl
import os
from pathlib import Path
import struct

def private_path(value):
    p=Path(value)
    if not p.is_absolute() or not p.is_relative_to('/home/AbeHiromu') or '..' in p.parts:
        raise ValueError('QUOTA_PRIVATE_PATH')
    if any(x.is_symlink() for x in (p,*p.parents)):raise ValueError('QUOTA_SYMLINK')
    return p

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
