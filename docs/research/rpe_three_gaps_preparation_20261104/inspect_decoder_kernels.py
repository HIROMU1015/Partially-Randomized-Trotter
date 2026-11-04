"""Read this process's NumPy loop resolution / ELF bindings; no accuracy claims.

Fixed ABI investigation for NumPy 1.26.4, glibc x86_64 only. No ptrace,
interposition, FP-mode changes, scientific imports or H4 data access.
Private NumPy layouts follow the capsule docstring, ndarraytypes.h and
legacy_array_method.c at v1.26.4. Unsupported layouts stop before dereference.
"""
import ctypes as C
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


def stop(reason):
    raise RuntimeError('UNKNOWN/STOP: ' + reason)


def output(*args):
    return subprocess.run(args, check=True, text=True, capture_output=True).stdout


class DlInfo(C.Structure):
    _fields_ = [('path', C.c_char_p), ('base', C.c_void_p),
                ('name', C.c_char_p), ('address', C.c_void_p)]


class CallInfo(C.Structure):
    _fields_ = [('loop', C.c_void_p), ('context', C.c_void_p), ('aux', C.c_void_p),
                ('requires_pyapi', C.c_byte), ('no_fp_errors', C.c_byte)]


class LegacyAux(C.Structure):
    _fields_ = [('base', C.c_void_p * 4), ('loop', C.c_void_p),
                ('data', C.c_void_p), ('pyerr_check', C.c_int)]


@lru_cache(None)
def symbols(path):
    return [p for line in output('readelf', '-Ws', path).splitlines()
            if len(p := line.split()) >= 8 and p[0].rstrip(':').isdigit()]


@lru_cache(None)
def executable(path):
    return 'EXEC (Executable file)' in output('readelf', '-h', path)


def location(address):
    if not address:
        return None
    dl = C.CDLL(None)
    dl.dladdr.argtypes = [C.c_void_p, C.POINTER(DlInfo)]
    info = DlInfo()
    if not dl.dladdr(address, C.byref(info)):
        stop('address outside identifiable ELF image')
    path = str(Path(info.path.decode()).resolve())
    bias = 0 if executable(path) else info.base
    value = address - bias
    return dict(path=path, elf_address=hex(value),
                symbols=sorted({p[7] for p in symbols(path) if int(p[1], 16) == value})), bias


def describe(address):
    return location(address)[0] if address else None


def check_abi():
    if (np.__version__ != '1.26.4' or platform.machine() != 'x86_64'
            or platform.libc_ver() != ('glibc', '2.35') or C.sizeof(C.c_void_p) != 8):
        stop('inspection ABI differs from the investigated target')


def resolved_loop(ufunc, dtype_names, strides):
    check_abi()
    dtypes, capsule = ufunc._resolve_dtypes_and_context(
        tuple(np.dtype(x) if x else None for x in dtype_names))
    ufunc._get_strided_loop(capsule, fixed_strides=strides)
    pointer = C.pythonapi.PyCapsule_GetPointer
    pointer.argtypes = [C.py_object, C.c_char_p]
    pointer.restype = C.c_void_p
    call = CallInfo.from_address(pointer(capsule, b'numpy_1.24_ufunc_call_info'))
    wrapper = describe(call.loop)
    if 'generic_wrapped_legacy_loop' not in wrapper['symbols'] or not call.aux:
        stop('not the verified legacy wrapper layout')
    aux = LegacyAux.from_address(call.aux)
    return dict(ufunc=ufunc.__name__, dtypes=[str(x) for x in dtypes],
                fixed_strides=list(strides), wrapper=wrapper,
                inner=describe(aux.loop), user_data=describe(aux.data))


def bindings(path, anchor, process=False):
    obj = C.CDLL(None) if process else C.CDLL(path)
    info, bias = location(C.cast(getattr(obj, anchor), C.c_void_p).value)
    path = info['path']
    # atan2's public wrapper calls an internal IFUNC through an IRELATIVE slot.
    resolvers = {p[1].lstrip('0') or '0' for p in symbols(path)
                 if p[3] == 'IFUNC' and p[7].split('@')[0] == '__atan2_finite'}
    result = []
    for line in output('readelf', '-Wr', path).splitlines():
        p = line.split()
        named = len(p) > 4 and p[4].split('@')[0] in ('fmod', 'cexp', 'atan2')
        internal = len(p) > 3 and 'IRELATIVE' in p[2] and p[-1] in resolvers
        if not (named or internal):
            continue
        target = C.c_void_p.from_address(bias + int(p[0], 16)).value
        result.append(dict(symbol=p[4] if named else '__atan2_finite IFUNC',
                           relocation=p[0], target=describe(target)))
    return dict(path=path, sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                bindings=result)


def fp_environment():
    # glibc x86_64 bits/fenv.h: x87 control at 0, MXCSR at 28, sizeof=32.
    check_abi()
    state = C.create_string_buffer(32)
    libc = C.CDLL(None)
    libc.fegetenv.argtypes = [C.c_void_p]
    if libc.fegetenv(state):
        stop('fegetenv failed')
    control = int.from_bytes(state.raw[:2], 'little')
    mxcsr = int.from_bytes(state.raw[28:32], 'little')
    return dict(x87_control=hex(control), mxcsr=hex(mxcsr),
                x87_nearest=(control & 0xc00) == 0,
                sse_nearest=(mxcsr & 0x6000) == 0,
                flush_to_zero=bool(mxcsr & 0x8000), denormals_are_zero=bool(mxcsr & 0x40),
                scope='observed current thread only; no FP-mode change')


def scalar_remainder_slot():
    check_abi()
    # Stable CPython typeslots.h: Py_nb_remainder == 34.
    slot = C.pythonapi.PyType_GetSlot
    slot.argtypes = [C.py_object, C.c_int]
    slot.restype = C.c_void_p
    return describe(slot(np.float64, 34))


def snapshot():
    check_abi()
    # Artificial calls resolve lazy math slots in our own process only.
    math.atan2(.25, .5)
    math.fmod(.25, 2.)
    library = '/usr/lib/x86_64-linux-gnu/libm.so.6'
    return dict(python=platform.python_version(), numpy=np.__version__,
                libc=platform.libc_ver(), machine=platform.machine(),
                fp=fp_environment(),
                scalar_remainder=scalar_remainder_slot(),
                loops=[resolved_loop(np.remainder, ('f8', 'f8', None), (8, 0, 8)),
                       resolved_loop(np.arctan2, ('f8', 'f8', None), (16, 16, 8)),
                       resolved_loop(np.arctan2, ('f8', 'f8', None), (8, 8, 8)),
                       resolved_loop(np.exp, ('c16', None), (16, 16))],
                images=[bindings(np.core._multiarray_umath.__file__, 'npy_divmod'),
                        bindings(sys.executable, 'Py_Initialize', process=True),
                        bindings(library, 'fmod')],
                precision_contract=None, H4_uniform_decoder_decision='UNKNOWN/STOP')


if __name__ == '__main__':
    print(json.dumps(snapshot(), indent=2))
