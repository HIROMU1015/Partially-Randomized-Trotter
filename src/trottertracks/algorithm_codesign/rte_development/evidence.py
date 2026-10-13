"""Exclusive, streaming output and UTF-8-safe lossless review chunks."""
import hashlib
import json
import os
from fractions import Fraction
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(65536), b''): h.update(b)
    return h.hexdigest()


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      default=lambda v: str(v) if isinstance(v, Fraction) else _bad(v))


def _bad(v): raise TypeError('unsupported evidence value: '+type(v).__name__)


def atomic_json(path, value, byte_cap=65536):
    """Bounded small receipt, exclusive final path and retained partial on error."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = (dumps(value)+'\n').encode()
    if len(data) > byte_cap: raise RuntimeError('receipt cap')
    partial = path.with_name(path.name+'.partial')
    with partial.open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    os.link(partial, path); partial.unlink()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def safe_relative(path):
    p = Path(path)
    if p.is_absolute() or '..' in p.parts: raise ValueError('relative evidence path required')
    return p


class EvidenceWriter:
    def __init__(self, directory, cap, guard=lambda: None, chunk_bytes=524288):
        if type(chunk_bytes) is not int or chunk_bytes < 4: raise ValueError('UTF-8 chunk capacity >=4')
        self.directory = Path(directory)
        self.cap, self.guard, self.chunk_bytes = cap, guard, chunk_bytes
        self.files = []

    def total_bytes(self):
        return sum(p.stat().st_size for p in self.directory.rglob('*') if p.is_file())

    def check(self, reserve=32768):
        self.guard()
        if self.total_bytes()+reserve > self.cap: raise RuntimeError('aggregate evidence output cap')

    def permit_bytes(self, n):
        self.check(reserve=32768+n)

    def write(self, relative, value):
        path = self.directory/safe_relative(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name+'.partial')
        encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False,
                                   default=lambda v: str(v) if isinstance(v,Fraction) else _bad(v))
        with partial.open('xb') as stream:
            for piece in encoder.iterencode(value):
                for start in range(0, len(piece), 16384):
                    data=piece[start:start+16384].encode('utf-8')
                    self.permit_bytes(len(data)); stream.write(data); stream.flush()
            stream.write(b'\n'); stream.flush(); os.fsync(stream.fileno())
        self.check(); os.link(partial,path); partial.unlink()
        identity = dict(path=str(path.relative_to(self.directory)), bytes=path.stat().st_size,
                        sha256=digest(path), chunks=[])
        if identity['bytes'] > self.chunk_bytes:
            with path.open('rb') as source:
                buf = b''; index = 0
                while True:
                    block = source.read(self.chunk_bytes)
                    if not block and not buf: break
                    buf += block
                    while len(buf) >= self.chunk_bytes or (not block and buf):
                        cut = min(len(buf), self.chunk_bytes)
                        try: buf[:cut].decode('utf-8')
                        except UnicodeDecodeError as exc:
                            if exc.reason != 'unexpected end of data': raise
                            cut = exc.start
                        data, buf = buf[:cut], buf[cut:]
                        part = path.with_name(path.name+f'.part{index:04d}.txt')
                        self.permit_bytes(len(data))
                        with part.open('xb') as stream:
                            stream.write(data); stream.flush(); os.fsync(stream.fileno())
                        identity['chunks'].append(dict(path=str(part.relative_to(self.directory)),
                            bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
                        index += 1
                        if block and len(buf) < self.chunk_bytes: break
            verify_chunks(self.directory, identity)
        self.files.append(identity); return identity


def verify_chunks(directory, identity):
    h = hashlib.sha256(); count = 0
    for row in identity['chunks']:
        data = (Path(directory)/safe_relative(row['path'])).read_bytes()
        data.decode('utf-8')
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ArithmeticError('chunk identity mismatch')
        h.update(data); count += len(data)
    if count != identity['bytes'] or h.hexdigest() != identity['sha256']:
        raise ArithmeticError('lossless reconstruction failed')
    return True
