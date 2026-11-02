"""Parent-only bounded JSON buffering; legacy evidence code stays immutable.

Encoder options, bytes, exclusive publication and review chunks are unchanged.
Full output-tree budget checks occur before each <=16 KiB write; the owner's
and ledger's periodic CPU/RSS/output checks continue independently.
"""
from fractions import Fraction
import hashlib
import json
import os
from ..rte_development.evidence import (
    EvidenceWriter, digest, safe_relative, verify_chunks, _bad,
)


class BatchedEvidenceWriter(EvidenceWriter):
    def __init__(self, directory, cap, guard=lambda: None, chunk_bytes=524288,
                 write_block_bytes=16384):
        if type(write_block_bytes) is not int or write_block_bytes != 16384:
            raise ValueError('fixed 16 KiB JSON write block required')
        super().__init__(directory, cap, guard, chunk_bytes)
        self.write_block_bytes = write_block_bytes

    def write(self, relative, value):
        path = self.directory/safe_relative(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name+'.partial')
        encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False,
                                   default=lambda v: str(v) if isinstance(v, Fraction) else _bad(v))
        pending = bytearray()

        def flush(stream, final=False):
            while len(pending) >= self.write_block_bytes or (final and pending):
                cut = min(len(pending), self.write_block_bytes)
                data = bytes(pending[:cut])
                try:
                    data.decode('utf-8')
                except UnicodeDecodeError as exc:
                    if exc.reason != 'unexpected end of data':
                        raise
                    cut = exc.start
                    data = bytes(pending[:cut])
                if not cut:
                    raise UnicodeError('cannot make a complete UTF-8 write block')
                self.permit_bytes(len(data))
                stream.write(data)
                stream.flush()
                del pending[:cut]

        with partial.open('xb') as stream:
            for piece in encoder.iterencode(value):
                # Limit temporary encoding to the legacy 16384-character slices.
                for start in range(0, len(piece), 16384):
                    pending.extend(piece[start:start+16384].encode('utf-8'))
                    flush(stream)
            pending.extend(b'\n')
            flush(stream, final=True)
            os.fsync(stream.fileno())
        self.check()
        os.link(partial, path)
        partial.unlink()
        identity = dict(path=str(path.relative_to(self.directory)), bytes=path.stat().st_size,
                        sha256=digest(path), chunks=[])
        # Preserve the exact legacy chunk algorithm and 512 KiB review boundary.
        if identity['bytes'] > self.chunk_bytes:
            with path.open('rb') as source:
                buf = b''
                index = 0
                while True:
                    block = source.read(self.chunk_bytes)
                    if not block and not buf:
                        break
                    buf += block
                    while len(buf) >= self.chunk_bytes or (not block and buf):
                        cut = min(len(buf), self.chunk_bytes)
                        try:
                            buf[:cut].decode('utf-8')
                        except UnicodeDecodeError as exc:
                            if exc.reason != 'unexpected end of data':
                                raise
                            cut = exc.start
                        data, buf = buf[:cut], buf[cut:]
                        part = path.with_name(path.name+f'.part{index:04d}.txt')
                        self.permit_bytes(len(data))
                        with part.open('xb') as stream:
                            stream.write(data)
                            stream.flush()
                            os.fsync(stream.fileno())
                        identity['chunks'].append(dict(path=str(part.relative_to(self.directory)),
                            bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
                        index += 1
                        if block and len(buf) < self.chunk_bytes:
                            break
            verify_chunks(self.directory, identity)
        self.files.append(identity)
        return identity
