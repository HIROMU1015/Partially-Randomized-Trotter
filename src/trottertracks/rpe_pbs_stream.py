"""Read the existing PBS spool JSON wire format with binary values on disk.

Only the two declared payload maps contain streamed base64 strings. Other
JSON stays ordinary metadata; canonical request identity is byte-compatible
with the existing gateway response receipts.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class DiskBlob:
    path: Path


class SpoolReader:
    def __init__(self, chunks, directory):
        self.chunks=iter(chunks);self.directory=Path(directory);self.buffer=b'';self.pos=0;self.counter=0

    def fill(self):
        while self.pos == len(self.buffer):
            self.buffer=next(self.chunks,b'');self.pos=0
            if not self.buffer:return False
        return True

    def peek(self):
        return self.buffer[self.pos:self.pos+1] if self.fill() else b''

    def take(self):
        c=self.peek()
        if c:self.pos+=1
        return c

    def whitespace(self):
        while (c:=self.peek()) and c in b' \t\n\r':self.pos+=1

    def expect(self,c):
        if self.take()!=c:raise ValueError('Malformed PBS spool JSON')

    def string(self):
        self.expect(b'"'); raw=bytearray(b'"')
        while True:
            c=self.take()
            if not c:raise ValueError('Truncated PBS string')
            raw.extend(c)
            if c==b'"':return json.loads(raw)
            if c==b'\\':
                escaped=self.take()
                if not escaped:raise ValueError('Truncated PBS escape')
                raw.extend(escaped)

    def blob(self):
        self.expect(b'"'); self.counter+=1; target=self.directory/str(self.counter)
        carry=b''
        with target.open('xb') as output:
            while True:
                if not self.fill():raise ValueError('Truncated PBS file payload')
                end=self.buffer.find(b'"',self.pos)
                final=end>=0
                part=self.buffer[self.pos:end if final else len(self.buffer)]
                self.pos=end+1 if final else len(self.buffer)
                value=carry+part
                length=len(value) if final else len(value)//4*4
                # Keep the final quartet until the closing quote, so padding
                # cannot be accepted in the middle of a larger string.
                if not final and length>=4:length-=4
                encoded,carry=value[:length],value[length:]
                if encoded:
                    raw=base64.b64decode(encoded,validate=True)
                    if base64.b64encode(raw)!=encoded or (not final and b'=' in encoded):
                        raise ValueError('Noncanonical PBS binary payload')
                    output.write(raw)
                if final:
                    if carry:raise ValueError('Truncated PBS base64 payload')
                    output.flush();os.fsync(output.fileno());break
        return DiskBlob(target)

    def value(self,path=()):
        self.whitespace(); c=self.peek()
        if len(path)==3 and path[:2] in (('payload','files'),('payload','blobs')):
            return self.blob()
        if c==b'"':return self.string()
        if c==b'{':
            self.take();self.whitespace(); result={}
            if self.peek()==b'}':self.take();return result
            while True:
                self.whitespace();key=self.string();self.whitespace();self.expect(b':')
                if key in result:raise ValueError('Duplicate PBS spool JSON key')
                result[key]=self.value((*path,key));self.whitespace();sep=self.take()
                if sep==b'}':return result
                if sep!=b',':raise ValueError('Malformed PBS object')
        if c==b'[':
            self.take();self.whitespace();result=[]
            if self.peek()==b']':self.take();return result
            while True:
                result.append(self.value((*path,len(result))));self.whitespace();sep=self.take()
                if sep==b']':return result
                if sep!=b',':raise ValueError('Malformed PBS array')
        raw=bytearray()
        while (c:=self.peek()) and c not in b' \t\r\n,]}':raw.extend(self.take())
        return json.loads(raw)

    def read(self):
        result=self.value();self.whitespace()
        if self.peek():raise ValueError('Trailing PBS spool data')
        return result


def canonical_digest(value):
    h=hashlib.sha256()
    def emit(v):
        if isinstance(v,DiskBlob):
            h.update(b'"')
            with v.path.open('rb') as source:
                for block in iter(lambda:source.read(3*256*1024),b''):h.update(base64.b64encode(block))
            h.update(b'"')
        elif isinstance(v,dict):
            h.update(b'{')
            for i,(key,val) in enumerate(sorted(v.items())):
                if i:h.update(b',')
                h.update(json.dumps(key).encode()+b':');emit(val)
            h.update(b'}')
        elif isinstance(v,list):
            h.update(b'[')
            for i,val in enumerate(v):
                if i:h.update(b',')
                emit(val)
            h.update(b']')
        else:h.update(json.dumps(v,separators=(',',':'),default=str).encode())
    emit(value)
    return h.hexdigest()
