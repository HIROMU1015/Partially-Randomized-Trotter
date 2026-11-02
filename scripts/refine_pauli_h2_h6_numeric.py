#!/usr/bin/env python3
"""Fresh numerical refinement outputs; never replace old state-action files."""
import importlib,json,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=types.ModuleType('hx_refinement');p.__path__=[str(ROOT/'src/trotterlib')];sys.modules[p.__name__]=p
V=importlib.import_module(p.__name__+'.pauli_vector_tight_enclosure')
def read(p):return json.loads(Path(p).read_text())
if __name__=='__main__':
 model,ref,spec,out=sys.argv[1:];s=read(spec);r=V.evaluate(read(model),read(ref),s['config'],s.get('law'))
 with Path(out).open('x') as f:json.dump(r,f,indent=2);f.write('\n')
