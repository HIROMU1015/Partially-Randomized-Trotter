#!/usr/bin/env python3
import importlib,json,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=types.ModuleType('hx_numeric');p.__path__=[str(ROOT/'src/trotterlib')];sys.modules[p.__name__]=p
V=importlib.import_module(p.__name__+'.pauli_vector_enclosure')
def read(p):return json.loads(Path(p).read_text())
if __name__=='__main__':
 mode,model_path,out_path,*rest=sys.argv[1:];model=read(model_path)
 if mode=='reference':result=V.reference_enclosure(model)
 elif mode=='target':
  spec=read(rest[0]);reference=read(rest[1]);result=V.evaluate(model,reference,spec['config'],spec.get('law'))
 else:raise ValueError(mode)
 path=Path(out_path)
 with path.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
