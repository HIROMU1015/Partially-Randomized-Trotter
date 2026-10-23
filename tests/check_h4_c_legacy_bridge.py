#!/usr/bin/env python3
"""Two-qubit supplied arrays only; frozen legacy preparation -> C export.

No H4 paths, chemistry, optimized state, reference, RNG or compilation calls.
"""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
LEGACY=Path('/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/non-df-h4-execution-foundation-20261023')
sys.path.insert(0,str(LEGACY/'src'))
from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.pauli_signal_adapter import PauliHamiltonian
from trotterlib.fixed_signal_evaluation import prepare_signal_adapter,FixedSignalConfig
s=importlib.util.spec_from_file_location('legacy_C_builder',ROOT/'src/trotterlib/h4_legacy_export.py');b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
s=importlib.util.spec_from_file_location('legacy_C_control',ROOT/'src/trotterlib/h4_certificate_control.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)

def main():
    destination=Path(sys.argv[1]);destination.mkdir(exist_ok=False)
    p=PauliHamiltonian.from_terms([('ZI',.25),('IX',.125),('II',.125)],num_qubits=2)
    d=DFHamiltonian(.125,np.diag([.125,.25]),np.array([.25,.125]),(np.diag([1.,0.]),np.diag([0.,1.])),dict(model_id='ARTIFICIAL_TWO_QUBIT_ONLY'))
    state=np.array([1.,0.,0.,0.],dtype=complex)
    targets=[dict(case=rep+'_'+method,axis=axis,epsilon_sig=eps,design_total_error_target_exact='1/100') for rep in ('pauli_jw','df_native') for method in ('PF','PR') for axis in ('cosine','sine') for eps in (.05,.01,.005,.001,.0005)]
    for rep,h in [('pauli_jw',p),('df_native',d)]:
        results={}
        for method in ('PF','PR'):
            a=prepare_signal_adapter(h,FixedSignalConfig(rep,method=method,T=.8,q=8,L_D=1 if method=='PR' else None,r=4,K=4))
            norm=dict(B=1.,log_B=0.,log_B_upper=1.,lambda_r=a.lambda_r)
            if a.randomized:
                import math
                norm['B']=math.exp(8*a.log_B_step)
            results[method]=dict(model=a.metadata(),config=dict(L_D=a.ld),signal=dict(signed_bias=dict(real=0.,imag=0.)),normalization=norm)
        # Literal independent supplied full matrix. No adapter.operators call.
        full=np.array([[.375,.125,0,0],[.125,.375,0,0],[0,0,-.125,.125],[0,0,.125,-.125]],complex) if rep=='pauli_jw' else np.diag([.125,.5,.5,.875]).astype(complex)
        model=b.build_export(p,d,state,full,results,targets,rep,c,'ARTIFICIAL_NOT_OLD_H4')
        model['artificial_only']=True;model['binding_hash']=c.digest({k:v for k,v in model.items() if k!='binding_hash'})
        c.save(destination/(rep+'.json'),model)
    print(json.dumps(dict(status='ARTIFICIAL_LEGACY_PREPARATION_EXPORTED',qubits=2,real_H4_calls=0)))
if __name__=='__main__':main()
