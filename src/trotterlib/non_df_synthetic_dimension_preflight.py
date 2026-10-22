"""Supplied synthetic eight-qubit circuits; never an H4 molecule."""
import os
from pathlib import Path
import time

import numpy as np
from scipy.linalg import expm

from .df_hamiltonian import DFHamiltonian
from .fixed_signal_evaluation import FixedSignalConfig,evaluate_fixed_signal,prepare_signal_adapter
from .h2_non_df_pilot import dense_law_mean
from .isolated_qiskit_execution import save
from .non_df_h4_readiness import structural_prefix
from .pauli_signal_adapter import PauliHamiltonian


def synthetic_dimension_case(name,destination):
    started=time.time()
    rep,method=name.rsplit("_",1)
    n=8
    if rep=="pauli_jw":
        terms=[("IIIIIIII",.05)]
        for i in range(n):
            label=["I"]*n;label[i]="Z";terms.append(("".join(label),.1/(i+1)))
        terms += [("XXIIIIII",.03),("IYYIIIII",-.02),("IIXXIIII",.015),("IIIZZIII",.01)]
        h=PauliHamiltonian.from_terms(terms,model_id="synthetic_eight_qubit_no_molecule")
        weights=[abs(c) for _,c in h.terms]
    else:
        one=np.diag(np.linspace(-.1,.1,n));one[0,1]=one[1,0]=.007
        gs=[]
        for i in range(3):
            g=np.diag(np.linspace(.02*(i+1),.09*(i+1),n));g[i,i+1]=g[i+1,i]=.004
            gs.append(g)
        h=DFHamiltonian(.05,one,np.array([.2,-.1,.05]),tuple(gs),dict(basis="supplied artificial arrays, no integrals"))
        from .df_partial_randomized_pf import rank_df_fragments
        weights=[f.weight for f in rank_df_fragments(h)]
    psi=np.zeros(1<<n,complex);psi[3]=1/np.sqrt(2);psi[12]=1j/np.sqrt(2)
    config=FixedSignalConfig(rep,method=method,T=.1,q=1,r=2,K=4,
        L_D=structural_prefix(weights) if method=="PR" else None,seed=1701,cost_samples=2)
    # Finite supplied artificial workload, independent of any H4 candidate.
    result=evaluate_fixed_signal(h,psi.copy(),config,epsilon_sig=.05,
        numerical_allowance={"cosine":1e-9,"sine":1e-9},log_B_margin=1e-9,
        evidence_kind="EMPIRICAL",evidence_ref="synthetic-dimension-check-only",verify_compiled=True,
        state_metadata=dict(kind="artificial_supplied_state",molecular=False),source_provenance=dict(scope="ARTIFICIAL_ONLY"))
    adapter=prepare_signal_adapter(h,config)
    full=adapter.operators()[0].toarray()
    exact=complex(np.vdot(psi,expm(-1j*config.T*full)@psi))
    mean=complex(np.vdot(psi,dense_law_mean(adapter,q=config.q)@psi))
    z=result["reference"]["signal"];s=result["signal"]["corrected"]
    errors=dict(reference=abs(exact-complex(z["real"],z["imag"])),
                law_mean=abs(mean-complex(s["real"],s["imag"])),
                compiled_probability=result["circuit_cost"]["compiled_signal_max_error"])
    if errors["reference"]>1e-10 or errors["law_mean"]>1e-10 or errors["compiled_probability"]>1e-9:
        raise ValueError("SYNTHETIC_DIMENSION_GATE")
    save(Path(destination)/"result.json",dict(status="PASS",case=name,artificial_only=True,
        molecule=None,num_system_qubits=n,num_wrapper_qubits=n+1,pid=os.getpid(),started=started,
        ended=time.time(),errors=errors,config=result["config"],CNG=result,
        claim="finite artificial dimension/interface check only; no H4 scientific validation"))
