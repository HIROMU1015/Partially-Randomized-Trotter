#!/usr/bin/env python3
"""Metadata inspection only. --execute always refuses, before reading H4."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
pkg=types.ModuleType('rho_prepare_private');pkg.__path__=[str(ROOT/'src/trotterlib')]
sys.modules['rho_prepare_private']=pkg
from rho_prepare_private import h4_central_rho_plan as P
C=P.C


def inspect_saved():
    targets,files,context=P.metadata_targets()
    P.BASE.mkdir(parents=True,exist_ok=True)
    C.save(P.BASE/'targets.json',dict(status='METADATA_AND_SHA_ONLY',targets=targets,real_H4_numeric_calls=0))
    from rho_prepare_private.h4_conditional_cng import csv_save
    csv_save(P.BASE/'targets.csv',targets)
    C.save(P.BASE/'contracts_DRAFT.json',P.drafts(targets))
    C.save(P.BASE/'state_ray.json',{k:v for k,v in context.items() if k!='guard_rows'})
    guard=[{k:r[k] for k in ('case','epsilon_sig','axis','rho_guard','rho_guard_exact','u_enc_exact','signed_bias_exact','scale')}
           for r in context['guard_rows']]
    C.save(P.BASE/'guard_inputs_quoted.json',dict(status='UNCHANGED_SAVED_VALUES_NO_RECALCULATION',axes=guard))
    for name in ('non_df_h4_d2_execution','non_df_h4_a_gate_order2','non_df_h4_a_axes'):
        for rel in ('seal.json','active_contract.json','verification.json','run01/grant.json','run01/call_ledger.json','run01/STOP.json','run01/native/STOP.json'):
            p=f'artifacts/{name}/2026-10-24/{rel}'
            C.require((ROOT/p).is_file(),'MISSING_OLD_AUDIT:'+p);files[p]=C.sha(ROOT/p)
    code=['src/trotterlib/'+n+'.py' for n in ('h4_d2_pauli_control','h4_a_gate','h4_a_cohort','h4_a_axes',
          'h4_central_cng','h4_conditional_cng','fixed_signal_evaluation','h4_ball_evidence','h4_local_enclosure',
          'h4_enclosure_prototype','h4_d1_adapter','h2_non_df_pilot_validation','h4_certificate_control','rte_event_audit','rte')]
    files.update({p:C.sha(ROOT/p) for p in code})
    C.save(P.BASE/'inputs.json',dict(kind='RHO_PREPARATION_METADATA_INPUTS',files=files,
        real_QPY_or_NPZ_decodes=0,scalar_error_or_shot_recalculations=0))
    preservation=dict(files)
    old='artifacts/non_df_h4_central_conditional_cng/2026-10-24'
    paths=subprocess.check_output(['git','ls-files','--',old],cwd=ROOT,text=True).splitlines()
    preservation.update({p:C.sha(ROOT/p) for p in paths})
    C.save(P.BASE/'preservation_before.json',preservation)
    C.save(P.BASE/'manifest_before.json',C.read(ROOT/'artifacts/validation_manifest.json'))
    C.require(not any(k.startswith(('qiskit','numpy','flint','pyscf','openfermion')) for k in sys.modules),'METADATA_IMPORTED_NUMERIC_BACKEND')
    C.save(P.BASE/'inspection.json',dict(status='PASS_METADATA_ONLY',targets=6,configurations=3,
        input_files=len(files),preservation_files=len(preservation),QPY_decodes=0,NPZ_decodes=0,
        quantum_or_numeric_backend_imports=0,real_state_actions=0,real_signal_evaluations=0,
        new_scientific_SOURCE=None,scientific_grants_issued=0,formal_certificate_accepted=False,
        policy_10_3='ABSENT_IN_AVAILABLE_2026-10-23_COPY; explicit current instruction applied',mandatory_STOP=True))
    print(json.dumps(dict(status='PREPARATION_METADATA_COMPLETE',targets=6,inputs=len(files),preserved=len(preservation),real_H4_calls=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--inspect',action='store_true');g.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if args.execute:
        try:P.execution_request()
        except PermissionError as e:p.exit(2,str(e)+'\n')
    else:inspect_saved()
