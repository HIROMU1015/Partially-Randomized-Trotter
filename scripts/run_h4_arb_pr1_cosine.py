#!/usr/bin/env python3
"""Reuse PR0 one-shot controller; PR1-specific saved-order2 static preflight."""
import argparse
import ast
import multiprocessing.popen_fork  # Pin the actual pair-worker startup code in runtime().
from pathlib import Path
import runpy
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
pkg=types.ModuleType('arb_pr1_run_private');pkg.__path__=[str(ROOT/'src/trotterlib')];sys.modules['arb_pr1_run_private']=pkg
from arb_pr1_run_private import h4_arb_pr1_cosine as A
C,B,BASE=A.C,A.B,A.BASE


def controller():
    old=runpy.run_path(str(ROOT/'scripts/run_h4_arb_pr0_cosine.py'))
    env={**old,**globals()}
    env['CODE']=old['CODE']+['src/trotterlib/h4_arb_pr1_cosine.py','src/trotterlib/h4_r1_order2.py',
        'scripts/run_h4_arb_pr1_cosine.py','tests/test_h4_arb_pr1_cosine.py',
        'tests/test_h4_r1_order2.py','tests/test_h4_r0_empirical.py',
        'artifacts/non_df_h4_central_arb_pr1_cosine/2026-10-25/prepare_metadata.py']
    tree=ast.parse((ROOT/'scripts/run_h4_arb_pr0_cosine.py').read_text())
    names=('resources','runtime','verify','prepare','check','worker','native_witness','launch')
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    C.require(len(nodes)==len(names),'REUSED_PR0_CONTROLLER_FUNCTIONS')
    module=ast.fix_missing_locations(A.ScopeStrings().visit(ast.Module(body=nodes,type_ignores=[])))
    exec(compile(module,'<PR0 authority/native controller with PR1 scope>','exec'),env)
    return env


def inspect():
    C.hashes({str(ROOT/p):h for p,h in C.read(BASE/'inputs.json')['files'].items()})
    target,model,join,circuits,binding=A.R1.preflight()  # Existing static-only R1 phase/SOURCE checks.
    plan=A.build_plan(model,join)
    program=B.descriptor_from_circuit(circuits['compiled_qpy'],axis='cosine')
    program['origin']=dict(qpy_sha256=C.sha(ROOT/target['compiled_qpy']),fingerprint=target['output_fingerprint'],
        decode_only=True,state_actions=0,compile=0)
    C.require(program['origin']['qpy_sha256']==A.QPY_SHA and program['origin']['fingerprint']==A.QPY_FP,'FIXED_PR1_COMPILED')
    guard=next(g for g in C.read(A.R1.P.BASE/'guard_inputs_quoted.json')['axes']
        if (g['case'],g['epsilon_sig'],g['axis'])==('pauli_jw_PR','0.0005','cosine'))
    state=C.read(A.R1.R0BASE/'run01/state_identity.json')
    C.save(BASE/'PR1_target_QUOTED.json',target);C.save(BASE/'PR1_guard_QUOTED.json',guard)
    C.save(BASE/'PR1_selected_C_plan_NO_EXECUTION.json',plan)
    C.save(BASE/'PR1_compiled_program_NO_EXECUTION.json',program);C.save(BASE/'state_identity_QUOTED.json',state)
    C.save(BASE/'structural_inspection.json',binding)
    A.load_inputs()
    C.save(BASE/'structural_preflight.json',dict(status='PASS_PR1_ACTUAL_QPY_PLAN_DESCRIPTOR_IDENTITY',QPY_decodes=4,
        state_actions=0,new_compile=0,compiled_sha256=A.QPY_SHA,fingerprint=A.QPY_FP,
        C_target_unscaled=True,B_saved_exact=A.SCALE,trajectory_id=A.TID,
        order2=binding['order2'],exact_plan_sha256=C.sha(BASE/'PR1_selected_C_plan_NO_EXECUTION.json'),
        descriptor_sha256=C.sha(BASE/'PR1_compiled_program_NO_EXECUTION.json')))
    # Complete the actual input manifest before scientific SOURCE/runtime/grant.
    inputs=C.read(BASE/'inputs.json')
    for name in ('PR1_target_QUOTED.json','PR1_guard_QUOTED.json','PR1_selected_C_plan_NO_EXECUTION.json',
            'PR1_compiled_program_NO_EXECUTION.json','state_identity_QUOTED.json'):
        inputs['files'][str((BASE/name).relative_to(ROOT))]=C.sha(BASE/name)
    C.save(BASE/'inputs.json',inputs,exclusive=False)
    print(A.R1.R.json.dumps(dict(status='PASS_PR1_STATIC_ONLY',QPY_decodes=4,state_actions=0)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    for mode in ('inspect','prepare','launch','worker'):group.add_argument('--'+mode,action='store_true')
    args=parser.parse_args();env=controller()
    if args.inspect:inspect()
    elif args.prepare:env['prepare']()
    elif args.launch:sys.exit(env['launch']())
    else:env['worker']()
