#!/usr/bin/env python3
"""Explicitly gated C production CLI. No H4 evaluation without a new valid grant."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]


def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    for key in ('inspect','issue-grant','execute','worker','bridge','witness'):g.add_argument('--'+key,type=Path)
    p.add_argument('--seal',type=Path);p.add_argument('--namespace',type=Path)
    p.add_argument('--representation',choices=['pauli_jw','df_native']);p.add_argument('--command-json')
    args=p.parse_args();ctl=load('C_control',ROOT/'src/trotterlib/h4_certificate_control.py')
    if args.inspect:
        c=ctl.validate_contract(ctl.read(args.inspect));print(json.dumps(dict(status='C_CONTRACT_INSPECTED_NO_DISPATCH',execution_authorized=c['execution_authorized'],grant_valid=c['grant_valid'],dispatch_enabled=c['dispatch_enabled'])));return
    if args.witness:
        if args.command_json is None:p.error('witness requires child command')
        sys.exit(ctl.witness(args.witness,json.loads(args.command_json)))
    if args.issue_grant:
        if not args.seal or not args.namespace:p.error('issue-grant requires --seal and --namespace')
        print(json.dumps(ctl.issue_grant(args.issue_grant,args.seal,args.namespace)));return
    if args.execute:print(json.dumps(ctl.launch(args.execute)));return
    ns=args.worker or args.bridge
    if not args.representation:p.error('worker/bridge requires representation')
    if args.bridge:
        bridge=load('C_legacy_export',ROOT/'src/trotterlib/h4_legacy_export.py')
        model=bridge.export(ns,args.representation,ctl)
        ctl.save(ns/(args.representation+'_model.json'),model);return
    c,seal=ctl.worker_authorize(ns,args.representation)
    budget=ctl.CallBudget(ns)
    try:
        budget.take('operator_preparation_reconstruction_batches')
        command=[c['legacy_python'],'-B',str(Path(__file__).resolve()),'--bridge',str(ns),'--representation',args.representation]
        # Fresh bridge witness persists its own native exit even if this worker dies.
        witness_cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'--witness',str(ns/(args.representation+'_bridge_native')),'--command-json',json.dumps(command)]
        import subprocess
        native=subprocess.run(witness_cmd,check=False)
        ctl.save(ns/(args.representation+'_bridge_witness_exit.json'),dict(exit_code=native.returncode,reaped=True,UTC=ctl.utc()))
        ctl.require(native.returncode==0,'LEGACY_BRIDGE_FAILED')
        budget.take('probability_mapping_batches')
        engine=load('C_ball_engine',ROOT/'src/trotterlib/h4_ball_evidence.py')
        model=ctl.read(ns/(args.representation+'_model.json'))
        numerical=engine.EnclosureEngine(model,budget)
        def record(bits,result):ctl.save(ns/(args.representation+'_tier_'+str(bits)+'.json'),result)
        def evaluate(bits):
            ctl.verify_seal(seal)
            return numerical.tier(bits)
        result=ctl.precision_schedule(evaluate,on_tier=record)
        ctl.verify_seal(seal)
        ctl.save(ns/(args.representation+'_result.json'),result)
        ctl.save(ns/(args.representation+'_worker_STOP.json'),dict(status='C_NUMERICAL_DIAGNOSTIC_STOP',numerical_status=result['status'],formal_certificate_accepted=False,old_N_G_changed=False))
    except BaseException as e:
        ctl.save(ns/(args.representation+'_partial.json'),dict(status='FAILED_NO_RETRY',error=repr(e),UTC=ctl.utc()))
        raise


if __name__=='__main__':main()
