#!/usr/bin/env python3
"""Pin metadata for the user's H8 preflight / H4 supplement permission.

Does not execute science. Every phase receives a fresh exclusive output/grant.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.h48_preflight_contract_v1 import (
    PREP,h4_plan,h8_input_plan,source_paths,profile,resources,observation_policy,
    aggregate_caps,registrations,verify_binding)
from trottertracks.resource_applicability.h4_common_panel_port_v1 import h4_parent
from trottertracks.resource_applicability.ax2b_h4_contract_v5 import snapshot_binding,file_hash
from trottertracks.resource_applicability.ax2a_preparation import digest

def publish(path,data):
    with Path(path).open('x') as f:f.write(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scope',choices=('H4_SIGNAL','H4_COST','H8_INPUT'),required=True)
    p.add_argument('--source',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--signals',type=Path)
    args=p.parse_args()
    if len(args.source)!=40:raise ValueError('SOURCE_SHA')
    subprocess.run(['git','merge-base','--is-ancestor',args.source,'HEAD'],cwd=ROOT,check=True)
    out=args.output.resolve()
    if out.exists() or (out.parent/(out.name+'.one_shot_lock')).exists():raise ValueError('EXCLUSIVE_OUTPUT_REQUIRED')
    base=ROOT/PREP;base.mkdir(parents=True,exist_ok=True)
    snap=snapshot_binding(ROOT)
    a=base/'h4_snapshot_receipt.json';b=base/'h4_df_receipt.json'
    if not a.exists():publish(a,snap)
    if not b.exists():publish(b,{'sha256':snap['sha256'],'policy':'existing rank12 input unchanged'})
    scope=args.scope
    m={'schema':'h48_one_shot_bound_phase_v1','scope':scope,'kind':scope,
       'source_commit':args.source,'source_hashes':{name:file_hash(ROOT/name) for name in source_paths(ROOT)},
       'profile':profile(),'plan':h8_input_plan() if scope=='H8_INPUT' else h4_plan(scope),
       'proposed_resources':resources(8 if scope=='H4_COST' else 1),
       'observation_policy':observation_policy(),'base_manifest':{'plan':{'caps_proposed':aggregate_caps()}},
       'snapshot':snap,'input_identity':h4_parent(ROOT),'output':str(out),
       'registrations':[{'index':0,'task':{'kind':scope}}],
       'H8_full_campaign_authorized':False,
       'preserved_H6_carry':{'known_compile':1716,'reserved_compile':20,'known_trajectory':842,
                           'reserved_trajectory':10,'refund':0,'reset':False},
       'H4_history':{'M1_actual_compile':12128,'M1_random_trajectory':6208,
                     'other_historical_actual':None,'no_reset_or_refund':True,
                     'production_H4_budget':'separate unchanged running lineage'},
       'phase_new_budget':{'H4_SIGNAL':{'signal_candidates':16,'compile':0,'trajectory':0},
                           'H4_COST':{'compile_upper':532,'trajectory_upper':256,'occurrence_upper':960},
                           'H8_INPUT':{'integral_build':1,'df_decomposition':1,'solver_matvec':10000,
                                       'compile':0,'trajectory':0}}[scope]}
    if scope=='H4_COST':
        if args.signals is None:raise ValueError('COMPLETE_SIGNAL_PHASE_REQUIRED')
        f=args.signals.resolve();signal_data=json.loads(f.read_text())
        terminal=json.loads((f.parent/'worker_terminal.json').read_text())
        if terminal.get('status')!='COMPLETE' or terminal.get('source_commit')!=args.source or terminal.get('scope')!='H4_SIGNAL':
            raise ValueError('SIGNAL_TERMINAL_REQUIRED')
        m['signal_summary_path']=str(f);m['signal_summary_sha256']=file_hash(f)
        m['registrations']=registrations(m,signal_data['signals'])
    grant={'schema':'h48_one_shot_user_authorization_v1','approved_by_user':True,
       'manifest_sha256':digest(m),'output':str(out),'H8_full_campaign_authorized':False,
       'authorization_basis':['準備ができたら開始してよい','開始対象は1と3（H8小規模事前検証・H4比較補完）'],
       'retry':False,'automatic_next_full_campaign':False,'fresh_resources_required':True}
    verify_binding(ROOT,m,grant)
    args.bundle.mkdir(parents=True,exist_ok=False)
    publish(args.bundle/'manifest.json',m);publish(args.bundle/'grant.json',grant)
    (args.bundle/'SHA256SUMS').write_text('\n'.join(file_hash(args.bundle/name)+'  '+name for name in ('manifest.json','grant.json'))+'\n')
    print(json.dumps({'scope':scope,'source':args.source,'bundle':str(args.bundle),
         'output':str(out),'registrations':len(m['registrations']),'new_science_executed':False}))

if __name__=='__main__':main()
