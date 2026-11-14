"""Bind six published complete packets with unchanged B intake; no risk/N calls.

Run from a private detached checkout of B 9c0e12d87 (never B's active worktree).
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

B='9c0e12d87bf83429e3411d295b16c47d913283d2'
DIR='docs/research/pf_shortlist_mean_completion_20261115/run01/'
LABELS=('FULL_PF_S2_16h_over_7','FULL_PF_Y4_16h_over_3','FULL_PF_M5_16h',
        'B0I_P7_S2_16h_over_7','B0I_P7_Y4_16h_over_3','B0I_P7_M5_16h')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--b-root',required=True);ap.add_argument('--packets-commit',required=True)
    args=ap.parse_args();root=Path(args.b_root).resolve()
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()!=B:
        raise ValueError('Pinned B checkout required')
    sys.path[:0]=[str(root/'src'),str(root/'tests')]
    from trottertracks import rpe_total_binding as T
    rows=[]
    for label in LABELS:
        packet_source=dict(commit=args.packets_commit,path=DIR+label+'/B_SIGNAL_INPUT.json')
        config_source=dict(commit=args.packets_commit,path=DIR+label+'/PLAN.json',selector=['configuration'])
        packet=T._selected(root,packet_source);config=T._selected(root,config_source)
        row=dict(label=label,configuration_id=config['id'],packet_source=packet_source,configuration_source=config_source)
        try:
            bridge=T.create(root,config,original_source=config_source)
            accepted=T.bind_signal(root,bridge,packet,original_packet_source=packet_source)
            if len(accepted['signal_packet']['means'])!=13 or any(v['raw'] is None for v in accepted['signal_packet']['means']):
                raise ValueError('Incomplete accepted means')
            kept=all(a['raw']==b['raw'] and a['corrected']==b['corrected'] and a['normalization']==b['normalization']
                     for a,b in zip(packet['means'],accepted['signal_packet']['means']))
            if not kept:raise ValueError('Intake mutated signal values')
            row.update(intake='PASS',mean_values_preserved=True,total_view_configuration_id=accepted['configuration']['id'],science_calls=accepted['science_calls'])
        except Exception as e:
            row.update(intake='FAIL',type=type(e).__name__,reason=str(e))
        rows.append(row)
    value=dict(B_source=B,packets_commit=args.packets_commit,rows=rows,
        new_mean_calls=0,risk_MSE_N_G_calls=0,sufficient_N=None,global_RMSE_bound=None,G=None,
        all_intake_pass=all(r['intake']=='PASS' for r in rows),mandatory_stop=True)
    print(json.dumps(value,ensure_ascii=False,indent=2))
    return 0 if value['all_intake_pass'] else 2


if __name__=='__main__':sys.exit(main())
