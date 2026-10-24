"""Quote six saved local enclosures; format endpoints, no science imports/calls."""
import csv
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,o):p.write_text(json.dumps(o,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
def decimal(q,*,upper=False):
    q=Q(q)
    with localcontext() as ctx:
        ctx.prec=24;ctx.rounding=ROUND_CEILING if upper else ROUND_FLOOR
        return str(Decimal(q.numerator)/Decimal(q.denominator))
def display_interval(interval):return dict(lower=decimal(interval['lower']),upper=decimal(interval['upper'],upper=True))

def summarize():
    paths={
        'PF_cosine':ROOT/'artifacts/non_df_h4_central_arb_pf_cosine/2026-10-25/run01',
        'PF_sine':BASE/'run01/PF_sine',
        'PR0_cosine':ROOT/'artifacts/non_df_h4_central_arb_pr0_cosine/2026-10-25/run01',
        'PR0_sine':BASE/'run01/PR0_sine',
        'PR1_cosine':ROOT/'artifacts/non_df_h4_central_arb_pr1_cosine/2026-10-25/run01',
        'PR1_sine':ROOT/'artifacts/non_df_h4_central_arb_pr1_sine/2026-10-25/run01'}
    rows=[]
    for key,ns in paths.items():
        result=read(ns/'result_summary.json');last=result['last'];rho=last['rho'];state=read(ns/'state_target_binding.json')
        signal=last['signal'] if key.startswith('PF_') else last['compiled']
        target=state['C_target_interval_exact'] if key.startswith('PF_') else last['target']['interval']
        nonnegative=dict(lower=str(max(Q(0),Q(rho['interval']['lower']))),upper=rho['interval']['upper'])
        row=dict(key=key,provenance='NEW' if key in ('PF_sine','PR0_sine') else 'OLD_QUOTED_NO_REEVALUATION',
            SOURCE=result['SOURCE'],result_summary=str((ns/'result_summary.json').relative_to(ROOT)),result_sha256=sha(ns/'result_summary.json'),
            state_binding_sha256=sha(ns/'state_target_binding.json'),STOP_sha256=sha(ns/'STOP.json'),
            C_target_interval_exact=target,compiled_interval_exact=signal['interval'],rho_raw_interval_exact=rho['interval'],
            rho_nonnegative_interval_exact=nonnegative,rho_upper_exact=nonnegative['upper'],rho_budget_exact=rho['rho_budget_exact'],
            rho_upper_decimal_outward=decimal(nonnegative['upper'],upper=True),rho_budget_decimal_outward=decimal(rho['rho_budget_exact'],upper=True),
            precision_bits=[t['precision_bits'] for t in result['tiers']],state_action_attempts=result['state_action_attempts'],
            residual_reductions=result['residual_reductions'],relation=result['relation'],B_saved_exact=rho['B_saved_exact'],
            raw_norm_exact=state['raw_norm_exact'],formal_certificate_accepted=False,PR_law_rho=None)
        assert result['relation']=='UPPER_WITHIN_BUDGET' and not result['formal_certificate_accepted']
        assert read(ns/'STOP.json')['status']=='COMPLETE_MANDATORY_STOP'
        rows.append(row)
    save(BASE/'six_local_results.json',dict(status='SIX_LOCAL_ENCLOSURES_COMPLETE',rows=rows,new_evaluations=2,
        old_four_operation='quote saved intervals and native status only, no scientific reevaluation',
        state_actions_this_run=3,residual_reductions_this_run=2,formal_certificate_accepted=False,PR_law_rho=None,
        old_N_G_changed=False,original_native='UNKNOWN_UNCHANGED',mandatory_STOP=True))
    columns=['key','provenance','rho_upper_exact','rho_upper_decimal_outward','rho_budget_exact','rho_budget_decimal_outward','precision_bits','state_action_attempts','relation','SOURCE','result_summary']
    with (BASE/'six_local_results.csv').open('w',newline='') as f:
        out=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');out.writeheader();out.writerows(rows)
    md='| 対象 | 軸 | 局所ρ上端（外向き表示） | 保存予算（概数） | bit | 状態作用 | 来歴 |\n|---|---|---:|---:|---:|---:|---|\n'
    for row in rows:
        name,axis=row['key'].split('_');md+=f"| {name} | {axis} | {row['rho_upper_decimal_outward']} | {row['rho_budget_decimal_outward']} | {row['precision_bits'][0]} | {row['state_action_attempts']} | {'今回' if row['provenance']=='NEW' else '保存値引用'} |\n"
    (BASE/'six_local_results.md').write_text(md)
    new=[r for r in rows if r['provenance']=='NEW'];display={}
    for row in new:
        ns=paths[row['key']];result=read(ns/'result_summary.json');last=result['last'];sig=last.get('signal',last.get('compiled'))
        display[row['key']]=dict(C_target=display_interval(row['C_target_interval_exact']),compiled=display_interval(row['compiled_interval_exact']),
            rho_raw=display_interval(row['rho_raw_interval_exact']),rho_nonnegative=display_interval(row['rho_nonnegative_interval_exact']),
            compiled_width_exact=sig['interval_width_exact'],compiled_width_decimal_outward=decimal(sig['interval_width_exact'],upper=True),
            rho_width_exact=last['rho']['interval_width_exact'],rho_width_decimal_outward=decimal(last['rho']['interval_width_exact'],upper=True),
            compiled_wall_seconds=sig['wall_seconds'],C_target_wall_seconds=0 if row['key']=='PF_sine' else last['target']['wall_seconds'])
    save(BASE/'reported_intervals.json',dict(operation='outward decimal formatting only; exact raw results authoritative',precision_decimal_digits=24,targets=display))
    save(BASE/'result_inventory.json',dict(scope='new run01 files only, finalized after native STOP',
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted((BASE/'run01').rglob('*')) if p.is_file()}))
    return rows

if __name__=='__main__':
    rows=summarize();print(json.dumps(dict(status='SAVED_ONLY_SUMMARY_COMPLETE',rows=len(rows),new_state_actions=3)))
