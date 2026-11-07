"""Conditional RPE branch/energy checks; never promotes a native guarantee."""
import json
import math
import subprocess
import numpy as np
from .track_c_mixed_formula import ROOT,OUT,common,save_json
from .track_c_h4_finite_bits import proposal_budget
from trotterlib.rpe_four_round_phase_validation import reconstruct_rpe_phase

def run():
    target=OUT/'RPE_CHECKS.json'
    if target.exists():raise RuntimeError('Do not overwrite RPE checks')
    summary=json.loads((OUT/'FULL_SUMMARY.json').read_text());c=common();records=[]
    for name in sorted({r['name'] for r in summary['choices']}):
        rows=[r['best'] for r in summary['choices'] if r['name']==name]
        assert len(rows)==18 and all(r is not None and r['feasible'] for r in rows)
        phases=[math.atan2(r['raw_signal']['imag'],r['raw_signal']['real']) for r in rows]
        decoded=reconstruct_rpe_phase(phases,[1<<m for m in range(18)])
        energy=-decoded['final_phase_estimate']/.02
        # E0 is used only after the oracle-free branch decoder returns.
        error=abs(energy-c['energy'])
        correct_branch=all(abs(z['base_phase_estimate']+.02*c['energy'])<=.4/(1<<m)+1e-10 for m,z in enumerate(decoded['rounds']))
        finite_bits=[]
        for r in rows:
            finite_bits.append(proposal_budget(0.,4*r['RTE_events'],218))
        records.append(dict(name=name,selected_rows=rows,source_mean_decode=decoded,decoded_energy=energy,energy_error=error,
            correct_branch_at_all_rounds=correct_branch,source_mean_energy_pass=error<=1.5936e-4,
            shots_total_X_plus_Y=sum(2*r['shots_per_axis'] for r in rows),
            primitive_surrogate_total=sum(r['primitive_shot_surrogate'] for r in rows),finite_bits=finite_bits))
    save_json(target,dict(status='CONDITIONAL_SOURCE_BRANCH_COMPLETE_NATIVE_QUALIFICATION_PENDING',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),records=records,
        delta0=.02,epsilon_energy=1.5936e-4,total_alpha=.05,alpha_per_axis=.05/36,total_phase_budget=.4,
        Hoeffding='N=ceil(4 log(1440)/(rho_lower^2 sin(beta_stat)^2)), independent fresh bounded X/Y outcomes.',
        branch_condition=3*.4<math.pi,energy_on_good_event_upper=.4/(.02*(1<<17)),
        interpretation='With a qualified complete signal disk and fresh bits, the union bound would be >=.95 success. These new families have conditional float64 source disks only: no native .95 theorem or experimental shots claimed.',
        sampling='Finite-bit proposal budgets are diagnostics; their small disks have not been added to a new native interval certificate.',
        price='Primitive totals mix no native gate metrics and are not quantum prices; no absolute G is certified.'))
    print('RPE_CHECKS_COMPLETE',flush=True)

if __name__=='__main__':run()
