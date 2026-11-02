"""Reproduce one unfinished SIGBUS trial with Python fatal-stack diagnostics.

Same fixed row/cohort/index/seed and scientific kernel, isolated diagnostic
output. This unfinished replay is not a new independent cost observation.
"""
import faulthandler
from pathlib import Path
import h46_b3 as b
from h46_b3_continue import cached_context


if __name__=='__main__':
    faulthandler.enable(all_threads=True)
    native=b.setup()
    row=next(r for r in b.read(b.PUBLIC/'native_selection.json')['candidates'] if r['id']=='2bba0344f5b65e19ead1')
    native.CONTEXT['H4']=cached_context('H4',[0])
    native.PRIVATE=b.PRIVATE/'diagnostic_SIGBUS_H4'
    native.PRIVATE.mkdir(exist_ok=True)
    result=native.cost_trial((row,'B3_EXPLORATION',0))
    b.save(native.PRIVATE/'diagnostic_result.json',dict(status=result['status'],same_fixed_seed_replay=True,
        adopted_into_main_cost_cohort=False,independent_cost_observation=False))
