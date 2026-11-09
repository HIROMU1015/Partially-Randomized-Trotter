"""T20 runner with the corrected action-only Aer adapter.

The original 8ca7233 execution and its technical STOP records stay intact.
No compiler, sampling, candidate, numerical or shot policy changes.
"""
import h46_full_h_t20_opt as study
from h46_full_h_t20_aer import simulate_statevector
study.aer_statevector=simulate_statevector
if __name__=='__main__':study.main()
