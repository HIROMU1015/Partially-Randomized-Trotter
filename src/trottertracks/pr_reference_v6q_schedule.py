"""Bind the fixed saved schedule to an opt-in Reference generation adapter.

No schedule, science input or quantum library is loaded on import. Calling
assemble_reference_for_saved_round on scientific events builds circuits and
needs separate execution permission. The adapter does not sample events or
compile circuits: supplied events must follow the preserved independent-bit
canonical law at the validated bit width, using the fixed cache/configuration.
"""
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import subprocess

from .pr_reference_v6q import phase_for_round

ROOT = Path(__file__).resolve().parents[2]
SCHEDULE_COMMIT = 'f2638a7845685a119619382b850441f5d8d33713'
SCHEDULE_PATH = 'artifacts/pr_reference_transfer_qualification/2026-11-06/STAGE4_RPE.json'


@dataclass(frozen=True)
class SavedRound:
    m: int
    q: int
    r: int
    bits: int


def _parse_schedule(document):
    """Project saved metadata only; never evaluate shots, signals or costs."""
    rows = document.get('rows')
    if not isinstance(rows, list) or len(rows) != 18:
        raise ValueError('The fixed schedule must contain exactly 18 rows')
    schedule = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not all(type(row.get(k)) is int for k in ('m', 'q', 'r', 'bits')):
            raise ValueError('Missing or noninteger round metadata')
        bound = SavedRound(*(row[k] for k in ('m', 'q', 'r', 'bits')))
        if bound.m != index or bound.q != 1 << index or bound.r < 1 or bound.bits < 1:
            raise ValueError('Missing, duplicate or inconsistent round metadata')
        schedule.append(bound)
    if schedule[17] != SavedRound(17, 131072, 32, 88):
        raise ValueError('The saved m17 does not match the qualified scope')
    return tuple(schedule)


@lru_cache(maxsize=1)
def saved_schedule():
    """Read the one immutable published schedule; no alternate ref or fill."""
    raw = subprocess.check_output(['git', 'show', SCHEDULE_COMMIT+':'+SCHEDULE_PATH], cwd=ROOT)
    return _parse_schedule(json.loads(raw))


def bind_round(m, *, q, r, bits, mode='legacy'):
    """Match caller metadata to the saved row before any scientific call.

    exact_binary_m17 is an opt-in policy for the schedule: its other 17 rows
    still select legacy. The underlying provider's exact mode is m17 only.
    """
    if mode not in ('legacy', 'exact_binary_m17'):
        raise ValueError('Unknown schedule phase mode')
    if type(m) is not int or not 0 <= m < 18:
        raise ValueError('Unknown round')
    if not all(type(v) is int for v in (q, r, bits)):
        raise ValueError('Round metadata must be builtin integers')
    expected = saved_schedule()[m]
    if SavedRound(m, q, r, bits) != expected:
        raise ValueError('Caller q/r/bits does not match the saved schedule')
    selected = 'exact_binary_m17' if mode == 'exact_binary_m17' and m == 17 else 'legacy'
    return expected, selected


def _legacy_modules():
    from . import pr_reference_interference_pilot as pilot
    from . import pr_reference_transfer_qualification as transfer
    return pilot, transfer


def assemble_reference_for_saved_round(m, *, q, r, bits, cfg, dist, events, mode='legacy'):
    """Reuse old assemble and gadget; change only the selected m17 P angle.

    Input metadata/configuration must describe the same fixed H4 SOURCE.
    Event-law provenance cannot be inferred from a single supplied trajectory.
    No event is drawn, copied, reordered or rescaled here. The old assemble
    still computes its legacy phase; reuse it rather than evaluating it twice.
    """
    metadata, selected = bind_round(m, q=q, r=r, bits=bits, mode=mode)
    if (cfg.rte_steps != metadata.r or cfg.evolution_time != .02
            or cfg.finite_taylor_order != 2 or dist.finite_taylor_order != 2
            or dist.orders != (0, 2)
            or cfg.dimensionless_step_time != dist.dimensionless_step_time):
        raise ValueError('RTE configuration does not match the fixed schedule/SOURCE')
    if len(events) != metadata.q*metadata.r:
        raise ValueError('Event sequence length does not match saved q*r')
    pilot, transfer = _legacy_modules()
    assembly = transfer.assemble(metadata.q, metadata.r, cfg, dist, events)
    f, g = assembly['F'], assembly['G']
    if f.num_qubits != 8 or g.num_qubits != 8:
        raise ValueError('Fixed H4 lanes must both have width 8')
    phase = (assembly['phase'] if selected == 'legacy' else
             phase_for_round(f, g, round_m=metadata.m, mode=selected))
    core = pilot.reference_core(f, g, phase)
    return dict(metadata=metadata, phase_mode=selected, assembly=assembly, phase=phase, core=core)


def reference_axes_for_saved_round(m, *, q, r, bits, cfg, dist, events, mode='legacy'):
    """Use one selected core/phase for both unchanged X/Y wrappers; no compile.

    The old downstream compiler/reset convention is external to this adapter;
    no reset, control, inverse or gate optimization is added here.
    """
    result = assemble_reference_for_saved_round(m, q=q, r=r, bits=bits,
        cfg=cfg, dist=dist, events=events, mode=mode)
    pilot, _ = _legacy_modules()
    result['axes'] = {axis:pilot.wrapper(result['core'], axis) for axis in ('X', 'Y')}
    return result
