"""Pure descriptor plans for the five fixed arms; no numerical operators."""
from dataclasses import dataclass
from fractions import Fraction

from trottertracks.b2d_adapter.adapter import (
    BlockExponential, DContext, DRun, TailOccurrence,
    _positive_integer, _rational, _expand_append_s2,
)
from trottertracks.b2d_adapter._published import (
    UPSTREAM_COMMIT, symmetric_s2_sequence,
)
from ._baselines import (
    trotter_2nd_list, yoshida_4th_list, morales_2025_y8m10b_list,
)

ARMS = ("A0", "A1", "A2", "A3", "A4")
FAMILIES = {
    "A0": "current_partial_s2",
    "A1": "standalone_D_S2",
    "A2": "Yoshida4",
    "A3": "published_Morales_Y8m10b",
    "A4": "published_Morales_YP8m8",
}


@dataclass(frozen=True)
class ArmPlan:
    arm: str
    n_d: int | None
    tail: TailOccurrence
    before_tail: tuple[BlockExponential, ...]
    after_tail: tuple[BlockExponential, ...]

    @property
    def append_operations(self):
        return self.before_tail + (self.tail,) + self.after_tail

    @property
    def mathematical_operations(self):
        return tuple(reversed(self.append_operations))

    @property
    def family(self):
        return FAMILIES[self.arm]

    @property
    def coefficient_source(self):
        return "PR:5d3ec883eeeabf69abfa128c9e169c59fc572149" if self.arm == "A0" else UPSTREAM_COMMIT


def make_arm_plan(arm: str, context: DContext, delta, n_d, tail: TailOccurrence):
    """Use the unchanged A4 DRun and baseline fixed-source generators.

    Scalar phases are separate: callers pay them once at total physical time,
    never per S2 stage. No fusion is applied across the opaque tail barrier.
    """
    if arm not in ARMS:
        raise ValueError("Exactly A0--A4 are supported.")
    delta = _rational(delta)
    if not isinstance(tail, TailOccurrence) or tail.context != context or tail.evolution_time != delta:
        raise ValueError("All arms require the same bound tail and time.")
    if arm == "A0":
        if n_d is not None:
            raise ValueError("A0 has no standalone D internal repetition parameter.")
        before = tuple(BlockExponential(b, delta / 2, context.controlled, context.ancilla_qubit)
                       for b in context.block_ids)
        after = tuple(reversed(before))
    else:
        n_d = _positive_integer(n_d)
        tau = delta / (2 * n_d)
        if arm == "A4":
            before = DRun(context, tau, n_d).append_operations()
        else:
            generator = {"A1": trotter_2nd_list, "A2": yoshida_4th_list,
                         "A3": morales_2025_y8m10b_list}[arm]
            mathematical_weights = tuple(map(Fraction, symmetric_s2_sequence(generator())))
            before = _expand_append_s2(context, mathematical_weights[::-1], tau) * n_d
        after = before  # Same signed direction; never the adjoint.
    return ArmPlan(arm, n_d, tail, before, after)
