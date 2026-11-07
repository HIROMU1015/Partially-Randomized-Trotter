"""C-S literal placement adapter; no law, synthesis, simulation, or price call.

Angles use the existing exact pi+radian representation. A supplied realization
is a coherent OPERATOR word, never a channel interpolation. This module only
preserves its position and lineage; it does not certify that word's mean/error.
The caller must lower control before declaring joint-space Pauli generators.
"""
from dataclasses import dataclass
from collections.abc import Mapping

from .synthesis_placement.wrapper_sequence import Angle


@dataclass(frozen=True)
class RotationSite:
    site_id: str
    generator: str
    angle: Angle
    role: str

    def __post_init__(self):
        if (not isinstance(self.site_id, str) or not self.site_id
                or not isinstance(self.generator, str) or not self.generator
                or any(p not in 'IXYZ' for p in self.generator)
                or not isinstance(self.angle, Angle)
                or self.role not in ('D_DIAGONAL', 'R_DIAGONAL', 'FIXED')):
            raise ValueError('explicit site, joint Pauli, exact angle and role required')


@dataclass(frozen=True)
class Replacement:
    original: RotationSite
    coarse: Angle

    @property
    def residual(self):
        return self.original.angle + self.coarse.scaled(-1)


def plan_sequence(sites, coarse_angles, *, arm):
    """S0 original, S1 all eligible, S2 D-only; no search/fusion/periodicity.

    The same externally chosen coarse table is required for all three arms.
    FIXED sites represent boundaries/basis/wrapper rotations, not eligible D/R.
    This is a literal rotation slice, not a full Hamiltonian or circuit loader.
    """
    sites = tuple(sites)
    if (arm not in ('S0', 'S1', 'S2') or not sites
            or any(not isinstance(s, RotationSite) for s in sites)
            or len({s.site_id for s in sites}) != len(sites)
            or len({len(s.generator) for s in sites}) != 1):
        raise ValueError('one ordered register and unique sites for S0/S1/S2 required')
    eligible = {s.site_id for s in sites if s.role != 'FIXED'}
    if (not isinstance(coarse_angles, Mapping) or set(coarse_angles) != eligible
            or any(not isinstance(a, Angle) for a in coarse_angles.values())):
        raise ValueError('complete common coarse table for eligible sites required')
    return tuple(Replacement(s, coarse_angles[s.site_id])
                 if arm == 'S1' and s.role != 'FIXED'
                 or arm == 'S2' and s.role == 'D_DIAGONAL' else s
                 for s in sites)


@dataclass(frozen=True)
class ResidualWord:
    site_id: str
    generator: str
    angle: Angle
    time_instructions: tuple
    semantics: str = 'COHERENT_OPERATOR'
    phase_retained: bool = True

    def __post_init__(self):
        if (type(self.time_instructions) is not tuple or not self.time_instructions
                or self.semantics != 'COHERENT_OPERATOR'
                or self.phase_retained is not True):
            raise ValueError('nonempty time-ordered coherent word with phase required')


@dataclass(frozen=True)
class Placed:
    site_id: str
    part: str
    operation: object


def splice_realization(plan, words):
    """Coarse then supplied residual, in the original slot, without reordering.

    Coefficients/proposals, independence, normalization, numerical/native error,
    confidence, and full prices remain the existing provider/caller's job.
    No equality to the exact target is asserted for a finite residual word.
    """
    plan = tuple(plan)
    sites = tuple(p.original if isinstance(p, Replacement) else p for p in plan)
    if (not sites or any(not isinstance(s, RotationSite) for s in sites)
            or len({s.site_id for s in sites}) != len(sites)
            or len({len(s.generator) for s in sites}) != 1):
        raise ValueError('valid ordered plan with unique sites required')
    needed = {p.original.site_id for p in plan
              if isinstance(p, Replacement) and not p.residual.zero}
    if not isinstance(words, Mapping) or set(words) != needed:
        raise ValueError('one residual word per nonzero selected slot; no extras')
    out = []
    for p in plan:
        if isinstance(p, RotationSite):
            out.append(Placed(p.site_id, 'original', p))
            continue
        s = p.original
        out.append(Placed(s.site_id, 'coarse', RotationSite(
            s.site_id, s.generator, p.coarse, s.role)))
        if not p.residual.zero:
            w = words[s.site_id]
            if (not isinstance(w, ResidualWord) or w.site_id != s.site_id
                    or w.generator != s.generator or w.angle != p.residual
                    or w.semantics != 'COHERENT_OPERATOR' or w.phase_retained is not True):
                raise ValueError('residual site/generator/angle/coherent-phase mismatch')
            out.extend(Placed(s.site_id, 'residual', instruction)
                       for instruction in w.time_instructions)
    return tuple(out)
