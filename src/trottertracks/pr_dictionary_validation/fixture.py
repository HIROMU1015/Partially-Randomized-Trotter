"""Exact specification/serialization. Materialization is a later B operation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
from typing import Any


def fraction(text: str) -> Fraction:
    if not isinstance(text, str) or not text or any(c.isspace() for c in text):
        raise ValueError('Exact fractions must be strings without whitespace')
    value = Fraction(text)
    canonical = str(value.numerator) if value.denominator == 1 else str(value)
    if text != canonical:
        raise ValueError('Fraction is not in canonical reduced form')
    return value


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class PlaneRotation:
    p: int
    q: int
    cosine: str
    sine: str

    def validate(self, modes: int) -> None:
        if not (0 <= self.p < self.q < modes):
            raise ValueError('Invalid rotation modes')
        c, s = fraction(self.cosine), fraction(self.sine)
        if c*c+s*s != 1:
            raise ValueError('Rotation is not exactly orthogonal')


@dataclass(frozen=True)
class FragmentSpec:
    fragment_id: str
    diagonal: tuple[str, ...]
    # Left-to-right MATRIX order, rightmost factor acts first.
    rotations: tuple[PlaneRotation, ...]


@dataclass(frozen=True)
class CompilerProfile:
    qiskit: str = '1.3.0'
    optimization_level: int = 1
    seed_transpiler: int = 17
    basis_gates: tuple[str, ...] = ('rz', 'sx', 'x', 'cx')
    approximation_degree: float = 1.0
    qubits_initially_zero: bool = False

    def validate(self) -> None:
        if asdict(self) != asdict(CompilerProfile()):
            raise ValueError('Compiler differs from P0 profile')


@dataclass(frozen=True)
class EvolutionContext:
    total_time: str
    q: int
    r: int
    cutoff: int

    def validate(self) -> None:
        if (self.total_time, self.q, self.r, self.cutoff) not in (
                ('1/5', 2, 2, 2), ('1/5', 1, 2, 0)):
            raise ValueError('Only the two unchanged P0 contexts are supported')

    @property
    def delta(self) -> Fraction:
        return fraction(self.total_time)/self.q

    @property
    def micro_time(self) -> Fraction:
        return self.delta/self.r


@dataclass(frozen=True)
class FixtureSpec:
    modes: int
    particles: int
    occupied: tuple[int, ...]
    scalar: str
    deterministic: tuple[str, ...]
    fragments: tuple[FragmentSpec, ...]
    identity_policy: str = 'faithful_identity_in_tail'
    mapping: str = 'JW_low_index_Z_prefix_little_endian_ancilla_last'
    frame_convention: str = 'B_adag_p_Bdag=sum_j_Vjp_adag_j;vacuum_phase=1'

    def validate(self) -> None:
        if (self.modes, self.particles, self.occupied) != (4, 2, (0, 1)):
            raise ValueError('P0 physical register/state required')
        if self.identity_policy != 'faithful_identity_in_tail':
            raise ValueError('Scalar/identity relocation is forbidden')
        if self.mapping != 'JW_low_index_Z_prefix_little_endian_ancilla_last':
            raise ValueError('Unknown JW/register convention')
        if self.frame_convention != 'B_adag_p_Bdag=sum_j_Vjp_adag_j;vacuum_phase=1':
            raise ValueError('Unknown Gaussian convention')
        if len(self.deterministic) != self.modes or len(self.fragments) != 2:
            raise ValueError('Fixed D block and two fragments required')
        if tuple(f.fragment_id for f in self.fragments) != ('F0', 'F1'):
            raise ValueError('Stable fragment order F0,F1 required')
        fraction(self.scalar)
        for x in self.deterministic:
            fraction(x)
        for f in self.fragments:
            if len(f.diagonal) != self.modes:
                raise ValueError('Diagonal size mismatch')
            for x in f.diagonal:
                fraction(x)
            for g in f.rotations:
                g.validate(self.modes)


@dataclass(frozen=True)
class FrozenTaskIdentity:
    fixture_sha256: str
    source_closure_sha256: str
    recipe_sha256: str
    environment_lock_sha256: str
    context: EvolutionContext
    compiler: CompilerProfile
    precision_policy: str

    def validate(self) -> None:
        for value in (self.fixture_sha256, self.source_closure_sha256,
                      self.recipe_sha256, self.environment_lock_sha256):
            if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Unbound identity')
        self.context.validate()
        self.compiler.validate()
        if self.precision_policy != 'exact_rational_complex128_ref90_128_no_screening':
            raise ValueError('Unbound precision policy')


def specification_record(spec: FixtureSpec) -> dict:
    spec.validate()
    return asdict(spec)


def specification_from_record(record: dict) -> FixtureSpec:
    value = dict(record)
    value['occupied'] = tuple(value['occupied'])
    value['deterministic'] = tuple(value['deterministic'])
    value['fragments'] = tuple(FragmentSpec(
        f['fragment_id'], tuple(f['diagonal']),
        tuple(PlaneRotation(**r) for r in f['rotations'])) for f in value['fragments'])
    spec = FixtureSpec(**value)
    spec.validate()
    if canonical_json(specification_record(spec)) != canonical_json(record):
        raise ValueError('Noncanonical/unknown specification fields')
    return spec


def p0_candidate() -> FixtureSpec:
    """Specification only; never called during I. No arrays or Pauli maps."""
    return FixtureSpec(4, 2, (0, 1), '1/37',
        ('1/20', '-1/30', '1/40', '-1/50'), (
        FragmentSpec('F0', ('1/5', '-1/7', '1/11', '0'), (
            PlaneRotation(0, 1, '3/5', '4/5'),
            PlaneRotation(1, 2, '5/13', '12/13'))),
        FragmentSpec('F1', ('0', '1/6', '-1/8', '1/10'), (
            PlaneRotation(2, 3, '8/17', '15/17'),
            PlaneRotation(0, 2, '7/25', '24/25'),
            PlaneRotation(0, 1, '9/41', '40/41')))))
