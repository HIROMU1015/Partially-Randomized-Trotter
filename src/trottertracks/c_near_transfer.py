"""Dimension/geometry adapter for the frozen finite-recipe comparison.

Import and design preflight are byte-only. Array helpers are numerical APIs for
synthetic tests now; real molecular use needs a later fixed scientific scope.
This module supplies no new optimizer, PF engine, compiler or execution permit.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

VERSION = 'near_recipe_transfer_adapter_v1'
STATUS = 'UNAPPROVED_DESIGN_NOT_EXECUTABLE'


@dataclass(frozen=True)
class System:
    norb: int
    rank: int
    ld: int
    distance_angstrom: float
    input_sha256: str
    basis: str = 'STO-3G'
    nalpha: int | None = None
    nbeta: int | None = None
    sector_dimension: int | None = None

    def __post_init__(self):
        if any(type(v) is not int for v in (self.norb, self.rank, self.ld)):
            raise ValueError('INTEGER_DIMENSIONS')
        if self.norb < 2 or self.norb % 2 or self.rank < 1 or not 0 <= self.ld < self.rank:
            raise ValueError('CLOSED_SHELL_NUMBER_CDF_DIMENSIONS')
        if (isinstance(self.distance_angstrom, bool) or not math.isfinite(self.distance_angstrom)
                or self.distance_angstrom <= 0 or self.basis.lower() != 'sto-3g'):
            raise ValueError('LINEAR_H_STO3G_GEOMETRY')
        if not isinstance(self.input_sha256, str) or re.fullmatch('[0-9a-f]{64}', self.input_sha256) is None:
            raise ValueError('INPUT_BYTE_IDENTITY')
        half = self.norb // 2
        dimension = math.comb(self.norb, half) ** 2
        for name, value in [('nalpha', half), ('nbeta', half), ('sector_dimension', dimension)]:
            supplied = getattr(self, name)
            if supplied is not None and (type(supplied) is not int or supplied != value):
                raise ValueError('SPIN_SECTOR_METADATA:' + name)
            object.__setattr__(self, name, value)

    def dimensions(self):
        return dict(sector=self.sector_dimension, system=1 << (2*self.norb),
                    controlled=1 << (2*self.norb+1),
                    sector_complex128_matrix_bytes=16*self.sector_dimension**2,
                    controlled_complex128_matrix_bytes=16*(1 << (4*self.norb+2)))

    def geometry(self):
        """Metadata only, same linear-chain convention as the existing input generator."""
        return [('H', (0., 0., i*self.distance_angstrom)) for i in range(self.norb)]


def validate_arrays(g, J, U, system):
    import numpy as np
    shape = (system.rank, system.norb, system.norb)
    if (g.shape != (system.norb,)*4 or J.shape != shape or U.shape != shape
            or any(a.dtype != np.float64 or not np.isfinite(a).all() for a in (g, J, U))):
        raise ValueError('TRANSFER_ARRAY_SHAPE_FINITE_BINARY64')
    if not np.array_equal(J, J.transpose(0, 2, 1)):
        raise ValueError('SIGNED_SYMMETRIC_J')
    if np.max(abs(U.transpose(0, 2, 1) @ U - np.eye(system.norb))) > 1e-10:
        raise ValueError('ORTHOGONAL_U')


def objective_specs(g, J, U, system, *, strength, epsilon):
    """Recompute scales ONCE at the new neutral start, never at endpoints.

    Uses the existing tensor map, PR mass and ENTRY oracle. A zero tensor scale
    is unresolved, not replaced by an arbitrary H4 number or a numerical floor.
    """
    import numpy as np
    from . import c_near_stable as existing
    validate_arrays(g, J, U, system)
    residual = existing.stable.linear.reconstruct(J, U) - g
    fscale = float(np.sum(residual * residual) / 2)
    if not math.isfinite(fscale) or fscale <= 0:
        raise ValueError('NEW_NEUTRAL_TENSOR_SCALE_UNRESOLVED')
    provisional = existing.near.pr.Objective(system.rank, system.norb, system.ld,
                                             fscale, 1., strength, epsilon)
    mscale = float(existing.near.pr.smooth_mass(J, provisional, np))
    pr = existing.near.pr.Objective(system.rank, system.norb, system.ld,
                                    fscale, mscale, strength, epsilon)
    entry = existing.near.initial_spec(J, ld=system.ld, loss_scale=fscale,
                                       strength=strength, epsilon=epsilon)
    return dict(PR=pr, ENTRY=entry)


def recipe_arm(name, g, cold, spec, system, physical, call, record, *, trajectory=None, **recipe_options):
    """Reuse the old arm wrapper, defaulting to its fixed 16/128 policy.

    This is a proposed recipe, not adopted transfer conditions. The wrapper is
    already dimension-generic after its input/spec assembly is separated out.
    Explicit task-local recipe_options may select a new permitted finite policy.
    No global monkeypatch or second reduced optimizer.
    """
    from . import c_near_practical as existing
    validate_arrays(g, cold['J'], cold['U'], system)
    if (spec.norb, spec.rank, spec.ld) != (system.norb, system.rank, system.ld):
        raise ValueError('SPEC_SYSTEM_IDENTITY')
    options = {} if trajectory is None else dict(trajectory=trajectory)
    return existing.run_arm(name, g, cold, spec, physical, call, record, **options, **recipe_options)


class PhysicalContext:
    """Generalize the H4 wrapper; all physical engines remain existing code."""
    def __init__(self, packet, system, *, observed_input_sha256):
        import numpy as np
        from . import c1_molecule_extension as existing
        if observed_input_sha256 != system.input_sha256:
            raise ValueError('WRONG_PHYSICAL_INPUT')
        n = system.norb
        if (set(packet) != {'h', 'g', 'constant'} or packet['h'].shape != (n, n)
                or packet['g'].shape != (n,)*4 or packet['constant'].shape != ()
                or any(a.dtype != np.float64 or not np.isfinite(a).all() for a in packet.values())
                or np.max(abs(packet['h']-packet['h'].T)) > 1e-10):
            raise ValueError('ORIGINAL_H_G_SCALAR_SCHEMA')
        g = packet['g']
        if max(np.max(abs(g-g.transpose(*axes))) for axes in
               [(1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)]) > 1e-10:
            raise ValueError('CHEMIST_ERI_SYMMETRIES')
        self.system, self.packet, self.engine = system, packet, existing
        self.context = existing.SectorCAR(n)
        if self.context.d != system.sector_dimension:
            raise ValueError('PHYSICAL_SECTOR_DIMENSION')
        self.physical = self.context.physical(**packet)

    def assess(self, J, U, *, candidate_sha256, budget_Ha):
        if (re.fullmatch('[0-9a-f]{64}', candidate_sha256) is None
                or not math.isfinite(budget_Ha) or budget_Ha <= 0):
            raise ValueError('CANDIDATE_BUDGET_BINDING')
        validate_arrays(self.packet['g'], J, U, self.system)
        row, generators = self.engine.rank_assessment(self.context, self.packet, J, U,
                                                       self.physical, self.system.rank)
        row.update(representation=self.engine.representation_gate(self.context, self.packet, J, U, generators),
                   input_sha256=self.system.input_sha256, candidate_sha256=candidate_sha256,
                   budget_Ha=budget_Ha, budget_PASS=row['sector_error']+row['numerical_guard'] <= budget_Ha,
                   kappa='original_g_corrected', scalar='nuclear_once',
                   actual_CA_achieved=None, ground_alias=None, PF_RTE=None, shots=None, total_RZ=None)
        return row, generators


def downstream_handoff(system, candidate_sha256, physical):
    """Bind a qualified endpoint to a NEW ground/alias stage, not old H4 values."""
    if (physical['input_sha256'] != system.input_sha256
            or physical['candidate_sha256'] != candidate_sha256
            or physical['actual_rank'] != system.rank):
        raise ValueError('DOWNSTREAM_CANDIDATE_IDENTITY')
    if (physical['budget_PASS'] is not True or
            physical['sector_error']+physical['numerical_guard'] > physical['budget_Ha']):
        raise ValueError('DOWNSTREAM_DF_BUDGET')
    return dict(input_sha256=system.input_sha256, candidate_sha256=candidate_sha256,
                rank=system.rank, original_frame_order=list(range(system.rank)), ld=system.ld,
                nalpha=system.nalpha, nbeta=system.nbeta, sector_dimension=system.sector_dimension,
                spin_order='interleaved_alpha_beta', kappa='original_g_corrected', scalar='nuclear_once',
                bit_order='existing_OpenFermion_to_Qiskit_bit_reverse',
                ground_alias=None, PF_RTE=None, shots=None, W=None, native_price=None, total_RZ=None,
                actual_CA_achieved=None, next_stage='new_same_candidate_ground_alias',
                state_model='own_DF_ground_state_oracle', state_preparation_cost='EXCLUDED',
                rigorous_enclosure=physical['rigorous_enclosure'])


def optimization_call_model(*, accepted_update_cap, total_trial_cap):
    """Derive existing paired-arm instrumentation for a supplied proposal.

    Initial input production/neutral fit/codec and downstream are separate.
    The values are logical work limits, never CPU/RAM/wall caps.
    """
    if (type(accepted_update_cap) is not int or type(total_trial_cap) is not int
            or accepted_update_cap < 1 or total_trial_cap < 1):
        raise ValueError('FINITE_RECIPE_BUDGET')
    points = 1 + total_trial_cap
    per_arm = dict(optimization_driver=1, inner_J_solve=3*points, precision_state=2*points,
                   reduced_admission=points, independent_terminal_assessment=accepted_update_cap+1,
                   physical_assessment=accepted_update_cap+1, endpoint_diagnostics=1, arm_export=1)
    common = dict(saved_load=1, initial_identity=1, physical_context=1,
                  common_physical_assessment=1, objective_setup=1, comparison=1, export=1)
    return dict(points_per_arm=points, per_arm=per_arm, common=common,
                maximum=sum(common.values())+2*sum(per_arm.values()),
                excludes=['input generation', 'neutral initialization/fit/codec', 'all downstream work'])


def preflight(root, contract_path):
    """Only design/policy/byte checks; not input numeric qualification."""
    root = Path(root).resolve()
    c = json.loads(Path(contract_path).read_text())
    if (c['status'] != STATUS or c['executable'] is not False or c['selected_system'] is not None
            or c['science_entry'] is not None or c['maximum_scientific_calls'] is not None
            or c['retry'] != 0 or c['resume'] != 0 or c['mandatory_stop'] is not True
            or c['comparison'] != 'II_practical_finite_recipe' or c['transfer'] != 'recipe_not_endpoint_parameters'):
        raise ValueError('DESIGN_NOT_EXECUTABLE_POLICY')
    def checked_path(path):
        value = (root/path).resolve()
        if not value.is_relative_to(root):
            raise ValueError('REPOSITORY_RELATIVE_IDENTITY')
        return value
    manifest_path = checked_path(c['technical_SOURCE'])
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != c['technical_SOURCE_sha256']:
        raise ValueError('TECHNICAL_SOURCE_MANIFEST')
    manifest = json.loads(manifest_path.read_text())
    for group in (manifest['files'], c['metadata_inputs']):
        for path, expected in group.items():
            if hashlib.sha256(checked_path(path).read_bytes()).hexdigest() != expected:
                raise ValueError('SOURCE_METADATA_DRIFT:' + path)
    if checked_path(c['future_output']).exists():
        raise ValueError('FRESH_NO_RETRY_RESUME')
    return dict(status='PASSIVE_DESIGN_PREFLIGHT_PASS', executable=False, scientific_calls=0,
                numeric_input_qualification='NOT_RUN', source_files=len(manifest['files']),
                metadata_files=len(c['metadata_inputs']), mandatory_stop=True)


def execute(*args, **kwargs):
    raise PermissionError('DESIGN_NOT_EXECUTABLE: fixed target, inputs and executable contract required')
