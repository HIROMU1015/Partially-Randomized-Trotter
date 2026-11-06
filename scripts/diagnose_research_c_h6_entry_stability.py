#!/usr/bin/env python3
"""The two existing invented failures only; no molecular input or run entry.

Reuse Clarabel, QR Newton, the original fixtures and physical gates. Decimal
evaluates supplied arrays, never optimizes them and never replaces acceptance.
"""
import argparse
from dataclasses import asdict
from decimal import Decimal, localcontext
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import svdvals

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests/management')]
from test_research_c_h6_conic_diagnostic import near_degenerate_chart
from trottertracks import c_h6_conic_diagnostic as conic
from trottertracks import c_h6_factored_newton as newton
from trottertracks import c_h6_fit_conditioning as fit


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def spectrum(matrix):
    s = svdvals(np.asarray(matrix, float))
    cutoff = np.finfo(float).eps*max(matrix.shape)*s[0]
    rank = int(np.count_nonzero(s > cutoff))
    return dict(shape=list(matrix.shape), singular_values=s.tolist(),
        diagnostic_cutoff=float(cutoff), effective_rank=rank,
        numerical_nullity=matrix.shape[1]-rank,
        largest=float(s[0]), smallest=float(s[-1]),
        diagnostic_only=True, directions_dropped=0)


def decimal_reference(chart, J, *, digits=80, affine_z=None):
    """Same exact binary64 g/J/U; scalar products in Decimal, not rounded A.

    Reuse the existing upper-triangle DF map and PR rational coefficient map.
    Symmetric tensor columns are shared, while every original g entry is kept.
    This extends the Decimal-from-float convention in test_c_j_u_stability.
    No optimizer, J projection, epsilon change, or scientific acceptance.
    """
    p, s = chart.p, chart.p.spec
    n, rank = s.norb, s.rank
    pairs = list(zip(*np.triu_indices(n)))
    with localcontext() as context:
        context.prec = digits
        d = lambda value: Decimal.from_float(float(value))
        u = [[[d(x) for x in row] for row in frame] for frame in p.mapping.U]
        if affine_z is None:
            j = [d(J[t, a, b]) for t in range(rank) for a, b in pairs]
        else:
            # Evaluate the existing cone coordinates before point() rounds J.
            # as_integer_ratio retains each longdouble W entry exactly.
            def exact(value):
                numerator, denominator = value.as_integer_ratio()
                return Decimal(numerator)/Decimal(denominator)
            z = [d(x) for x in affine_z]
            j = [(d(x)+sum((exact(w)*v for w, v in zip(row, z)), Decimal(0)))/d(weight)
                 for x, row, weight in zip(chart.x0, chart.W, p.w)]
        P = [[[u[t][a][i]*u[t][b][i] for a, b in pairs]
              for i in range(n)] for t in range(rank)]
        columns = []
        for t in range(rank):
            for i, k in pairs:
                columns.append([P[t][i][a]*P[t][k][b] +
                    (P[t][k][a]*P[t][i][b] if i != k else Decimal(0))
                    for a in range(len(pairs)) for b in range(len(pairs))])
        residual_sums, square, residual = [], Decimal(0), np.empty_like(p.g, dtype=np.longdouble)
        for a, (v, w) in enumerate(pairs):
            for b, (x, y) in enumerate(pairs):
                row = a*len(pairs)+b
                tensor = sum((column[row]*coef for column, coef in zip(columns, j)), Decimal(0))
                total = Decimal(0)
                for pq in ([(v, w)] if v == w else [(v, w), (w, v)]):
                    for rs in ([(x, y)] if x == y else [(x, y), (y, x)]):
                        index = (*pq, *rs)
                        r = tensor-d(p.g[index]); total += r; square += r*r
                        residual[index] = np.longdouble(str(r))
                residual_sums.append(total)
        weights = [d(x) for x in p.w]
        gradient = [sum((a*b for a, b in zip(column, residual_sums)), Decimal(0))
                    /weight/d(s.loss_scale) for column, weight in zip(columns, weights)]
        if getattr(p, 'is_entry', False):
            tail = s.ld*len(pairs)
            eta = d(s.entry_epsilon)
            ew = [d(x) for x in np.tile(p.entry_weights, rank-s.ld)]
            h = [(x*x+eta*eta).sqrt() for x in j[tail:]]
            mass = sum((weight*x for weight, x in zip(ew, h)), Decimal(0))
            derivative = [Decimal(0)]*tail + [weight*x/y for weight, x, y in zip(ew, j[tail:], h)]
        else:
            T = p.mapping.coefficient_matrix()[s.ld*n*(2*n+1):].tocsr()
            coeff, h = [], []
            for row in range(T.shape[0]):
                a, b = T.indptr[row:row+2]
                value = sum((d(T.data[k])*j[T.indices[k]] for k in range(a, b)), Decimal(0))
                coeff.append(value); h.append((value*value+d(s.epsilon)**2).sqrt())
            mass = sum(h, Decimal(0)); derivative = [Decimal(0)]*len(j)
            for row in range(T.shape[0]):
                a, b = T.indptr[row:row+2]
                for k in range(a, b):
                    derivative[T.indices[k]] += d(T.data[k])*coeff[row]/h[row]
        multiplier = 2*d(chart.strength)*mass/d(s.mass_scale)**2
        gradient = np.array([float(g+multiplier*dp/weight)
                             for g, dp, weight in zip(gradient, derivative, weights)])
        value = square/(2*d(s.loss_scale))+d(chart.strength)*(mass/d(s.mass_scale))**2
    return dict(objective=float(value), gradient=gradient, residual=residual,
                decimal_digits=digits, optimized=False, scientific_acceptance=False,
                point='supplied binary64 J' if affine_z is None else 'unrounded existing affine coordinates')


def evaluate(chart, J, *, reference=True):
    p, s = chart.p, chart.p.spec
    parts = fit.near.parts_gradient_J if getattr(p, 'is_entry', False) else fit.inner.stable.precision.parts_gradient_J
    binary, gb, rb = parts(p.g, J, p.mapping.U, s, extended=False)
    extended, ge, re = parts(p.g, J, p.mapping.U, s, extended=True)
    gbmax, gemax = float(np.max(abs(gb))), float(np.max(abs(ge)))
    gate = fit.inner.stable.old.gradient_gate(max(gbmax, gemax), chart.policy.j_gradient_infinity, chart.policy)
    row = dict(J_norm=float(np.linalg.norm(J)), J_max=float(np.max(abs(J))),
        maximum_binary64_spacing=float(np.max(abs(np.spacing(J)))),
        binary64_objective=binary['objective'], extended_objective=extended['objective'],
        binary64_gradient_infinity=gbmax, extended_gradient_infinity=gemax,
        gradient_difference_infinity=float(np.max(abs(gb-ge))),
        residual_difference_infinity=float(np.max(abs(rb-re))),
        original_J_gate=gate, reference_is_not_acceptance=True)
    if reference:
        high = decimal_reference(chart, J)
        row.update(decimal_objective=high['objective'], decimal_digits=high['decimal_digits'],
            decimal_gradient_infinity=float(np.max(abs(high['gradient']))),
            binary64_gradient_error_vs_decimal=float(np.max(abs(gb-high['gradient']))),
            extended_gradient_error_vs_decimal=float(np.max(abs(ge-high['gradient']))),
            binary64_residual_error_vs_decimal=float(np.max(abs(rb-high['residual']))),
            extended_residual_error_vs_decimal=float(np.max(abs(re-high['residual']))))
    return row


def diagnose(arm, out):
    out.mkdir(parents=True, exist_ok=False)
    tick = time.monotonic(); c = near_degenerate_chart(arm); p = c.p
    captured, trace, cone = {}, [], {}
    J, meta = conic.solve_and_polish(c, record=lambda row, z: trace.append(row),
        record_endpoint=lambda info, arrays: captured.update(arrays),
        record_conic=lambda info, z: cone.update(z=z))
    local = fit.Chart(p, J, c.policy); zero = np.zeros(local.dimension)
    checked_J = local.point(zero)[1]  # exactly the original failing test's point
    D, gradient, rhs = newton.factor(local, zero)
    step, linear = newton.direction(D, rhs)
    next_J = local.point(step)[1]
    origin = c.point(np.zeros(c.dimension))[1]
    direction = p.pack(checked_J-origin).astype(np.longdouble)
    head_columns = p.spec.ld*p.spec.norb*(p.spec.norb+1)//2
    affine = decimal_reference(c, None, affine_z=cone['z'])
    _, _, residual0, coeff0, _, _, _ = c._primal(np.zeros(c.dimension))
    rmodel = residual0.ravel()+c.A@cone['z'].astype(np.longdouble)
    cmodel = coeff0+c.T@cone['z'].astype(np.longdouble)
    hmodel = np.hypot(cmodel, np.asarray(c.epsilon, np.longdouble))
    mmodel = np.sum(hmodel, dtype=np.longdouble)
    model_gradient = c.physical_A.T@rmodel/p.spec.loss_scale + (
        2*c.strength*mmodel/p.spec.mass_scale**2*(c.physical_T.T@(cmodel/hmodel)))
    rays = []
    for exponent in (0, 4, 8, 12, 16, 24, 32, 40, 48, 56):
        alpha = 2.**(-exponent); z = alpha*step; moved = local.point(z)[1]
        rays.append(dict(alpha=alpha,
            J_displacement_infinity=float(np.max(abs(moved-checked_J))),
            centered_objective_change=local.centered(z)[0],
            quadratic_predicted_change=float(gradient@z+.5*np.sum((D@z)**2)),
            same_binary64_J=bool(np.array_equal(moved, checked_J))))
    row = dict(arm=arm, fixture='existing rank14 near-degenerate head and nonzero tail',
        dimensions=dict(norb=6, rank=14, ld=4, packed_coefficients=len(p.w), chart=c.dimension),
        objective_spec=asdict(p.spec), head=fit.head_report(p),
        physical_map=spectrum(c.physical_A), quotient_metric=spectrum(np.vstack((
            c.A/np.sqrt(np.longdouble(p.spec.loss_scale)),
            np.sqrt(2*np.longdouble(c.strength))*c.T/p.spec.mass_scale))),
        Newton_factor=spectrum(D), conic_and_polish=meta,
        cone_affine_reference=dict(objective=affine['objective'],
            gradient_infinity=float(np.max(abs(affine['gradient']))),
            point=affine['point'], decimal_digits=affine['decimal_digits'],
            optimized=False, scientific_acceptance=False),
        cone_extended_affine_model=dict(
            objective=float(np.sum(rmodel*rmodel)/(2*p.spec.loss_scale)+c.strength*(mmodel/p.spec.mass_scale)**2),
            physical_gradient_infinity=float(np.max(abs(model_gradient)))),
        initial=evaluate(c, origin, reference=False),
        conic_decoded_start=evaluate(c, captured['starting_J'], reference=False),
        returned_J=evaluate(c, J, reference=False), checked_J=evaluate(c, checked_J),
        repacking_change_infinity=float(np.max(abs(J-checked_J))),
        cancelling_displacement=dict(packed_norm=float(np.linalg.norm(direction)),
            mapped_tensor_norm=fit.fro(c.physical_A@direction),
            unpenalized_head_norm=float(np.linalg.norm(direction[:head_columns])),
            penalized_tail_norm=float(np.linalg.norm(direction[head_columns:]))),
        endpoint_Newton=dict(**linear, slope=float(gradient@step),
            coordinate_gradient_infinity=float(np.max(abs(gradient))),
            full_step_changes_binary64_J=not np.array_equal(next_J, checked_J),
            full_step_centered_value=local.centered(step)[0],
            full_step=evaluate(c, next_J, reference=False)),
        fixed_Newton_ray=rays, polish_trace=trace, real_molecular_calls=0, scientific_eligibility=False,
        new_solver_designed=False, objective_epsilon_and_gates_unchanged=True,
        wall_seconds=time.monotonic()-tick)
    np.savez(out/'arrays.npz', g=p.g, initial_J=origin, U=p.mapping.U,
        returned_J=J, checked_J=checked_J, conic_z=cone['z'], **captured)
    save(out/'diagnosis.json', row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('PR', 'ENTRY'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    row = diagnose(args.arm, args.out)
    print(json.dumps(dict(arm=args.arm, gate=row['checked_J']['original_J_gate'],
        decimal_gradient=row['checked_J']['decimal_gradient_infinity'],
        wall_seconds=row['wall_seconds'])), flush=True)


if __name__ == '__main__':
    main()
