#!/usr/bin/env python3
"""Artificial fixed-time technical evaluation; no molecular computation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from trotterlib.fixed_signal_evaluation import FixedSignalConfig, evaluate_fixed_signal
from trotterlib.fixed_signal_io import load_artificial_input, source_provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hamiltonian-type", choices=["df_native", "pauli_jw"], required=True)
    parser.add_argument("--method", choices=["PF", "PR", "B0", "B0I"], default="PF")
    parser.add_argument("--T", type=float, default=.2)
    parser.add_argument("--q", type=int, default=1)
    parser.add_argument("--L-D", type=int)
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cost-samples", type=int, default=8)
    parser.add_argument("--epsilon-sig", type=float, required=True)
    parser.add_argument("--numerical-allowance", type=float,
                        help="Declared empirical combined candidate/reference allowance per axis")
    parser.add_argument("--log-B-margin", type=float,
                        help="Declared empirical rounding margin for the log normalization")
    parser.add_argument("--evidence-ref", help="Required when declaring empirical numerical evidence")
    parser.add_argument("--verify-compiled", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; existing results are preserved")
    root = Path(__file__).resolve().parents[1]
    h, state, inputs = load_artificial_input(args.input, args.hamiltonian_type)
    config = FixedSignalConfig(args.hamiltonian_type, method=args.method, T=args.T, q=args.q,
        L_D=args.L_D, r=args.r, K=args.K, seed=args.seed, cost_samples=args.cost_samples)
    allowance = {a: args.numerical_allowance for a in ("cosine", "sine")}
    source = source_provenance(root)
    source["input"] = inputs
    result = evaluate_fixed_signal(h, state, config, epsilon_sig=args.epsilon_sig,
        numerical_allowance=allowance, log_B_margin=args.log_B_margin,
        evidence_kind="EMPIRICAL" if args.numerical_allowance is not None else "UNKNOWN",
        evidence_ref=args.evidence_ref, state_metadata=inputs["state_generation"],
        source_provenance=source, verify_compiled=args.verify_compiled)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=result["validation"]["status"],
                          N=result["shots"]["N_total"], G=result["total_RZ"],
                          output=str(args.output)), ensure_ascii=False))


if __name__ == "__main__":
    main()
