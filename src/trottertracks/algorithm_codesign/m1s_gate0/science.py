"""UNEXECUTED numerical implementation, accessible only after the authority guard.

No module-level numerical imports, chemistry generation, eigensolver, sampling,
circuits, or simulation. Every numerical method belongs to a permit-bound engine.
The independent reconstruction uses occupation-bit fermionic actions; the adapter
uses exact JW algebra (including A_l*A_l contractions), without term compression.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import time

from .control import BASIS, H_SHA, INPUT_SHA, digest, require_permit


class _TechnicalStop(ValueError):
    pass


class _Engine:
    def __init__(self, root, permit):
        require_permit(permit)
        self.config = permit.config
        self.audit = permit.audit
        self.root = Path(root)
        self.eps = self.config["precision"]
        self.gates = []
        self.contributions = 0
        self.started = time.monotonic()
        self.n = 8
        self.identity = "I" * self.n
        self.flip_cache, self.phase_cache = {}, {}
        # No numerical import is reachable without a valid permit.
        import numpy as np
        if np.__version__ != self.config["runtime"]["numpy"]:
            raise _TechnicalStop("NumPy version mismatch")
        self.np = np
        self.indices = np.arange(1 << self.n, dtype=np.int64)

    def gate(self, name, condition, value=None, limit=None, split=None):
        # Preserve failure records as strict JSON even when arithmetic is nonfinite.
        def safe(item):
            if isinstance(item, float) and not math.isfinite(item):
                return str(item)
            if isinstance(item, dict):
                return {k: safe(v) for k, v in item.items()}
            if isinstance(item, (list, tuple)):
                return [safe(v) for v in item]
            return item
        record = {"name": name, "split": split, "status": "PASS" if condition else "STOP",
                  "value": safe(value), "limit": safe(limit)}
        self.gates.append(record)
        if not condition:
            raise _TechnicalStop(f"{name}: {value}, limit={limit}")

    def finite(self, name, value):
        self.gate(name + "_finite", bool(self.np.all(self.np.isfinite(value))))

    def _sum(self, values):
        values = list(values)
        return complex(math.fsum(complex(x).real for x in values),
                       math.fsum(complex(x).imag for x in values))

    def _combine(self, terms):
        grouped = {}
        for word, coeff in terms:
            self.contributions += 1
            if self.contributions > self.config["limits"]["max_product_contributions"]:
                raise _TechnicalStop("Pauli contribution instruction limit")
            grouped.setdefault(word, []).append(coeff)
        out = {word: self._sum(grouped[word]) for word in sorted(grouped)}
        out = {word: coeff for word, coeff in out.items() if coeff != 0}
        if len(out) > self.config["limits"]["max_pauli_terms"]:
            raise _TechnicalStop("Pauli table term bound exceeded")
        return out

    def _multiply(self, left, right):
        # Pauli product order is left @ right, never a symmetrized product.
        table = {("X", "Y"): (1j, "Z"), ("Y", "X"): (-1j, "Z"),
                 ("Y", "Z"): (1j, "X"), ("Z", "Y"): (-1j, "X"),
                 ("Z", "X"): (1j, "Y"), ("X", "Z"): (-1j, "Y")}

        def terms():
            for a, ca in sorted(left.items()):
                for b, cb in sorted(right.items()):
                    phase, letters = 1, []
                    for x, y in zip(a, b):
                        if x == "I":
                            letters.append(y)
                        elif y == "I":
                            letters.append(x)
                        elif x == y:
                            letters.append("I")
                        else:
                            factor, z = table[x, y]
                            phase *= factor
                            letters.append(z)
                    yield "".join(letters), phase * ca * cb
        return self._combine(terms())

    def _jw_ladder(self, mode, creation):
        prefix = "Z" * mode
        suffix = "I" * (self.n - mode - 1)
        return {prefix + "X" + suffix: 0.5,
                prefix + "Y" + suffix: -0.5j if creation else 0.5j}

    def _one_body_pauli(self, matrix):
        terms = []
        for p in range(self.n):
            for q in range(self.n):
                coeff = matrix[p, q]
                if coeff != 0:
                    for word, value in self.bilinears[p, q].items():
                        terms.append((word, coeff * value))
        return self._combine(terms)

    def _occupation_ladder(self, mode, creation):
        """Independent reference, source columns and target rows, mode0 MSB."""
        np = self.np
        matrix = np.zeros((256, 256), dtype=np.complex128)
        bit = 1 << (self.n - 1 - mode)
        before = ((1 << self.n) - 1) ^ ((1 << (self.n - mode)) - 1)
        for source in range(256):
            occupied = bool(source & bit)
            if occupied == creation:
                continue
            target = source ^ bit
            matrix[target, source] = -1 if (source & before).bit_count() % 2 else 1
        return matrix

    def _reference_one_body(self, coeff):
        """Bit-based a_p^dagger a_q; no Pauli maps are used in this path."""
        np = self.np
        matrix = np.zeros((256, 256), dtype=np.complex128)
        for q in range(self.n):
            qbit = 1 << (self.n - 1 - q)
            qbefore = ((1 << self.n) - 1) ^ ((1 << (self.n - q)) - 1)
            for source in range(256):
                if not source & qbit:
                    continue
                removed = source ^ qbit
                sign_q = -1 if (source & qbefore).bit_count() % 2 else 1
                for p in range(self.n):
                    pbit = 1 << (self.n - 1 - p)
                    if removed & pbit:
                        continue
                    pbefore = ((1 << self.n) - 1) ^ ((1 << (self.n - p)) - 1)
                    sign_p = -1 if (removed & pbefore).bit_count() % 2 else 1
                    matrix[removed | pbit, source] += sign_q * sign_p * coeff[p, q]
        return matrix

    def _apply_pauli(self, word, vector):
        np = self.np
        xmask = zmask = ny = 0
        for mode, letter in enumerate(word):
            bit = 1 << (self.n - 1 - mode)
            if letter in "XY":
                xmask |= bit
            if letter in "YZ":
                zmask |= bit
            ny += letter == "Y"
        if xmask not in self.flip_cache:
            self.flip_cache[xmask] = self.indices ^ xmask
        if zmask not in self.phase_cache:
            self.phase_cache[zmask] = np.asarray(
                [-1 if (i & zmask).bit_count() % 2 else 1 for i in range(256)], dtype=np.float64)
        out = np.empty_like(vector)
        phase = (1j ** ny) * self.phase_cache[zmask]
        if vector.ndim == 2:
            phase = phase[:, None]
        out[self.flip_cache[xmask]] = phase * vector
        return out

    def _pauli_matrix(self, mapping):
        np = self.np
        result = np.zeros((256, 256), dtype=np.complex128)
        eye = np.eye(256, dtype=np.complex128)
        for word, coeff in sorted(mapping.items()):
            result += coeff * self._apply_pauli(word, eye)
        return result

    def _real_map(self, mapping, label):
        # Projection is permitted only after its L1 operator-error ledger passes.
        # No approximate real coefficient is dropped, even if tiny.
        imag_l1 = math.fsum(abs(complex(c).imag) for c in mapping.values())
        self.gate(label + "_imaginary_L1", imag_l1 <= self.eps["eps_imag"],
                  imag_l1, self.eps["eps_imag"])
        real = {p: float(complex(c).real) for p, c in mapping.items() if complex(c).real != 0}
        return real, imag_l1

    def _norm(self, matrix):
        # Frobenius for matrices; Euclidean for vectors. No eigensolver/SVD.
        return float(self.np.linalg.norm(matrix))

    def load(self):
        np = self.np
        path = self.root / self.config["input"]["path"]
        raw = path.read_bytes()
        self.gate("input_raw_identity", len(raw) == 8652 and digest(raw) == INPUT_SHA,
                  {"bytes": len(raw), "sha256": digest(raw)})
        # Decode these exact verified bytes, avoiding a later path/bytes race.
        import io
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            expected = {"constant", "one_body", "lambdas", "g_matrices", "state_vector",
                        "sector_basis_indices", "sector_state_vector", "metadata_json"}
            self.gate("input_members", set(data.files) == expected)
            metadata = json.loads(str(data["metadata_json"].item()))
            shapes = {"constant": (), "one_body": (8, 8), "lambdas": (12,),
                      "g_matrices": (12, 8, 8), "state_vector": (256,),
                      "sector_basis_indices": (36,), "sector_state_vector": (36,)}
            arrays = {k: data[k].copy() for k in shapes}
        self.gate("input_hamiltonian_hash", metadata["hamiltonian_hash"] == H_SHA)
        self.gate("input_model_metadata", metadata["model"] == "linear_H4"
                  and metadata["distance_angstrom"] == 1.0 and metadata["basis"] == "sto-3g"
                  and metadata["reference_rank"] == 12)
        self.gate("saved_df_order", metadata["fragment_order"] == list(range(12)))
        dtypes = {"constant": "<f8", "one_body": "<c16", "lambdas": "<f8",
                  "g_matrices": "<c16", "state_vector": "<c16",
                  "sector_basis_indices": "<i8", "sector_state_vector": "<c16"}
        for key, shape in shapes.items():
            self.gate(key + "_shape", arrays[key].shape == shape)
            self.gate(key + "_dtype", arrays[key].dtype.str == dtypes[key])
            self.finite(key, arrays[key])
        self.constant = float(arrays["constant"])
        self.one = arrays["one_body"]
        self.lambdas = arrays["lambdas"]
        self.gs = arrays["g_matrices"]
        self.state = arrays["state_vector"]  # Never normalized or phase-adjusted.
        self.gate("saved_state_basis", self.config["basis"] == BASIS)
        expected_indices = [i for i in range(256)
                            if sum(bool(i & (1 << (7 - m))) for m in (0, 2, 4, 6)) == 2
                            and sum(bool(i & (1 << (7 - m))) for m in (1, 3, 5, 7)) == 2]
        self.gate("sector_basis_order", np.array_equal(arrays["sector_basis_indices"], expected_indices))
        embedded = np.zeros(256, dtype=np.complex128)
        embedded[expected_indices] = arrays["sector_state_vector"]
        emb_error = self._norm(embedded - self.state)
        self.gate("state_sector_embedding", emb_error <= self.eps["eps_basis"], emb_error, self.eps["eps_basis"])
        norm2 = np.vdot(self.state, self.state)
        norm_error = abs(float(norm2.real) - 1)
        self.gate("state_normalization", norm_error <= self.eps["eps_state_norm"],
                  norm_error, self.eps["eps_state_norm"])
        self.norm2 = float(norm2.real)
        self.gate("state_positive_norm", self.norm2 > 0)
        self.metadata = metadata
        self.input_receipt = {"path": self.config["input"]["path"], "bytes": len(raw),
                              "sha256": digest(raw), "stored_hamiltonian_hash": H_SHA,
                              "stored_state_hash": metadata["state_hash"]}

    def prepare(self):
        np = self.np
        for label, matrix in [("one_body", self.one), *[(f"g_{i}", g) for i, g in enumerate(self.gs)]]:
            error = self._norm(matrix - matrix.conj().T)
            self.gate(label + "_input_hermiticity_F", error <= self.eps["eps_hermiticity"],
                      error, self.eps["eps_hermiticity"])
        # Reuse each JW bilinear and each squared DF block across the 3 splits.
        self.bilinears = {(p, q): self._multiply(self._jw_ladder(p, True), self._jw_ladder(q, False))
                          for p in range(8) for q in range(8)}
        for mode in range(8):
            for creation in (False, True):
                err = self._norm(self._pauli_matrix(self._jw_ladder(mode, creation))
                                 - self._occupation_ladder(mode, creation))
                self.gate(f"basis_ladder_{mode}_{creation}", err <= self.eps["eps_basis"], err, self.eps["eps_basis"])
        self.one_map = self._combine([(self.identity, self.constant),
                                     *self._one_body_pauli(self.one).items()])
        self.base_reference = self.constant * np.eye(256) + self._reference_one_body(self.one)
        self.block_maps, self.block_reference = [], []
        for index in range(12):
            A = self._one_body_pauli(self.gs[index])
            squared = self._multiply(A, A)  # Contains all contractions without normal ordering.
            self.block_maps.append({word: self.lambdas[index] * coeff for word, coeff in squared.items()})
            refA = self._reference_one_body(self.gs[index])
            self.block_reference.append(self.lambdas[index] * (refA @ refA))
        self.full_reference = self.base_reference + sum(self.block_reference)
        herr = self._norm(self.full_reference - self.full_reference.conj().T)
        self.gate("full_H_hermiticity_F", herr <= self.eps["eps_hermiticity"], herr, self.eps["eps_hermiticity"])
        Hpsi = self.full_reference @ self.state
        energy = np.vdot(self.state, Hpsi) / self.norm2
        self.gate("state_energy_imaginary", abs(float(energy.imag)) <= self.eps["eps_imag"],
                  abs(float(energy.imag)), self.eps["eps_imag"])
        residual = self._norm(Hpsi - energy.real * self.state) / math.sqrt(self.norm2)
        self.gate("full_H_state_residual_2", residual <= self.eps["eps_state_residual"],
                  residual, self.eps["eps_state_residual"])
        self.state_receipt = {"norm_squared": self.norm2, "rayleigh_energy": float(energy.real),
                              "full_H_residual_2": residual,
                              "label": self.config["state"]["label"],
                              "full_Fock_ground_certified": False,
                              "full_H_energy_nondegeneracy_certified": False,
                              "pair_sum_condition_certified": False,
                              "theorem_applicability": "NOT_CERTIFIED"}

    def split(self, count):
        np = self.np
        split = f"G{count}"
        dcomplex = self._combine(list(self.one_map.items()) +
                                 [item for block in self.block_maps[:count] for item in block.items()])
        rcomplex = self._combine([item for block in self.block_maps[count:] for item in block.items()])
        dreal, dimag = self._real_map(dcomplex, split + "_HD")
        rreal, rimag = self._real_map(rcomplex, split + "_HR")
        identity_before_d, moved = dreal.get(self.identity, 0.0), rreal.pop(self.identity, 0.0)
        dreal[self.identity] = math.fsum([identity_before_d, moved])
        hd, hr = self._pauli_matrix(dreal), self._pauli_matrix(rreal)
        refd = self.base_reference + sum(self.block_reference[:count])
        refr = sum(self.block_reference[count:])
        identity_matrix = np.eye(256)
        reconstruction = {"full_F": self._norm(hd + hr - self.full_reference),
                          "HD_after_move_F": self._norm(hd - refd - moved * identity_matrix),
                          "HR_after_move_F": self._norm(hr - refr + moved * identity_matrix)}
        for name, error in reconstruction.items():
            self.gate("reconstruction_" + name, error <= self.eps["eps_H"], error, self.eps["eps_H"], split)
        for name, matrix in (("HD", hd), ("HR", hr)):
            error = self._norm(matrix - matrix.conj().T)
            self.gate(name + "_hermiticity_F", error <= self.eps["eps_hermiticity"], error,
                      self.eps["eps_hermiticity"], split)
        phase_error = self._norm((hd - refd) + (hr - refr))
        self.gate("identity_phase_ledger_F", phase_error <= self.eps["eps_phase"], phase_error,
                  self.eps["eps_phase"], split)
        lam = math.fsum(abs(coeff) for coeff in rreal.values())
        self.finite(split + "_lambda", lam)
        table = [{"word": word, "coefficient_hex": float(coeff).hex(),
                  "signed_event_sign": 1 if coeff > 0 else -1,
                  "probability": abs(coeff) / lam} for word, coeff in sorted(rreal.items())] if lam else []
        table_bytes = json.dumps(table, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        row = {"split": split, "HD_df_indices": list(range(count)), "HR_df_indices": list(range(count, 12)),
               "lambda_R": lam, "canonical_table": table, "canonical_table_sha256": digest(table_bytes),
               "identity_policy": "move_to_hd", "phase_ledger": {
                   "DF_constant": self.constant, "HD_Pauli_scalar_before": identity_before_d,
                   "HR_Pauli_scalar_before": moved, "HD_Pauli_scalar_after": dreal[self.identity],
                   "HR_Pauli_scalar_after": 0.0,
                   "relative_phase_convention": "positive i T_rel H",
                   "HD_added_phase_per_T_rel": moved, "HR_removed_phase_per_T_rel": -moved,
                   "net_relative_phase_change_per_T_rel": 0.0,
                   "HD_half_added_phase_per_delta": moved / 4,
                   "finite_step_phase_experiment": "NOT_PERFORMED"},
               "reconstruction": reconstruction,
               "error_ledger": {"HD_discarded_imaginary_L1": dimag, "HR_discarded_imaginary_L1": rimag,
                                "approximate_real_coefficient_cutoff": 0.0,
                                "coefficient_zero_policy": "exact_zero_only",
                                "configured_precision": self.eps,
                                "error_budgets_are_rigorous_bounds": False},
               "domain": self.state_receipt,
               "scientific_decision": "REQUIRES_USER_GPT_JUDGMENT",
               "finite_step_dressing": "NOT_EVALUATED", "native_shots_resources": "NOT_ESTABLISHED"}
        if lam == 0:
            row.update(W=None, v=None, rho=None, rho_interval=None, metric_status="EMPTY_TAIL_NOT_APPLICABLE")
            return row
        probabilities = [event["probability"] for event in table]
        self.finite(split + "_probabilities", probabilities)
        prob_error = abs(math.fsum(probabilities) - 1.0)
        self.gate("probability_normalization", all(p >= 0 for p in probabilities)
                  and prob_error <= self.eps["eps_prob"], prob_error, self.eps["eps_prob"], split)
        # Expectations use the original saved vector and its explicit norm squared.
        # This quotient does not overwrite/normalize the stored vector.
        hpsi = (hr @ self.state) / lam
        h2mean = float(np.vdot(hpsi, hpsi).real) / self.norm2
        W = 1.0 - h2mean
        means = []
        for event in table:
            value = event["signed_event_sign"] * np.vdot(self.state, self._apply_pauli(event["word"], self.state)) / self.norm2
            self.gate("Pauli_expectation_imaginary", abs(float(value.imag)) <= self.eps["eps_imag"],
                      abs(float(value.imag)), self.eps["eps_imag"], split)
            means.append(float(value.real))
        mean = math.fsum(p * a for p, a in zip(probabilities, means))
        v = math.fsum(p * (a - mean) ** 2 for p, a in zip(probabilities, means))
        self.finite(split + "_W_v", [W, v])
        mean_matrix = float(np.vdot(self.state, hpsi).real) / self.norm2
        ew, ev = self.eps["W_error_budget"], self.eps["v_error_budget"]
        self.gate("Pauli_mean_consistency", abs(mean - mean_matrix) <= self.eps["eps_W"],
                  abs(mean - mean_matrix), self.eps["eps_W"], split)
        self.gate("W_internal_range", -ew <= W <= 1 + ew, W, ew, split)
        self.gate("v_internal_range", -ev <= v <= W + ew + ev, v, ev, split)
        wlo, whi = max(0.0, W - ew), min(1.0, W + ew)
        vlo, vhi = max(0.0, v - ev), min(1.0, v + ev)
        self.gate("W_v_interval_consistency", wlo <= whi and vlo <= vhi and vlo <= whi,
                  {"W": [wlo, whi], "v": [vlo, vhi]}, split=split)
        row.update(W=W, v=v, W_interval=[wlo, whi], v_interval=[vlo, vhi], rho=None, rho_interval=None)
        if W == 0.0:
            classification = "W_ZERO_NUMERICAL_NOT_APPLICABLE"
        elif wlo <= 0:
            classification = "DENOMINATOR_INTERVAL_CONTAINS_ZERO_UNDETERMINED"
        elif wlo <= self.eps["W_floor"]:
            classification = "SMALL_DENOMINATOR_UNDETERMINED"
        else:
            rho = (W + v) / (2 * W)
            lo, hi = 0.5 * (1 + vlo / whi), 0.5 * (1 + vhi / wlo)
            uncertainty = max(abs(rho - lo), abs(hi - rho))
            self.gate("rho_internal_range", 0.5 - self.eps["eps_rho"] <= rho <= 1 + self.eps["eps_rho"],
                      rho, self.eps["eps_rho"], split)
            row.update(rho=rho, rho_interval=[lo, hi], rho_error_budget_uncertainty=uncertainty)
            classification = "HEADROOM_DIAGNOSTIC_ONLY" if uncertainty <= self.eps["eps_rho"] else "RHO_PRECISION_UNRESOLVED"
        row["metric_status"] = classification
        row["error_ledger"]["interval_interpretation"] = "engineering sensitivity to adopted W/v budgets; no rigorous eigenstate error enclosure"
        row["comparators"] = {"equal_total_event_positions": "shared r=R; ordinary r=R; independent each branch R/2",
                              "shared_leading_coefficient_per_lambda2_over_R": (W + v) / 4,
                              "ordinary_and_independent_per_lambda2_over_R": W / 2,
                              "finite_R_delta_signal_computed": False,
                              "single_rate_equality_condition": "SP_NOT_CERTIFIED",
                              "Jensen_floor_condition": "FULL_H_NONDEGENERACY_NOT_CERTIFIED"}
        return row


def run_authorized(root, permit):
    """Future execution entry. This function has NOT been run in source preparation."""
    require_permit(permit)
    limits = permit.config["limits"]
    # Process must be isolated; apply aggregate memory/file bounds before NumPy.
    def lower_limit(kind, requested):
        soft, hard = resource.getrlimit(kind)
        cap = requested if hard == resource.RLIM_INFINITY else min(requested, hard)
        if soft != resource.RLIM_INFINITY:
            cap = min(cap, soft)
        resource.setrlimit(kind, (cap, hard))
    lower_limit(resource.RLIMIT_AS, limits["memory_bytes"])
    lower_limit(resource.RLIMIT_FSIZE, limits["output_bytes"])
    output = Path(root) / permit.audit["output_directory"]
    # Exclusive new directory is the result container, not a separate runner marker.
    # Its presence prevents an accidental retry; no authority/marker is issued here.
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "m1s_gate0_result_v1", "status": "TECHNICAL_STOP",
              "audit": permit.audit, "config": permit.config, "rows": [], "gates": [],
              "new_scientific_execution": True, "automatic_scientific_GO_STOP": False,
              "domain_applicability": "NUMERICAL_ORACLE_ONLY_THEOREM_NOT_CERTIFIED",
              "unperformed": ["sampling", "simulation", "spectrum", "sigma", "gap", "coupling",
                              "circuits", "compile", "resources", "T1c", "Q4", "Q5"],
              "mandatory_STOP": True}
    engine = None
    try:
        engine = _Engine(root, permit)
        engine.load()
        engine.prepare()
        report["input"] = engine.input_receipt
        report["state"] = engine.state_receipt
        for count in (3, 6, 9):
            try:
                report["rows"].append(engine.split(count))
            except _TechnicalStop as exc:
                report["rows"].append({"split": f"G{count}", "metric_status": "TECHNICAL_STOP", "reason": str(exc)})
        report["status"] = ("TECHNICAL_STOP" if any(r["metric_status"] == "TECHNICAL_STOP" for r in report["rows"])
                            else "HEADROOM_DIAGNOSTICS_COMPLETE_AWAITING_SCIENTIFIC_JUDGMENT")
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "reason": str(exc)}
    finally:
        completed = {r["split"] for r in report["rows"]}
        report["rows"].extend({"split": s, "metric_status": "NOT_EVALUATED_GLOBAL_TECHNICAL_STOP"}
                              for s in ("G3", "G6", "G9") if s not in completed)
        if engine is not None:
            report["gates"] = engine.gates
            report["resources"] = {"elapsed_seconds": time.monotonic() - engine.started,
                                   "maxrss_linux_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                   "product_contributions": engine.contributions,
                                   "python": platform.python_version(), "numpy": engine.np.__version__,
                                   "workers": 1, "backend": "cpu", "wall_time_cap": None}
        raw = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
        if len(raw) > limits["output_bytes"]:
            raise _TechnicalStop("Result exceeds adopted output byte bound; no implicit truncation")
        with (output / "result.json").open("xb") as stream:
            stream.write(raw)
        with (output / "result.sha256").open("x") as stream:
            stream.write(hashlib.sha256(raw).hexdigest() + "\n")
    return {"status": report["status"], "result_directory": str(output), "mandatory_STOP": True}
