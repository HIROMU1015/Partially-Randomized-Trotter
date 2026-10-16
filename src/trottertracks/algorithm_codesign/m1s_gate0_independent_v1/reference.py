"""UNEXECUTED independent reference kernels. Standard library only.

Coefficient layer: exact Gaussian integers / powers of two, exterior-algebra
occupation states, explicit CAR contraction, complete Pauli trace via FWHT.
State layer: Decimal complex pairs, 80 digits; conditional, not a certificate.
Construction is blocked in this preparation release by control.require_execution.
"""
from bisect import bisect_left
from decimal import Decimal, localcontext, ROUND_HALF_EVEN
from fractions import Fraction
import ast
import hashlib
import io
import json
import math
import struct
import zipfile

from .control import require_execution, INPUT_SHA, H_SHA, POLICY


class Reference:
    def __init__(self, authorization=None):
        require_execution(authorization)  # No live permit issuer in this release.
        self.events = 0
        self.transitions = {}
        self.occupations = [tuple(m for m in range(8) if (i >> (7-m)) & 1)
                            for i in range(256)]
        self.occupation_index = {s: i for i, s in enumerate(self.occupations)}

    def tick(self, count=1):
        self.events += count
        if self.events > POLICY["resources"]["max_primitive_events"]:
            raise RuntimeError("UNDETERMINED: primitive event limit")

    def add(self, a, b):
        self.tick()
        c = (a[0]+b[0], a[1]+b[1])
        if max(abs(x).bit_length() for x in c) > POLICY["resources"]["max_exact_integer_bits"]:
            raise RuntimeError("UNDETERMINED: exact integer bit limit")
        return c

    def mul(self, a, b):
        self.tick()
        return self.add((a[0]*b[0], a[0]*b[1]), (-a[1]*b[1], a[1]*b[0]))

    def put(self, matrix, key, value):
        value = self.add(matrix.get(key, (0, 0)), value)
        if value == (0, 0):
            matrix.pop(key, None)
        else:
            matrix[key] = value

    def decode(self, raw):
        """Independent bounded NPY decoder; no NumPy, pickle or saved code."""
        if len(raw) != 8652 or hashlib.sha256(raw).hexdigest() != INPUT_SHA:
            raise ValueError("UNDETERMINED: raw input identity")
        shapes = {"constant": (), "one_body": (8, 8), "lambdas": (12,),
                  "g_matrices": (12, 8, 8), "state_vector": (256,),
                  "sector_basis_indices": (36,), "sector_state_vector": (36,)}
        kinds = {"constant": "<f8", "lambdas": "<f8", "sector_basis_indices": "<i8"}
        result = {}
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            names = [k+".npy" for k in (*shapes, "metadata_json")]
            if sorted(archive.namelist()) != sorted(names):
                raise ValueError("UNDETERMINED: archive members or duplicates")
            if sum(i.file_size for i in archive.infolist()) > POLICY["resources"]["max_input_decode_bytes"]:
                raise ValueError("UNDETERMINED: decoded input byte cap")
            for key in (*shapes, "metadata_json"):
                blob = archive.read(key+".npy")
                if blob[:6] != b"\x93NUMPY" or blob[6:8] not in (b"\x01\x00", b"\x02\x00"):
                    raise ValueError("unsupported NPY version")
                hstart = 10 if blob[6] == 1 else 12
                hlen = int.from_bytes(blob[8:hstart], "little")
                if hlen > 8192 or hstart+hlen > len(blob):
                    raise ValueError("invalid NPY header")
                header = ast.literal_eval(blob[hstart:hstart+hlen].decode("latin1"))
                if set(header) != {"descr", "shape", "fortran_order"} or header["fortran_order"] is not False:
                    raise ValueError("unsupported NPY layout")
                payload = blob[hstart+hlen:]
                if key == "metadata_json":
                    descr = header["descr"]
                    if header["shape"] != () or not descr.startswith("<U") or len(payload) != 4*int(descr[2:]):
                        raise ValueError("metadata Unicode scalar required")
                    result[key] = json.loads(payload.decode("utf-32-le").rstrip("\0"))
                    continue
                if header["shape"] != shapes[key] or header["descr"] != kinds.get(key, "<c16"):
                    raise ValueError("input shape or dtype")
                count = math.prod(shapes[key])
                code = {"<f8": "d", "<c16": "dd", "<i8": "q"}[header["descr"]]
                if len(payload) != count*struct.calcsize("<"+code):
                    raise ValueError("input payload length")
                values = list(struct.iter_unpack("<"+code, payload))
                result[key] = [complex(*v) if code == "dd" else v[0] for v in values]
                if not all(math.isfinite(complex(v).real) and math.isfinite(complex(v).imag)
                           for v in result[key]):
                    raise ValueError("nonfinite input")
        meta = result["metadata_json"]
        required = {"hamiltonian_hash": H_SHA, "model": "linear_H4", "distance_angstrom": 1.0,
                    "basis": "sto-3g", "reference_rank": 12, "fragment_order": list(range(12))}
        if any(meta.get(k) != v for k, v in required.items()):
            raise ValueError("input model metadata")
        # H/state metadata alone is not an identity certificate; entire NPZ hash is.
        return result

    def dyadic(self, values):
        pairs = [(Fraction(float(complex(v).real)), Fraction(float(complex(v).imag))) for v in values]
        exponent = max(x.denominator.bit_length()-1 for p in pairs for x in p)
        return exponent, [(int(a*(1 << exponent)), int(b*(1 << exponent))) for a, b in pairs]

    def action(self, operators):
        """Operators listed left to right; application proceeds right to left.

        Creation inserts into the sorted wedge; annihilation removes. The list
        position supplies the fermion sign, independently of JW/mask products.
        """
        if operators not in self.transitions:
            entries = []
            for col, initial in enumerate(self.occupations):
                state, sign = list(initial), 1
                for mode, create in reversed(operators):
                    pos = bisect_left(state, mode)
                    occupied = pos < len(state) and state[pos] == mode
                    if occupied == create:
                        break
                    sign *= -1 if pos % 2 else 1
                    if create:
                        state.insert(pos, mode)
                    else:
                        state.pop(pos)
                else:
                    entries.append((self.occupation_index[tuple(state)], col, sign))
                self.tick()
            self.transitions[operators] = entries
        return self.transitions[operators]

    def one_body(self, g):
        exponent, coeff = self.dyadic(g)
        matrix = {}
        for p in range(8):
            for q in range(8):
                a = coeff[8*p+q]
                if a != (0, 0):
                    for row, col, sign in self.action(((p, True), (q, False))):
                        self.put(matrix, (row, col), (sign*a[0], sign*a[1]))
        return exponent, matrix

    def square(self, matrix):
        exponent, data = matrix
        by_col, out = {}, {}
        for (row, col), value in data.items():
            by_col.setdefault(col, []).append((row, value))
        for (middle, col), b in data.items():
            for row, a in by_col.get(middle, ()):
                self.put(out, (row, col), self.mul(a, b))
        return 2*exponent, out

    def contracted_square(self, g):
        """A²=Σ g_pq g_rs (δ_qr a†p a_s - a†p a†r a_q a_s)."""
        exponent, coeff = self.dyadic(g)
        out = {}
        for p in range(8):
            for q in range(8):
                for r in range(8):
                    for s in range(8):
                        a = self.mul(coeff[8*p+q], coeff[8*r+s])
                        if a == (0, 0):
                            continue
                        if q == r:
                            for row, col, sign in self.action(((p, True), (s, False))):
                                self.put(out, (row, col), (sign*a[0], sign*a[1]))
                        for row, col, sign in self.action(((p, True), (r, True), (q, False), (s, False))):
                            self.put(out, (row, col), (-sign*a[0], -sign*a[1]))
        return 2*exponent, out

    def scale(self, matrix, scalar):
        e, values = self.dyadic([scalar])
        return matrix[0]+e, {k: self.mul(v, values[0]) for k, v in matrix[1].items()
                             if values[0] != (0, 0)}

    def combine(self, matrices):
        exponent = max(m[0] for m in matrices)
        out = {}
        for e, data in matrices:
            for key, (a, b) in data.items():
                self.put(out, key, (a << (exponent-e), b << (exponent-e)))
        return exponent, out

    def scalar_matrix(self, scalar):
        exponent, value = self.dyadic([scalar])
        return exponent, { (i, i): value[0] for i in range(256) if value[0] != (0, 0) }

    def hermitian(self, matrix):
        data = matrix[1]
        return all(data.get((col, row), (0, 0)) == (v[0], -v[1])
                   for (row, col), v in data.items())

    def ladder_correspondence(self):
        # Tensor product of local 2x2 matrices, with mode0 the leftmost axis.
        # A local creation matrix has only row1,col0; parity prefixes are Z.
        for mode in range(8):
            for create in (False, True):
                tensor = {(0, 0): (1, 0)}
                for axis in range(8):
                    local = ([(0, 0, 1), (1, 1, -1)] if axis < mode else
                             [(int(create), int(not create), 1)] if axis == mode else
                             [(0, 0, 1), (1, 1, 1)])
                    tensor = {(2*r+a, 2*c+b): (v[0]*sign, 0)
                              for (r, c), v in tensor.items() for a, b, sign in local}
                    self.tick(len(tensor))
                exterior = {(r, c): (s, 0) for r, c, s in self.action(((mode, create),))}
                if tensor != exterior:
                    raise ValueError("FAIL: fermionic/JW tensor ladder correspondence")

    def trace_pauli(self, matrix):
        """All 4^8 coefficients, including exact zeros, via XOR diagonals.

        c(x,z)=(-i)^popcount(x&z)/256 Σ_s (-1)^popcount(s&z) H[s xor x,s].
        The FWHT uses integer additions only; this is not a JW polynomial DAG.
        """
        exponent, data = matrix
        out, zero_bitmap = {}, bytearray(8192)
        for x in range(256):
            diagonal = [data.get((s ^ x, s), (0, 0)) for s in range(256)]
            stride = 1
            while stride < 256:
                for start in range(0, 256, 2*stride):
                    for j in range(start, start+stride):
                        a, b = diagonal[j], diagonal[j+stride]
                        diagonal[j] = self.add(a, b)
                        diagonal[j+stride] = self.add(a, (-b[0], -b[1]))
                stride *= 2
            for z, value in enumerate(diagonal):
                phase = ((1, 0), (0, -1), (-1, 0), (0, 1))[(x & z).bit_count() % 4]
                a, b = self.mul(phase, value)
                if b:
                    raise ValueError("FAIL: exact Pauli coefficient imaginary part")
                index = 256*x+z
                if a == 0:
                    zero_bitmap[index // 8] |= 1 << (index % 8)
                else:
                    word = "".join("IXZY"[((x >> j) & 1) + 2*((z >> j) & 1)] for j in range(7, -1, -1))
                    out[word] = Fraction(a, 1 << (exponent+8))
        return out, {"cells_examined": 65536, "exact_nonzero": len(out),
                     "zero_bitmap_hex": zero_bitmap.hex(),
                     "zero_bitmap_sha256": hashlib.sha256(zero_bitmap).hexdigest()}

    def prepare_coefficients(self, arrays):
        self.ladder_correspondence()
        one = self.one_body(arrays["one_body"])
        if not self.hermitian(one):
            raise ValueError("FAIL: exact one-body Hermiticity")
        base = self.combine([one, self.scalar_matrix(arrays["constant"][0])])
        blocks = []
        for ell in range(12):
            g = arrays["g_matrices"][64*ell:64*(ell+1)]
            a = self.one_body(g)
            if not self.hermitian(a):
                raise ValueError("FAIL: exact factor Hermiticity")
            squared = self.square(a)
            if squared != self.contracted_square(g):
                raise ValueError("FAIL: A_l² contraction")
            blocks.append(self.scale(squared, arrays["lambdas"][ell]))
        full = self.combine([base, *blocks])
        full_map, full_support = self.trace_pauli(full)
        splits = {}
        for count in (3, 6, 9):
            tail = self.combine(blocks[count:])
            law, coverage = self.trace_pauli(tail)
            moved = law.pop("I"*8, Fraction(0))
            # Fraction scalar cannot pass through binary64: shift exact integer diagonal.
            e, data = tail
            shifted_e = max(e, moved.denominator.bit_length()-1)
            integer_move = moved*(1 << shifted_e)
            shifted = {k: (a << (shifted_e-e), b << (shifted_e-e)) for k, (a, b) in data.items()}
            for i in range(256):
                self.put(shifted, (i, i), (-int(integer_move), 0))
            hd = {p: full_map.get(p, 0)-law.get(p, 0) for p in set(full_map) | set(law)}
            hd = {p: c for p, c in hd.items() if c}
            lam = sum(abs(c) for c in law.values())
            splits[count] = {"law": law, "lambda": lam, "moved": moved, "hd": hd,
                             "matrix": (shifted_e, shifted), "support": coverage}
        return {"full_matrix": full, "full_map": full_map,
                "full_support": full_support, "splits": splits}

    def dc(self, value):
        return Decimal(value.numerator) / Decimal(value.denominator)

    def cmul(self, a, b):
        self.tick()
        return a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0]

    def dot(self, a, b):
        terms = [self.cmul((x[0], -x[1]), y) for x, y in zip(a, b)]
        return sum(t[0] for t in terms), sum(t[1] for t in terms)

    def matvec(self, matrix, vector):
        exponent, data = matrix
        scale = Decimal(2) ** (-exponent)
        out = [(Decimal(0), Decimal(0)) for _ in range(256)]
        for (row, col), (a, b) in sorted(data.items()):
            x, y = self.cmul((Decimal(a)*scale, Decimal(b)*scale), vector[col])
            out[row] = out[row][0]+x, out[row][1]+y
        return out

    def tensor_action(self, word, vector):
        # Apply local I/X/Y/Z to strided pairs. No v5 bitmask/phase cache.
        out = list(vector)
        for axis, letter in enumerate(word):
            stride = 2 ** (7-axis)
            for start in range(0, 256, 2*stride):
                for j in range(start, start+stride):
                    a, b = out[j], out[j+stride]
                    self.tick()
                    if letter == "X":
                        out[j], out[j+stride] = b, a
                    elif letter == "Y":
                        out[j], out[j+stride] = (b[1], -b[0]), (-a[1], a[0])
                    elif letter == "Z":
                        out[j+stride] = -b[0], -b[1]
        return out

    def state_metrics(self, arrays, prepared, count):
        with localcontext() as ctx:
            ctx.prec, ctx.rounding = 80, ROUND_HALF_EVEN
            vector = [(Decimal.from_float(z.real), Decimal.from_float(z.imag)) for z in arrays["state_vector"]]
            expected = [i for i, s in enumerate(self.occupations)
                        if sum(m % 2 == 0 for m in s) == 2 and sum(m % 2 == 1 for m in s) == 2]
            if arrays["sector_basis_indices"] != expected:
                raise ValueError("FAIL: occupation/sector basis order")
            embedded = [0j for _ in range(256)]
            for index, value in zip(expected, arrays["sector_state_vector"]):
                embedded[index] = value
            if embedded != arrays["state_vector"]:
                raise ValueError("FAIL: exact sector embedding")
            norm2, norm_imag = self.dot(vector, vector)
            if norm2 <= 0 or norm_imag or abs(norm2-1) > Decimal(2) ** -40:
                raise ValueError("FAIL: state norm²")
            full_psi = self.matvec(prepared["full_matrix"], vector)
            energy_re, energy_im = self.dot(vector, full_psi)
            energy = energy_re/norm2
            residual_vector = [(a-energy*x, b-energy*y)
                               for (a, b), (x, y) in zip(full_psi, vector)]
            residual = (self.dot(residual_vector, residual_vector)[0]/norm2).sqrt()
            split = prepared["splits"][count]
            if split["lambda"] <= 0:
                return {"status": "UNDETERMINED", "reason": "empty tail"}
            lam = self.dc(split["lambda"])
            hpsi = [(a/lam, b/lam) for a, b in self.matvec(split["matrix"], vector)]
            W = 1-self.dot(hpsi, hpsi)[0]/norm2
            events = []
            for word, coefficient in sorted(split["law"].items()):
                re, im = self.dot(vector, self.tensor_action(word, vector))
                if abs(im/norm2) > Decimal(2) ** -80:
                    raise ValueError("UNDETERMINED: reference expectation imaginary residue")
                events.append((self.dc(abs(coefficient)/split["lambda"]),
                               (1 if coefficient > 0 else -1)*re/norm2))
            mean = sum(p*a for p, a in events)
            variance = sum(p*(a-mean)**2 for p, a in events)
            if abs(mean-self.dot(vector, hpsi)[0]/norm2) > Decimal(2) ** -80:
                raise ValueError("UNDETERMINED: matrix/law mean disagreement")
            allowance = Decimal(2) ** -80
            if W < -allowance or W > 1+allowance or variance < -allowance or variance > W+2*allowance:
                raise ValueError("UNDETERMINED: reference W/v range")
            result = {"status": "PASS", "evidence": "CONDITIONAL_NOT_CERTIFIED",
                      "norm2": str(norm2), "W": str(W), "v": str(variance),
                      "rayleigh_energy_Ha": str(energy), "energy_imaginary_Ha": str(energy_im/norm2),
                      "full_H_residual_2_Ha": str(residual),
                      "exact_eigenstate_or_theorem_applicability_certified": False,
                      "rho": None, "reference_allowance": str(allowance)}
            if W <= allowance:
                result.update(status="UNDETERMINED", reason="W small or denominator interval contains zero")
            else:
                rho = (W+variance)/(2*W)
                lo = (1+(variance-allowance)/(W+allowance))/2
                hi = (1+(variance+allowance)/(W-allowance))/2
                if max(rho-lo, hi-rho) > Decimal(2) ** -12 or rho < Decimal("0.5") or rho > 1:
                    result.update(status="UNDETERMINED", reason="rho reference resolution/domain")
                else:
                    result.update(rho=str(rho), rho_interval=[str(lo), str(hi)])
            return result

    def coefficient_allowance(self, arrays, count):
        # Re-derive the input-only scale with rational arithmetic, without v5 Budget.
        scale = Fraction(0)
        for ell in range(count, 12):
            g = arrays["g_matrices"][64*ell:64*(ell+1)]
            mass = sum(abs(Fraction(z.real))+abs(Fraction(z.imag)) for z in g)
            scale += abs(Fraction(arrays["lambdas"][ell]))*mass*mass
        t = Fraction(713+12-count, 2**53)
        return t/(1-t)*scale

    def compare_law(self, arrays, split, row, count):
        """Union-support L1 comparisons in Ha; exact saved-float law relations."""
        table = row["canonical_table"]
        words = [e["word"] for e in table]
        if words != sorted(set(words)) or any(len(w) != 8 or set(w)-set("IXYZ") or w == "I"*8 for w in words):
            raise ValueError("FAIL: canonical words/order/duplicates/identity")
        table_raw = json.dumps(table, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if hashlib.sha256(table_raw).hexdigest() != row["canonical_table_sha256"]:
            raise ValueError("UNDETERMINED: comparator table hash")
        approx = {e["word"]: Fraction(float.fromhex(e["coefficient_hex"])) for e in table}
        if any(not c for c in approx.values()):
            raise ValueError("FAIL: saved zero coefficient was not removed")
        exact = split["law"]
        missing, spurious = sorted(set(exact)-set(approx)), sorted(set(approx)-set(exact))
        flipped = sorted(w for w in set(exact) & set(approx) if exact[w]*approx[w] < 0)
        error = sum(abs(exact.get(w, 0)-approx.get(w, 0)) for w in set(exact) | set(approx))
        alpha = self.coefficient_allowance(arrays, count)
        report = {"status": "PASS", "coefficient_L1_error_Ha": str(error),
                  "coefficient_L1_allowance_Ha": str(alpha),
                  "exact_only_missing": missing, "computed_only_spurious": spurious,
                  "sign_flips": flipped, "exact_support_status": "PASS",
                  "lambda_exact_Ha": str(split["lambda"])}
        # Conservative independent candidate: any support/sign discrepancy stops,
        # even though v5's approximate-law L1 theorem can accommodate small ones.
        if missing or spurious or flipped:
            report.update(status="FAIL", exact_support_status="FAIL",
                          reason="strict support/sign discrepancy requires investigation")
        if error > alpha:
            report.update(status="FAIL", reason="coefficient L1 envelope exceeded")
        lam_float = row["lambda_R"]
        if type(lam_float) not in (int, float) or not math.isfinite(lam_float) or lam_float <= 0:
            return dict(report, status="UNDETERMINED", reason="saved lambda nonpositive/nonfinite")
        lam = Fraction(lam_float)
        saved_mass = sum(abs(c) for c in approx.values())
        t = Fraction(65536+32, 2**53)
        summation = t/(1-t)*saved_mass
        if abs(lam-saved_mass) > summation or abs(lam-split["lambda"]) > alpha+summation:
            report.update(status="FAIL", reason="lambda_R mismatch")
        probability_mass = Fraction(0)
        for event in table:
            c = approx[event["word"]]
            if type(event["signed_event_sign"]) is not int or event["signed_event_sign"] != (1 if c > 0 else -1):
                report.update(status="FAIL", reason="signed event sign mismatch")
            p = event["probability"]
            if type(p) not in (int, float) or not math.isfinite(p) or p < 0:
                raise ValueError("FAIL: probability finite/nonnegative")
            if p != float(abs(c)/lam):
                report.update(status="FAIL", reason="probability is not correctly rounded abs(c_hat)/lambda_hat")
            probability_mass += Fraction(p)
        if abs(probability_mass-1) > Fraction(1, 2**46):
            report.update(status="FAIL", reason="saved probability mass")
        report["probability_mass_exact_saved_floats"] = str(probability_mass)
        phase = row["phase_ledger"]
        before, moved, after = (Fraction(phase[k]) for k in
            ("HD_Pauli_scalar_before", "HR_Pauli_scalar_before", "HD_Pauli_scalar_after"))
        shift_roundoff = Fraction(1, 2**53)*(abs(before)+abs(moved))
        if (abs(moved-split["moved"]) > alpha or abs(after-before-moved) > shift_roundoff
                or phase["HR_Pauli_scalar_after"] != 0
                or Fraction(phase["HD_added_phase_per_T_rel"]) != moved
                or Fraction(phase["HR_removed_phase_per_T_rel"]) != -moved
                or phase["net_relative_phase_change_per_T_rel"] != 0
                or Fraction(phase["HD_half_added_phase_per_delta"]) != moved/4):
            report.update(status="FAIL", reason="identity transfer/phase ledger")
        report["identity_move_exact_Ha"] = str(split["moved"])
        report["HD_nonidentity_comparison"] = "UNDETERMINED_NOT_SAVED_BY_V5"
        return report

    def compare_metrics(self, reference, saved):
        with localcontext() as ctx:
            ctx.prec, ctx.rounding = 80, ROUND_HALF_EVEN
            eps = Decimal(2) ** -80
            out = {"status": reference["status"], "checks": {},
                   "evidence": "CONDITIONAL_NOT_CERTIFIED"}
            for name, power in (("W", -24), ("v", -24), ("rho", -12)):
                if reference.get(name) is None or saved.get(name) is None:
                    status, diff = "UNDETERMINED", None
                elif type(saved[name]) not in (int, float) or not math.isfinite(saved[name]):
                    status, diff = "FAIL", "NONFINITE_OR_NONNUMERIC"
                else:
                    diff = abs(Decimal.from_float(float(saved[name]))-Decimal(reference[name]))
                    # Include the separately fixed reference allowance, never tune it.
                    radius = eps
                    if name == "rho":
                        lo, hi = map(Decimal, reference["rho_interval"])
                        radius = max(Decimal(reference[name])-lo, hi-Decimal(reference[name]))
                    status = "PASS" if diff+radius <= Decimal(2) ** power else "FAIL"
                    diff = str(diff)
                out["checks"][name] = {"status": status, "absolute_difference": diff,
                                       "units": "dimensionless", "tolerance": "2^"+str(power)}
            statuses = [out["status"], *(c["status"] for c in out["checks"].values())]
            out["status"] = "FAIL" if "FAIL" in statuses else "UNDETERMINED" if "UNDETERMINED" in statuses else "PASS"
            return out

    def evaluate(self, input_raw, comparator_raw, expected_comparator_sha256):
        """In-memory result builder for a FUTURE separately authorized adapter.

        This release has no executable caller. The future adapter must enforce
        provenance, isolation, aggregate caps and exclusive result persistence.
        """
        if hashlib.sha256(comparator_raw).hexdigest() != expected_comparator_sha256:
            raise ValueError("UNDETERMINED: prebound comparator identity")
        def pairs(items):
            out = {}
            for k, v in items:
                if k in out:
                    raise ValueError("duplicate comparator JSON key")
                out[k] = v
            return out
        saved = json.loads(comparator_raw, object_pairs_hook=pairs,
                           parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
        if (saved.get("schema") != "m1s_gate0_result_v5"
                or saved.get("input", {}).get("sha256") != INPUT_SHA
                or saved.get("config", {}).get("source_commit") != "69de022dc8142f703423b1ce0bd42f05e123954e"
                or saved.get("config", {}).get("source_sha256", {}).get(
                    "src/trottertracks/algorithm_codesign/m1s_gate0_v5/science.py") !=
                    "9a1d42916e972cb9b39af462678db1a70888fcab22a5de4f98d11fab0bc1a0ab"):
            raise ValueError("UNDETERMINED: comparator scope/source/input provenance")
        if [r.get("split") for r in saved["rows"]] != ["G3", "G6", "G9"]:
            raise ValueError("UNDETERMINED: all three ordered comparator rows required")
        arrays = self.decode(input_raw)
        prepared = self.prepare_coefficients(arrays)
        report = {"schema": "m1s_gate0_independent_result_v1", "status": "UNDETERMINED",
                  "input_sha256": INPUT_SHA, "comparator_sha256": expected_comparator_sha256,
                  "rows": [], "full_support": prepared["full_support"],
                  "coefficient_evidence": "EXACT_SAVED_BINARY64_ALGEBRA_IF_SOURCE_CORRECT",
                  "state_metric_evidence": "CONDITIONAL_NOT_CERTIFIED",
                  "mandatory_STOP": True, "unperformed": ["v5_reexecution", "sampling", "circuits", "compile", "finite_step", "scientific_GO"]}
        for count, row in zip((3, 6, 9), saved["rows"]):
            split = prepared["splits"][count]
            entry = {"split": "G"+str(count), "support": split["support"],
                     "exact_canonical_law": [{"word": w, "coefficient_Ha": str(c),
                         "probability": str(abs(c)/split["lambda"]), "sign": 1 if c > 0 else -1}
                         for w, c in sorted(split["law"].items())],
                     "exact_HD_coefficients_Ha": {w: str(c) for w, c in sorted(split["hd"].items())}}
            if not row.get("canonical_table") or row.get("phase_ledger") is None:
                entry.update(status="UNDETERMINED", reason="comparator law/identity fields missing")
            else:
                entry["law_comparison"] = self.compare_law(arrays, split, row, count)
                if entry["law_comparison"]["status"] != "PASS":
                    entry["status"] = entry["law_comparison"]["status"]
                else:
                    entry["reference_metrics"] = self.state_metrics(arrays, prepared, count)
                    entry["metric_comparison"] = self.compare_metrics(entry["reference_metrics"], row)
                    entry["status"] = entry["metric_comparison"]["status"]
            report["rows"].append(entry)
            if entry["status"] != "PASS":
                break  # STOP at first discrepancy; no implicit diagnosis/retry.
        completed = {r["split"] for r in report["rows"]}
        report["rows"].extend({"split": s, "status": "UNDETERMINED", "reason": "NOT_EVALUATED_AFTER_STOP"}
                              for s in ("G3", "G6", "G9") if s not in completed)
        statuses = [r["status"] for r in report["rows"]]
        report["status"] = "FAIL" if "FAIL" in statuses else "UNDETERMINED" if "UNDETERMINED" in statuses else "PASS"
        report["primitive_events"] = self.events
        return report
