"""UNEXECUTED independent reference, completed from published halted draft.

Occupation/CAR matrices and full Pauli traces use exact Gaussian integers and
powers of two. All state moments, W/v/rho and comparisons use exact rational
arithmetic. Decimal is for bounded display only, never a PASS tolerance.
No v5 helper imports; saved comparator values enter only compare_saved(), after
all independent references and their digest have been completed.
"""
from bisect import bisect_left
from decimal import Decimal, localcontext
from fractions import Fraction
import ast
import hashlib
import io
import json
import math
import struct
import zipfile
from .control import require_execution, INPUT_SHA, H_SHA, POLICY, read_json


class Reference:
    def __init__(self, authorization=None):
        require_execution(authorization)
        self.events = 0
        self.transitions = {}
        self.pauli_transitions = {}
        self.pauli_expectations = {}
        self.progress = {'phase':'INITIALIZED', 'completed_reference_splits':[], 'comparisons':[]}
        self.occupations = [tuple(m for m in range(8) if (i >> (7-m)) & 1)
                            for i in range(256)]
        self.occupation_index = {s:i for i,s in enumerate(self.occupations)}

    def tick(self, count=1):
        self.events += count
        if self.events > POLICY['resources']['max_primitive_events']:
            raise RuntimeError('UNDETERMINED: primitive event cap')

    def integer(self, value):
        if type(value) is not int or abs(value).bit_length() > POLICY['resources']['max_exact_integer_bits']:
            raise RuntimeError('UNDETERMINED: exact integer bit cap/type')
        return value

    def q(self, value):
        value = Fraction(value)
        self.integer(value.numerator)
        self.integer(value.denominator)
        return value

    def add(self, a, b):
        self.tick()
        return self.integer(a[0]+b[0]), self.integer(a[1]+b[1])

    def mul(self, a, b):
        self.tick()
        return self.add((a[0]*b[0], a[0]*b[1]), (-a[1]*b[1], a[1]*b[0]))

    def put(self, matrix, key, value):
        value = self.add(matrix.get(key,(0,0)),value)
        if value == (0,0):
            matrix.pop(key,None)
        else:
            matrix[key] = value
        if len(matrix) > POLICY['resources']['max_matrix_cells']:
            raise RuntimeError('UNDETERMINED: matrix cell cap')

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
            if len(self.transitions) >= POLICY['resources']['max_transition_cache_entries']:
                raise RuntimeError('UNDETERMINED: fermion transition cache cap')
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

    def shifted_identity(self, matrix, amount):
        exponent, data = matrix
        e = max(exponent,amount.denominator.bit_length()-1)
        move = int(amount*(1 << e))
        out = {k:(a << (e-exponent),b << (e-exponent)) for k,(a,b) in data.items()}
        for i in range(256):
            self.put(out,(i,i),(move,0))
        return e,out

    def tensor_word(self, word):
        # Independent literal local matrices, no JW polynomial or v5 mask cache.
        if word not in self.pauli_transitions:
            local = {'I':[(0,0,(1,0)),(1,1,(1,0))],
                     'X':[(0,1,(1,0)),(1,0,(1,0))],
                     'Y':[(0,1,(0,-1)),(1,0,(0,1))],
                     'Z':[(0,0,(1,0)),(1,1,(-1,0))]}
            entries = [(0,0,(1,0))]
            for letter in word:
                entries = [(2*r+a,2*c+b,self.mul(v,w))
                           for r,c,v in entries for a,b,w in local[letter]]
            if len(self.pauli_transitions) >= POLICY['resources']['max_pauli_cache_entries']:
                self.pauli_transitions.pop(next(iter(self.pauli_transitions)))
            self.pauli_transitions[word] = entries
        return self.pauli_transitions[word]

    def reconstruct(self, coefficients):
        exponent = max((c.denominator.bit_length()-1 for c in coefficients.values()),default=0)
        out = {}
        for word,c in sorted(coefficients.items()):
            numerator = int(c*(1 << exponent))
            for row,col,phase in self.tensor_word(word):
                self.put(out,(row,col),self.mul((numerator,0),phase))
        return exponent,out

    def matrix_equal(self, a, b):
        e = max(a[0],b[0])
        return all((a[1].get(k,(0,0))[0] << (e-a[0]),a[1].get(k,(0,0))[1] << (e-a[0])) ==
                   (b[1].get(k,(0,0))[0] << (e-b[0]),b[1].get(k,(0,0))[1] << (e-b[0]))
                   for k in set(a[1]) | set(b[1]))

    def matrix_witness(self, label, a, b):
        self.progress['failure_witness'] = {'label':label,'left':self.matrix_digest(a),
                                            'right':self.matrix_digest(b)}

    def matrix_digest(self, matrix):
        digest = hashlib.sha256()
        digest.update(('exponent='+str(matrix[0])+'\n').encode())
        for (r,c),(a,b) in sorted(matrix[1].items()):
            digest.update((str(r)+','+str(c)+':'+hex(a)+','+hex(b)+'\n').encode())
        return {'dyadic_exponent':matrix[0],'nonzero_cells':len(matrix[1]),'sha256':digest.hexdigest()}

    def coefficient_digest(self, coefficients):
        digest = hashlib.sha256()
        for w,c in sorted(coefficients.items()):
            digest.update((w+':'+hex(c.numerator)+'/'+hex(c.denominator)+'\n').encode())
        return digest.hexdigest()

    def encoded(self, value):
        value = self.q(value)
        with localcontext() as ctx:
            ctx.prec = 32
            display = str(Decimal(value.numerator)/Decimal(value.denominator))
        return {'numerator_hex':hex(value.numerator),'denominator_hex':hex(value.denominator),
                'decimal_display_only':display}

    def prepare_coefficients(self, arrays):
        self.ladder_correspondence()
        one = self.one_body(arrays['one_body'])
        if not self.hermitian(one):
            raise ValueError('FAIL: exact one-body Hermiticity')
        base = self.combine([one,self.scalar_matrix(arrays['constant'][0])])
        blocks = []
        for ell in range(12):
            g = arrays['g_matrices'][64*ell:64*(ell+1)]
            a = self.one_body(g)
            if not self.hermitian(a):
                raise ValueError('FAIL: exact factor Hermiticity')
            squared = self.square(a)
            contracted = self.contracted_square(g)
            if squared != contracted:
                self.matrix_witness('A_squared_vs_CAR_block_'+str(ell),squared,contracted)
                raise ValueError('FAIL: A_l squared/CAR contraction')
            blocks.append(self.scale(squared,arrays['lambdas'][ell]))
            self.progress['completed_exact_blocks'] = ell+1
        full = self.combine([base,*blocks])
        full_map,full_support = self.trace_pauli(full)
        reconstructed_full = self.reconstruct(full_map)
        if not self.matrix_equal(full,reconstructed_full):
            self.matrix_witness('full_trace_tensor',full,reconstructed_full)
            raise ValueError('FAIL: full-H trace/tensor reconstruction')
        splits = {}
        for count in (3,6,9):
            tail = self.combine(blocks[count:])
            law,coverage = self.trace_pauli(tail)
            moved = law.pop('I'*8,Fraction(0))
            hr = self.shifted_identity(tail,-moved)
            hd_before = self.combine([base,*blocks[:count]])
            hd_matrix = self.shifted_identity(hd_before,moved)
            hd,hd_support = self.trace_pauli(hd_matrix)
            difference = {w:full_map.get(w,0)-law.get(w,0) for w in set(full_map)|set(law)}
            difference = {w:c for w,c in difference.items() if c}
            if hd != difference or not self.matrix_equal(self.combine([hd_matrix,hr]),full):
                raise ValueError('FAIL: exact full/HD/HR/identity decomposition')
            if not self.matrix_equal(hr,self.reconstruct(law)) or not self.matrix_equal(hd_matrix,self.reconstruct(hd)):
                raise ValueError('FAIL: split trace/tensor reconstruction')
            bitmap = bytearray.fromhex(coverage['zero_bitmap_hex'])
            bitmap[0] |= 1
            coverage.update(exact_nonzero=len(law),zero_bitmap_hex=bitmap.hex(),zero_bitmap_sha256=hashlib.sha256(bitmap).hexdigest(),identity_removed=True)
            lam = self.q(sum(abs(c) for c in law.values()))
            splits[count] = {'law':law,'lambda':lam,'moved':moved,'hd':hd,
                             'hd_scalar_before':hd.get('I'*8,Fraction(0))-moved,
                             'matrix':hr,'hd_matrix':hd_matrix,'support':coverage,'hd_support':hd_support,
                             'coefficient_allowance':self.coefficient_allowance(arrays,count)}
            self.progress['completed_exact_splits'] = ['G'+str(k) for k in splits]
        return {'full_matrix':full,'full_map':full_map,'full_support':full_support,'splits':splits,
                'global_coefficient_allowance':self.global_coefficient_allowance(arrays)}

    def dot(self, a, b):
        total = (0,0)
        for x,y in zip(a,b):
            total = self.add(total,self.mul((x[0],-x[1]),y))
        return total

    def matvec(self, matrix, vector):
        out = [(0,0) for _ in range(256)]
        for (row,col),value in sorted(matrix[1].items()):
            out[row] = self.add(out[row],self.mul(value,vector[col]))
        return out

    def tensor_action(self, word, vector):
        # Strided local-state pairs: independently implemented from tensor_word.
        out = list(vector)
        for axis,letter in enumerate(word):
            stride = 2 ** (7-axis)
            for start in range(0,256,2*stride):
                for j in range(start,start+stride):
                    a,b = out[j],out[j+stride]
                    self.tick()
                    if letter == 'X':
                        out[j],out[j+stride] = b,a
                    elif letter == 'Y':
                        out[j],out[j+stride] = (b[1],-b[0]),(-a[1],a[0])
                    elif letter == 'Z':
                        out[j+stride] = -b[0],-b[1]
        return out

    def state_metrics(self, arrays, prepared, count):
        exponent,vector = self.dyadic(arrays['state_vector'])
        expected = [i for i,s in enumerate(self.occupations)
                    if sum(m % 2 == 0 for m in s)==2 and sum(m % 2 == 1 for m in s)==2]
        if arrays['sector_basis_indices'] != expected:
            raise ValueError('FAIL: occupation/sector basis ordering')
        embedded = [0j for _ in range(256)]
        for index,value in zip(expected,arrays['sector_state_vector']):
            embedded[index] = value
        if embedded != arrays['state_vector']:
            raise ValueError('FAIL: exact sector embedding')
        norm,imag = self.dot(vector,vector)
        norm2 = self.q(Fraction(norm,1 << (2*exponent)))
        if norm <= 0 or imag or abs(norm2-1) > Fraction(1,1 << 40):
            raise ValueError('FAIL: state norm quality')
        full = prepared['full_matrix']
        full_psi = self.matvec(full,vector)
        er,ei = self.dot(vector,full_psi)
        if ei:
            raise ValueError('FAIL: exact Hermitian energy imaginary part')
        energy = self.q(Fraction(er,(1 << full[0])*norm))
        residual2 = self.q(Fraction(self.dot(full_psi,full_psi)[0],(1 << (2*full[0]))*norm)-energy*energy)
        if residual2 < 0:
            raise ValueError('FAIL: exact residual squared sign')
        oracle_quality = residual2 <= Fraction(1,10**16)
        split = prepared['splits'][count]
        lam = split['lambda']
        if lam <= 0:
            return {'status':'UNDETERMINED','reason':'empty tail','W':None,'v':None,'rho':None,'norm2':norm2}
        hpsi = self.matvec(split['matrix'],vector)
        W = self.q(1-Fraction(self.dot(hpsi,hpsi)[0],(1 << (2*split['matrix'][0]))*norm)/(lam*lam))
        ce = max(c.denominator.bit_length()-1 for c in split['law'].values())
        mass,first,second = 0,0,0
        for word,coefficient in sorted(split['law'].items()):
            if word not in self.pauli_expectations:
                ar,ai = self.dot(vector,self.tensor_action(word,vector))
                if ai:
                    raise ValueError('FAIL: exact Pauli expectation imaginary part')
                self.pauli_expectations[word] = ar
            ar = self.pauli_expectations[word]
            ci = int(coefficient*(1 << ce))
            mass = self.integer(mass+abs(ci))
            first = self.integer(first+ci*ar)
            second = self.integer(second+abs(ci)*ar*ar)
        if Fraction(mass,1 << ce) != lam:
            raise ValueError('FAIL: exact law mass')
        mean = self.q(Fraction(first,mass*norm))
        variance = self.q(Fraction(second,mass*norm*norm)-mean*mean)
        mean_matrix,mean_imag = self.dot(vector,hpsi)
        if mean_imag or mean != Fraction(mean_matrix,(1 << split['matrix'][0])*norm)/lam:
            raise ValueError('FAIL: exact independent matrix/law mean')
        if not 0 <= variance <= W <= 1:
            raise ValueError('FAIL: exact W/v algebraic range')
        rho = self.q((W+variance)/(2*W)) if W > 0 else None
        return {'status':'PASS' if rho is not None and oracle_quality else 'UNDETERMINED',
                'reason':None if rho is not None and oracle_quality else 'zero W or exact absolute oracle quality unresolved',
                'norm2':norm2,'W':W,'v':variance,'rho':rho,'rayleigh_energy_Ha':energy,
                'full_H_residual_squared_Ha2':residual2,'exact_oracle_absolute_quality':oracle_quality,
                'reference_error_allowance':Fraction(0)}

    def coefficient_allowance(self, arrays, count):
        scale = Fraction(0)
        for ell in range(count,12):
            g = arrays['g_matrices'][64*ell:64*(ell+1)]
            mass = sum(abs(Fraction(z.real))+abs(Fraction(z.imag)) for z in g)
            scale += abs(Fraction(arrays['lambdas'][ell]))*mass*mass
        t = Fraction(713+12-count,1 << 53)
        return self.q(t/(1-t)*scale)

    def global_coefficient_allowance(self, arrays):
        scale = abs(Fraction(arrays['constant'][0]))+sum(abs(Fraction(z.real))+abs(Fraction(z.imag)) for z in arrays['one_body'])
        for ell in range(12):
            mass = sum(abs(Fraction(z.real))+abs(Fraction(z.imag)) for z in arrays['g_matrices'][64*ell:64*(ell+1)])
            scale += abs(Fraction(arrays['lambdas'][ell]))*mass*mass
        a,b = Fraction(24000064,1 << 53),Fraction(65568,1 << 53)
        return self.q(2*a/(1-a)*scale*(1+b/(1-b)))

    def public_metrics(self, metrics):
        return {k:self.encoded(v) if isinstance(v,Fraction) else v for k,v in metrics.items()}

    def build(self, input_raw):
        # This entire phase accepts only the frozen NPZ, never comparator bytes.
        self.progress['phase'] = 'INDEPENDENT_REFERENCE'
        arrays = self.decode(input_raw)
        prepared = self.prepare_coefficients(arrays)
        metrics = {}
        public = {'full_matrix':self.matrix_digest(prepared['full_matrix']),
                  'full_coefficients_sha256':self.coefficient_digest(prepared['full_map']),
                  'full_support':prepared['full_support'],'splits':{}}
        self.progress['partial_reference'] = public
        for count in (3,6,9):
            metrics[count] = self.state_metrics(arrays,prepared,count)
            split = prepared['splits'][count]
            public['splits']['G'+str(count)] = {
                'HR_matrix':self.matrix_digest(split['matrix']),'HD_matrix':self.matrix_digest(split['hd_matrix']),
                'HR_coefficients_sha256':self.coefficient_digest(split['law']),
                'HD_coefficients_sha256':self.coefficient_digest(split['hd']),
                'HR_support':split['support'],'HD_support':split['hd_support'],
                'identity_moved_Ha':self.encoded(split['moved']),'lambda_R_Ha':self.encoded(split['lambda']),
                'metrics':self.public_metrics(metrics[count])}
            self.progress['completed_reference_splits'].append('G'+str(count))
        raw = json.dumps(public,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        self.progress.update(phase='INDEPENDENT_REFERENCE_FROZEN',independent_reference_sha256=hashlib.sha256(raw).hexdigest())
        return {'arrays':arrays,'prepared':prepared,'metrics':metrics,'public':public,
                'independent_reference_sha256':self.progress['independent_reference_sha256']}

    def rounded_binary64(self, value, exact):
        # Integer nearest-even cell test; never reuse Python float division.
        if type(value) is not float or not math.isfinite(value):
            return False
        bits = struct.unpack('<Q',struct.pack('<d',value))[0]
        if bits >> 63:
            return self.rounded_binary64(-value,-exact)
        if bits >= 0x7fefffffffffffff or exact < 0:
            return False
        q = Fraction(value)
        previous = Fraction(struct.unpack('<d',struct.pack('<Q',bits-1))[0]) if bits else Fraction(0)
        following = Fraction(struct.unpack('<d',struct.pack('<Q',bits+1))[0])
        lower,upper = (previous+q)/2,(q+following)/2
        return (lower <= exact <= upper) if bits % 2 == 0 else (lower < exact < upper)

    def difference_F_squared(self, a, b):
        e = max(a[0],b[0]);total = 0
        for k in set(a[1]) | set(b[1]):
            x,y = a[1].get(k,(0,0)),b[1].get(k,(0,0))
            dr,di = (x[0] << (e-a[0]))-(y[0] << (e-b[0])),(x[1] << (e-a[0]))-(y[1] << (e-b[0]))
            total = self.integer(total+dr*dr+di*di)
            self.tick()
        return self.q(Fraction(total,1 << (2*e)))

    def compare_law(self, prepared, count, row):
        split = prepared['splits'][count]
        table = row['canonical_table'];words = [e['word'] for e in table]
        if words != sorted(set(words)) or any(len(w)!=8 or set(w)-set('IXYZ') or w=='I'*8 for w in words):
            raise ValueError('FAIL: comparator canonical word/order/identity')
        raw = json.dumps(table,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        if hashlib.sha256(raw).hexdigest() != row['canonical_table_sha256']:
            raise ValueError('UNDETERMINED: comparator table digest')
        approx = {e['word']:Fraction(float.fromhex(e['coefficient_hex'])) for e in table}
        if any(not x for x in approx.values()):
            raise ValueError('FAIL: saved zero coefficient present')
        exact = split['law'];union = sorted(set(exact)|set(approx));alpha = split['coefficient_allowance']
        missing,spurious = sorted(set(exact)-set(approx)),sorted(set(approx)-set(exact))
        flipped = sorted(w for w in set(exact)&set(approx) if exact[w]*approx[w] < 0)
        error = self.q(sum(abs(exact.get(w,0)-approx.get(w,0)) for w in union))
        status = 'FAIL' if error > alpha else 'UNDETERMINED' if missing or spurious or flipped else 'PASS'
        witness_words = [w for w in union if exact.get(w,0)!=approx.get(w,0)][:64]
        report = {'status':status,'coefficient_L1_error_Ha':self.encoded(error),'coefficient_L1_allowance_Ha':self.encoded(alpha),
                  'support_missing_count':len(missing),'support_spurious_count':len(spurious),'sign_flip_count':len(flipped),
                  'support_or_sign_witnesses':{'missing':missing[:64],'spurious':spurious[:64],'flipped':flipped[:64]},
                  'coefficient_difference_witnesses':[{'word':w,'exact_Ha':self.encoded(exact.get(w,Fraction(0))),
                       'saved_Ha':self.encoded(approx.get(w,Fraction(0)))} for w in witness_words]}
        lam_float = row['lambda_R']
        if type(lam_float) is not float or not math.isfinite(lam_float) or lam_float <= 0:
            return dict(report,status='FAIL',reason='invalid saved lambda')
        lam = Fraction(lam_float);mass = sum(abs(c) for c in approx.values())
        t = Fraction(65568,1 << 53);summation = t/(1-t)*mass
        if abs(lam-mass) > summation or abs(lam-split['lambda']) > alpha+summation:
            report.update(status='FAIL',reason='lambda discrepancy')
        probability_mass = Fraction(0)
        for event in table:
            coeff = approx[event['word']];prob = event['probability']
            if type(event['signed_event_sign']) is not int or event['signed_event_sign'] != (1 if coeff > 0 else -1) or type(prob) is not float or not math.isfinite(prob) or prob < 0:
                raise ValueError('FAIL: saved sign/probability structure')
            if not self.rounded_binary64(prob,abs(coeff)/lam):
                report.update(status='FAIL',reason='probability rounding relation')
            probability_mass += Fraction(prob)
        if abs(probability_mass-1) > Fraction(1,1 << 46):
            report.update(status='FAIL',reason='probability mass quality')
        report['saved_probability_mass'] = self.encoded(probability_mass)
        saved_matrix = self.reconstruct(approx)
        f2 = self.difference_F_squared(saved_matrix,split['matrix'])
        if f2 > 256*alpha*alpha:
            report.update(status='FAIL',reason='HR matrix Frobenius envelope')
        report['HR_matrix_F_squared_Ha2'] = self.encoded(f2)
        phase = row['phase_ledger']
        before,moved,after = (Fraction(phase[k]) for k in ('HD_Pauli_scalar_before','HR_Pauli_scalar_before','HD_Pauli_scalar_after'))
        global_alpha = prepared['global_coefficient_allowance']
        t_sum = Fraction(34,1 << 53)
        identity_sum_error = self.q(abs(after-before-moved))
        identity_sum_allowance = self.q(t_sum/(1-t_sum)*(abs(before)+abs(moved)))
        report['identity_sum_error_Ha'] = self.encoded(identity_sum_error)
        report['identity_sum_allowance_Ha'] = self.encoded(identity_sum_allowance)
        report['identity_sum_nearest_even_relation_diagnostic_only'] = self.rounded_binary64(float(phase['HD_Pauli_scalar_after']),before+moved)
        if abs(moved-split['moved']) > alpha or abs(before-split['hd_scalar_before']) > global_alpha or identity_sum_error > identity_sum_allowance or phase['HR_Pauli_scalar_after'] != 0 or Fraction(phase['HD_added_phase_per_T_rel']) != moved or Fraction(phase['HR_removed_phase_per_T_rel']) != -moved or phase['net_relative_phase_change_per_T_rel'] != 0 or Fraction(phase['HD_half_added_phase_per_delta']) != moved/4:
            report.update(status='FAIL',reason='identity/phase ledger discrepancy')
        report['saved_full_HD_nonidentity_match'] = 'UNDETERMINED_NOT_SAVED_BY_V5'
        return report

    def compare_metrics(self, exact, saved):
        out = {'status':exact['status'],'checks':{},'reference_arithmetic':'EXACT_RATIONAL_IF_SOURCE_CORRECT'}
        for key,power in [('W',24),('v',24),('rho',12)]:
            value = saved.get(key)
            if exact.get(key) is None:
                status,diff = 'UNDETERMINED',None
            elif type(value) is not float or not math.isfinite(value):
                status,diff = 'FAIL',None
            else:
                diff = self.q(abs(Fraction(value)-exact[key]));status = 'PASS' if diff <= Fraction(1,1 << power) else 'FAIL'
            out['checks'][key] = {'status':status,'absolute_difference':self.encoded(diff) if diff is not None else None,'tolerance':'2^-'+str(power),'units':'dimensionless'}
        states = [out['status'],*[x['status'] for x in out['checks'].values()]]
        out['status'] = 'FAIL' if 'FAIL' in states else 'UNDETERMINED' if 'UNDETERMINED' in states else 'PASS'
        return out

    def compare_saved(self, independent, comparator_raw):
        if self.progress['phase'] != 'INDEPENDENT_REFERENCE_FROZEN':
            raise RuntimeError('UNDETERMINED: reference not frozen before comparator')
        if hashlib.sha256(comparator_raw).hexdigest() != POLICY['comparator']['sha256']:
            raise ValueError('UNDETERMINED: comparator bytes')
        # First comparator JSON parse: construction/metrics have all completed.
        saved = read_json(comparator_raw)
        if saved.get('schema') != 'm1s_gate0_result_v5' or saved.get('input',{}).get('sha256') != INPUT_SHA or saved.get('config',{}).get('source_commit') != POLICY['v5_source']['commit'] or [r.get('split') for r in saved.get('rows',[])] != ['G3','G6','G9']:
            raise ValueError('UNDETERMINED: comparator provenance/split scope')
        rows = []
        for count,row in zip((3,6,9),saved['rows']):
            law = self.compare_law(independent['prepared'],count,row)
            metrics = self.compare_metrics(independent['metrics'][count],row)
            statuses = [law['status'],metrics['status']]
            status = 'FAIL' if 'FAIL' in statuses else 'UNDETERMINED' if 'UNDETERMINED' in statuses else 'PASS'
            entry = {'split':'G'+str(count),'available_comparator_status':status,'law_comparison':law,'metric_comparison':metrics,
                     'full_requested_scope_status':'FAIL' if status=='FAIL' else 'UNDETERMINED',
                     'missing_comparator':'v5 full-H matrix / HD nonidentity coefficients and matrix were not saved'}
            rows.append(entry);self.progress['comparisons'].append(entry)
            if status != 'PASS':
                break
        done = {r['split'] for r in rows}
        rows.extend({'split':s,'available_comparator_status':'UNDETERMINED','full_requested_scope_status':'UNDETERMINED','reason':'NOT_EVALUATED_AFTER_STOP'} for s in ('G3','G6','G9') if s not in done)
        statuses = [r['available_comparator_status'] for r in rows]
        primary = 'FAIL' if 'FAIL' in statuses else 'UNDETERMINED' if 'UNDETERMINED' in statuses else 'PASS'
        return {'status':'FAIL' if primary=='FAIL' else 'UNDETERMINED','available_comparator_status':primary,'rows':rows,
                'full_scope_limitation':'Missing frozen full/HD comparator prevents full requested scope PASS',
                'independent_reference_sha256':independent['independent_reference_sha256'],'reference':independent['public']}
