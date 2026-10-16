"""New permit-bound input layer, AST-reusing saved-bit algebra from S4787cf90.

Reuse is explicit, not an independent reimplementation. No Pauli trace, W/v/rho
generation, old permits or old execution paths. Evaluated only in future pilot.
"""
from bisect import bisect_left
from fractions import Fraction
import ast,hashlib,io,json,math,struct,zipfile
from .control import POLICY,require_permit,read_json
INPUT_SHA=POLICY['input']['sha256']
H_SHA='de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424'

class ExactInput:
    def __init__(self,permit):
        require_permit(permit)
        self.events=0;self.transitions={};self.pauli_transitions={}
        self.occupations=[tuple(m for m in range(8) if (i>>(7-m))&1) for i in range(256)]
        self.occupation_index={s:i for i,s in enumerate(self.occupations)}
    def tick(self,count=1):
        self.events+=count
        if self.events>POLICY['resources']['max_exact_events']:raise RuntimeError('exact-event cap')
    def integer(self,value):
        if type(value) is not int or abs(value).bit_length()>POLICY['resources']['max_exact_integer_bits']:raise RuntimeError('exact integer bit cap')
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

    def matrix_digest(self, matrix):
        digest = hashlib.sha256()
        digest.update(('exponent='+str(matrix[0])+'\n').encode())
        for (r,c),(a,b) in sorted(matrix[1].items()):
            digest.update((str(r)+','+str(c)+':'+hex(a)+','+hex(b)+'\n').encode())
        return {'dyadic_exponent':matrix[0],'nonzero_cells':len(matrix[1]),'sha256':digest.hexdigest()}

    def targets(self,root):
        arrays=self.decode((root/POLICY['input']['path']).read_bytes())
        raw=(root/POLICY['saved_law']['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=POLICY['saved_law']['sha256']:raise ValueError('saved law bytes')
        saved=read_json(raw)
        if [x['split'] for x in saved['rows']]!=['G3','G6','G9']:raise ValueError('saved split ordering')
        base=self.combine([self.one_body(arrays['one_body']),self.scalar_matrix(arrays['constant'][0])])
        blocks=[self.scale(self.square(self.one_body(arrays['g_matrices'][64*l:64*(l+1)])),arrays['lambdas'][l]) for l in range(12)]
        full=self.combine([base,*blocks])
        if not all(self.hermitian(x) for x in [full,base,*blocks]):raise ValueError('exact Hermiticity')
        expected=[i for i in range(256) if sum(bool(i&(1<<(7-m))) for m in [0,2,4,6])==2 and sum(bool(i&(1<<(7-m))) for m in [1,3,5,7])==2]
        if arrays['sector_basis_indices']!=expected:raise ValueError('basis ordering')
        embedded=[0j]*256
        for i,z in zip(expected,arrays['sector_state_vector']):embedded[i]=z
        if embedded!=arrays['state_vector']:raise ValueError('state embedding')
        norm=sum(Fraction(z.real)**2+Fraction(z.imag)**2 for z in arrays['state_vector'])
        if norm<=0 or abs(norm-1)>Fraction(1,1<<40):raise ValueError('state norm')
        result={}
        for count,row in zip([3,6,9],saved['rows']):
            table=row['canonical_table'];canonical=json.dumps(table,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
            if hashlib.sha256(canonical).hexdigest()!=row['canonical_table_sha256']:raise ValueError('Pauli table hash')
            coeff={e['word']:Fraction(float.fromhex(e['coefficient_hex'])) for e in table}
            if len(coeff)>POLICY['resources']['max_saved_law_terms']:raise RuntimeError('saved-law term cap; no coefficient cutoff')
            if len(coeff)!=len(table) or list(coeff)!=sorted(coeff) or any(len(w)!=8 or set(w)-set('IXYZ') or w=='I'*8 or v==0 for w,v in coeff.items()):raise ValueError('canonical nonidentity signed law')
            lam=sum(abs(x) for x in coeff.values())
            if lam<=0:raise ValueError('empty tail')
            moved=row['phase_ledger']['HR_Pauli_scalar_before']
            hd=self.combine([base,*blocks[:count],self.scalar_matrix(moved)])
            hr=self.reconstruct(coeff)
            mismatch=self.combine([hd,hr,(full[0],{k:(-a,-b) for k,(a,b) in full[1].items()})])
            # Exact row-sum upper bound on ||HD+HR_saved-H_full||_2 for Hermitian difference.
            rows=[Fraction(0)]*256
            for (i,j),(a,b) in mismatch[1].items():rows[i]+=Fraction(abs(a)+abs(b),1<<mismatch[0])
            result['G'+str(count)]={'HD':hd,'HR':hr,'lambda':lam,'coefficients':coeff,
                'model_difference_operator_bound_Ha':max(rows),'saved_table_sha256':row['canonical_table_sha256'],
                'old_W_v_rho':{k:row[k] for k in ['W','v','rho']}}
        return {'full':full,'state':arrays['state_vector'],'norm2':norm,'splits':result,
                'full_matrix_identity':self.matrix_digest(full),'exact_events':self.events}
