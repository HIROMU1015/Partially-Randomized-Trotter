"""Read fixed numeric QPY v13 as data only. Never import Qiskit or build circuits.

Binary field order follows installed Qiskit 1.3.0 qpy/formats.py and
qpy/binary_io/circuits.py. Unsupported conditions/calibrations/layouts fail.
Symbolic/numpy values remain opaque bytes, never evaluated or unpickled.
"""
from collections import Counter
import hashlib
import math
import struct
import h2_pr_identity_recovery as r


class Reader:
    def __init__(self, raw):
        self.raw, self.pos = raw, 0

    def take(self, count):
        r.require(type(count) is int and 0 <= count <= len(self.raw) - self.pos, 'QPY_TRUNCATED_OR_SIZE')
        result = self.raw[self.pos:self.pos+count]; self.pos += count
        return result

    def unpack(self, fmt):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))

    def text(self, count):
        return self.take(count).decode('utf8')

    def value(self, kind, raw, *, instruction=False):
        # QPY v13 deliberately uses little endian for instruction numeric
        # parameters, but network endian for the global phase value.
        endian = '<' if instruction else '!'
        if kind == b'f':
            r.require(len(raw) == 8, 'QPY_FLOAT_SIZE')
            number=struct.unpack(endian+'d',raw)[0]
            r.require(math.isfinite(number),'QPY_NONFINITE_FLOAT')
            return dict(kind='float', hex=number.hex())
        if kind == b'i':
            r.require(len(raw) == 8, 'QPY_INTEGER_SIZE')
            return dict(kind='integer', value=struct.unpack(endian+'q', raw)[0])
        return dict(kind='opaque:'+kind.decode('ascii'), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())

    def instruction(self, nq, nc):
        ns, ls, np, nqs, ncs, condition, crs, cv, controls, ctrlstate = self.unpack('!HHHIIBHqII')
        name = self.text(ns); label = self.text(ls); cr = self.text(crs)
        r.require(condition == 0 and crs == 0, 'QPY_CONDITION_UNSUPPORTED')
        wires=[];clbits=[]
        for _ in range(nqs if nq is not None else 0):
            kind, idx = self.unpack('!cI'); r.require(kind == b'q' and idx < nq, 'QPY_QUBIT_WIRE')
            wires.append(idx)
        for _ in range(ncs if nc is not None else 0):
            kind, idx = self.unpack('!cI'); r.require(kind == b'c' and idx < nc, 'QPY_CLBIT_WIRE')
            clbits.append(idx)
        params=[]
        for _ in range(np):
            kind, size=self.unpack('!cQ');params.append(self.value(kind,self.take(size),instruction=True))
        return dict(name=name,label=label,wires=wires,clbits=clbits,params=params,
                    controls=controls,ctrl_state=ctrlstate)

    def circuit(self):
        ns, phasekind, ps, nq, nc, ms, nr, ni, nv = self.unpack('!HcHIIQIQI')
        name=self.text(ns);phase=self.value(phasekind,self.take(ps));metadata=r.loads(self.take(ms))
        r.require(nv == 0, 'QPY_CLASSICAL_VARS_UNSUPPORTED')
        for _ in range(nr):
            kind, standalone, size, namesize, incircuit=self.unpack('!c?IH?')
            self.take(namesize);self.take(8*size)
            r.require(kind in (b'q',b'c'), 'QPY_REGISTER_KIND')
        count,=self.unpack('!Q');custom={}
        for _ in range(count):
            size, kind, dnq, dnc, hasbody, bodysize, controls, ctrlstate, basesize=self.unpack('!HcII?QIIQ')
            customname=self.text(size)
            r.require(customname not in custom and not customname.startswith('###PauliEvolutionGate_'), 'QPY_CUSTOM_KIND')
            body=None
            if hasbody:
                parser=Reader(self.take(bodysize));body=parser.circuit()
                r.require(parser.pos == len(parser.raw), 'QPY_CUSTOM_TRAILING')
            else:r.require(bodysize == 0, 'QPY_CUSTOM_BODY_SIZE')
            base=self.take(basesize); base_operation=None
            if base:
                parser=Reader(base);base_operation=parser.instruction(None,None)
                r.require(parser.pos==len(base),'QPY_BASE_TRAILING')
            custom[customname]=dict(kind=kind.decode('ascii'),nq=dnq,nc=dnc,body=body,
                controls=controls,ctrl_state=ctrlstate,base_bytes=len(base),base_sha256=hashlib.sha256(base).hexdigest(),
                base_operation=base_operation)
        instructions=[self.instruction(nq,nc) for _ in range(ni)]
        cals,=self.unpack('!H');r.require(cals == 0, 'QPY_CALIBRATIONS_UNSUPPORTED')
        exists, initial, mapping, final, extra, inputcount=self.unpack('!?iiiIi')
        r.require(exists is False and extra == 0, 'QPY_LAYOUT_UNSUPPORTED')
        return dict(name=name,nq=nq,nc=nc,phase=phase,metadata=metadata,
                    custom=custom,instructions=instructions)


def read_qpy(raw):
    p=Reader(raw);magic,version,major,minor,patch,count,encoding=p.unpack('!6sBBBBQc')
    r.require(magic==b'QISKIT' and version==13 and (major,minor,patch)==(1,3,0) and count==1,
              'FIXED_QPY_13_QISKIT_130_SINGLE_CIRCUIT')
    r.require(encoding in (b'e',b'p'), 'QPY_SYMBOLIC_ENCODING')
    kind=p.take(1);r.require(kind==b'q','QPY_CIRCUIT_TYPE')
    result=p.circuit();r.require(p.pos==len(raw),'QPY_TRAILING_BYTES')
    result['qpy_version']=version
    return result


def stream_digest(circuit):
    """Serialized stream identity, distinct from Qiskit's recursive fingerprint."""
    return r.digest(dict(nq=circuit['nq'],nc=circuit['nc'],phase=circuit['phase'],
        instructions=circuit['instructions'],custom=circuit['custom']))


def summary(circuit):
    return dict(nq=circuit['nq'],nc=circuit['nc'],phase=circuit['phase'],
        instruction_count=len(circuit['instructions']),
        names=dict(Counter('CUSTOM' if x['name'] in circuit['custom'] else x['name'] for x in circuit['instructions'])),
        custom_definitions=len(circuit['custom']),metadata_keys=list(circuit['metadata'] or {}),
        stream_sha256=stream_digest(circuit),qpy_version=circuit.get('qpy_version',13),
        circuit_object_created=False,scientific_libraries_imported=False)


def structural_digest(circuit, instructions=None):
    """UUID/label-independent structural byte fields, not an operator proof.

    Custom operations are expanded as data descriptors of their saved bodies;
    no gate definition, inverse, matrix or new circuit is computed.
    """
    cache={}
    def instruction(item):
        name=item['name'];definition=None
        if name in circuit['custom']:
            if name not in cache:
                data=circuit['custom'][name]
                cache[name]=dict(controls=data['controls'],ctrl_state=data['ctrl_state'],
                    base=data['base_operation'],body=structural_digest(data['body']) if data['body'] else None)
            definition=cache[name];name='CUSTOM'
        return dict(name=name,definition=definition,wires=item['wires'],clbits=item['clbits'],
                    params=item['params'],controls=item['controls'],ctrl_state=item['ctrl_state'])
    return r.digest(dict(nq=circuit['nq'],phase=circuit['phase'],
        instructions=[instruction(item) for item in (circuit['instructions'] if instructions is None else instructions)]))


def event_central_join(circuit, events):
    expected=[]
    for stream in events:
        for event in stream:
            for app in event['application_sequence']:
                if app['is_identity']:continue
                support=app['diagonal_pauli_support']
                r.require(len(support) in (1,2),'EVENT_PAULI_SUPPORT')
                if app['role']=='product':
                    expected.extend(dict(name='CZGate',wires=[4,q],params=[]) for q in support)
                else:
                    r.require(app['role']=='rotation','EVENT_ROLE')
                    # Exact saved-value field conversion specified by original
                    # unsigned_rotation_angle / _append_central_application.
                    angle=2.0*(app['coefficient_sign']*event['rotation_angle'])
                    expected.append(dict(name='CRZGate' if len(support)==1 else 'CONTROLLED_RZZ',
                        wires=[4,*support],params=[dict(kind='float',hex=angle.hex())]))
    actual=[]
    for item in circuit['instructions']:
        name=item['name']
        if name in circuit['custom']:
            data=circuit['custom'][name];base=data['base_operation'];body=data['body']
            r.require(data['controls']==1 and data['ctrl_state']==1 and base and base['name']=='RZZGate',
                      'QPY_CONTROLLED_RZZ_BASE')
            r.require(body is not None and [i['name'] for i in body['instructions']]==['CCXGate','CRZGate','CCXGate'] and
                [i['wires'] for i in body['instructions']]==[[0,1,2],[0,2],[0,1,2]] and
                body['instructions'][1]['params']==item['params']==base['params'],'QPY_CONTROLLED_RZZ_BODY')
            name='CONTROLLED_RZZ'
        elif name not in ('CRZGate','CZGate'):continue
        actual.append(dict(name=name,wires=item['wires'],params=item['params']))
    r.require(actual==expected,'EVENT_CENTRAL_ORDER_SUPPORT_ANGLE')
    return dict(status='EVENT_CENTRAL_ORDER_SUPPORT_ANGLE_MATCH',central_applications=len(expected),
        scope='Saved central Pauli/control fields only; full basis/reuse/phase stream proof remains separate')


def gaussian_matches(circuit, receipt):
    names={'p':'PhaseGate','xx_plus_yy':'XXPlusYYGate'}
    def key(item):return (item['name'],item['wires'],[v.get('hex') for v in item['params']])
    stream=[key(x) for x in circuit['instructions']]
    rows=[]
    for group in receipt['operation_stream']:
        expected=[(names[o['gate']],o['wires'],o['params']) for o in group['operations']]
        rows.append(dict(rotation_sha256=group['rotation_sha256'],operation_count=len(expected),
            forward_exact_occurrences=sum(stream[i:i+len(expected)]==expected for i in range(len(stream)-len(expected)+1))))
    return rows


def event_basis_join(events,receipt):
    """Join decimal repr strings to binary64 hex; no rotation is recomputed."""
    known={r.digest(group['operations']) for group in receipt['operation_stream']}
    total=0
    for stream in events:
        for event in stream:
            for app in event['application_sequence']:
                ops=[]
                for item in app['basis_change_operations']:
                    numbers=[float(x) for x in item['parameters']]
                    r.require(all(math.isfinite(x) for x in numbers),'EVENT_NONFINITE_BASIS_PARAMETER')
                    ops.append(dict(gate=item['name'],params=[x.hex() for x in numbers],wires=item['qubits']))
                if ops:r.require(r.digest(ops) in known,'EVENT_BASIS_VS_COMMON_OPERATION_BYTES')
                total+=1
    return dict(status='ALL_SAVED_EVENT_REGISTERED_BASIS_FIELDS_MATCH_COMMON_RECEIPT',
                applications=total,conversion='saved decimal repr to identical binary64 hex',
                rotation_regenerated=False)
