"""Strict F3 document codec and lossless sidecar around driver projection.

No scientific imports. Compatibility is derived from fixed source syntax.
Unrepresentable/ambiguous keys are rejected, never silently overwritten.
"""
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json


class Rejected(ValueError):
    pass


def need(ok, reason):
    if not ok:
        raise Rejected(reason)


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()


def digest_bytes(value):
    return sha256(value).hexdigest()


def _pairs(rows):
    result = {}
    for key, value in rows:
        need(key not in result, 'DUPLICATE_JSON_KEY')
        result[key] = value
    return result


def plain(document):
    need(type(document) is str, 'DOCUMENT_STRING_REQUIRED')
    try:
        return json.loads(document, object_pairs_hook=_pairs,
                          parse_constant=lambda x: (_ for _ in ()).throw(Rejected('NONFINITE_JSON')))
    except (TypeError, ValueError) as exc:
        raise Rejected('INVALID_JSON:' + str(exc)) from exc


def _hex(value):
    need(type(value) is str, 'HEX_STRING_REQUIRED')
    try:
        integer = int(value, 16)
    except ValueError as exc:
        raise Rejected('INVALID_HEX') from exc
    need(hex(integer) == value, 'NONCANONICAL_HEX')
    return integer


def pack(value):
    t = type(value)
    if t is Fraction:
        return {'$rat': [hex(value.numerator), hex(value.denominator)]}
    if t is int:
        return {'$int': hex(value)}
    if value is None or t in (str, bool):
        return value
    if t is tuple:
        return {'$tuple': [pack(v) for v in value]}
    if t is list:
        return [pack(v) for v in value]
    if t is dict:
        # An ordinary dictionary spelling a codec tag cannot round-trip.
        need(not any(type(k) is str and k.startswith('$') for k in value), 'RESERVED_CODEC_KEY')
        if all(type(k) is str for k in value):
            return {k: pack(v) for k, v in value.items()}
        items = [[pack(k), pack(v)] for k, v in value.items()]
        items.sort(key=lambda row: json.dumps(row[0], sort_keys=True))
        return {'$map': items}
    raise Rejected('UNSUPPORTED_EXACT_TYPE:' + t.__name__)


def unpack(value):
    t = type(value)
    if value is None or t in (str, bool):
        return value
    if t is list:
        return [unpack(v) for v in value]
    need(t is dict, 'NONEXACT_JSON_NUMBER')
    if set(value) == {'$int'}:
        return _hex(value['$int'])
    if set(value) == {'$rat'}:
        row = value['$rat']
        need(type(row) is list and len(row) == 2, 'RATIONAL_PAIR')
        n, d = map(_hex, row)
        need(d > 0, 'POSITIVE_DENOMINATOR')
        result = Fraction(n, d)
        need((result.numerator, result.denominator) == (n, d), 'NONCANONICAL_RATIONAL')
        return result
    if set(value) == {'$tuple'}:
        need(type(value['$tuple']) is list, 'TUPLE_ARRAY')
        return tuple(unpack(v) for v in value['$tuple'])
    if set(value) == {'$map'}:
        need(type(value['$map']) is list, 'MAP_ARRAY')
        result = {}
        for row in value['$map']:
            need(type(row) is list and len(row) == 2, 'MAP_PAIR')
            key, item = unpack(row[0]), unpack(row[1])
            try:
                need(key not in result, 'DUPLICATE_OR_EQUAL_MAP_KEY')
                result[key] = item
            except TypeError as exc:
                raise Rejected('UNHASHABLE_MAP_KEY') from exc
        return result
    need(not any(k.startswith('$') for k in value), 'UNKNOWN_CODEC_VARIANT')
    return {k: unpack(v) for k, v in value.items()}


def canonical(value):
    return json_bytes(pack(value)).decode()


def fingerprint(value):
    return digest_bytes(canonical(value).encode())


@dataclass(frozen=True)
class Document:
    document: str
    digest: str

    @classmethod
    def create(cls, value):
        document = canonical(value)
        return cls(document, digest_bytes(document.encode()))

    def read(self):
        need(type(self.document) is str and type(self.digest) is str, 'DOCUMENT_TYPES')
        need(digest_bytes(self.document.encode()) == self.digest, 'EVIDENCE_DIGEST_MISMATCH')
        value = unpack(plain(self.document))
        need(canonical(value) == self.document, 'NONCANONICAL_F3_DOCUMENT')
        return value

    def transport(self):
        self.read()
        return dict(document=self.document, digest=self.digest)


def read_transport(value):
    need(type(value) is dict and set(value) == {'document', 'digest'}, 'DOCUMENT_TRANSPORT_FIELDS')
    record = Document(**value)
    record.read()
    return record


def driver_value(value):
    """Exact driver.exact -> audit._pack; list/tuple collapse is deliberate.

    Reject str(key) collisions that the original driver would overwrite.
    Original typed keys/containers remain in the F3 sidecar.
    """
    t = type(value)
    if t is Fraction:
        return {'rat': [hex(value.numerator), hex(value.denominator)]}
    if t is int:
        return {'int': hex(value)}
    if value is None or t in (str, bool):
        return value
    if t in (tuple, list):
        return [driver_value(v) for v in value]
    need(t is dict, 'UNSUPPORTED_DRIVER_TYPE')
    result = {}
    for key, item in value.items():
        name = str(key)
        need(name not in result, 'DRIVER_MAP_KEY_COLLISION')
        result[name] = driver_value(item)
    return result


def driver_canonical(value):
    return json_bytes(driver_value(value)).decode()


def convert(record, *, source_identity):
    need(type(record) is Document, 'EXACT_DOCUMENT_REQUIRED')
    need(type(source_identity) is str and bool(source_identity), 'SOURCE_IDENTITY_REQUIRED')
    value = record.read()
    projection = driver_canonical(value)
    return dict(schema='b3_exact_codec_sidecar_v1', source_identity=source_identity,
                f3=record.transport(), driver_document=projection,
                driver_digest=digest_bytes(projection.encode()),
                preservation='F3_SOURCE_DOCUMENT_IS_LOSSLESS; DRIVER_PROJECTION_IS_NOT',
                original_fingerprint=record.digest)


def restore(sidecar, *, source_identity):
    need(type(sidecar) is dict and set(sidecar) == {
        'schema', 'source_identity', 'f3', 'driver_document', 'driver_digest',
        'preservation', 'original_fingerprint'}, 'CODEC_SIDECAR_FIELDS')
    record = read_transport(sidecar['f3'])
    need(json_bytes(sidecar) == json_bytes(convert(record, source_identity=source_identity)),
         'CODEC_SIDECAR_MISMATCH')
    return record


@dataclass(frozen=True)
class DriverDocument:
    """Already-normalized driver bytes; earlier tuple/key types are unknown."""
    text: str
    digest: str

    @classmethod
    def create(cls, value):
        text=driver_canonical(value)
        return cls(text,digest_bytes(text.encode()))

    def read(self):
        need(type(self.text) is str and type(self.digest) is str and
             digest_bytes(self.text.encode())==self.digest,'DRIVER_DOCUMENT_DIGEST')
        def decode(value):
            if value is None or type(value) in (str,bool):return value
            if type(value) is list:return [decode(v) for v in value]
            need(type(value) is dict,'DRIVER_NONEXACT_NUMBER')
            if set(value)=={'int'}:return _hex(value['int'])
            if set(value)=={'rat'}:
                pair=value['rat'];need(type(pair) is list and len(pair)==2,'DRIVER_RATIONAL_PAIR')
                n,d=map(_hex,pair);need(d>0,'DRIVER_DENOMINATOR');result=Fraction(n,d)
                need((result.numerator,result.denominator)==(n,d),'DRIVER_RATIONAL_CANONICAL')
                return result
            return {k:decode(v) for k,v in value.items()}
        value=decode(plain(self.text))
        need(driver_canonical(value)==self.text,'DRIVER_CANONICAL_BYTES')
        return value

    def transport(self):
        self.read()
        return dict(text=self.text,digest=self.digest)
