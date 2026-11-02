"""Versioned static binding specification. Names are data, never load targets."""
from ..g6_w2c_b3_api_binding_v1.static import binding as previous_binding


def specification():
    result=previous_binding()
    result.update(schema='b3_static_real_adapter_binding_v1',
        dispatch_implemented=False,
        bundle_injection='versioned synthetic document adapter inside fixed scheduler/pair/F01/Storage V2',
        codec_adapter='g6_w2c_b3_real_adapter_v1.codec: original F3 bytes plus driver projection',
        result_adapter='g6_w2c_b3_real_adapter_v1.result: exact documents only; no science object/callable',
        canonical_preflight='CANONICAL_PREFLIGHT_NOT_IMPLEMENTED retained; positive canonical never sent to F3.prepare',
        actual_runtime_compatibility='NOT_ESTABLISHED',
        science_loader='ABSENT',real_factory='ABSENT',real_authority='ABSENT')
    return result
