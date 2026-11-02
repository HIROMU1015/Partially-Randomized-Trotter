"""Reconstruct one lost event log, verified against saved circuit fingerprints.

Only the saved-seed event stream and native input fingerprints are rebuilt.
No price acquisition, signal, simulation or compile is repeated. The completed
result remains byte-identical; the empty original archive is retained.
"""
import gzip
import hashlib
import io
import math
import h46_b3 as b
from h46_b3_durable import save,sync_file,sync_directory


def run():
    native=b.setup()
    r=next(r for r in b.read(b.PUBLIC/'native_selection.json')['candidates'] if r['id']=='2c5aa96f886f95f73ad3')
    directory=b.PRIVATE/'cost/B3_EXPLORATION'/r['id']/'4'
    result_path=directory/'result.json';before=b.signal.sha(result_path);old=b.read(result_path)
    ctx=b.native_context(r['system'],[0]);prep=ctx.preparations[0]
    requests,seeds,laws=native.sample_requests(r,prep,'B3_EXPLORATION',4)
    scalar=ctx.ham.constant+prep.extracted_identity_coefficient
    from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
    from trottertracks.resource_applicability import ax2b_native_df_v5 as lower
    bindings={}
    with use_fingerprint_reuse() as fp:
        evolution=native.compose(r,prep,requests,scalar,fp.fingerprint)
        for axis in ('cosine','sine'):
            wrapper=lower.build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=True,
                max_instructions=len(evolution.circuit.data)+4)
            identity=fp.fingerprint(wrapper)
            cached=b.read(b.PRIVATE/'compiled_cache'/(identity+'.json'))
            if cached['metrics']!=old['prices'][axis] or cached['compiler']!=native.COMPILER_HASH:
                raise ValueError('RECONSTRUCTED_EVENT_SAVED_PRICE_BINDING')
            bindings[axis]=identity
    events=[[event.to_dict() for event in request.rte_occurrence.events] for request in requests]
    orders={str(k):sum(e['taylor_order']==k for batch in events for e in batch) for k in range(0,r['K']+1,2)}
    log_B=sum(len(req.rte_occurrence.events)*math.log(req.rte_distribution.exact_finite_distribution) for req in requests)
    if orders!=old['orders'] or abs(log_B-old['log_B'])>1e-10 or prep.preparation_hash!=old['preparation_hash']:
        raise ValueError('RECONSTRUCTED_EVENT_LAW_BINDING')
    raw=directory/'events_PRIVATE.reconstructed.json'
    save(raw,dict(events=events,seeds=seeds,conditional=False,cohort='B3_EXPLORATION'))
    data=raw.read_bytes();compressed=gzip.compress(data,compresslevel=1,mtime=0)
    if gzip.decompress(compressed)!=data: raise ValueError('RECONSTRUCTED_GZIP_ROUNDTRIP')
    target=raw.with_suffix('.json.gz');target.write_bytes(compressed);sync_file(target);sync_directory(directory)
    raw.unlink();sync_directory(directory)
    if b.signal.sha(result_path)!=before: raise ValueError('COMPLETED_PRICE_CHANGED')
    record=dict(status='PASS',original_event_archive_empty_preserved=True,
        original_byte_identity_claimed=False,fixed_seed_reconstruction=True,
        saved_native_input_fingerprints=bindings,result_sha256_unchanged=before,
        event_sha256=hashlib.sha256(data).hexdigest(),new_independent_price_observations=0,
        compile_calls=0,signal_calls=0,simulation_calls=0)
    save(directory/'event_reconstruction.json',record)
    save(b.PUBLIC/'event_log_reconstruction.json',dict(candidate=r,index=4,**record))
    return record


if __name__=='__main__':
    import json
    print(json.dumps(run()))
