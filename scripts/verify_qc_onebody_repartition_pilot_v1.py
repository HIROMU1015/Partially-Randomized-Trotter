#!/usr/bin/env python3
"""Independent audit of one issued molecular attempt; not a science launcher."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from trottertracks.ground_pf_repartition.qc_onebody_repartition_execution_v1 import ARTIFACT,OUTPUT,SCOPE,digest,write_json,preflight


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();directory=args.directory.resolve()
    try:
        if directory!=(ROOT/OUTPUT).resolve():raise ValueError('REGISTERED_ATTEMPT_DIRECTORY_REQUIRED')
        preflight(ROOT)  # stdlib identity only; all executable payload still frozen
        marker=json.loads((directory/'attempt_consumed.json').read_text())
        if marker.get('scope_id')!=SCOPE or marker.get('attempt')!=1 or marker.get('retry')!=0:
            raise ValueError('ISSUED_ATTEMPT_MARKER_REQUIRED')
        raw=json.loads((directory/'raw.json').read_text());seal=json.loads((directory/'selector_seal.json').read_text())
        if digest(directory/'raw.npz')!=raw['raw_npz_SHA256'] or digest(directory/'selector_seal.json')!=raw['selector_seal_SHA256']:
            raise ValueError('RAW_OR_SEAL_CHANGED')
        identity=json.loads((directory/'result_identity.json').read_text())
        if identity['contract_SHA256']!=digest(ROOT/ARTIFACT/'contract.json') or identity['source_manifest_SHA256']!=digest(ROOT/ARTIFACT/'source_manifest.json') or identity['authorization_SHA256']!=marker['authorization_SHA256']:
            raise ValueError('RESULT_FIXED_IDENTITY')
        if (directory/'audit_consumed.json').exists():raise ValueError('ONE_AUDIT_ATTEMPT_NO_RETRY')
        write_json(directory/'audit_consumed.json',{'scope_id':SCOPE,'attempt':1,'retry':0})
        import numpy as np
        from trottertracks.ground_pf_repartition.qc_onebody_repartition_checker_v1 import audit_arrays
        with np.load(directory/'raw.npz',allow_pickle=False) as z:data={k:np.array(z[k]) for k in z.files}
        contract=json.loads((ROOT/ARTIFACT/'contract.json').read_text())
        input_identity=json.loads((ROOT/ARTIFACT/'input_identity.json').read_text())
        metadata=input_identity['saved_metadata']
        # Raw science arrays must be the exact frozen coefficient/state arrays,
        # not merely an arbitrary H that happens to pass the internal tests.
        import hashlib
        def array_sha(a):
            a=np.ascontiguousarray(a);h=hashlib.sha256();h.update(a.dtype.str.encode())
            h.update(json.dumps(list(a.shape),separators=(',',':')).encode());h.update(a.tobytes());return h.hexdigest()
        for key,stored in [('one_body','one_body'),('lambdas','lambdas'),('state','sector_state_vector')]:
            if array_sha(data[key])!=metadata[stored+'_hash']:raise ValueError('RAW_INPUT_'+key)
        if [array_sha(g) for g in data['g_matrices']]!=metadata['g_matrix_hashes'] or array_sha(data['labels'])!=metadata['sector']['basis_indices_hash'] or float(data['constant']).hex()!=metadata['constant_hex']:
            raise ValueError('RAW_INPUT_FRAGMENT_SECTOR_SCALAR')
        audit=audit_arrays(data,raw,seal,contract);write_json(directory/'independent_audit.json',audit)
        print(json.dumps(audit,indent=2));return 0
    except (ValueError,RuntimeError,FileNotFoundError) as error:
        failure={'status':'INDEPENDENT_AUDIT_STOP_NO_RETRY','reason':str(error),'mandatory_stop':True}
        if (directory/'audit_consumed.json').exists():write_json(directory/'independent_audit.json',failure)
        print(json.dumps(failure),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
