"""Stdlib-only accepted-bundle checks. Scientific decoding is separately gated."""
import ast
from dataclasses import fields
from pathlib import Path
import struct
import zipfile
import h2_common_production_contract as old
import h2_diagnostics_binding as seal

ORIGIN_SOURCE='35444654f9159a5d85debdbc577feab0590e2610'
ORIGIN_CONTRACT='f71e52ce6f42cc17d1a97f0cbdfac311da57c80b49ee8ea69ebe34fb4d45bfd9'
ORIGIN_COMPLETION='cc7ebfee979b2e34eb4327aa8a8c6e20ea53bc46871f97c83d0391db5e7ec90f'
require=old.require


def adaptive(record):
    require(set(record)=={f.name for f in fields(old.policy.AdaptiveInputs)},'ADAPTIVE_INPUT_SCHEMA')
    kwargs=dict(record)
    for name in ('reference_error','corrected_signal_closure','log_B_discrepancy','maximum_norm'):
        kwargs[name]=old.policy.Observation(**kwargs[name])
    for name in ('path_signal_errors','path_vector_errors'):
        kwargs[name]=tuple(old.policy.Observation(**x) for x in kwargs[name])
    return old.policy.AdaptiveInputs(**kwargs)


def archive_metadata(path,columns):
    """Read NPY headers and integer basis bytes; never decode signal vectors."""
    expected={'primary.npy':((6,columns),'<c16',16),
              'independent.npy':((6,columns),'<c16',16),
              'occupation_basis.npy':((6,),'<i8',8),'qindices.npy':((6,),'<i8',8)}
    result={}
    with zipfile.ZipFile(path) as archive:
        require(len(archive.namelist())==4 and set(archive.namelist())==set(expected),'NPZ_MEMBERS')
        for name,(shape,dtype,itemsize) in expected.items():
            with archive.open(name) as f:
                require(f.read(6)==b'\x93NUMPY','NPY_MAGIC')
                version=tuple(f.read(2));require(version in ((1,0),(2,0),(3,0)),'NPY_VERSION')
                fmt='<H' if version==(1,0) else '<I'
                n=struct.unpack(fmt,f.read(struct.calcsize(fmt)))[0]
                header=ast.literal_eval(f.read(n).decode('utf-8' if version==(3,0) else 'latin1'))
                require(set(header)=={'descr','fortran_order','shape'} and
                        header['descr']==dtype and header['fortran_order'] is False and
                        type(header['shape']) is tuple and all(type(x) is int for x in header['shape']) and
                        header['shape']==shape,'NPZ_SHAPE_DTYPE_REPRESENTATION')
                size=itemsize
                for dim in shape:size*=dim
                prefix=6+2+struct.calcsize(fmt)+n
                require(archive.getinfo(name).file_size==prefix+size,'NPZ_PAYLOAD_SIZE')
                if name in ('occupation_basis.npy','qindices.npy'):
                    values=(3,5,6,9,10,12) if name=='occupation_basis.npy' else (12,10,6,9,5,3)
                    require(f.read()==struct.pack('<6q',*values),'OCCUPATION_BASIS_WIRE_ORDER')
                result[name]=dict(shape=list(shape),dtype=dtype,fortran_order=False)
    return result


def origin_source_check(cfg,roots):
    require(set(roots)=={'runner','acquisition','provider'},'ORIGIN_SOURCE_ROLES')
    commits={'runner':ORIGIN_SOURCE,'acquisition':old.ACQUISITION,'provider':old.PROVIDER}
    sealed={r['path']:r['sha256'] for r in cfg['files']}
    for role,commit in commits.items():
        rows=[r for r in cfg['source_files'] if r['root']==role]
        require(bool(rows),'ORIGIN_SOURCE_CLOSURE')
        for row,(_,raw) in zip(rows,seal.git_batch(commit,[r['path'] for r in rows])):
            p=(Path(roots[role])/row['path']).resolve()
            require(p.is_relative_to(Path(roots[role]).resolve()) and
                    sealed.get(str(p))==row['sha256']==seal.sha(p),'ORIGIN_SOURCE_PATH')
            import hashlib
            require(hashlib.sha256(raw).hexdigest()==row['sha256'],'ORIGIN_GIT_SOURCE_BYTES')


def inspect_origin(ref,consumer=None):
    require(set(ref)=={'contract','run_root','completion_sha256','approval','grant','source_roots'},'ORIGIN_REFERENCE_SCHEMA')
    cfg=old.read(ref['contract']);run=Path(ref['run_root']).resolve()
    require(seal.digest(cfg)==ORIGIN_CONTRACT and cfg['source_commit']==ORIGIN_SOURCE and
            cfg['branch']=='DIAGNOSTICS' and str(run)==cfg['run_root'],'PINNED_ORIGIN_CONTRACT_SOURCE')
    require(ref['completion_sha256']==ORIGIN_COMPLETION,'PINNED_ORIGIN_COMPLETION')
    try:receipt=seal.accepted_diagnostics_receipt(run,cfg,completion_sha256=ORIGIN_COMPLETION)
    except (ValueError,OSError,KeyError) as exc:
        raise old.Stop('ACCEPTED_ORIGIN:'+str(exc)) from exc
    grant=old.read(ref['grant']);approval=old.read(ref['approval']);marker=old.read(run/'CONSUMED.json')
    require(old.policy.strict_equal(marker['grant'],grant) and
            grant['approval_sha256']==seal.digest(approval)==marker['approval_sha256'] and
            approval['issuer']=='USER_EXPLICIT' and approval['execution_authorized'] is True and
            approval['contract_sha256']==ORIGIN_CONTRACT,'ORIGIN_GRANT_APPROVAL_MARKER')
    require(approval['policies_adopted']==[cfg[k] for k in ('native_policy','allowance_policy','reference_policy')],
            'ORIGIN_POLICY_ADOPTION')
    require(old.policy.strict_equal(cfg['candidates'],[old.asdict(x) for x in old.policy.CANDIDATES]) and
            old.policy.strict_equal(cfg['physical'],old.policy.PHYSICAL) and
            old.policy.strict_equal(cfg['compiler'],old.policy.COMPILER) and
            old.policy.strict_equal(cfg['binding'],old.BINDING),'ORIGIN_FIXED_SCIENCE')
    old.seal_check(cfg);origin_source_check(cfg,ref['source_roots'])
    if consumer is not None:
        for key in ('physical','compiler','binding','candidates','native_policy','allowance_policy','reference_policy','inputs'):
            require(old.policy.strict_equal(consumer[key],cfg[key]),'CONSUMER_ORIGIN_IDENTITY:'+key)
        runtime_keys=('python_sha256','python_target','qiskit','numpy','inventory_sha256','sys_path','closure_mode')
        require(all(consumer['runtime'][k]==cfg['runtime'][k] for k in runtime_keys),'CONSUMER_RUNTIME_IDENTITY')
    preparations={p:old.read(run/'results'/('preparation_'+str(p)+'.json')) for p in (1,3)}
    for prefix,row in preparations.items():
        require(type(row['prefix']) is int and row['prefix']==prefix and row['native_policy']==cfg['native_policy'] and
                seal.digest(row['operation_stream'])==row['operations_sha256'],'ORIGIN_PREPARATION_IDENTITY')
    input_digest=seal.digest(dict(binding=cfg['binding'],files=cfg['files'],series=old.SERIES))
    records={};headers={}
    reference_id='OLD_FP:'+old.sha(cfg['inputs']['old_reference'])
    for candidate in old.policy.CANDIDATES:
        cid=candidate.id;row=old.read(run/'results'/(cid+'_diagnostics.json'));pr=candidate.method=='PR'
        inputs=adaptive(row['inputs']);inputs.validate(reference_id)
        require(inputs.domain=='SAVED_SCIENCE' and inputs.candidate_id==cid and
                inputs.input_binding_sha256==input_digest,'ORIGIN_ADAPTIVE_INPUT_BINDING')
        require(row['preparation_sha256']==preparations[candidate.prefix]['preparation_sha256'] and
                row['operations_sha256']==preparations[candidate.prefix]['operations_sha256'],
                'CANDIDATE_PREFIX_PREPARATION')
        require(row['signal_columns']==(['corrected','raw'] if pr else ['corrected']) and
                len(row['primary_signals'])==len(row['independent_signals'])==(2 if pr else 1),
                'CORRECTED_RAW_COLUMNS')
        require(len(inputs.path_signal_errors)==len(inputs.path_vector_errors)==(2 if pr else 1),
                'ADAPTIVE_PATH_COVERAGE')
        require(row['reference_binding']['reference_id']==reference_id and
                row['reference_binding']['reference_recomputed'] is False and
                row['reference_binding']['reference_error_certified'] is False,'CONDITIONAL_REFERENCE_IDENTITY')
        require(set(row['reference_binding'].get('bias',{}))=={'real','imag'},'REFERENCE_BIAS_SCHEMA')
        for value in row['reference_binding']['bias'].values():old.policy.number(value,'saved_bias',nonnegative=False)
        old.policy.number(row['log_B'],'saved_log_B')
        headers[cid]=archive_metadata(run/'results'/(cid+'_vectors.npz'),2 if pr else 1)
        records[cid]=row
    return dict(config=cfg,records=records,preparations=preparations,headers=headers,
                receipt=receipt,files=receipt['files'],prior_grant_reusable=False,
                signal_vectors_evaluated=False)


def load_for_backend(backend):
    """Only the new authorized bundle_load call may decode scientific vectors."""
    import h2_accepted_compare_contract as c
    s=backend.s
    require(type(s) is c.Session and s.active=='bundle_load' and s.used['bundle_load']==1,
            'AUTHORIZED_BUNDLE_LOAD_REQUIRED')
    plan=inspect_origin(s.cfg['accepted_diagnostics'],s.cfg);run=Path(s.cfg['accepted_diagnostics']['run_root'])
    for prefix,old_prep in plan['preparations'].items():
        require(backend.preps[prefix].preparation_hash==old_prep['preparation_sha256'] and
                backend.operations[prefix]==old_prep['operations_sha256'],'RECREATED_PREPARATION_IDENTITY')
    result={}
    for candidate in c.policy.CANDIDATES:
        cid=candidate.id;row=plan['records'][cid]
        with backend.np.load(run/'results'/(cid+'_vectors.npz'),allow_pickle=False) as archive:
            require(backend.np.array_equal(archive['occupation_basis'],backend.ctx.oracle.basis) and
                    backend.np.array_equal(archive['qindices'],backend.ctx.oracle.qindices),'RUNTIME_BASIS_IDENTITY')
            primary=archive['primary'].copy();independent=archive['independent'].copy()
            require(backend.np.isfinite(primary).all() and backend.np.isfinite(independent).all(),
                    'FINITE_ARCHIVED_VECTORS')
        # Exact byte copy, explicitly described as inherited evidence in reuse receipt.
        for suffix in ('_diagnostics.json','_vectors.npz'):
            original=run/'results'/(cid+suffix);target=s.root/'results'/(cid+suffix)
            with target.open('xb') as f:f.write(original.read_bytes())
        result[cid]=dict(inputs=adaptive(row['inputs']),
            primary=dict(log_B=row['log_B'],vectors=primary),independent=dict(vectors=independent),
            bias=row['reference_binding']['bias'],origin_completion_sha256=ORIGIN_COMPLETION)
    c.save(s.root/'results/accepted_diagnostics_reuse.json',dict(
        origin_contract_sha256=ORIGIN_CONTRACT,origin_completion_sha256=ORIGIN_COMPLETION,
        inherited_result_files=plan['files'],path_recomputed=False,reference_rebound=False,
        old_grant_reusable=False,source_vector_files_unchanged=True))
    return result
