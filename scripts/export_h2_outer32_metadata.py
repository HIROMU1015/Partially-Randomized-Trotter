"""Static, stdlib-only projection of specifically located H2 saved metadata.

Never imports scientific packages or executes an old runner/audit. All numeric
JSON tokens are retained verbatim; only public path strings and secret fields
are transformed. No shot, error, cost, variance or rank is calculated.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

SOURCE = '4ec26921c6872ed6e8667f167074e1e405cf1a3f'
PLAN = '4dda5e68815f8359b3890dc7a9e301956a631d6cea500e8a90703224db719373'
AUDIT = '38ad4d594a3be9289bdf1697d9a37a6d677384899a3320666c75761a304f7723'
BASE = '366b2fd24f7e86aa82b5b518b6e3774520f65596'
TARGETS = {'H2O32_c7ec590a32a859bc': 33, 'H2O32_8a2ecdd0294ffb74': 34}
OLD = ('B1_p3_q4_2nd', 'B1_p3_q2_4th')
GUARD = 'H2_QISKIT130_SYMBOLIC_ZERO_SCALAR_GUARD_V1'
FIXED = dict(audit=BASE, acquisition_SOURCE=SOURCE,
 acquisition_result='a05f4ed19e87e1eb65f7fc346385b5b4eae69845',
 accounting_SOURCE='9a77d2b0a0b87e246e85f085b37f5923c18e27bb',
 accounting_result='8613b3ecda2030a677b1690fe7aa3da23dec4873',
 old_result='829d697ae7905b3ea90a61da9d2d4da15d8b757a',
 old_compile_SOURCE='cf99e4452df96291d7e9d37d76d2e91be5cbc413',
 old_provider_SOURCE='217c971dd537b1cc7fbc13742bc0f6c263ff7f45')


class Number(str):
    """Original JSON number spelling, not a newly evaluated floating value."""


def loads(text):
    return json.loads(text, parse_float=Number)


def encode(value, level=0):
    if isinstance(value, Number):
        if not re.fullmatch(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?', value):
            raise ValueError('INVALID_JSON_NUMBER')
        return str(value)
    if isinstance(value, dict):
        return '{' + ','.join('\n' + '  '*(level+1) + json.dumps(k, ensure_ascii=False) + ': ' + encode(v, level+1)
            for k, v in value.items()) + ('\n'+'  '*level if value else '') + '}'
    if isinstance(value, list):
        return '[' + ','.join('\n'+'  '*(level+1)+encode(v, level+1) for v in value) + ('\n'+'  '*level if value else '') + ']'
    if isinstance(value, float):
        raise ValueError('NO_NEW_FLOAT_ARITHMETIC_OR_CONVERSION')
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def require(ok, label):
    if not ok:
        raise ValueError(label)


class Export:
    def __init__(self, repo, destination, private_map):
        self.repo, self.dest = Path(repo).resolve(), Path(destination).resolve()
        self.private_map = Path(private_map).resolve()
        self.roots, self.origins, self.data, self.links, self.checks, self.gaps = {}, {}, {}, [], [], []
        self.initial = {}

    def root(self, path, token):
        self.roots[str(Path(path).absolute())] = token

    def token(self, text):
        if isinstance(text, Number):
            return text
        if not text.startswith('/'):
            return text
        for root, token in sorted(self.roots.items(), key=lambda x: -len(x[0])):
            if text==root or text.startswith(root+'/'):
                return '${'+token+'}'+text[len(root):]
        if text.startswith('/'):
            p = Path(text)
            if text.startswith('/home/'):
                parts = p.parts
                root = str(Path(*parts[:5])) if len(parts)>5 else str(p.parent)
            else:
                root = '/' + p.parts[1]
            tag = 'OTHER_ROOT_' + sha_bytes(root.encode())[:12]
            self.root(root, tag)
            text = '${'+tag+'}'+text[len(root):]
        return text

    def public(self, value):
        if isinstance(value, Number):
            return value
        if isinstance(value, str):
            return self.token(value)
        if isinstance(value, dict):
            return {self.token(k): (v if k=='pointer' else self.public(v)) for k, v in value.items()
                if not (k.lower() in ('grant', 'token', 'credential', 'credentials', 'private_key', 'environment', 'env')
                    or ('seed' in k.lower() and k not in ('seed_cohort', 'seed_transpiler') and v is not None))}
        if isinstance(value, list):
            return [self.public(v) for v in value]
        return value

    def origin(self, path, role):
        p = Path(path).absolute()
        key = str(p)
        if key not in self.origins:
            fid = 'O' + str(len(self.origins)+1).zfill(5)
            digest = sha_file(p)
            self.initial[key] = digest
            self.origins[key] = dict(id=fid, locator=self.token(key), roles=[role],
                original_sha256=digest, bytes=p.stat().st_size, original_published=False)
        elif role not in self.origins[key]['roles']:
            self.origins[key]['roles'].append(role)
        return self.origins[key]['id']

    def read(self, path, role):
        self.origin(path, role)
        key = str(Path(path).absolute())
        if key not in self.data:
            self.data[key] = loads(Path(path).read_text())
        return self.data[key]

    def pointer(self, value, pointer):
        for part in pointer.split('/')[1:]:
            part = part.replace('~1', '/').replace('~0', '~')
            value = value[int(part)] if isinstance(value, list) else value[part]
        return value

    def get(self, path, pointer='', role='metadata'):
        value = self.pointer(self.read(path, role), pointer)
        return dict(origin=self.origin(path, role), pointer=pointer, saved=self.public(value))

    def write(self, rel, value):
        p = self.dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        projected=self.public(value)
        payload = (encode(projected)+'\n').encode()
        # Check exact numbers and structure survive projection serialization.
        require(loads(payload)==projected, 'SERIALIZATION_EXACT:'+rel)
        self.checks.append('SERIALIZATION_EXACT:'+rel)
        p.write_bytes(payload)
        return rel

    def check(self, ok, label):
        require(ok, label)
        self.checks.append(label)

    def gap(self, item, reason):
        self.gaps.append(dict(item=item, value=None, reason=reason))

    def git(self, commit, path):
        r = subprocess.run(['git', 'show', commit+':'+path], cwd=self.repo, capture_output=True)
        return r.stdout if r.returncode == 0 else None

    def run(self, pipeline_path, config_path, old_spec_path):
        pipeline_path, config_path, old_spec_path = map(Path, (pipeline_path, config_path, old_spec_path))
        self.root(pipeline_path.parent, 'NEW_EXECUTION')
        cfg = loads(config_path.read_text())
        oldbase = Path(cfg['inputs']['D_result']['path']).parents[3]
        self.root(oldbase, 'OLD_H2_PRIVATE')
        self.root(cfg['provider_root'], 'NEW_PROVIDER')
        self.root(Path(cfg['runtime']['python']).parent.parent, 'RUNTIME_ENV')
        self.root(self.repo, 'EXPORT_WORKTREE')
        for row in cfg['sealed_files']:
            if row['role'].startswith('H2_SOURCE:'):
                rel = row['role'].split(':',1)[1]
                self.root(row['path'][:-len(rel)].rstrip('/'), 'ACQUISITION_WORKTREE')
                break
        pipe = self.read(pipeline_path, 'original_pipeline_config')
        cfg = self.read(config_path, 'original_exploration_config')
        spec = self.read(old_spec_path, 'old_full_PF_original_spec')
        self.origin(cfg['marker'], 'old_consumed_marker_hash_only')
        for p in Path(cfg['marker']).parent.parent.glob('*grant*.json'):
            self.origin(p, 'old_grant_hash_only_never_published')
        self.root(spec['native_config']['provider_root'], 'OLD_PROVIDER')
        self.check(cfg['SOURCE']==pipe['SOURCE']==SOURCE, 'FIXED_ACQUISITION_SOURCE')
        self.check(cfg['stage']=='H2_PR_EXPLORATION', 'EXPLORATION_ONLY')
        self.check(sha_file(cfg['plan_path'])==cfg['plan_sha256']==PLAN, 'FIXED_PLAN_SHA')
        self.check(self.read(cfg['plan_path'], 'exploration_plan')==cfg['plan'], 'PLAN_CONFIG_IDENTICAL')
        self.check(self.read(pipe['plan_path'], 'pipeline_plan')==pipe['plan']
            and sha_file(pipe['plan_path'])==pipe['plan_sha256'], 'PIPELINE_PLAN_SHA')
        # Static invocation chain; no original config/grant or seed values exported.
        chain = {k:self.get(config_path,'/'+k) for k in ('SOURCE','stage','plan_sha256','runtime','provider_root',
            'signal_path','reference_path','run_root','output','binding','preparation_hash','compiler_phase_policy','storage')}
        chain['pipeline_SOURCE'] = self.get(pipeline_path,'/SOURCE')
        chain['pipeline_plan_sha256'] = self.get(pipeline_path,'/plan_sha256')
        chain['historical_locator_chain'] = 'Prior H2 confirmation config /origin_config_path -> original exploration config -> original pipeline/config input locators'
        self.write('acquisition_context.json', chain)
        audit_path = Path(cfg['run_root'])/'end_audit.json'
        audit = self.read(audit_path, 'original_exploration_end_audit')
        self.check(sha_file(audit_path)==AUDIT, 'FIXED_END_AUDIT_SHA')
        self.check(audit['SOURCE']==SOURCE and audit['plan_sha256']==PLAN and not audit['errors'], 'ORIGINAL_AUDIT_IDENTITY')
        accepted = {r['id']:(i,r) for i,r in enumerate(audit['results'])}
        self.check(len(accepted)==len(audit['results']), 'AUDIT_RESULT_UNIQUENESS')
        self.write('stage_end_metadata.json', {
            'end_audit':self.get(audit_path),
            'compiler_policy_end_audit':self.get(Path(cfg['run_root'])/'compiler_policy_end_audit.json'),
            'completion_witness':self.get(Path(cfg['run_root'])/'completion_witness.json'),
            'native_exit':self.get(cfg['supervisor']['exit']),
            'scope':'Historical stage-wide 525-task audit copied as metadata; only 67 task receipts newly acquired here. No old audit rerun.'})
        self.runtime(cfg, spec, config_path)
        signal_path = Path(cfg['signal_path'])
        signals = self.read(signal_path, 'original_outer32_signals')
        picked = [(i,r) for i,r in enumerate(signals['records']) if r['candidate']['id'] in TARGETS]
        self.check(len(picked)==2 and {r['candidate']['id'] for _,r in picked}==set(TARGETS), 'TWO_SIGNAL_RECORDS')
        self.write('signal_records_2.json', dict(header={k:self.get(signal_path,'/'+k) for k in signals if k!='records'},
            records=[self.get(signal_path,'/records/'+str(i)) for i,_ in picked],
            numeric_policy='Original numeric JSON tokens copied verbatim; no bias/error/allowance/shot recomputation.'))
        for _,r in picked:
            for name in ('path_vector_difference','work_action_count'):
                self.gap(r['candidate']['id']+':'+name, 'No separately named saved diagnostic in this signal record; vector closure and maximum_intermediate_norm are exported as stored.')
        snapshot_receipt = Path(cfg['inputs']['snapshot_receipt']['path'])
        # Scalar/array hash and shape metadata only; no array decode.
        reference = dict(saved_declaration=self.get(signal_path,'/reference_accuracy'),
            reference_sha256=self.get(signal_path,'/reference_sha256'),
            reference_file=dict(locator=self.token(cfg['reference_path']),bytes=Path(cfg['reference_path']).stat().st_size,
                current_sha256=sha_file(cfg['reference_path']),content_published=False,decoded=False),
            old_reference=self.get(cfg['inputs']['FP_result']['path'],'/reference'),
            old_snapshot_receipt=self.get(snapshot_receipt),
            accuracy_guarantee=None)
        self.check(reference['reference_file']['current_sha256']==signals['reference_sha256'], 'REFERENCE_SAVED_SHA')
        self.origin(cfg['reference_path'],'reference_binary_hash_only_not_decoded')
        self.write('reference_metadata.json', reference)
        self.gap('reference_error_bound','Saved reference is conditional, not a rigorous error bound; no reconstruction or new validation.')
        prep = dict(binding=self.get(config_path,'/binding'), fullprep3=self.get(signal_path,'/full_preparation_hash'),
            cut1=self.get(signal_path,'/preparation_hash'), task_preparation=self.get(config_path,'/preparation_hash'),
            old_preparations=self.get(oldbase/'attempts/D_run02/output/preparations.json'),
            ordering_basis_partition_scalar={'stored_new_preparation_object':None,
                'reason':'No serialized full/cut preparation object referenced by config; hashes and binding saved. Native reconstruction prohibited.',
                'static_source':'4ec26921:scripts/h2_outer32_runtime.py:context',
                'declarations':'fullprep=prepare(ham,3); cut=prepare(ham,1); PhysicalSector.spin_sector(4,1,1); deterministic_fragment_indices==(0,); original scalar plus tail identity once. SOURCE declarations, not a new preparation receipt.'},
            cross_method_hash_equality_claimed=False)
        self.write('preparation_metadata.json', prep)
        self.gap('new_full_cut_preparation_serialization','Saved fullprep3/cut1 hashes exist; complete block/partition/basis preparation objects were not found among config-bound metadata. SOURCE recipe is separate evidence.')
        tasks=[t for t in cfg['tasks'] if t['candidate']['id'] in TARGETS]
        self.check(Counter(t['candidate']['id'] for t in tasks)==Counter(TARGETS), 'EXACT_N33_N34')
        self.check(len({t['id'] for t in tasks})==67, '67_UNIQUE_TASKS')
        rows=[]; storage_rows=[]; archive_count=0
        check_names=('saved_normalization_log_B','native_low','native_leakage','independent_occupation_vector',
            'hadamard_cosine','compiled_vector_cosine','hadamard_sine','compiled_vector_sine')
        for t in tasks:
            folder=Path(cfg['output'])/t['id']; rp=folder/'result.json';r=self.read(rp,'accepted_task_result')
            for p in Path(cfg['ledger']).glob(t['id']+'__*.json'):
                self.origin(p, 'existing_target_ledger_hash_only')
            ai,ar=accepted[t['id']]
            self.check(sha_file(rp)==ar['result_sha256'], 'ACCEPTED_RESULT_SHA:'+t['id'])
            self.check(r['task']==t and r['SOURCE']==SOURCE and r['plan_sha256']==PLAN
                and r['seed_cohort']=='EXPLORATION', 'RESULT_IDENTITY:'+t['id'])
            self.check(r['preparation_hash']==cfg['preparation_hash'] and r['input_binding']==cfg['binding'], 'TASK_PREPARATION_BINDING:'+t['id'])
            entry=dict(task=self.public(t), result=self.get(rp),
                acceptance=self.get(audit_path,'/results/'+str(ai)), axes=[], saved_correctness=[])
            for name in check_names:
                entry['saved_correctness'].append(self.get(folder/(name+'.json')))
                self.check(self.read(folder/(name+'.json'),'saved_correctness')==next(x for x in r['checks'] if x['check']==name), 'CHECK_RECEIPT_COPY:'+t['id']+':'+name)
            sa=folder/'storage_audit.json';s=self.read(sa,'saved_storage_audit')
            self.check(sha_file(sa)==r['storage_audit_sha256'], 'STORAGE_AUDIT_SHA:'+t['id'])
            sr=dict(task_id=t['id'],storage_audit=self.get(sa),axes=[])
            for i,axis in enumerate(('cosine','sine')):
                cost=folder/('cost_'+axis+'.json');costv=self.read(cost,'axis_cost_receipt')
                self.check(costv==r['axis_costs'][i] and costv['axis']==axis, 'COST_RESULT_JOIN:'+t['id']+':'+axis)
                self.check(costv['preparation_hash']==cfg['preparation_hash'] and costv['compiler_settings_hash']==cfg['binding']['compiler'], 'AXIS_PROFILE_PREPARATION:'+t['id']+':'+axis)
                guard=folder/('compiler_phase_guard_'+str(i)+'.json');gv=self.read(guard,'saved_compiler_guard')
                self.check(gv['policy']==GUARD and gv['unchanged_gate_stream'] and not gv['state_fit'], 'GUARD_SAVED_STATUS:'+t['id']+':'+axis)
                self.check(all(p['policy']==GUARD and p['unchanged_gate_stream'] and not p['input_or_output_state_fit'] and not p['postcompile_phase_correction'] for p in gv['passes']), 'GUARD_PASSES:'+t['id']+':'+axis)
                rel='compiler_guard_receipts/'+t['id']+'__'+axis+'.json';self.write(rel,self.get(guard))
                entry['axes'].append(dict(axis=axis,cost=self.get(cost),guard_projection=rel))
                qd=folder/('quarantine_'+axis);manifests=[]
                for logical in ('wrapper.qpy','compiled.qpy','vectors.npz'):
                    mp=qd/(logical+'.archive.json');m=self.read(mp,'saved_archive_manifest')
                    ap=qd/m['archive_file'];current=sha_file(ap);self.origin(ap,'opaque_archive_SHA_only_no_decode')
                    self.check(current==m['codec_receipt']['stored']['sha256'], 'ARCHIVE_CURRENT_SHA:'+t['id']+':'+axis+':'+logical)
                    release=qd/(logical+'.released.json');rv=self.read(release,'historical_raw_release_receipt')
                    self.check(rv['manifest_sha256']==sha_file(mp), 'RAW_RELEASE_MANIFEST_SHA:'+t['id']+':'+axis+':'+logical)
                    if logical=='compiled.qpy':self.check(m['codec_receipt']['original']['sha256']==costv['compiled_SHA'], 'COMPILED_ORIGINAL_SHA:'+t['id']+':'+axis)
                    manifests.append(dict(manifest=self.get(mp),release=self.get(release),current_archive_sha256=current,
                        raw_bytes_restored_in_this_task=False))
                    archive_count+=1
                sr['axes'].append(dict(axis=axis,archives=manifests))
            rows.append(entry);storage_rows.append(sr)
        self.write('task_acceptance_67.json',dict(tasks=rows,task_count=len(rows),axis_count=2*len(rows),cohort='EXPLORATION',
            audit_original_sha256=AUDIT,other_cohorts_pooled=False,correctness_reperformed=False))
        self.write('storage_records.json',dict(tasks=storage_rows,opaque_archives_SHA_checked=archive_count,
            note='Historical original-byte restoration PASS copied; current compressed bytes SHA checked. No decompression, array decode or scientific native validation rerun.'))
        self.gap('compiler_guard_historical_hash_binding', 'Current guard files are joined by exact task/axis and stage policy audit. Original accepted result does not itself contain guard-file SHA; this export adds current original SHA, not a retroactive historical byte attestation.')
        self.old_pf(cfg,spec,oldbase)
        prior=self.repo/'artifacts/resource_applicability/track_a_pf_order_reversal_existing_evidence_audit/2026-10-23'
        for name in ('REPORT.md','private_metadata_audit.json','acquisition_manifest.json','evidence_rows_75.csv','completion_status.json'):
            self.origin(prior/name,'prior_audit_reused_without_rerun')
        self.write('missing_evidence.json',dict(status='PARTIAL_WITH_GAPS',items=self.gaps,
            Comparison_C_scientifically_adopted=False,allowance_policy_adopted=False))
        # All visited originals are hashed before/after. No old raw or control writes.
        changes=[]
        for p,digest in self.initial.items():
            if not Path(p).exists() or sha_file(p)!=digest:changes.append(self.token(p))
        self.check(not changes, 'ALL_ACQUIRED_ORIGINAL_BYTES_UNCHANGED')
        self.write('origin_inventory.json',dict(roots=[dict(root_token='${'+v+'}',meaning='Stable original root; absolute mapping retained privately') for v in sorted(set(self.roots.values()))],
            originals=list(self.origins.values()),mapping_published=False))
        self.write('completion_status.json',dict(status='PARTIAL_WITH_GAPS',metadata_task_coverage=67,
            metadata_axis_coverage=134,originals_unchanged=True,new_scientific_calls=0,new_numerical_analysis=0,
            shot_recalculation=0,new_compile=0,new_policy_adoption=False,mandatory_STOP=True,
            next_actor='USER_OR_RECEIVING_CODEX',next_action='Use exported saved metadata to resume static joins; separately decide any allowance-policy or comparison-C work.'))
        self.write('STOP.json',dict(status='MANDATORY_STOP',metadata_export_only=True,next_scientific_phase_authorized=False,
            old_STOPs_preserved=True,new_grant_issued=False,scientific_calls=0))
        self.write('technical_verification.json',dict(status='STATIC_CHECKS_PASS_WITH_DECLARED_EVIDENCE_GAPS',
            checks=self.checks,check_count=len(self.checks),origin_files=len(self.origins),
            source_imports_executed=0,scientific_calls=0,numerical_reanalysis=0,shot_recalculation=0,
            original_changes=changes,numeric_transfer='ORIGINAL_JSON_FLOAT_TOKENS_VERBATIM',
            separate_AI_review=False,limited_to_known_locator_closure=True))
        self.private_map.parent.mkdir(parents=True,exist_ok=True)
        self.private_map.write_text(json.dumps({'roots':self.roots,'original_sha256':self.initial},indent=2))
        print(json.dumps(dict(status='PARTIAL_WITH_GAPS',tasks=67,axes=134,origins=len(self.origins),checks=len(self.checks),gaps=len(self.gaps))))

    def runtime(self,cfg,spec,config_path):
        seal=cfg['sealed_files'];provider=Path(cfg['provider_root'])
        def inspect(row):
            p=Path(row['path']);current=sha_file(p) if p.is_file() else None
            out=dict(saved=self.public(row),current_sha256=current,
                matches_saved=current==row['sha256'],current_bytes=p.stat().st_size if p.is_file() else None)
            if row['role'].startswith('H2_SOURCE:'):
                rel=row['role'].split(':',1)[1];blob=self.git(SOURCE,rel)
                out.update(fixed_Git_SOURCE=SOURCE,fixed_Git_path=rel,fixed_Git_sha256=sha_bytes(blob) if blob is not None else None)
            if p.is_relative_to(provider):
                rel=p.relative_to(provider).as_posix();out['provider_git_comparisons']=[]
                for commit in (cfg['runtime']['provider_SOURCE'],FIXED['old_provider_SOURCE'],SOURCE):
                    blob=self.git(commit,rel)
                    out['provider_git_comparisons'].append(dict(commit=commit,path=rel,
                        sha256=sha_bytes(blob) if blob is not None else None,matches_saved=sha_bytes(blob)==row['sha256'] if blob is not None else None))
            return out
        # Available CPU determines independent hash workers; no scientific work.
        with ThreadPoolExecutor(max_workers=min(len(seal),len(os.sched_getaffinity(0)))) as pool:
            inventory=list(pool.map(inspect,seal))
        for original,current in zip(seal,inventory):
            if current['current_sha256'] is not None:
                # Include every named seal in preservation without publishing
                # its content (in particular seed registries and approval).
                path=original['path'];self.initial[path]=current['current_sha256']
        for row in inventory:
            if not row['matches_saved']:self.gap('runtime_changed:'+row['saved']['path'],'Current byte differs or is missing. Historical sealed hash remains authoritative for the old run; current runtime is not substituted.')
        source_rows=[r for r in inventory if 'fixed_Git_SOURCE' in r]
        self.check(all(r['fixed_Git_sha256']==r['saved']['sha256'] for r in source_rows), 'ALL_ACQUISITION_GIT_SOURCE_SEALS')
        packages=[]
        for row in seal:
            if Path(row['path']).name in ('METADATA','PKG-INFO'):
                p=Path(row['path']);headers=p.read_text(errors='replace').splitlines()
                packages.append(dict(origin=self.origin(p,'saved_package_distribution_identity'),
                    headers=[x for x in headers if x.startswith(('Name:','Version:'))],saved_sha256=row['sha256']))
        supplemental=[]
        relevant=('src/trotterlib/df_partial_s2.py','src/trotterlib/df_hamiltonian.py',
            'src/trotterlib/pr2_s0_s1_validation.py','src/trotterlib/product_formula.py','src/trotterlib/rte.py',
            'src/trotterlib/rte_compiled_cost.py','src/trottertracks/resource_applicability/ax2a_preparation.py',
            'src/trottertracks/resource_applicability/ax2b_native_df_v5.py')
        sealed_index={r['path']:r for r in seal}
        for rel in relevant:
            p=provider/rel;entry=dict(path=self.token(str(p)),historically_sealed=str(p) in sealed_index,
                historical_sha256=sealed_index.get(str(p),{}).get('sha256'),current_sha256=sha_file(p) if p.exists() else None,
                current_bytes_prove_historical_load=False)
            if p.exists():self.origin(p,'explicit_native_provider_static_hash')
            supplemental.append(entry)
            if not entry['historically_sealed']:self.gap('unsealed_native_provider:'+rel,'Present current provider bytes do not supply a historical import/hash attestation. Scientific SOURCE alone is not full external provider closure.')
        old_manifest=Path(spec['runtime_manifest']);self.check(sha_file(old_manifest)==spec['runtime_manifest_sha256'],'OLD_RUNTIME_MANIFEST_SHA')
        self.gap('per_worker_loaded_module_manifest','No per-worker actual-import manifest referenced by the original config. Imported-runtime seal roles and SOURCE origin checks are distinct from a complete observed import graph.')
        self.write('runtime_provider_manifest.json',dict(runtime=self.get(config_path,'/runtime'),
            sealed_inventory_origin=dict(origin=self.origin(config_path,'sealed_runtime_inventory'),pointer='/sealed_files'),
            historical_sealed_files=inventory,distribution_identity=packages,
            supplemental_native_files=supplemental,
            old_runtime_manifest=self.get(old_manifest),old_python_saved_sha256=spec['python_sha256'],
            provider_source_and_executed_scientific_SOURCE_are_distinct=True,
            current_bytes_are_not_historical_load_attestation=True,complete_actual_import_manifest=None,
            known_fixed_commits=FIXED))

    def old_pf(self,cfg,spec,oldbase):
        dp=Path(cfg['inputs']['D_result']['path']);d=self.read(dp,'old_full_PF_result')
        folder=dp.parent/'compile_workers';rows=[]
        for target in OLD:
            i,m=next((i,m) for i,m in enumerate(d['measurements']) if m['cell_id']==target)
            taskfolder=folder/target
            tr=taskfolder/'task_result.json';v=self.read(tr,'old_full_PF_task_result')
            self.check(v==m,'OLD_TASK_ACCEPTED_MEASUREMENT:'+target)
            row=dict(candidate=target,measurement=self.get(dp,'/measurements/'+str(i)),
                task_result=self.get(tr),control=self.get(taskfolder/'control.json'),axes=[],
                compiled_action_phase_receipt=None,compiled_circuit_saved_locator=None,compiled_vector_saved_locator=None)
            for axis in ('cosine','sine'):
                p=taskfolder/('measured_'+axis+'.json');self.check(self.read(p,'old_axis_measurement')==m['axes'][axis], 'OLD_AXIS_JOIN:'+target+':'+axis)
                row['axes'].append(dict(axis=axis,measurement=self.get(p),
                    compile_reservation=self.get(taskfolder/('compile_'+axis+'_reserved.json'))))
            # Bounded inspection of the exact two worker output folders.
            row['bounded_worker_file_inventory']=[self.token(str(p)) for p in sorted(taskfolder.rglob('*')) if p.is_file()]
            self.gap(target+':compiled_action_phase','No compiled-vector/phase receipt or saved QPY/vector found in this exact worker folder; saved native control PASS is not a compiled action gate.')
            rows.append(row)
        self.write('old_pf_receipts.json',dict(compile_SOURCE=self.get(dp,'/SOURCE'),binding=self.get(dp,'/binding'),
            reference=self.get(dp,'/reference'),rows=rows,
            completion=self.get(dp.parent.parent/'completion_audit.json'),
            native_exit=self.get(dp.parent.parent/'native_exit.json'),
            result_sha256=sha_file(dp),SOURCE_from_original_spec=self.get(oldbase/'D_activation01/D_spec_EXECUTION.json','/SOURCE'),
            provider_SOURCE=FIXED['old_provider_SOURCE'],compiled_gate_not_inferred_from_new_PR=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pipeline',required=True);p.add_argument('--exploration',required=True)
    p.add_argument('--old-spec',required=True);p.add_argument('--output',required=True);p.add_argument('--private-map',required=True)
    args=p.parse_args();repo=Path(__file__).resolve().parents[1]
    require(not Path(args.output).exists(),'NEW_EXPORT_DESTINATION_REQUIRED')
    ex=Export(repo,args.output,args.private_map);ex.run(args.pipeline,args.exploration,args.old_spec)


if __name__=='__main__':main()
