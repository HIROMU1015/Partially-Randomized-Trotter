"""Static bytes/AST evidence only; never imports scientific modules."""
from pathlib import Path
from hashlib import sha256
import json,subprocess,sys,ast
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.dont_write_bytecode=True
from trottertracks.algorithm_codesign.g6_w2c_b3_api_binding_v1 import static as S
from trottertracks.algorithm_codesign.g6_w2c_b3_real_adapter_v1.binding import specification


def write(name,value):
    (HERE/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def main():
    refs={'integration':'62967269e611b61e73b70abe20e998d9c7eaaf50',
          'independent':'5c1e13fa74b3c5e4f3166babed785274844655d0',
          'api_binding':'82896d53fa97235fa848d6c0833eb193af8b99d6',
          'api_independent':'5e862a70f2d74defe72dd39205d38c2a4f8ebeaf',
          'global_bundle':'d24cccbecdf0928fe977080dd7b28c591df0089e',
          'storage_v2':'596781daeba253418ba35f4b7964312a54caf233',
          'science':'76ca3c4089b201d55da0fede08e1f77733ac38a5',
          'common_policy':'7eff8a33bef4ee1f29698ab10c6da46988dc8db5'}
    inventory=S.inventory(ROOT)
    inventory['semantics']=specification()
    for path,row in inventory['modules'].items():
        raw=subprocess.check_output(['git','show',refs['science']+':'+path],cwd=ROOT)
        assert sha256(raw).hexdigest()==row['sha256']
    write('scientific_api_binding_manifest.json',dict(schema='b3_real_adapter_static_binding_v1',
        references=refs,inventory=inventory,confirmed='SOURCE_BYTES_AND_AST_ONLY',
        statically_inferred=['future document binding to generator/session','observation capture shape'],
        execution_only=['actual type/law/bit consumption','live evidence/requery compatibility'],
        unimplemented=['positive canonical F3 preflight','real dispatch/loader','real authority'],
        old_inventory_count=sum(len(x['entries']) for x in inventory['modules'].values())))
    oldpaths=subprocess.check_output(['git','ls-tree','-r','--name-only',refs['integration'],
        '--','src/trottertracks/algorithm_codesign','tests/tracks/algorithm_codesign'],cwd=ROOT,text=True).splitlines()
    hashes={}
    for path in oldpaths:
        if not Path(ROOT/path).is_file():continue
        raw=subprocess.check_output(['git','show',refs['integration']+':'+path],cwd=ROOT)
        current=(ROOT/path).read_bytes()
        assert raw==current,path
        hashes[path]=dict(bytes=len(raw),sha256=sha256(raw).hexdigest())
    write('preservation_before.json',dict(base=refs['integration'],unchanged_original_paths=hashes,
        method='Known project SOURCE/tests only; no other-worktree freeze or broad temp search',
        other_worktrees_modified=False,old_raw_grant_ledger_marker_STOP_modified=False))
    fixed=json.loads((ROOT/'docs/research/track_b_g6_w2c_b3_v2_production_bundle_bridge_candidate_20261019/fixed_scientific_reference.json').read_text())
    write('bitstream_contract_candidate.json',dict(schema='b3_bits_contract_candidate_v1',
        algorithm='SHA256(UTF8 domain || uint64 big-endian counter), concatenate blocks, take high width bits; discard remainder per call',
        artificial_namespace='ARTIFICIAL_REAL_ADAPTER_V1',scientific_namespaces_reference_only=[
            'B3_FULL_N_NEW_20261019_V1','PROPOSED_B3_LIMITED_G7_PILOT_20261019_V1'],
        identity_fields=['source_release','input','scope','claim','run','mode','arm','layer','axis','phase','pass','local_trial','global_trial'],
        seed_fields=['namespace','arm','axis'],mode_in_global_identity=True,mode_in_seed=False,
        shared_prefix=['S/P1','A/B','correctness/profile','baseline/instrumented'],
        scientific_fixed_reference=fixed['scientific_reference']['input'],
        precision_reference=fixed['scientific_reference']['policy'],
        angle_reference=fixed['fixed']['angle'],actual_scientific_bits_generated=0,
        real_variable_draw_equivalence='NOT_ESTABLISHED',IID='NOT_ESTABLISHED'))
    write('codec_conversion_manifest.json',dict(schema='b3_codec_conversion_v1',
        source='F3 $int/$rat/$tuple/$map canonical exact JSON',target='driver rat/int, tuple/list collapse, str map keys',
        lossless='original document/digest/source identity retained in typed sidecar',
        rejection=['noncanonical hex/rational','bool as int','unknown reserved tag','duplicate key','str-key collision',
                   'missing/unexpected field','wrong digest/source','unsupported type'],
        signed_values=True,map_order='F3 json.dumps(packed key,sort_keys=True); string maps sorted in canonical JSON',
        scientific_compatibility='NOT_ESTABLISHED'))
    write('result_adapter_contract.json',dict(schema='b3_result_adapter_contract_v1',
        variants={'EVENT':'ACCEPTED -> EVENT','ZERO/RAW_NONREDUCED':'ZERO_R','ZERO/COIN_REJECT':'ZERO_C',
                  'IDENTITY':'preserved; no driver zero projection','deterministic EVENT query':'EVENT_QUERY, no sampled-trial projection',
                  'failure':'category/reason/stage/input/details/original bytes, terminal STOP'},
        canonical='G7 observation/group/raw/child + canonical audit/requery; never F3 facade',
        full_B='F3-shaped sample/angle documents',full_A='G7 dict/None + captured trace/packet provenance',
        none_alone='REJECTED_INFORMATION_INSUFFICIENT',real_dispatch=False))
    write('authority_contract_candidate.json',dict(schema='b3_authority_candidate_v1',artificial_only=True,
        formal_authority=False,required_bindings=['science release','input','precision','runtime','seed','global scope/claim',
            'worker lineage','storage destination','contract digest','monitoring','STOP','retry0/resume false','global call/result/log budget'],
        fake_store='fresh private SQLite one winner, committed spent before ACK, STOP no refund',
        closed=dict(content_adopted=False,run_ready=False,contract_grants_execution=False,science_release=None,
                    scientific_adoption=None,artificial_only=True,mandatory_STOP=True),
        unresolved_external=['trusted store owner','keys/revocation','production atomicity','ACK uncertainty recovery policy',
                             'power durability','runtime containment'],
        real_credentials_keys_grants_claims=0))
    monitor=S.monitoring();monitor['new_adapter_functions']=['fixture.build','bits.ArtificialBits.bits','result.adapt_f3',
        'result.adapt_g7','result.angle_binding','codec.convert','codec.restore','codec.canonical']
    monitor['available_via']='fixed CachedObserver selects all pinned Python paths in instrumented pass'
    monitor['aggregation']='caller parent CPU + all waited worker CPU once; coarse/fine are alternatives; stages/IPC never added'
    write('monitoring_manifest_candidate.json',monitor)
    print('static packet generated; modules=',len(inventory['modules']),'entries=',sum(len(x['entries']) for x in inventory['modules'].values()))
if __name__=='__main__':main()
