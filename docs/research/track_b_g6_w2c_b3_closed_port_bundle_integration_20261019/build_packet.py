"""Technical report assembly from artificial evidence only."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import difflib,json,shutil,subprocess,sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SOURCE='deade78a19f5b23859d16816d3a7bfb37765dd80'
BASE='d24cccbecdf0928fe977080dd7b28c591df0089e'
PORT='82896d53fa97235fa848d6c0833eb193af8b99d6'
BRANCH='track-b-g6-w2c-b3-closed-port-bundle-integration-20261019'
sys.path.insert(0,str(ROOT/'src'));sys.dont_write_bytecode=True
def read(path):return json.loads((HERE/path).read_text())
def dump(path,value):(HERE/path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def main():
    from trottertracks.algorithm_codesign.g6_w2c_b3_port_bundle_v1 import contract as K
    from trottertracks.algorithm_codesign.g6_w2c_b3_port_bundle_v1 import offline as O
    matrix=read('runs_final/matrix.json');tests=read('runs_tests_final/results.json');reg=read('regressions/results.json');extra=read('additional/results.json');restored=read('restored_verification.json');archive=read('evidence/archive_manifest.json')
    c=K.candidate(ROOT);rows={x['case']:x for x in matrix['cases']}
    assert tests['PASS']==tests['run']==134 and tests['FAIL']==tests['ERROR']==tests['SKIP']==0
    assert all(x['PASS']==x['run']and not x['forbidden_imports']for x in reg)
    assert all(x['measured_git_SOURCE_commit']==SOURCE for x in matrix['cases'])
    assert all(not x.get('forbidden_imports',[])and not x.get('consumer_forbidden_imports')for x in matrix['cases'])
    assert restored['status']=='PASS'and len(restored['results'])==101
    assert rows['normal_bundle_n256']['calls']['completed']==16896
    dump('technical_contract_candidate.json',c)
    sample=O._read(Path(rows['normal_bundle_n256']['root'])/'output/manifest.json');dump('integration_manifest_candidate.json',sample);dump('endpoint_binding_manifest.json',c['endpoint_binding'])
    counts=Counter(x['verification_status']for x in matrix['cases']);mutcounts=Counter(x['status']for x in tests['mutations'])
    acceptance=dict(schema='b3_global_port_end_to_end_matrix_v1',SOURCE=SOURCE,new_tests={k:tests[k]for k in ('run','PASS','FAIL','ERROR','SKIP','collected')},cases=dict(total=53,classifications=dict(counts),observations=matrix['cases']),
        mutations=dict(total=45,classifications=dict(mutcounts),semantic_port_only=10,semantic_port_storage_pass_required=True,observations=tests['mutations']),additional=extra,
        fixed_regressions=reg,historical_845='fixed reference only; not rerun or added',counts_are_not_added_to_historic_counts=True)
    dump('artificial_acceptance_matrix.json',acceptance)
    callrecords=[];byterecords=[];dispatch=[];examples=[]
    for name in ('normal_bundle_small','normal_bundle_n256','variant_payload_propagation'):
        r=rows[name];v=read('runs_final/'+name+'_verification.json');root=Path(r['root']);final=O._read(root/'output/final.json')
        assert v['port_dispatch']['verified_trials']==v['port_dispatch']['scheduler_complete_joined']==final['planned']
        callrecords.append(dict(case=name,calls=v['calls'],states=final['states'],conditions=v['conditions'],mode_arm_layer_axis=v['port_dispatch']['cell_mode_counts'],offsets=O._read(root/'output/manifest.json')['tasks'],history=v['history'],one_claim=v['admission']))
        groups=Counter();files=[]
        for p in (root/'output').iterdir():
            if not p.is_file():continue
            category=('trace'if p.name=='operations.jsonl'else'journal'if p.name=='control.jsonl'else'statefile'if p.name=='trial_states.bin'else'pair_payload_and_F01'if p.suffix=='.jsonl'else'diagnostic'if p.name=='diagnostic.json'else'finalization'if p.name in ('final.json','delivery.json','marker.json')else'receipts'if'receipt'in p.name else'candidate_summary_and_fake_snapshot')
            groups[category]+=p.stat().st_size;files.append(dict(path='output/'+p.name,category=category,bytes=p.stat().st_size,sha256=sha256(p.read_bytes()).hexdigest()))
        groups['caller_observation']+=(root/'external_observation.json').stat().st_size
        byterecords.append(dict(case=name,global_budget=v['output_budget'],condition_attribution=v['condition_byte_attribution'],components=dict(groups),files=files,cost=v['cost']))
        dispatch.append(dict(case=name,run_id=final['run_id'],port_binding=final['port_binding'],candidate_binding=final['candidate_binding'],manifest_binding=final['manifest_binding'],**v['port_dispatch']))
        if name=='normal_bundle_small':
            for row in final['pairs']:
                if row['task']['phase_index']%3:continue
                for envelope in O.V.Reader(root/'output'/(row['task']['key']+'.jsonl')):
                    p=envelope.payload
                    if p.get('trial_id','').endswith('/0'):
                        examples.append(dict(case=name,pair=row['task']['key'],payload=O.V.pack(p)))
    dump('global_call_accounting.json',dict(schema='b3_global_port_call_accounting_v1',normal_cases=callrecords,science_calls=0))
    old=dict(source_commit=BASE,measured_SOURCE='e099dcc461247e81eff981867dc28c3f3b16df5e',n256_result_bytes=15502271,n256_log_bytes=43190850,reference='docs/research/track_b_g6_w2c_b3_v2_production_bundle_bridge_candidate_20261019/cpu_wall_and_output_accounting.md')
    full=next(x for x in byterecords if x['case']=='normal_bundle_n256')
    compare=dict(result_bytes=full['global_budget']['result']['physical']-old['n256_result_bytes'],log_bytes=full['global_budget']['log']['physical']-old['n256_log_bytes'],matched_scientific_cost_comparison=False,interpretation='different artificial payload/law shape and SOURCE; descriptive storage delta only')
    dump('global_byte_accounting.json',dict(schema='b3_global_port_byte_accounting_v1',normal_cases=byterecords,old_artificial_reference=old,n256_descriptive_delta=compare,cap_expanded=False))
    dump('port_dispatch_provenance.json',dict(schema='b3_global_port_dispatch_proof_v1',SOURCE=SOURCE,endpoint_binding=c['endpoint_binding'],normal_cases=dispatch,examples=examples,proof_scope='all normal trials: independently replayed artificial bits/event/evidence and joined scheduler ENTER/COMPLETE; F01 and post-close accepted separately',not_scientific_equivalence=True))
    fixed=[]
    for path,h in K.FIXED_PINS.items():assert sha256((ROOT/path).read_bytes()).hexdigest()==h;fixed.append(dict(path=path,sha256=h,unchanged=True))
    dump('source_inventory.json',dict(schema='b3_global_port_source_inventory_v1',SOURCE=SOURCE,base=BASE,fixed_port=PORT,source_pins=c['source_pins'],input_pins=c['scientific_reference']['input_pins'],runtime=c['python'],worker_runtime=c['worker_runtime'],fixed_bundle_and_port=fixed,postprocessing_helpers_not_measured_runtime=True))
    observations=[]
    for item in read('preservation_before.json'):
        status=subprocess.check_output(['git','status','--porcelain=v1','-z'],cwd=item['path']);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=item['path'],text=True).strip()
        observations.append(dict(item,after_HEAD=head,after_status_sha256=sha256(status).hexdigest(),observationally_unchanged=head==item['HEAD']and sha256(status).hexdigest()==item['status_sha256'],unrelated_progress_is_not_execution_gate=True))
    dump('preservation_after.json',observations)
    zero={k:0 for k in ('real_G7_F3_import','real_scientific_sampling','scientific_namespace_bitstream','new_scientific_calculation','saved_scientific_numeric_reanalysis','science_SOURCE_adoption','real_authorization_grant_claim_issuance_consumption','native_quantum_build_compile_synthesis','old_grant_reuse','old_STOP_removed','subagents_used')}
    verification=dict(schema='b3_global_port_final_verification_v1',status='ARTIFICIAL_END_TO_END_INTEGRATION_COMPLETE_UNADOPTED_EXECUTION_CLOSED',SOURCE=SOURCE,base=BASE,fixed_port=PORT,
        new_tests={k:tests[k]for k in ('run','PASS','FAIL','ERROR','SKIP')},cases=dict(counts),mutations=dict(total=45,classifications=dict(mutcounts),semantic_port_only=10),additional=extra,
        regressions=[{k:x[k]for k in ('label','run','PASS','FAIL','ERROR','SKIP','HEAD')}for x in reg],normal_n16_calls=1536,normal_n256_calls=16896,all_normal_trials_port_dispatched=True,
        restored_raw_members=restored['members'],restored_classifications=101,archive_bytes=archive['archive_bytes'],archive_SHA256=archive['archive_sha256'],archive_parts=len(archive['parts']),zeros=zero,
        fixed_SOURCE_bytes_preserved=True,protected_worktrees_observationally_unchanged=all(x['observationally_unchanged']for x in observations),
        content_adopted=False,run_ready=False,contract_grants_execution=False,science_release=None,scientific_adoption=None,retry=0,resume=False,mandatory_STOP=True,
        scientific_GO=False,separate_reviewer_used=False,server_offline_checker='same primary agent self-check',local_parallel_unpublished_results_used=False,
        unresolved_technical_discrepancies_in_executed_scope=[],next_owners=['USER','GPT','LOCAL_CODEX'])
    dump('verification.json',verification)
    fullcost=full['cost'];budget=full['global_budget']
    common=f'''固定人工測定SOURCE：`{SOURCE}`。直接基点Global Bundle `{BASE}`、選択取得API Port `{PORT}`、既存独立受入 `6bffb5b5819918698b522270ff19b255766d3072`。Branch `{BRANCH}`。

ARTIFICIAL_ONLY / UNADOPTED / EXECUTION_CLOSED。サーバーCodex主担当による技術自己検証。実G7/F3 import・科学bits・sampling・新科学計算・数値再解析・正式science/authority・native処理0。mandatory STOP。
'''
    docs={
    'README.md':'''# Global Bundle × API Port 人工End-to-End統合

固定Portを実際のGlobal Bundle workerへ接続した。単一global claim/coordinator/call/byte予算、6phase、S→P1、Storage V2/STOP/post-close joinを通り、n16=1536・n256=16896の全trialについてPort dispatchとscheduler COMPLETEをrawから結合した。別Port gridを連続実行して統合と呼ぶ構造ではない。

新統合版134 tests（旧107のassertionを新SOURCE向けに移植し27件追加、別module ID）は全PASS。53cases（bundle正常3・条件のみ2・技術失敗22・証拠不完全26）、45 mutation拒否（10件は旧storage検査を通る偽Port証拠）、追加negative3件。固定回帰107/73/41/57は別件数。実科学同値・性能・正式採択は主張しない。

読む順：[handoff](handoff.md)、[構造](integration_architecture.md)、[注入口](endpoint_injection_design.md)、[受入結果](artificial_end_to_end_acceptance.md)、[不整合と限界](producer_consumer_discrepancies.md)、[未決事項](pending_and_required_decisions.md)。JSONはcontract/manifest/endpoint binding、call/byte会計、全trial dispatch証明、SOURCE/runtime、verification、publication receiptを収録。
''',
    'integration_architecture.md':'''# 統合構造

旧bridge.execute_artificialはF.ArtificialFactory(runtime)を生成し、_conditionの全run_phaseへF.bits_factoryを直接渡していた。このhardwireが外部Port不在の原因。新subpackage g6_w2c_b3_port_bundle_v1/bridge.pyへ固定Bridgeをversioned派生し、Fを新adapter、Kを新contractへ切替。scheduler/worker/ControlledEndpoint/run_pair/Stream/fused codec/CachedObserver/coalesced client/Storage V2は固定実装を再利用する。

```mermaid
flowchart TD
 A[単一fake claim SPENT] --> B[単一global coordinator / 48pair / budget]
 B --> C[S phase0..2]
 C -->|S完了| D[P1 phase3..5]
 C -->|失敗| Z[Global STOP / drain]
 D --> W[固定spawn worker / permit ENTER]
 W --> P[新adapter → 固定ArtificialEndpoint / ArtificialBits]
 P --> E[固定pair engine / canonical / F01]
 E --> T[global state / counter / trusted trace / journal]
 Z --> T
 T --> F[STOP / storage close / receipt / marker / delivery / caller]
 F --> V[Producer-free offline storage + Port proof]
```

全S/P1は同じclaim、uint8 statefile、cursor/ticket、trace/journal、64GiB/256MiBの一つのbudgetを共有する。conditionごとの独立実行や独立grantは作らない。
''',
    'endpoint_injection_design.md':'''# Versioned Endpoint注入口

adapter.ArtificialFactoryは固定Port ArtificialEndpoint/ArtificialBitsだけを構築する。任意factory/callable・real module loaderは受け付けない。runtime/source/context seal、module location、taskの厳密canonical bytesをspawn constructorで確認。元Port6filesと旧Bundle52filesの公開hash58件をcontractへ明示固定し、改変後candidate再生成による再固定も拒否する。

人工EndpointのOutcome/evidenceは固定Port由来。adapterはdispatch metadataと人工event tag PORT_V1/PORT_V2を付加し、元Portのconsume/result validationを保持する。tag変更は人工接続確認用のversioned fixture差で、科学law変更ではない。内側Portがsampleしたこと、Bitsが消費されたこと、保存raw/event/evidence/angle/hashが一致することをofflineの独立surrogate実装で全正常trial確認する。

F01にはPortFailureのcategory/stage/reason/mandatory STOPと元例外を保存。generator/audit/angle/requery/invalid result/consume failureをzeroへ変更しない。固定Portの科学mappingやprecisionを黙って補完しない。canonical Bは元overlay、Full BはF3 facadeという静的差を保持するが、今回は両方人工surrogate。
''',
    'worker_and_spawn_binding.md':'''# Worker・spawn binding

同じ固定run_phaseがS/P1のspawn workerへ新ArtificialFactoryとbits_factoryを渡す。Sは逐次pair、P1は固定8pair構造。workerはsource/runtimeを検査してからconstructorへ進む。必要contextはrun/task plan、variant/人工fault、candidate/manifest/runtime/endpoint digestのみ。fake storeオブジェクト、approval、grant、claim操作APIは渡さない。manifestのdigestと公開非credential識別子は実認可ではない。

親はvalidate前、workerは既存BlockScienceおよびPort.install_guardで禁止科学moduleをload前拒否。既loadも拒否。SOURCE hash・factory class/path・task/global identity・artificial_onlyを照合する。worker PIDはdispatch evidence、親worker report/control journalと結合。real factoryへ切り替えるbooleanやunseal実装はない。

将来の正式Port/runtime/authority releaseは別SOURCEで採択が必要。Python実行環境自体に悪意あるmemory patchがないという技術前提を超えるsecurity sandboxの証明はしない。
''',
    'global_identity_and_seed_prefix.md':'''# Global identity・seed-prefix

run/condition/task/arm/layer/axis/phase/pass/local trial/global ID/global offset/seed domainを分離。global IDはrun/mode/local ID、offsetは48-task planの単調range。dispatch metadata、ENTER/COMPLETE trace、physical state、raw trial identityが同じslotへ到達することを照合する。

固定科学参照はP5_general_order、p=(1/5,3/10,1/2)、x=5/7、m=5、sigma=+1。root256/probability160、元eta/rho/alpha/precisionとA6 math-onlyを変更しない。

人工bitsは固定PortのARTIFICIAL_API_BINDING_V1/arm/axis、SHA256(seed UTF8 || uint64 big-endian counter)、bits160は各block上位160、下位96をdiscard。mode/layer/phase/passはseedへ足さない。S/P1、A/B、correctness/profile、baseline/instrumentedのprefixは一致、armとX/Yは分離。counter/width/value/consumed total/event/evidence digestをofflineで再現する。

人工canonical2draw、人工Full ZERO_R1/ZERO_C2/EVENT3drawというsurrogate仕様。実G7の可変degree law・実F3 trace・fresh IID・angle enclosureの同値証明ではない。科学namespaceは文字列参照のみで生成0。失敗prefixは正常比較から除外。
''',
    'admission_one_shot_and_stop.md':'''# Admission・one-shot・STOP

private新規FAKE_ONLY SQLiteだけを使用。単一global scope/claim、BEGIN IMMEDIATE/unique constraint、SPENT commit後ACK。32raceでwinner1、double/spent/STOP後/retry/resumeを拒否。commit ACK不明でも消費済みを返却せず、可能なfake STOPだけ記録する。実store、旧spent、grant/authorizationへアクセスしない。

S Port failureでglobal STOP、P1開始禁止。P1 failureでS receipt/rawを保持。同claimでP1だけ再開しない。親arrival順、STOP後ENTER禁止、STOP前inflight drain、START_RESERVEDとbody entryの違い、UNKNOWN・未ACK bytes非返却を固定Storage V2から引き継ぐ。

snapshot/historyは保存された人工状態の照合であり、未保存全atomic transition、実trusted store、電源断durabilityの証明ではない。STOP保存先別の失敗・不完全は結果へ残し、成功へ格上げしない。
''',
    'storage_v2_and_post_close_acceptance.md':'''# Storage V2・post-close受入

RAM shadow intent、physical statefile、ordered trusted prefix、journal、pair/F01、byte credit、summary/condition receipt/global receipt、diagnostic、fake STOP/spent、storage close、marker、final/delivery/callerを一つのglobal会計で結合する。新Port evidenceは既存pair streamへ入り、別sidecar/gridの証拠を代替しない。

receiptはstorage close後、markerはreceipt close後。final4MiB/delivery4096/caller4096のcreditを初めに予約する。自己ACKはprojectionと明示し、外側のdelivery/caller観測が揃うまで完全成功としない。diagnosticはcomplete=false/final_acceptance=false。marker単独・close後失敗・caller failure隠蔽を拒否する。

RESERVE/ENTER/COMPLETE保存故障、partial trace、statefile、caps、ACK欠落、summary/receipt/diagnostic/STOP/marker/close/delivery故障を検証。26casesは証拠不完全を維持。empty summary/receiptをPort拡張checkerが再parseしてintegrityへ誤分類した開発不具合を修正し、保存されない証拠を補完しない。caller witness自身のcloseとfsync/power durabilityは未証明。
''',
    'global_call_and_byte_accounting.md':f'''# Global call・byte会計

n16は各条件256+256+256=768、S/P1計1536。n256は各条件256+4096+4096=8448、計16896。各正常でplanned=reserved=started=completedが総数に一致、failed/unknown/cancelled0。各mode/arm/layer/axisと全offsetをglobal_call_accounting.jsonに保存。

n256 resultは{budget['result']['physical']:,}B、logは{budget['log']['physical']:,}B。両方reserved=observed=physical、unknown0。64GiB/256MiB capは拡大なし。component別はpair payload/F01、trace、journal、statefile、receipts、diagnostic、summary/input/snapshot、finalization/callerに分解。global_byte_accounting.jsonに全file SHA/bytesとcondition/shared attributionを保存。

旧人工Bundle固定記録との差はresult {compare['result_bytes']:+,}B、log {compare['log_bytes']:+,}B。人工Portのpayloadとlaw形状、SOURCEが違う記述比較で、matched science比較や実G7保存上界ではない。
''',
    'cpu_wall_and_monitoring.md':f'''# CPU/wall・monitoring

n256人工観測：parent {fullcost['parent_cpu_ns']/1e9:.9f}s、全worker {fullcost['all_worker_cpu_ns']/1e9:.9f}s、total {fullcost['total_cpu_ns']/1e9:.9f}s、global outer wall {fullcost['global_wall_ns']/1e9:.9f}s。callerのbridge開始前からdelivery close後のwindow。caller観測文書自身の保存とoffline検査は後処理。

parent＋全workerを一度だけ足す。pair/IPC/observer/stageをouterへ再加算せず、pair wall和をglobal wallへしない。条件CPU/wallとshared setup/finalizationを区別し、未配賦を二重に配らない。constructor/source checks、generator＋audit、angle＋requery、evidence、serialization/hash/write/flush/close、IPC、CachedObserver/coarse/fine代替partition、ownRSS/HWM/IO、retained bytesを記録。

audit/angle/requery個別Python関数境界はfine observerで観測可能だが、isolated observer overhead、C有理数bit演算、親の全個別I/O費用、unique continuous RAM peakは未識別。実G7/F3 local coefficient/child law/precision/angle requery費用は未計測。S→P1順序、共有負荷、cache交絡を保持し、科学CPU予測・speedup・母平均CIへ拡張しない。fused同士なのでfused単独効果は識別不能。

64CPU・十分な空きRAMと低PSIを確認し、独立人工casesを8プロセスで並列化。固定P1 worker構造は保持。共有OS/cgroup/package/service/他process変更0。無関係worktreeのHEAD/statusは観測であり停止gateではない。
''',
    'source_and_contract_delta.md':'''# SOURCE・contract差分

新namespaceはsrc/trottertracks/algorithm_codesign/g6_w2c_b3_port_bundle_v1/。bridgeは旧Bundleのversioned派生で、imports/guard、runtime人工fixture、factory context、summary/receipt/finalのport_bindingを変更した。storage_consumerは旧offlineを派生し新candidate/manifest/seed schemaを認める。全差分をbridge_delta.patch/storage_consumer_delta.patchに保存。scheduler/Storage V2/state/trace/counter/one-shot/post-close algorithmは固定再利用。

API Port6filesは82896d53から同bytesで選択取得。port原型、科学SOURCE、旧Bundle/V2/checker/契約/results/raw/grant/ledger/marker/STOPを変更しない。旧branchをmerge/rebaseせず、local独立受入の未公開SOURCE・途中結果を入力にしない。

新schema b3_closed_port_global_bundle_candidate_v1 / b3_closed_port_global_bundle_manifest_v1。固定n256/16896、16cell/6phase、S→P1、fused/CachedObserver、64GiB/256MiB、retry0/resume falseを保持。JSON schemaは最小closed shape、contract.guardとofflineの厳密検査が規範。実認可を与えない。
''',
    'artificial_end_to_end_acceptance.md':f'''# 人工End-to-End受入

| 対象 | 最終結果 |
|---|---|
|新integration tests|134 PASS / FAIL0 / ERROR0 / SKIP0（移植107＋統合追加27）|
|53 integrated cases|正常bundle3、condition-only2、technical failure22、incomplete26|
|n16/n256|1536/16896全trialでPort evidence＋scheduler COMPLETE join|
|45 mutation|integrity42 / incomplete3、うちPort-only semantic10は旧storage検査PASSを要求|
|追加negative3|S失敗後P1 ENTER偽造、P1失敗時S receipt喪失、mode入りseedを拒否|
|固定回帰|Bundle107、API73、Storage41、既存独立57：各全PASS、別件数|
|fresh restore|{len(archive['members'])} members / 101分類一致|

raw archiveは{archive['archive_bytes']:,}B、{len(archive['parts'])}分割、SHA256 `{archive['archive_sha256']}`。全member inventoryと復元手順をevidence/archive_manifest.json / verify_restored.pyへ保存。各trialの人工bit replay→kind/event/weight→evidence/angle full/hash→global ENTER/COMPLETE→state/counter→receiptsを独立ロジックで確認。payload PORT_V1→V2はbit列不変のままevent/raw hashへ伝播。

旧107/73/845や原版622/1・補正版623を新件数へ加算しない。旧co_qualname差は固定履歴のまま。技術失敗/incompleteの期待分類一致は完全成功の証明ではない。
''',
    'producer_consumer_discrepancies.md':'''# Producer/consumer不一致・修正履歴

最終封印SOURCE deade78aの53ケースと45mutationは期待分類に一致。実施範囲で未解決の技術不整合なし。

開発時：疎checkoutのpilot schema不足は固定Git bytes補完で解消。Port拡張consumerがwrite failureで残ったempty summary/receiptを再parseしintegrityへ誤分類したため、保存失敗/不完全の元判定を保つよう修正した。Reader iterable testとworker runtime entrypoint allowlistもtest実装を修正した。

unsealed development3途中でcandidate pinを追加してしまい、一部worker/consumerがSOURCE mismatchを検知した。途中分を最終PASSへ使わず履歴保存し、固定development4、公開sealed SOURCEの順で全cases/testsを実行し直した。封印後SOURCEは変更なし。正常計算を意図的に中断・再起動していない。

固定Bundle107回帰の初回は旧runs_final（開発41case）をdefault参照して96PASS/7FAIL/4ERRORとなった。原本を変えず、封印済みruns_sealed（43cases、e099dcc）へ入力を直し107PASS。API73/Storage41/既存独立57も全PASS。初回ログ/件数と原因をdevelopment_diagnosticsに保存し、最終へ加算しない。

offline checkerは同じ主担当の自己検証。全故障一般、実G7/F3同値、実authority/power-loss、別物理hostを証明しない。
''',
    'pending_and_required_decisions.md':'''# 未決事項・次の担当

ユーザー/GPT：人工統合結果と並行ローカル独立受入を照合し、正式science SOURCE、real API/bitstream、authority/bootstrap、runtime、監視・保存・STOP・予算・seed・CPU estimand/coverageを採択するか判断する。このSOURCEは未採択・閉鎖のまま。bool反転による科学移行は禁止設計。

後続ローカルCodex：最終公開HEAD/bytes/SHA、選択Port6files/固定旧58pins、contract/schema、worker/runtime/task/source、全trial dispatch proof、人工bits/prefix、global one-shot/STOP/UNKNOWN、V2/caller join、134tests/53cases/45mutations/追加3、fresh restoreを独立受入する。サーバー自身のoffline判定を別reviewerと数えない。

実G7/F3 law/bit/angleの実行同値、path率、accept/reject、有理数bit長、local coefficient/child law/requery費用、実CPU/storage、Full-N可否は未測定。real trusted store atomic/ACK/STOP/durabilityとcaller自身closeも未確立。native guarantee/algorithm効果/性能/科学的GOは今回認定しない。

サーバーCodexは公開・保全照合後mandatory STOP。正式production、correctness、実科学pilot16896、Full-N31710848へ自動移行しない。
''',
    'handoff.md':'''# Q1〜Q10 技術handoff

Q1：固定Bundle/API Port/独立受入のSOURCE/契約をGit object/bytes/SHAで照合。local未公開結果の利用・merge/rebaseなし。

Q2：新versioned bridge/contract/adapter/storage_consumer/offlineを追加。旧factory hardwireの接続先を固定人工Portへ切替え、旧SOURCEは変更なし。

Q3：全正常trialのPort source/factory/PID/contract/manifest/runtime/global ID/offset/consumeがpair raw/evidence full/hashへ入り、ENTER/COMPLETE trace、state、summary/receipt/callerへ結合。port_dispatch_provenance.jsonとraw参照。

Q4：n16/n256のplanned/reserved/started/completed=1536/16896、UNKNOWN等0。単一global bundle。

Q5：modeはIDのみ、人工seedはarm/axis。S/P1・A/B/pass/prefixとcounter/width/value、event/weight/angle/canonical/F01を照合。

Q6：one-shot、race/spent/ACK不明、S失敗後P1禁止、P1時S保持、inflight drain/UNKNOWN・byte非返却を検証。

Q7：RAM/physical/trusted prefix/journal/receipts/diagnostic/marker/STOP/close/delivery/callerを結合。保存不足26casesは不完全を維持。

Q8：新134PASS、53cases（正常bundle3・条件のみ2・技術失敗22・不完全26）、45mutations＋追加3拒否。固定回帰107/73/41/57別件数。最終実施範囲で未解決不整合なし。

Q9：公開SOURCEとraw/contract一式を後続ローカルCodex独立受入へ。ユーザー/GPTは並行独立結果と照合して正式release/実API/authority/実行条件を別判断。

Q10：実G7/F3 import/sampling、科学bits、新科学計算/数値再解析、science採択、実authorization/grant/claim、native quantum処理0。旧grant/STOP保全、subagent0。

最終公開HEAD・bytes/SHAはpublication_receipt.jsonと最終チャットを参照。receipt自身は後続receipt-only commitの外部読戻しで確認し、自己hashを循環記載しない。mandatory STOP。
'''}
    for name,body in docs.items():
        title,rest=body.split('\n',1);(HERE/name).write_text(title+'\n\n'+common+'\n'+rest.lstrip())
    print(json.dumps(dict(status=verification['status'],docs=len(docs),calls=16896,output=budget,mutations=dict(mutcounts))))
if __name__=='__main__':main()
