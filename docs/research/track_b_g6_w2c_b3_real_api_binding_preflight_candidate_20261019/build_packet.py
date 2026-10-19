"""Assemble technical self-check evidence; no scientific numeric analysis."""
from pathlib import Path
import collections,hashlib,json,subprocess
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SOURCE='78c571005fe766c61b32dc867e006531099ff608'
BASE='d24cccbecdf0928fe977080dd7b28c591df0089e'
INDEPENDENT='6bffb5b5819918698b522270ff19b255766d3072'
BRANCH='track-b-g6-w2c-b3-real-api-binding-preflight-candidate-20261019'

def read(p):return json.loads((HERE/p).read_text())
def dump(p,x):(HERE/p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')

def main():
    unit=read('acceptance/unit.json');ports=read('acceptance/port_cases.json');mutations=read('acceptance/mutations.json')
    bundle=read('acceptance/bundle_matrix.json');legacy=read('regressions/legacy/results.json')
    fixed=read('regressions/storage_and_independent/results.json');bt=read('regressions/bundle/results.json')
    restored=read('restored_verification.json');archive=read('evidence/archive_manifest.json');candidate=read('technical_contract_candidate.json')
    assert unit['PASS']==unit['run']==73 and unit['FAIL']==unit['ERROR']==unit['SKIP']==0
    assert all(x['pass_count']==x['run'] and not x['forbidden_imports']for x in fixed)
    assert all(not x['forbidden_imports']for mode in legacy.values()for x in mode['suites'])
    assert legacy['original']['pass_count']==622 and legacy['original']['fail']==1
    assert legacy['corrected']['pass_count']==623 and legacy['corrected']['fail']==0
    assert bt['pass_count']==107 and restored['status']=='PASS'
    assert ports[1]['calls']==16896
    assert all(x['measured_SOURCE']==SOURCE for x in [json.loads((Path(y['root'])/'port_report.json').read_text()) for y in ports])
    counts=dict(collections.Counter(x['verification_status']for x in bundle['cases']))
    mcounts=dict(collections.Counter(x['status']for x in mutations))
    matrix=dict(schema='b3_API_artificial_acceptance_matrix_v1',
        unit=dict(run=73,PASS=73,FAIL=0,ERROR=0,SKIP=0,ids=unit['collected']),
        new_port_cases=[{k:x[k]for k in ('case','status','calls')} for x in ports],
        new_offline_mutations=dict(cases=22,classifications=mcounts,observations=mutations),
        unchanged_bundle=dict(cases=43,classifications=counts,tests=107,
            overlays=35,overlays_included_in_107=True,source_commit=BASE),
        fixed_existing_regressions=dict(corrected_legacy=623,storage_v2=41,prior_independent_v2=57,bundle=107,
            distinct_current_versioned_IDs=828,local_845='fixed external reference only, not rerun or added'),
        original_label_result=dict(PASS=622,FAIL=1,not_erased=True,not_added=True),
        new_counts_not_added_to_historic_counts=True,scope='same primary agent technical self-check')
    extra=read('acceptance/additional_controls.json')
    assert len(extra['controls'])==4 and all(x['PASS']for x in extra['controls']) and not extra['forbidden_imports']
    matrix['additional_postprocessing_controls']=extra
    dump('artificial_acceptance_matrix.json',matrix)
    after=[]
    for x in read('preservation_before.json'):
        p=Path(x['path']);status=subprocess.check_output(['git','status','--porcelain=v1','-z'],cwd=p)
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip()
        after.append(dict(x,after_HEAD=head,after_status_sha256=hashlib.sha256(status).hexdigest(),
            observation_unchanged=head==x['HEAD'] and hashlib.sha256(status).hexdigest()==x['status_sha256'],
            unrelated_worktree_change_is_not_a_failure_or_execution_gate=True))
    dump('preservation_after.json',after)
    zeros={k:0 for k in ('real_G7_F3_import','real_G7_F3_sampling','scientific_namespace_bits_generated',
                         'new_scientific_calculation','saved_scientific_numeric_reanalysis','science_SOURCE_adoption',
                         'real_authorization_grant_claim_issuance_consumption','old_spent_scope_reuse',
                         'native_quantum_build_compile_synthesis','subagents_used','production_gate_unlocked')}
    verification=dict(schema='b3_API_closed_candidate_verification_v1',
        status='ARTIFICIAL_PORT_ACCEPTANCE_COMPLETE_WITH_PENDING_REAL_BINDING_EXECUTION_CLOSED',
        measured_SOURCE=SOURCE,base_commit=BASE,independent_reference=INDEPENDENT,
        new_unit=matrix['unit'],new_port_cases=matrix['new_port_cases'],new_offline_mutations=dict(cases=22,classifications=mcounts),
        additional_postprocessing_controls=extra,
        existing_regressions=matrix['fixed_existing_regressions'],bundle_classifications=counts,
        original_label_result=matrix['original_label_result'],raw_archive=dict(members=len(archive['members']),bytes=archive['archive_bytes'],sha256=archive['archive_sha256'],parts=len(archive['parts'])),
        restored_classifications=len(restored['results']),restored_status=restored['status'],
        fixed_runtime_bytes_equal=True,fixed_input_bytes_equal=True,all_protected_worktree_observations_unchanged=all(x['observation_unchanged']for x in after),
        zeros=zeros,run_ready=False,content_adopted=False,science_release=None,scientific_adoption=None,
        contract_grants_execution=False,scientific_GO=False,mandatory_STOP=True,
        unchanged_bundle_endpoint_injection='UNAVAILABLE_BY_FIXED_DESIGN; not implemented or falsely tested',
        author_self_check=True,independent_reviewer_used=False,local_Codex_independent_acceptance='NEXT_SEPARATE_TASK',
        unresolved_technical_defects_in_tested_scope=[],
        pending=['formal scientific adoption and release','future separate bundle endpoint-injection source',
                 'real bits consumption/law/angle execution equivalence','real authority bootstrap and durability',
                 'actual science path coverage/cost/storage and Full-N feasibility','CPU estimand and measurement resolution'])
    dump('verification.json',verification)
    source_url=f'https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/{SOURCE}'
    preamble=f'''固定SOURCE：[{SOURCE}]({source_url})。直接基点 `{BASE}`。ローカル独立受入 `{INDEPENDENT}` は固定参照のみでmerge/rebaseしない。

今回の範囲は静的読取・AST・hash照合と人工技術検証。実科学入口閉鎖、科学的採択なし、mandatory STOP。サーバーCodex主担当一名による自己検証であり、別reviewerの独立受入ではない。
'''
    docs={
    'README.md':'''# 実G7/F3 API binding・authority・monitoring 閉鎖候補

実APIの静的mapping、別versioned port、人工bits/endpoint、fake authority、preflight、Producerと分離したoffline判定を追加した。実科学へdispatchするSOURCEは作成していない。

新規73 tests PASS、4 portケース（正常2・技術失敗2）、新offline mutation22件拒否。既存回帰は補正版623＋V2 41＋既存独立57＋Bundle107＝828の別versioned ID。過去828/845へ新件数を重複加算しない。

**接続上の未解決点**：固定Global Bundleは組込みfixtureをhardwireし外部endpointを受けない。今回のportは固定pair engineへ人工結合し、変更していないBundleの保存・STOP回帰と証拠を併置した。新portが固定Bundle内部で実行されたとは主張しない。将来の実注入口は別SOURCEの採択対象。

入口：[handoff](handoff.md)、[API調査](scientific_api_static_inventory.md)、[binding設計](real_api_binding_design.md)、[受入結果](artificial_acceptance_results.md)、[未決事項](pending_and_required_decisions.md)。machine inventory、contract、manifest、authority、monitoring、acceptance matrix、SOURCE inventory、verification、publication receiptを同じpacketに保存する。
''',
    'scientific_api_static_inventory.md':'''# 静的API inventory

8 moduleの201 class/function/nested entryをASTだけで採取した。scientific_api_inventory.jsonにpath、SHA256、行、signature、docstring、return/raise構文、個別SOURCE segment SHAを保存。返却構文は動的schema推論ではない。意味論・型・副作用・失敗境界は同JSONのsemanticsとapi_binding_manifest_candidate.jsonに記録した。

| Arm | A | B |
|---|---|---|
|ordinary|make_generator → CanonicalGenerator.sample|既存LegacyEndpointのcanonical_snapshot/AuditParity/event_witness、event(group_index,raw_word,child)再照会、enclose_atan|
|partial_return_tail|同上・partial groups＋ordinary tail|同上|
|closed_P3_tail|同上・closed P3 groups＋ordinary tail|同上|
|full_return|FullReturnGenerator.sample|F3 prepare → GeneratorSession.sample/angle_for_event|

F3.prepareは正の入力のcanonical3方式をCANONICAL_PREFLIGHT_NOT_IMPLEMENTEDで拒否する。存在しないcanonical F3 facadeを仮定しない。sigmaはprovider/angle contextでありG7 constructorの引数ではない。X/Yは実行identity/seed domainで、G7の測定axis引数ではない。

G7はevent dictまたはFullのNoneを返す。Full Noneはraw nonreduced/coin rejectをtrace/packet境界で区別し、例外をzeroへ変換しない。word/raw_word/circuit_time_wordはtuple[int]、ratio/coefficient/proposal/weightはFraction、child/phaseはint。weight×proposal=coefficient。finite rational approximationと理想のirrational normalizerを区別する。

F3はEvidence(document,digest)のEVENT/ZERO/IDENTITYとcontext/traceを返す。F3Failureはcategory/reason/stage/input/detailsとSTOPを持ちsession terminal。固定正入力ではIDENTITY経路を実行しない。F3 hex $rat/$tuple codecとdriver exact codecは別documentで、bytes互換を仮定しない。

Fullはorder→raw labels→reduced判定→local packet→acceptance→accepted時child。packetはReturnKernel.digital_parent→parent→word_series/convolution/_mass、root/envelope/child law/weight。sampleは同packet再照会を避け、B angleはevent→packetを再照会してfingerprintを照合する。setupのfirst_passage/returnsを保持し、成長するpacket cacheはない。共有状態とbits消費、session失敗、副作用のsource file readを区別する。providerはIR記述でありnative実行は範囲外。
''',
    'real_api_binding_design.md':'''# 閉鎖port設計

新namespaceは src/trottertracks/algorithm_codesign/g6_w2c_b3_api_binding_v1/。static.pyがSOURCE path/name/signature inventory/入力/precision/監視を結ぶ。ports.pyが人工専用ArtificialEndpoint.sample(bits,trial_id)、angle_for(result)、shared_identity、ArtificialBitsを実装。人工結果は固定parallel driverのOutcomeへ渡し、固定run_pair/fused codec/CachedObserver/F01を再利用する。

実module loader/callable resolver/科学dispatchは実装していない。production_entry、real_bits、real_authorityは引数とbooleanに無関係に拒否。人工factory/bitsとfake storeの型・namespaceを限定する。人工event/angleはsurrogateであり、実law、F3 Evidence、角度enclosureと等価ではない。

exercise.pyは48pair・6phaseのport interface gridであり、Global Bundleを再実装しない。Sを逐次、P1を8 spawn、pairごとfresh workerで人工APIの形状を検証する。global admission/call/byte schedulerやStorage V2を書き直さない。固定Bundleはその既存run_case/run_matrix/run_testsを別試験として実行した。portのsidecar保存量はBundle予算へ混入させず、既存Bundle内へ新portを注入できたという結論を出さない。

offline.pyはports/exercise/static/authority/Producer/schedulerをimportせず、固定offline codec/F01/cost primitivesからidentity/consumption/prefix/claim snapshot/callerを別ロジックで照合する。後続のローカルCodex独立受入は別作業。
''',
    'science_input_and_precision_binding.md':'''# 固定入力・precision binding

P5_general_order、p=(1/5,3/10,1/2)、x=5/7、m=5、sigma=+1。root_bits256、probability_bits160、eta/rho各1/10^12、epsilon_axis1/200、primitive_error1/10^6、alpha_axis1/320、familywise alpha1/20を固定参照から保持。A6はtau1/10^6、K_phi32、J32のmath-only。値の再計算、再最適化、保存科学値の新規数値再解析はしない。

technical_contract_candidate.jsonに元reference全体、SOURCE/input pins、runtime executable/stdlib/extension hashesを保持。preflightは実SOURCEとAST digest、入力bytes、runtimeと完全candidate一致を確認する。source_inventory.jsonは固定runtime52 filesと入力10 filesの保全を記録。必須固定資料34件のGit object/SHAはfixed_reference_inventory.json。

G7のroot interval、largest-remainder dyadic law、downward acceptance、有限rational coefficientとproposal/weightの意味論を変えない。F3はcaller-supplied precisionのみで適応選択/再試行なし、P0〜P4と実lawのP5 binding、同event requeryとhalf-angle/alternating atanを保持する設計参照。今回はそれらを実行していない。canonical Bは既存overlayの別監査でありFull F3 preflightと同一certificateではない。
''',
    'seed_and_bitstream_binding.md':'''# Seed・bits・identity

Bundle、SOURCE release、科学input、scope/claim、run、mode、arm/layer/axis、phase/pass/trial、seed domainを分離。global trial IDにはrun/modeとlocal trial IDを含める。mode/layer/phase/passは科学seed domainへ追加しない。同arm/axisのS/P1、A/B、baseline/instrumented、correctness/profile対応prefixをそろえ、armとX/Yは分離する。

科学候補namespace PROPOSED_B3_LIMITED_G7_PILOT_20261019_V1 は文字列参照のみ。実bitsは生成しない。新人工namespaceはARTIFICIAL_API_BINDING_V1/arm/axisのみ。SHA256(seed UTF8 || counter big-endian uint64)、各bits(160)で256-bit blockの上位160を返し下位96を捨てる。consumed_bitsとcounter、width、value_hex、discardをtrace化。独立したoffline実装が人工bitsだけを再照合する。

人工canonicalは2 draws/attempt、人工FullはZERO_R1、ZERO_C2、EVENT3 draws。これはG7の可変degree/conditional/rejection lawを模倣した同値証明ではない。実G7/F3 consumption equivalence、fresh IID、科学path coverageはpending。失敗prefixは正常比較から除外。retry0/resume false、source/hash/domain/consume mismatch拒否を試験した。
''',
    'authority_and_one_shot_design.md':'''# Authority・one-shot候補

authority.pyはformal release(null)、SOURCE candidate、科学reference/input hash、runtime、contract、scope/claim、call/result/log budgets、seed、retry/resume、worker policy、storage destination、monitoring/STOP digestをbindingする。authority_binding_candidate.jsonは架空の将来fake rootを使う静的例であり認可ではない。

実行試験は新規private /tmp/b3-bundle-artificial-* だけで固定FakeStoreを生成する。BEGIN IMMEDIATE/unique scope+claim、SPENT commit後ACK、UNKNOWN ACKでも返却しない。32競合のwinner1、二重/STOP後/spent再利用、19 binding mutation、人工approval mismatchを拒否。実store/grant/旧scopeを探索・消費・変更しない。fake approvalはbindingの人工tokenで、署名済み実認可ではない。

実authority/bootstrap、trusted store所有・失効・鍵、全worker lineage、承認文書、正式grantの発行/消費・耐久性は後続の別SOURCEと明示認可が必要。保存snapshot/historyは記録された人工状態を照合するもので、未保存atomic履歴・power loss・実authorityの証明ではない。
''',
    'monitoring_and_cost_design.md':'''# Monitoring・CPU/wall設計

| 項目 | 現時点の取得境界 |
|---|---|
|parent＋全worker CPU、global outer wall、条件wall|固定Bundleに既存instrumentation。pair CPU/IPC/observer/stagesを再加算しない。pair wall和をglobal wallとしない|
|setup/online、generator＋audit、angle＋requery|固定pair coarse stage。fine function observerは別partition|
|Full local coefficient/child law/requery|静的にpacket/parent/word_series/_mass/child_law/event/enclose_atanの関数境界あり。実値は実API接続後のみ|
|Fraction bit演算費用、observer単独費用|現状識別不能。C算術はPython callerへ帰属しbit単位費用へ分解しない|
|serialization/hash/write/flush/close、IPC|固定pair/Bundle記録。親の全個別I/O exclusive費用はnull|
|own RSS/HWM/IO、output bytes|既存記録。RSS samples/HWMはunique continuous global peakではない|
|bit consumption|新人工adapter traceとoffline replay。実G7値はnull|
|inflight/STOP/drain/UNKNOWN credit|固定Bundle trusted trace/RAM/physical/journal join|

新port n256人工観測はparent0.121358936s、全worker38.674677s、total38.796035936s、outer wall21.738776636s。port exerciser自身の値で、固定Bundleや実科学費用の値ではない。worker preflight/constructorとprocess setupは全worker CPUへ含まれ、pairのtimed region外の分をpair stageへ推定配賦しない。observer込み、coarse/fineを足さない。setup不足・未知費用は0へ置き換えない。

S→P1固定順、共有host負荷、cache等の交絡がある。実G7 path率、accept/reject、有理数bit長、requery頻度、保存量、CPU分布、Full-N費用はpending。CPU母平均CI、fused単独改善率、量子confidenceからCPU CIは導かない。資源確認64CPU・空きRAM十分・CPU PSI0の観測下で合理的並列化。共有OS/cgroup/package/service/他processは変更0。
''',
    'storage_stop_and_final_acceptance.md':'''# Storage・STOP・最終受入

固定Global BundleのRAM intent、physical statefile、trusted ordered trace prefix、journal、summary、receipt、diagnostic、marker、deliveryとcaller observationを既存offline checkerで結合した。marker単独、receipt-before-close、close/caller失敗、ACK欠損、partial write、RAM/physical mismatchを成功にしない。保存されなかった証拠をsnapshotから補完しない。

単一global claim/call/byte budget、親arrival順、STOP後ENTER禁止、STOP前inflightの合法drain、START_RESERVEDとbody entryの区別、UNKNOWN非返却、保存先別STOP成否を変更しない。S失敗後P1開始禁止、P1失敗時S保存も固定43-case matrixで検証。障害の26件はINCOMPLETE_EVIDENCEのままで、testの期待分類一致と完全成功を混同しない。

新port側はpair stream close→report close→caller witnessを照合し、F01失敗のinflight sidecarは完了prefixへ数えない。これは固定BundleのStorage V2証明の代用品ではない。port interface gridはphase単位で停止し同phase workerが完了し得る。global permit/STOP後ENTERの保証は別の固定Bundle試験に限定し、新port入りglobal schedulerを検証したとは表示しない。

caller witness自身のclose/power failure、fsync durability、実trusted storeのpower-loss履歴は未確立。原STOP/marker/旧one-shotを解除・再使用しない。
''',
    'preflight_and_production_gate.md':'''# Preflight・閉鎖gate

SOURCE62 pins（固定52＋新runtime/harness/test10）、科学入力10 pins、AST inventory digest、runtime executable/library、candidate/schema/closed flags/seed/予算の完全一致を照合する。親と各人工workerでscience MetaPathFinderを設置し、g7_generator/g7_provider/return_aggregation/g6_f3_*のimport要求をload前に拒否。既loadも拒否。科学importを実行してから判定する試験は行わない。

production_entry、real_bits、real_authorityにdispatch処理は存在せず、無条件例外。run_ready/content_adopted/contract_grants_execution=false、science_release/scientific_adoption=null。boolean反転、fake authority/bits転用、grantなし/偽grant要求は解除手段にならない。将来は別正式SOURCE、input/runtime/authority/監視と保存範囲の採択が必要。

新規negative controlsはSOURCE/runtime/signature/入力/contract/schema、missing/wrong result、arm/layer/axis/event、consume/domain、retry/resume、親/spawn import block、F01 generator/audit/angle/requery、fake race/commit-ACK UNKNOWN/spent/approval等。offlineは別process import boundaryも試験した。Pythonコードの悪意ある直接改変に耐えるOS security sandboxを主張するものではない。
''',
    'source_and_contract_delta.md':'''# SOURCE・contract差分

直接基点d24cccb、独立受入6bffb5b5は参照のみ。追加namespace/API port/人工bits/fake authority/monitor/preflight/offline/harness/testと本packetだけを新規作成する。既存科学6 files、固定runtime52 files、旧Bundle/V2/checker/契約/raw/grant/ledger/marker/STOPは固定bytes。新subpackageに置いたため、旧Bundle candidateのg6_w2c_b3_*.py globを変えない。

新contractは未採択。元n256、16cells、correctness256＋baseline4096＋instrumented4096=8448/条件、S/P1計16896、S→P1、6phase、fused/CachedObserver、64GiB result/256MiB log、retry0/resume falseを静的参照。人工小規模n16とsurrogate用sidecarは別試験形状であり科学予算の変更ではない。

旧Global Bundleの外部endpoint注入口がない点を明示した。今回その改造/再実装/monkeypatchは行わない。別実科学SOURCEに必要な注入口とauthority/runtime/STOP/storage bindingは未採択事項。

開発時の疎checkout依存不足は固定資料を専用worktreeへ補完して解消。旧legacy_restoration.json生成差分は元bytesへ戻し、途中診断は新packetのdevelopmentに保全。初期SOURCE190d7d5fの比較でbool/intを同一視し得る点を発見し、canonical bytes比較と整数型guardを追加した別SOURCE78c57100へ固定し直した。初期SOURCE/人工証拠は開発履歴として保持し、最終SOURCEで全新規人工検証をやり直した。package/restore/docs/publicationのpostprocessing helpersは最終content commitへ記録し、人工測定SOURCEとは区別する。
''',
    'artificial_acceptance_results.md':f'''# 人工受入結果

| 対象 | 結果 |
|---|---|
|新規unit tests|73 PASS / 0 FAIL / 0 ERROR / 0 SKIP|
|新port正常grid|n16:1536 calls、n256:16896 calls、各48pair/6phase/16cells|
|新port失敗grid|S generator:224 completed、P1 angle:992 completed、いずれもtyped technical failure。失敗prefixを正常比較へ混入しない|
|新offline mutation|22拒否：INTEGRITY_MISMATCH18、INCOMPLETE_EVIDENCE4|
|変更していないBundle43cases|bundle complete2、condition-only2、technical failure13、incomplete26|
|Bundle旧tests|107 PASS、うち35 overlaysを含む|
|既存回帰|補正版623＋Storage V2 41＋既存独立V2 57＋Bundle107＝828の別versioned ID|
|旧原版label|Python3.12 原版622 PASS/1 FAIL、補正版623 PASS。原版とcompiled差分を保持|

Local独立845 PASSは固定参照6bffb5b5の証拠で今回新規実行した件数へ足さない。旧件数/新unit/ケース/mutationは重複加算しない。テストPASSは期待した拒否を含み、evidence不完全ケースが完全受入になったという意味ではない。

全人工raw/overlaysは分割archive {len(archive['parts'])} parts、{archive['archive_bytes']:,} bytes、{len(archive['members'])} members、SHA256 `{archive['archive_sha256']}`。fresh restore後の{len(restored['results'])}分類を再照合。rawの全member SHAはevidence/archive_manifest.json、復元手順はverify_restored.py。新offlineは同じ主担当の自己検証。検証した人工範囲で未修正の技術不整合なし。
''',
    'pending_and_required_decisions.md':'''# 未決事項・次の担当

ユーザー/GPT：V2 RAM/physical/trusted-prefix/UNKNOWN/STOP/post-close契約、単一global authority、科学SOURCE/release、input/precision、seed namespace/消費同値、n256/16896、順序・CPU estimand・coverage/精度・保存方針を正式採択するか判断する。A6 math-onlyと条件付きquery-modelを実物native保証へ拡張しない。

ユーザー/GPT：固定Bundleには外部endpoint注入口がない。今回の静的port＋pair-engine人工結合と固定Bundle回帰の証拠分離を評価し、必要なら後続の別versioned SOURCEで注入口/real bits/authority/監視を統合するscopeを明示する。現在のboolean反転や旧SOURCE改変で実科学へ進む手順はない。

後続ローカルCodex：固定公開HEAD/全bytes、8-module静的inventoryとsignature/mapping、source/input/runtime pins、artificial prefix/消費/identity、fake race/ACK/spent/STOP、73 tests＋22 mutations、既存828の区別、archive fresh restore、post-close/F01/UNKNOWN結合、closed import/gate、未接続の境界と未計測値を独立受入する。別指示で行い、今回の主担当自己検証を独立reviewと数えない。

実G7/F3 law/bit/angle実行同値、実path分布、accepted/reject、有理数bit長、angle requery、科学storage/CPU、Full-N31,710,848の実行可能性は未測定。実authorityのatomic/ACK/STOP/power durability、caller witness自身のcloseも未確立。実科学correctness/pilot/Full-N/native処理には別明示認可が必要。

今回Codexは公開・照合と保全確認後mandatory STOP。次処理へ自動移行しない。
''',
    'handoff.md':'''# 技術handoff・mandatory STOP

Q1：固定Bundle/独立受入/G7/F3をGit objectとhashで照合。SOURCE/input/runtime pins保存、旧bytes保全。

Q2：四方式・A/B・X/Yのmappingを静的に確定。canonical Bは既存LegacyEndpoint overlay、Full BはF3 facade。科学pathの実行確認は0。

Q3：versioned閉鎖portと人工実装を固定pair engineで検証。固定Bundleへ新endpointを渡す入口は存在せず、その部分は未接続。既存Bundleの改造/再実装なし。

Q4：modeをglobal IDに含めseed domainへ入れない。arm/axis分離、対応S/P1・A/B/pass/prefixと人工consumeを照合。科学namespace bits生成0。

Q5：候補SOURCE/input/runtime/contract/budget/seed/worker/storage/STOPのfake binding、race/one-shot/ACK UNKNOWN/spentを検証。正式authority/grantとのdispatch境界は閉鎖。

Q6：固定BundleのRAM/physical/trusted-prefix/UNKNOWN非返却/post-close joinを43cases/107testsで回帰。新portは別のpair/sidecar/caller証拠で、Storage V2代替証明にしない。

Q7：parent＋全worker/outer wallとcoarse/fine/storage/IO/RSSを設計参照。局所query関数境界は静的確認のみ。実費用はnull、bit演算/observer単独/unique peakは識別不能。

Q8：新73 PASS、正常2/技術失敗2、22mutation拒否。既存補正版828 PASS、旧原版622/1保持、独立845は固定参照。人工検証範囲に未修正不整合なし。endpoint注入口は将来SOURCEの採択事項。

Q9：ユーザー/GPTが正式科学・契約・release・authorityと後続scopeを判断。後続ローカルCodexが本packetを独立受入。現candidateは実行認可なし。

Q10：実G7/F3 import/sampling、科学namespace bits、新科学計算、保存科学値の新規数値再解析、科学SOURCE採択、実authorization/grant/claim発行・消費、native quantum build/compile/synthesisすべて0。サブエージェント0。詳細はverification.json。

公開branchと最終HEADはpublication_receipt.jsonおよび最終チャットの固定URLを参照。SOURCE候補とpostprocessing content/publication receiptのcommitを区別する。mandatory STOP。
'''}
    for name,body in docs.items():
        cpu=ports[1]['cpu']
        body=body.replace('parent0.121358936s、全worker38.674677s、total38.796035936s、outer wall21.738776636s',
            f"parent{cpu['parent_cpu_ns']/1e9:.9f}s、全worker{cpu['all_worker_cpu_ns']/1e9:.9f}s、total{cpu['total_cpu_ns']/1e9:.9f}s、outer wall{cpu['wall_ns']/1e9:.9f}s")
        if name=='artificial_acceptance_results.md':
            body+='\n追加postprocessing controls4件もPASS（approvalのみ不一致、grantなし、fake bits/authority実物のreal port転用拒否）。73 unit testsへ加算しない。\n'
        title,rest=body.split('\n',1)
        (HERE/name).write_text(title+'\n\n'+preamble+'\n'+rest.lstrip())
    print(json.dumps(dict(status=verification['status'],docs=len(docs),restored=len(restored['results']),preservation_unchanged=verification['all_protected_worktree_observations_unchanged'])))
if __name__=='__main__':main()
