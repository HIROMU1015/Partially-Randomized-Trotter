"""Build the factual handoff from final artificial machine reports only."""
from hashlib import sha256
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def load(name):return json.loads((HERE/name).read_text())
def save(name,value):(HERE/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def md(name,text):(HERE/name).write_text(text.strip()+'\n')
def main():
    matrix=load('acceptance_delivery/matrix.json');new=load('new_tests_delivery.json');legacy=load('legacy_corrected_final/results.json');mutations=load('mutations_delivery.json');extra=load('reservation_before_after_independent.json')
    assert matrix['counts']=={'TECHNICAL_FAILURE':40,'INCOMPLETE_EVIDENCE':66,'VERIFIED_ARTIFICIAL_COMPLETE':2}
    assert len(matrix['cases'])==108 and matrix['physical_state_exact_cases']==82 and all(not r['issues'] and r['classification_equal'] for r in matrix['cases'])
    assert new['successful'] and new['run']==new['passed']==57 and legacy['successful'] and legacy['total']==legacy['pass_count']==623
    ids=new['collected']+[x for suite in legacy['suites'] for x in suite['collected']];assert len(ids)==len(set(ids))==680
    assert len(mutations)==51 and len(extra['new_external_writer_faults'])==2
    save('artificial_108_case_verification.json',dict(matrix,independent_SOURCE_inventory='source_inventory.json',detailed_reports='acceptance_delivery/'))
    save('independent_fault_injection_results.json',dict(schema='b3_v2_independent_faults_v1',new_external_writer_fault_cases=extra['new_external_writer_faults'],external_fault_case_count=2,offline_consumer_mutation_count=51,offline_consumer_mutations=mutations,mutation_cases_are_included_in_57_new_tests=True,raw_archive_manifest='new_artificial_raw_manifest.json',producer_preexisting_108_cases_not_counted_as_new_tests=True,real_scientific_calls=0))
    md('README.md',r'''
# Storage V2 サーバー独立技術受入

**人工技術受入完了。証拠不完全66件を維持し、科学実行は閉鎖したままmandatory STOP。**

固定Producer公開HEADは `596781daeba253418ba35f4b7964312a54caf233`、人工測定SOURCEは `45bf46b4a833e71f138903fb511f2741e54043b7`。旧独立検証器 `49a0dcc9c8ec6dfc96eba5bb862cfdcb3d0fc866` を直接基点に、別セッション・専用branch/worktreeで実装した。同じサーバー上の検証であり、異なる物理環境での外部再現や第三者査読ではない。サブエージェントは使用していない。

|区分|独立結果|
|---|---:|
|公開人工raw|正常2／技術失敗40／証拠不完全66、合計108|
|物理stateとcounterのexact照合|82ケース|
|旧指定10suite・原文のまま|622 PASS／ラベル期待値不整合1 FAIL|
|その期待値だけを明示補正した別versioned harness|623 PASS|
|新規独立tests|57 PASS、うちnegative/mutation 51ケース|
|上記versioned回帰の重複しないtest ID|680 PASS、FAIL/ERROR/SKIP 0|
|別枠の新規外部writer故障|S/P1の2ケース、独立再読は両方INCOMPLETE_EVIDENCE|

Producer既済820件、公開raw108件、外部writer故障2件を680へ加算していない。元testの失敗を消したり原版623 PASSと表示したりしない。詳細は [回帰区分](artificial_regression_results.md) と [旧assertion最小再現](legacy_label_assertion_reproduction.json)。

[Q1–Q10／次担当](handoff.md)、[SOURCE/schema](source_and_schema_verification.md)、[予約会計](v2_reservation_accounting_acceptance.md)、[trace/STOP](v2_trace_and_stop_acceptance.md)、[物理証拠境界](v2_physical_evidence_boundaries.md)、[費用・予算](cost_and_budget_verification.md)、[不一致](producer_consumer_discrepancies.md)、[残る証拠・契約判断](pending_evidence.md)。

機械結果は [108件matrix](artificial_108_case_verification.json)、[各ケース詳細](acceptance_delivery/)、[新規fault/mutation](independent_fault_injection_results.json)、[修正前後比較](reservation_before_after_independent.json)、[最終tests](new_tests_delivery.json)、[検証総括](verification.json)、[保全](preservation_verification.json)、[公開照合](publication_receipt.json)。

独立consumerは `g6_w2c_b3_storage_independent_{identity,trace,adapter}_v2.py`。旧global core、pair/F01、CPU partitionを再利用し、V2固有部分をこのセッションで実装した。Producer/scheduler/Admission/Producer補助checkerを独立consumer processへimportしていない。Producer実行があるのは別processの人工writer故障と、指定済み人工回帰だけである。

## 再現

Python 3.12と標準ライブラリだけを用いる。fresh checkoutでは、必要なら同じrepositoryから指定のProducer公開commitと測定SOURCE commitを `git fetch --no-tags origin <完全SHA>` で取得する。既存出力への上書きや同じ人工storeの再実行は行わず、新しい作業領域を使用する。

```bash
python3 docs/research/track_b_g6_w2c_b3_v2_independent_acceptance_server_20261019/prepare_evidence.py
python3 docs/research/track_b_g6_w2c_b3_v2_independent_acceptance_server_20261019/run_acceptance.py reproduce
python3 docs/research/track_b_g6_w2c_b3_v2_independent_acceptance_server_20261019/run_new_tests.py reproduce
python3 docs/research/track_b_g6_w2c_b3_v2_independent_acceptance_server_20261019/run_legacy_regressions.py reproduce
```

新規faultの固定runtimeは `prepare_injection_runtime.py` で新設し、`run_independent_writer_fault.py S` と `... P1` をそれぞれ未存在の保存先へ実行する。`verify_additional_evidence.py` が別processで独立再読する。runtimeの119ファイルは測定SOURCEからそのまま取り出し、科学moduleはimportしない。人工Admissionが要求する `/tmp/b3-r1-artificial-*` 領域を守る。初回の名前不適合による拒否もpreflight logとして保持した。

新規51 negative raw＋2 writer rawは5分割partと [manifest](new_artificial_raw_manifest.json) に保存した。次のhelperは実行せず復元するだけで、273ファイルのbytes/SHA256往復一致を確認済み。

```bash
python3 docs/research/track_b_g6_w2c_b3_v2_independent_acceptance_server_20261019/restore_evidence.py --new-directory /tmp/b3-v2-independent-new-restore
```

Producer公開108 rawは元の固定公開archiveを参照する。`prepare_evidence.py` は公開part、archive SHA256、758 memberのbytes/SHA256を独立に検証して新設領域へ復元する。元archiveをこのpacketで再生成したものへ置換しない。

正式science SOURCE採択、production bridge、実G7/F3 sampling/import、新科学計算、科学保存値の数値再解析、実authorization/grant/claim、quantum native build/compile/synthesisはすべて0。pilot n=256・S/P1計16,896 callsは未採択候補、Full-N 31,710,848 callsも未実行。次の担当はユーザーとGPT。
''')
    md('source_and_schema_verification.md',r'''
# 固定SOURCE・schema・公開bytes

Producer公開HEAD 596781daeba253418ba35f4b7964312a54caf233 と旧独立branch HEAD 49a0dcc9c8ec6dfc96eba5bb862cfdcb3d0fc866 を `git ls-remote` で照合した。公開HEADと測定SOURCE 45bf46b4a833e71f138903fb511f2741e54043b7 は役割を区別し、inventoryの `at_measured_SOURCE=true` の全ファイルについて両commitのbytes一致も検証した。

最初の82公開ファイルは固定raw GitHub URLとGit blobを照合した。補足として修正前5ファイルと正式共通指示7eff8a33bef4ee1f29698ab10c6da46988dc8db5の1ファイルも照合した。根拠はpublic_source_and_restore_verification.jsonとsupplemental_public_bytes_verification.json。SOURCE、schema、runner、保存資料、分割part、各ケースcontract/manifestのつながりを参照できる。

4partから14,511,204 bytesの元archiveを独立復元した。SHA256は `ecb6ca11da78bf924ec08d82e93c35c41f1d3516fa6d9fe25cb9f7be91119936`。全758 memberは公開manifestのbytes/SHA256と一致。復元は専用未存在領域で行い、既存rawは変更していない。

V2 candidate schemaは `b3_pilot_storage_repair_candidate_v2`、storage protocolは `MONOTONE_ACCOUNTING_EXPLICIT_INCOMPLETE_EVIDENCE_V2`。result/summary/receipt/diagnosticはそれぞれ別schemaとして検査する。旧templateからschema/protocolと指定3pinの差だけを独立導出し、その他の候補条件は厳密一致を要求する。public rawの信頼済みcandidate bytesをmanifest SHAへbindしてからconsumerを起動する。V1独立adapterをV2のaliasへ書き換えていない。

新規writer runtimeではSOURCE pinsは同一、runtime_rootだけが新設領域となる。人工contract/Python identityの変更は明示的に別referenceへbindし、科学条件・proposal・SOURCE/input pinsは変えない。内側contractのscope/claim/runtime/schema/source rolesは旧独立identity coreで検証する。

保存されたPython executable/library metadataは現在の同一サーバーのbytesとも全一致した。ただし過去に実際にloadされたruntimeを独立証明したことにはしない。元のprivate approval/atomic admission履歴もsnapshotだけでは証明しない。
''')
    md('v2_reservation_accounting_acceptance.md',r'''
# SRC_TRACE_RESERVE_ACCOUNTING

修正前公開SOURCE 76ca3c4089b201d55da0fede08e1f77733ac38a5 と保存rawを保全して再読した。planned768、物理state START_RESERVED1、report reserved0。独立accounting_v3は `GLOBAL_STATE_COUNTER_BINDING` として拒否した。旧SOURCEのSHA256はV2側に保護された旧SOURCEとも一致する。

公開V2のS/P1 original_reserve_writerはともにreserved1、started0、CANCELLED_BEFORE_START1、complete=false。物理stateのcountsとreport、global offsetとpair cursorを独立照合した。RESERVEの完全なtraceが保存されていないため判定はINCOMPLETE_EVIDENCEであり、正常成功ではない。

追加で別processの外部writerを実装した。固定Producerを改変せず、最初のRESERVEで実file.writeがOSErrorを投げるその時点のRAMを観測する。S/P1ともcounter=cursor=active cursor=1、ticket/index/stateが同じslotを指し、START_RESERVED1、started0を確認。これは最終reportだけを補正したという観測ではない。保存rawはProducer補助checkerを呼ばず、別の独立consumerで再読し、最終reserved1/started0/cancelled1とINCOMPLETE_EVIDENCEを再確認した。

予約callをNOT_STARTEDへ戻さず、予約creditを返却せず、新規ENTERを許可しない。重複RESERVE、ticket/cursor/global identityの不整合、未証明COMPLETE、STOP後ENTER、旧不整合のreport再導入は新consumerが拒否する。COMPLETE trace保存失敗はcompleted0・UNKNOWNへ残り、未開始trialをSTARTEDへ補完しない。

state保存失敗では予約後の保存notificationがSTOPの後に並ぶ場合がある。既存RAM ticketの申告を参照する限定的なconsumer正規化を明示し、物理的な予約順序を独立証明したとは扱わずINCOMPLETE_EVIDENCEを維持する。正常stateの新規予約をSTOP後に許可する例外ではない。

結論は固定人工条件における修正確認であり、任意の科学実行やhardware障害への形式証明ではない。詳細はreservation_before_after_independent.jsonと新規raw archive。
''')
    md('v2_trace_and_stop_acceptance.md',r'''
# Ordered trace・STOP・完了証拠

独立readerでcanonical envelope/payload、seq/run/trial identity、SHA256 chain、prefixのbytes/rows/chain/SHA256、row境界を検査する。trusted prefixを超えるtailは物理bytesとして記録し、操作が確定した履歴には使わない。欠損reportでは物理的にchainが成立するprefixを読めても、ACK済みtrusted prefixが確立したとは表示しない。

旧global coreを使ってphase barrier、予約/entry/terminal、重複、STOP後ENTER禁止、合法なin-flight drainを検査する。保存失敗によりENTER traceだけが残っても、state保存と許可が完了したとは推定しない。state保存失敗に続くCANCELはこの区別を明示してINCOMPLETE_EVIDENCEへ残す。

親arrival順序とrepresentative failureはcontrol/report間で照合し、worker/pair identity・global offset・F01と物理prefixを検査する。P1 cooperative drainの8開始・7COMPLETE・1FAILを独立再確認した。cooperative STOPをworker全員の同時物理停止とは呼ばない。強制killや共有環境の変更は行わない。

STOPのtrace保存、journal保存、fake Admission保存は別々に扱う。Admission STOP保存失敗ケースはspent snapshotの状態を保持し、成功marker欠損やrecord_errorsを正常に隠さない。snapshotはatomic historyそのものではない。retry/resumeなし・同scopeのspent拒否は固定SOURCE静的確認と指定人工Admission回帰による。

正常成功はreport、全trial state/trace、pair/F01、summary、receipt、diagnostic、marker、Admission STOPと保存errorのない状態を結合して初めて認める。summary/receipt/prefixをhashで結合し、final resultとsnapshotの時刻境界を区別する。summaryのCPU snapshotはfinal report以前なので同じ値と仮定せず、既存partitionと外側host境界を検査する。

正常ケースではtraceのresult reserve/ACK末尾をsummary→receipt→diagnosticの実bytesに結合し、journal各行のlog credit/ACKとも順序照合する。diagnostic ACKがmarker用journal reserveより先でなければ拒否する。順序を崩してchainを作り直した独立mutationも拒否した。

markerはreceipt・diagnostic・Admission STOPの後、storage closeより前に発行される。後続close失敗ではmarkerが残り得るが、final reportは失敗となり、後続STOP/保存不完全も独立に検出する。markerの存在だけをfinal acceptanceへ格上げしない。この複数証拠を結合する意味論は将来の正式契約で明示する必要がある。
''')
    md('v2_physical_evidence_boundaries.md',r'''
# RAM claimと物理証拠

108ケース中82件では完全と申告された物理state bytesからcountsを再計算し、report・shadow SHA256・pair別cursor・global reservedと照合した。14件はstate保存不完全、12件はfinal report欠損であり、同じexact state/report証明には格上げしていない。

RAM stateはProducerが保持するmonotone intentである。保存失敗のstate_errors、statefile_complete、physical witness不足を明示し、保存されなかったRAMを独立に復元できたとは扱わない。現在の完全な最終stateも、それだけで全transition履歴を証明しない。state close失敗時に読めるbytesが一致しても、closeが成功したという証明にはしない。

66件のINCOMPLETE_EVIDENCEは意図的に維持した。少なくともtrace、state、journal、host report、closeのいずれかの証拠限界が残る。保存不能を正常完了や費用0へ補完しない。UNKNOWNの実entry/開始時刻を推定せず、不明なcall/byte creditを返却しない。

fake SQLiteはimmutable read-only URIでquick_check、single store/scope/claim、contract/source/input binding、spent/STOP flagsを検査する。実authority storeは開かない。保存snapshotとProducer観測だけからatomic transition順序、当時のprivate approval、歴史runtime load、power-failure耐久性を独立に証明することはできない。新規writer故障時のRAM観測も別の新規人工証拠として区別する。
''')
    md('cost_and_budget_verification.md',r'''
# 費用・予算の独立照合

final host reportが残る96件は既存独立cost partitionを使い、host parent＋全worker CPU、outer global wall、pair coarse/fine/setup/online/IPC、memory/RSS/HWM/IOの保存境界を検査した。12件のreport欠損ではtotal CPU/global wallをnullとして残し、0へ補完しない。失敗ケースの費用は正常比較へ混ぜない。

pair CPUをworker CPUへ再加算せず、coarseとfineを合算せず、IPCをouter totalへ加算せず、parent/worker wallをglobal wallへ足さない。CachedObserver fine partitionは代替の測定表現であり、observer単独のoverheadは未取得としてnullを維持する。sampled RSS/HWMやshared-page sumをunique memoryの厳密peakと呼ばない。

新RAM shadowはplanned slot当たり1 byteの追加RAM。物理state保持長とは別であり、logへ二度課金しない。正常人工stateは768 retained bytes、累積pwrite trafficは3072 bytesである。serialization/hash/write/flush/close、fault notification、diagnostic等の親処理はhost parent CPU/wallに含まれるが、各処理のexclusive costを独立測定済みとは主張しない。

log/resultのreserved、ACK observed、physical retained bytes、unknown creditを照合し、完全traceでは非trace credit＋trace自身のbytesを一度だけ課金してexact一致を要求する。不完全traceではprefixの下限と物理bytesを照合し、prefix後の会計を独立証明へ格上げしない。未ACK creditはreserved−ACKのまま保持する。diagnosticもresult creditに含め、無料にしない。

固定候補のresult 64GiB/log 256MiB、call/byte cap、retry0/resume falseは無断変更していない。人工fault用の小さいcapは公開fixtureの指定された設定としてそのまま検証した。実行資源は利用可能な64 CPU・十分な空きRAMを確認してraw再読と指定回帰を並列化した。共有OS/kernel/cgroup/quota/packageや他ユーザーのprocessは変更していない。
''')
    md('producer_consumer_discrepancies.md',r'''
# 検出事項と境界

最終独立consumerは公開108件の分類と一致し、今回の人工scopeで新しいProducer不整合は検出しなかった。旧SRC_TRACE_RESERVE_ACCOUNTINGは保全rawから独立再検出し、V2のS/P1 rawと新規writer観測で修正を確認した。これは科学実行のGOではない。

固定の旧10suiteを原文で実行したところ622 PASS/1 FAILとなり、過去の「623 PASS」をそのまま再現できなかった。失敗testは `CandidateAcceptance.test_cached_observer_call_count_label_exclusive_partition` の `endswith(':leaf')` で、固定observerはco_qualnameを使うため実labelが `...<locals>.leaf` となる。両observerの全計測行とpartitionは一致する。legacy_label_assertion_reproduction.jsonに実値を保存した。

旧test/consumer SOURCEを改変せず、新規versioned harness内でこの1個のsuffixだけを `.<locals>.leaf` へ明示補正した。その結果は623 PASS。原版のFAILと補正後PASSは別記録であり、原版623 PASSとは表示しない。正式な旧test修正を取り込むかは後続保守で判断できる。

開発中のconsumer誤りは今回のscopeで修正した。summary/final CPUの異なるsnapshot、exact Fraction表現、保存失敗後の予約notification、state closeの不確実性、異常STOP入力に対するIndexError、markerより前のdiagnostic ACK、canonical envelopeを扱う。初期matrix/testログは診断として保存し、最終結果へ加算していない。最初の旧回帰では限定checkoutの依存不足もあり、そのログを保持した。

Producer固定SOURCE、旧SOURCE/結果、旧V1 consumerは変更していない。新規consumerの作者自身の検査を、別reviewerによる独立レビューとは呼ばない。今回の「独立」はProducer修正セッションと異なるconsumer実装・検証経路を指す。
''')
    md('pending_evidence.md',r'''
# 残る証拠と次の契約判断

- 66件の保存不完全。失われたstate/trace/journal transaction、RAM shadowの物理証明、report欠損時のCPU/wall、未知entry時刻は未確認のまま残る。
- 歴史runtime load、元private approval、fake Admissionの完全なatomic transition履歴、power-failure durabilityは今回のsnapshotだけでは独立証明できない。
- 旧47件/18件等の歴史pendingは再分類していない。Producer側820既済testを今回の独立件数に加えていない。
- 原文623回帰中1件の期待値不整合と、今回versioned harnessの明示補正をどう旧testへ反映するかは後続保守事項。旧SOURCEは保全済み。
- V2のRAM/physical区別、trusted prefix、UNKNOWN非返却、diagnostic/receipt/markerの結合、close後の失敗検出を正式技術契約へ採るかはユーザー/GPTの判断が必要。
- Production bundle bridgeは未実装。S/P1を二条件global identity/one-shot/call budgetで結合する正式SOURCE・contract、CPU性能のestimand、承認/監視/STOPは別段階。
- n=256、S/P1計16,896 callsは未採択第一候補。実G7/F3、新科学計算、Full-N、science SOURCE採択、実authorization/grant/claimへの許可は本packetにない。

次の担当：**ユーザーとGPT**。本packetとQ1–Q10を確認し、V2技術契約差分・旧test保守・production bridge準備へ進む範囲を判断する。科学実行を選ぶ場合は、その範囲と固定SOURCE/入力/契約を別途明示承認する。**Codexは公開照合後mandatory STOP**し、自動で次段階へ移らない。
''')
    md('artificial_regression_results.md',r'''
# 人工検証の件数

|集合|PASS|FAIL/ERROR/SKIP|扱い|
|---|---:|---|---|
|固定旧runner・依存を揃えた原文実行|622|1/0/0|legacy_regressions_r60ws1q6。元FAIL保持|
|新versioned harnessによる旧10suite|623|0/0/0|co_qualname suffix期待値1箇所だけ明示補正|
|新規独立consumer tests|57|0/0/0|new_tests_delivery.json。51 mutation/negativeを含む|
|上記versioned回帰の重複しないID|680|0/0/0|623＋57。原文実行と二重加算しない|
|公開rawの独立検証|108|正常2／技術失敗40／証拠不完全66|unit test件数へ加えない|
|新規外部writer fault|2|S/P1とも期待した非成功状態|unit test件数へ加えない|
|Producer側既済unit|820（申告）|今回の件数へ不加算|Producer側資料の固定参照|

51 mutationにはSOURCE/runtime/contract/schema、scope/claim、counter/cursor/ticket、global/trial identity、RESERVE/COMPLETE重複、STOP後ENTER、新規予約、chain/canonical/prefix/tail、ACK/refund/byte cap/trace自己課金/diagnostic、summary/receipt/marker順序、CPU/wall、保存失敗の隠蔽等を含む。単に同じProducer関数を再実行して独立と呼んでいない。保存済み親/worker crash、in-flight drain、UNKNOWN、F01は108件の独立core経路でも確認した。

consumer processのProducer/scheduler/Admission/補助checker/science importは0。指定人工回帰・外部writer故障のProducer processでも禁止science moduleを検査し0を確認。raw元入力はread-only、SQLiteはimmutable read-only。mutationは新規copyだけで行った。

初期の依存不足・consumer修正前の診断は別名のログ/JSONへ保全し、最終SOURCEでPASSした件数と混ぜない。新rawの273ファイルは5partへ保存し、復元bytes/SHA256の往復一致を確認済み。
''')
    md('handoff.md',r'''
# Q1–Q10：独立技術受入と次担当

**結論：固定Storage V2の人工技術受入を完了した。正常2件、技術失敗40件、証拠不完全66件を独立に確認し、科学実行は閉鎖したままmandatory STOPする。** 同じサーバー上の別セッション・別worktreeによる技術検証であり、異なる物理環境での外部再現や第三者査読ではない。

**Q1. 固定SOURCE等は照合できたか。** できた。公開596781daeba253418ba35f4b7964312a54caf233／測定45bf46b4a833e71f138903fb511f2741e54043b7を区別してSOURCE/schema/inventoryを検証した。最初の82公開ファイルと補足6ファイルはGit blobと一致。108ケースの4part archiveと758 memberのbytes/SHA256も一致した。

**Q2. V2独立adapterと独立性。** 新しいidentity/trace/adapterとacceptance harnessを実装した。旧global core・pair/F01・CPU partitionを使い、V2の意味論を別実装した。独立consumerはProducer/scheduler/Admission/Producer補助checkerをimportせず、Producerの保存済み判定は結果比較だけに用いた。旧V1検証器は変更していない。

**Q3. 108件の結果。** 正常 `VERIFIED_ARTIFICIAL_COMPLETE` 2、`TECHNICAL_FAILURE` 40、`INCOMPLETE_EVIDENCE` 66。全件で申告分類と一致した。66件を正常PASSへ変えていない。

**Q4. 予約会計修正。** 固定旧rawは予約state1/report reserved0を `GLOBAL_STATE_COUNTER_BINDING` として独立拒否。V2のS/P1保存rawはreserved1/started0/cancelled1。さらに新規外部writer故障S/P1で実write例外時点のcounter/cursor/ticket/stateの一致を観測し、保存rawも別consumerで再確認した。両方ともtrace不完全なのでINCOMPLETE_EVIDENCE。固定人工条件で修正を確認した。

**Q5. RAM/physical/prefix/UNKNOWN/STOPの証明範囲。** 82件は物理stateからcounts、shadow hash、cursor等を照合。14 state保存不完全＋12 report欠損は同じ証明へ格上げしない。trusted prefix後のtail、未保存RAM、未知entry時刻、snapshotだけからのatomic historyは未確認のまま残す。P1 drainの8開始/7COMPLETE/1FAILを確認。STOP後ENTER、重複、UNKNOWN成功化を拒否する。

**Q6. Summary/receipt/diagnostic/marker。** schema/identity/hash/prefix、snapshot境界、byte credit/ACKとmarker順序を結合して検査した。diagnosticはpre-closeの非成功記録で、receiptの代用にはしない。markerはstorage close前に出るため、後続close失敗では残り得るが最終受入は拒否する。単一ファイルの存在を成功の証明にしない。

**Q7. 費用と予算。** 保存reportの96件は独立cost boundary/partitionを検証、12件はCPU/wall未知をnullで維持。parent＋worker CPUとouter wall、coarse/fine代替、ACK/physical/reserved、trace自己課金、diagnostic、RAM shadowとstate保持/累積pwriteの区別を確認した。個々の親保存処理のexclusive cost、observer単独overhead、unique memory peakは未取得として残す。候補予算は無断変更していない。

**Q8. Testsと新不具合。** 原文の旧623件は622 PASS/ラベル期待値不整合1 FAIL。元SOURCEを保全し、その期待値だけを明示補正した別versioned harnessでは623 PASS。新独立57件もPASS、重複しないversioned回帰は680 PASS/FAIL・ERROR・SKIP各0。51 negativeは57件の内数。公開108 rawと新規writer故障2件、Producer既済820件を加算していない。新しいProducer不整合は今回のscopeで検出しなかった。consumer実装上の不具合は修正・再検証し、初期記録も保持した。

**Q9. Production bridge前の残件。** 66件と歴史pendingの証拠限界を維持する。V2のRAM/physical/prefix/UNKNOWN/marker/close意味論の正式契約採択、旧test期待値の保守、二条件global bundle/one-shot/call budgetを結合するproduction bridge、CPU比較estimand、正式SOURCE/承認/監視条件の判断が必要。n=256・二条件16,896 callsは未採択候補のまま。

**Q10. 禁止科学実行等は0か。** 実G7/F3 sampling/import、新科学計算、保存科学値数値再解析、science SOURCE採択、実authorization/grant/claim発行・消費、quantum native build/compile/synthesisはすべて0。旧grant/claim再利用、既存STOP解除も0。新fake storeだけを人工検証に使用した。

公開内容・remote HEAD・公開bytes/SHA256の記録はpublication_receipt.json。保全監査はpreservation_verification.json。元dirty/untrackedの内容全件hash snapshotを取得したとは主張せず、親statusの前後一致、固定の使用SOURCEと旧consumerのbytes一致、および書込scopeを区別して記録した。

**次の作業（未実行）**：**ユーザーとGPT**が本packetを確認し、V2の正式技術契約差分・旧test保守・production bridge準備の範囲を判断する。科学実行には別の明示承認が必要。**Codex**はcommit/pushと公開照合後mandatory STOPし、次段階へ自動移行しない。
''')
    source_paths=[ROOT/'src/trottertracks/algorithm_codesign'/('g6_w2c_b3_storage_independent_'+name+'_v2.py') for name in ('identity','trace','adapter')]+[ROOT/'tests/tracks/algorithm_codesign/test_g6_w2c_b3_storage_independent_v2.py']+sorted(HERE.glob('*.py'))
    rows=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha256(p.read_bytes()).hexdigest()) for p in source_paths]
    save('source_inventory.json',dict(schema='b3_v2_independent_SOURCE_inventory_v1',base_commit='49a0dcc9c8ec6dfc96eba5bb862cfdcb3d0fc866',producer_public_HEAD=matrix['producer_public_HEAD'],producer_measured_SOURCE=matrix['measured_SOURCE'],old_producer='76ca3c4089b201d55da0fede08e1f77733ac38a5',common_instructions_commit='7eff8a33bef4ee1f29698ab10c6da46988dc8db5',files=rows,old_independent_checker_modified=False,versioned_harness_runtime_assertion_correction='one co_qualname suffix assertion, recorded in legacy_corrected_final/results.json',consumer_forbidden_imports=[],same_server=True,subagents_used=False))
    save('verification.json',dict(schema='b3_v2_final_independent_verification_v1',status='ARTIFICIAL_TECHNICAL_ACCEPTANCE_COMPLETE_WITH_EXPLICIT_EVIDENCE_LIMITS',public_case_counts=matrix['counts'],physical_state_exact_cases=82,old_original_regressions=dict(pass_count=622,fail=1,error=0,skip=0,issue='co_qualname suffix assertion'),versioned_legacy_regressions=dict(pass_count=623,fail=0,error=0,skip=0,explicit_assertion_corrections=1),new_independent_regressions=dict(pass_count=57,fail=0,error=0,skip=0),unique_versioned_test_ids=680,offline_mutations=51,offline_mutations_are_in_new_test_count=True,new_external_writer_faults=2,producer_preexisting_test_count_not_added=820,source_inventory='source_inventory.json',new_Producer_defect_detected=False,independent_consumer_forbidden_imports=[],real_G7_F3_sampling=0,real_G7_F3_imports=0,new_scientific_calculations=0,saved_scientific_numerical_reanalysis=0,science_SOURCE_adoption=0,real_authorizations=0,real_grants=0,real_claims_issued_or_consumed=0,quantum_native_build_compile_synthesis=0,old_STOP_released=0,production_bridge='NOT_IMPLEMENTED',pilot_profile_n256_two_condition_calls16896='UNADOPTED_CANDIDATE',full_N_calls31710848='NOT_EXECUTED',scientific_GO=False,same_physical_server=True,external_environment_reproduction=False,third_party_peer_review=False,subagents_used=False,mandatory_STOP=True,next_owner=['USER','GPT']))
    print('Built Q1-Q10 and Evidence Packet; unique versioned test IDs:',len(ids))
if __name__=='__main__':main()
