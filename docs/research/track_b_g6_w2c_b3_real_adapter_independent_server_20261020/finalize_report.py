"""Final report after the user's explicit driver/helper scope amendment."""
from collections import Counter
import json
from pathlib import Path
import independent as I

def read(n):return json.loads((I.HERE/n).read_text())
def save(n,x):(I.HERE/n).write_text(json.dumps(x,indent=2)+'\n')
def md(n,x):(I.HERE/n).write_text(x.strip()+'\n')

def main():
    probes=read('independent_probe_verification.json');assert (probes['tests'],probes['PASS'],probes['FAIL'],probes['not_executed'])==(214,214,0,0)
    for folder,count in [('final_unit',102),('final_integration',30),('final_regression_retry2',107)]:
        x=read(folder+'/results.json');assert x['run']==x['PASS']==count and x['FAIL']==x['ERROR']==x['SKIP']==0
        assert x['core_scientific_modules_loaded']==x['core_import_attempts']==[]
    fresh=read('fresh_independent_verification.json');assert [x['status'] for x in fresh]==['VERIFIED_ARTIFICIAL_BUNDLE_COMPLETE','TECHNICAL_FAILURE']
    core=read('core_import_observations.json');assert core['actual_core_imports']==core['actual_core_import_attempts']==0
    gate=read('parent_worker_gate_verification.json');assert gate['denied_calls']==30
    current=read('verification.json')
    current.update(status='INDEPENDENT_TECHNICAL_ACCEPTANCE_COMPLETE_ARTIFICIAL_ONLY',acceptance_complete=True,
        independent_new_checks=dict(codec=49,result_context_angle_audit=148,bits=16,parent_gate=1,fake_authority=10,total=224,PASS=224,FAIL=0),
        parent_spawn_gate_denied_calls=30,parent_spawn_gate_counted_separately_from_tests=True,
        guarded_process_observations=core['process_observations'],blocked_independent_dynamic_checks=0,
        Producer_132_tests_independently_rerun=dict(unit_PASS=102,integration_PASS=30,FAIL=0,ERROR=0,SKIP=0,source='b7ad344401b10395c49e4cfa806909282859bab5'),
        old_107_tests_independently_rerun=dict(PASS=107,FAIL=0,ERROR=0,SKIP=0,source='b7ad344401b10395c49e4cfa806909282859bab5'),
        extra_fresh_artificial_cases=dict(total=2,normal=1,typed_terminal_failure=1),
        scope_amendment=dict(explicit_user_authorization=True,allowed_artificial_modules=['g6_w2c_b3_driver','g6_w2c_cost_preparation'],
                             remaining_static_only_modules=sorted(I.SCIENCE),scientific_authorization=False),
        unresolved=[],next_actors=['USER_AND_GPT'])
    current['prohibited_operations'].pop('all_user_static_only_modules_loaded',None)
    current['prohibited_operations']['current_static_only_core_modules_loaded']=0
    current['prohibited_operations']['current_core_import_attempts']=0
    save('verification.json',current)
    for group,name in [('codec','independent_codec_verification.json'),('result','independent_result_verification.json')]:
        rows=[r for r in probes['results'] if r['group']==group]
        save(name,dict(PASS=len(rows),FAIL=0,source='b7ad344401b10395c49e4cfa806909282859bab5',
                       subject_process='candidate_probe.py',independent_oracle='independent.py',shared_candidate_validator_in_oracle=False,checks=rows))
    bit=read('independent_bitstream_verification.json')
    bit.update(dynamic_candidate_boundaries='PASS',boundary_widths_pending=[],boundary_widths_verified=[1,160,256,257,512,513],
        independent_bits_checks=16,failure_position_and_terminal_stop=True,namespace_denied_before_generation=True)
    save('independent_bitstream_verification.json',bit)
    dispatch=read('independent_dispatch_verification.json');dispatch.update(new_producer_spawn_execution=2,fresh_case_results=fresh,
        parent_spawn_gate_verification='parent_worker_gate_verification.json',core_import_observations='core_import_observations.json')
    save('independent_dispatch_verification.json',dispatch)
    dev=read('development_record.json');dev.update(scope_amendment_resolved_original_hold=True,
        final_subject=dict(PASS=214,FAIL=0,not_executed=0),
        regression_initial_startup='Missing fixed execution_contract_candidate.json in temporary runtime; fixed blob acquired, no Producer edits',
        regression_retry1=dict(PASS=102,ERROR=5,reason='Three missing fixed fixture/schema paths; original result retained'),
        regression_retry2=dict(PASS=107,FAIL=0,ERROR=0,SKIP=0,source='b7ad344401b10395c49e4cfa806909282859bab5'))
    save('development_record.json',dev)
    md('README.md','''
# Track B W2c-B3：サーバー内独立技術受入

**人工技術受入完了。実科学未採択、production gate閉鎖。Mandatory STOP。**

検証対象HEAD `80b7bf61fe0a735d4031bb719c4b3e788611c3ba`、測定SOURCE `b7ad344401b10395c49e4cfa806909282859bab5`、artifact `024d8812e97c5d541391533680fdbb3aee4cf6d8e51f020d3302a00767af42cb`。
独立基点 `5c1e13fa74b3c5e4f3166babed785274844655d0`。branch `track-b-g6-w2c-b3-real-adapter-independent-server-20261020-r2`。固定Producerのmerge/rebase/修正はしていない。

- 新独立check **224 PASS**：Codec49、Result/Context/Angle/Audit148、Bits16、parent gate1、fake authority10。
- 親/spawn workerのproduction拒否 **30 calls**は別集計。guard記録57 process、禁止core科学moduleのload/import試行0。
- 保存人工22 case：正常2、技術失敗13、証拠不完全7。semantic mutation20件は独立に全件拒否。
- n256全16,896 trial完了/join一致。full1,536・hash-only15,360。未保存原document直接取得は主張しない。
- 最終SOURCEでProducer unit102/integration30と旧差分回帰107を再実行し、全PASS。新scopeの人工Bundle2件（正常n16/Angle失敗）も独立検証。
- SOURCE89/input10、科学8 module/201 entryの静的照合、archive994 member、content82 file、旧291 SOURCE/test bytes一致・保全。

[Q1〜Q10](handoff.md)、[機械判定](verification.json)、[公開照合](publication_receipt.json)、[次の担当](pending_and_required_decisions.md)。
旧独立1,038・Producer132・旧107・今回224・20 mutation・22保存case・追加2人工caseは合算しない。

当初の§3Eはdriver/helperも静的限定だったため、依存をload前に遮断し保留した。
その後の利用者の明示修正で、この2 moduleの人工用途だけを許可して残試験を完了した。
[範囲修正と元の依存経路](import_scope_finding.md)・`*_prior_scope.json`に旧判断を保持する。

## 独立性・再現

別Codexセッション・新worktree・独自parser/canonicalization/hash/Result oracleを使用。AI sub-agent0、物理host独立性は主張しない。
`independent.py`は候補Codec/Result/Fixture/Offlineをimportしない。既存独立Storage V2 consumerは保存・STOP・会計だけへ再利用。
`candidate_probe.py`は別processのsubjectであり、期待値生成には使わない。

`audit_source.py`→`run_audit.py`が固定Git archiveと22case/20mutationを検証。
`run_probes.py`が214独立probe、`authority_probe.py`が新FAKE_ONLY scope、`run_fixed_tests.py`がProducer132/旧107を別集計。
`fresh_bundle.py`が親/worker preload guardと新規人工2scope、`static_audit.py`が固定差分・保全を確認。
`finalize_report.py`は利用者修正後の最終判定を生成する。`build_report.py`は元scopeでの保留記録のgeneratorとして残す。
原rawの再取得には固定Producer archiveを使い、新rawは[fresh archive/manifest](fresh_evidence_manifest.json)に保存した。
一時runtime `/tmp/b3-independent-server-r2-20261020/runtime` は固定blobから構成し、再実行では新scope/outputを使う。
''')
    md('seed_and_bits_acceptance.md','''
# 人工seed / bits / consumption

独自bit-string SHA256 oracleで公開16,896 trialをreplayし、UTF8 seed＋uint64 big-endian counter、high160/discard96、累積consumption、人工Canonical2draw/Full3-5drawが一致。
Modeをglobal IDに含め、seedから除外。S/P1・A/B・phase/pass prefixとarm/X-Yの8domain分離も一致した。

利用者のdriver/helper人工実行許可後、候補Bitsにwidth1/160/256/257/512/513を渡す独立試験を実行した。
block連結、高位bits、discard、counter/consumption、invalid widthのfailure position/terminal STOP、namespace改変の消費前拒否の16 check PASS。
科学namespaceは文字列参照のみで、生成はARTIFICIAL_REAL_ADAPTER_V1だけ。
実G7のdegree依存draw、conditional law、reject率、fresh IID、actual科学consumption同値は未確認。
[結果](independent_bitstream_verification.json)、[214 probe](independent_probe_verification.json)。
''')
    md('global_bundle_dispatch_acceptance.md','''
# Global Bundle dispatch

独立oracleが原F3 document/sidecar/Result/Seal/Angle/query/bitsを再構成し、factory/runtime/candidate/manifest/PID/task/global ID/offsetをRESERVE/ENTER/COMPLETE、state/journal、receipt/delivery/callerへ結合した。
n256はplanned/reserved/started/completed16,896、unknown/cancelled0、全16,896 join。EVENT14,728/ZERO_R1,020/ZERO_C1,148は人工fixture件数。
full1,536とhash-only15,360を区別した。hash-onlyは独立人工再構成と保存hashの一致であり、未保存元documentを取得した証拠ではない。

追加許可後、新しい人工scopeで固定Producerの正常n16 Bundle（1,536 calls完了）とAngle失敗を実行した。
両方を別processの独立checkerで判定し、正常/TECHNICAL_FAILUREに一致した。旧scope/claim/rawを再消費・上書きしていない。
親とspawn worker双方でcore6科学guardをsubject import前に設置し、production入口/real factory/real bitsを30calls拒否、57process記録でcore科学load/import試行0。
過去processの外部attestationや実科学互換性は主張しない。
[dispatch結果](independent_dispatch_verification.json)、[新raw manifest](fresh_evidence_manifest.json)、[guard観測](core_import_observations.json)。
''')
    md('storage_stop_and_authority_acceptance.md','''
# Storage / STOP / authority

旧独立Storage V2 consumerでRAM intent/physical state/trusted trace prefix/journal/単一call-result-log budgetを検査。
UNKNOWN非返却、STOP後ENTER禁止、inflight drain、S失敗後P1禁止、P1失敗時S保持、receipt/marker/close/delivery/callerを確認した。
保存22caseの正常2/技術失敗13/証拠不完全7を維持し、missing evidenceを補完しなかった。
追加のfresh Angle失敗もterminal STOP/F01、S失敗後P1禁止を独立確認した。

新FAKE_ONLY scope2件でatomic race32中1winner、double claim/SPENT/ACK unknown/binding mismatch/STOP後再利用の10 check PASS。
原DBをauthority_evidenceへ保存。実authority/store/grantには触れていない。
親とspawn workerでclosed entry/real factory/bitsにboolean/fake grant等を渡す30callを拒否し、core科学moduleのloadとimport試行0。
closed flags false/null/true指定を維持。real store所有・鍵/失効・実atomicity/ACK運用・power durabilityは未確立。
[authority結果](independent_authority_verification.json)、[gate](parent_worker_gate_verification.json)、[保全](pins_and_preservation_verification.json)。
''')
    md('regression_and_source_delta.md','''
# 最終SOURCEと旧回帰

Producerの旧107PASSはSOURCE `8d5135ee77e6cd06d8b2ed8aa0b9ee50a41a0e36`。
最終 `b7ad344401b10395c49e4cfa806909282859bab5`との差分は7path/144insertion/4deletionで、Canonical Seal/angle contextとtestsが中心。旧4test file/291旧SOURCE-test bytesは不変。

今回、利用者のdriver/helper人工許可後、**最終SOURCEで旧107件を全再実行：107 PASS、FAIL/ERROR/SKIP0**。
Producer新unit102/integration30も最終SOURCEで再実行し全PASS。これらを新独立224checkへ加算しない。
Canonical補完はCodec/Result/Seal/Angle独立197probe、全保存Layer B raw、fresh人工Bundleで重点検証した。
Bits/seed/STOP/Storageも今回のprobe・107回帰・22保存case・fresh失敗caseで確認し、重要契約を未実行のまま省略していない。

一時runtimeのfixture不足により初回startup失敗、次に102PASS/5ERRORを記録した。
不足3fixture/schemaと原contractを固定Git blobから補い、Producer/source/expectedを変更せず107PASSへ到達した。
中間失敗はdevelopment_recordとfinal_regression_retry1へ保持し、別の追加成功件数として合算しない。
[差分](final_source_delta.patch)、[最終結果](final_regression_retry2/results.json)、[集計](regression_summary.json)。
''')
    md('producer_consumer_discrepancies.md','''
# 不一致と解決履歴

API測定SHAは指示文と原inventoryが不一致。原資料の`78c571005fe766c61b32dc867e006531099ff608`を採用しexact fetchした。
当初、利用者§3Eは8module静的限定、Producer guardはcore6module対象で、人工runtimeはcost preparation helperへ依存していた。
独立guardはhelperをload前に遮断し保留。その後ユーザーがdriver/helperの人工用途のみを明示許可し、core6禁止を維持して残試験を完了した。
旧判断を*_prior_scope.jsonとimport_scope_finding.mdへ保持する。固定Producerや既存STOPの変更ではない。

回帰一時runtimeの不足fixtureは原Git blobから補い、最終107PASS。途中の5ERRORを保持した。
保存人工22case/20mutationのProducer結論は独立oracleでも一致。未解決の人工技術不整合は0。
物理host独立性、実科学law/live型/atan保証/科学費用、real authorityは未確立。
''')
    md('pending_and_required_decisions.md','''
# 次の担当と必要な判断

**ユーザー・GPT**：本packetの独立人工技術受入を踏まえ、科学SOURCE/input/precisionの正式採択、新real接続SOURCE、live互換性/実correctnessのscope・coverage・予算・STOP・新authorizationを判断する。
実trusted store所有・鍵/失効・atomicity/ACK不明運用・power durability/runtime containmentの外部条件も残る。
GPT科学レビュー開始、科学SOURCE採択、実科学計算の実行承認は別事項。

**Codex（今回）**：公開bytes/remote照合後mandatory STOP。
**Codex（新たな明示指示・必要な承認後）**：採択された次scopeだけを新versioned SOURCE/入力/新grantで準備・検証する。現在は未開始。
実G7/F3 core import/sampling・科学bits・科学再解析・native・実authority・correctness・limited pilot・Full-N・production解除へ自動進行しない。
driver/helper人工用途の許可は、そのどれも承認しない。既存STOP/claimは保持する。
''')
    md('handoff.md','''
# 最終報告 Q1〜Q10

**独立人工技術受入完了。実科学未採択、production gate閉鎖。Mandatory STOP。**

1. **Q1 固定照合**：公開HEAD `80b7bf61fe0a735d4031bb719c4b3e788611c3ba`、測定SOURCE `b7ad344401b10395c49e4cfa806909282859bab5`、content `9bbf87f47688340c1d52899b0a5c614c2d8932ef`、artifactを分離。SOURCE89/input10、科学8module/201entry、content82file、archive994memberのbytes/SHA一致。API測定SHAは原inventoryの`78c571005fe766c61b32dc867e006531099ff608`へ訂正記録した。
2. **Q2 独立性**：別session/worktree・独自exact parser/canonicalization/hash/Result/Angle/bits oracle。候補Codec/Result validatorをoracleへimportせず、Storage V2の既存独立consumerだけ再利用。subject別process、物理host独立性は主張しない。
3. **Q3 Codec**：49独立check PASS。signed/large integer、exact Fraction、tuple/list、nested/heterogeneous mapの受理subsetでsidecar付きround-trip成立。driver単独はtuple/key情報を失う。非canonical/collision/unsupported型を拒否。実live Evidence互換性は未確認。
4. **Q4 Result等**：148独立check PASS。EVENT/ZERO_R/ZERO_C/IDENTITY/EVENT_QUERY/typed failure、Canonical/Full context、Angle/query/Sealを区別。Noneから理由を推測せず、原failureをF01 terminal STOPへ保持。人工Fraction区間整合を実atan数学保証へ格上げしない。
5. **Q5 Bits**：16独立check PASS。width1/160/256/257/512/513、SHA256-counter/high bits/discard/consumption/failure-position、科学namespaceの消費前拒否を確認。公開16,896trialの8domainとS/P1/A/B/phase/pass prefix一致、Modeはglobal IDへ入りseedから除外。実variable consumption/IID/lawの同値は未確認。
6. **Q6 dispatch**：保存rawの原document/sidecar/Adapter→worker PID/global offset→RESERVE/ENTER/COMPLETE/state→receipt/delivery/callerを独立結合。n256全16,896完了、full1,536/hash-only15,360。追加新scopeの正常n16/Angle失敗も実spawn経路から独立確認。hash-onlyの未保存元document取得は主張しない。
7. **Q7 mutation**：保存20件は独立Storage checkerを通り、新独立Adapter oracleで20件すべて拒否。変更fieldを元rawとの差分から分類。追加negativeは独立probeへ別集計。core科学の新不具合は未確定。当初のhelper scope衝突は利用者の明示修正で解決し履歴を保持した。
8. **Q8 counts/regression**：新独立224check全PASS（49+148+16+gate1+fake10）。親/spawn gate30拒否callは別集計、57process観測でcore科学load/import試行0。保存22caseは正常2/技術失敗13/証拠不完全7。Producer102unit+30integrationと旧107回帰を最終SOURCEで再実行し全PASS。旧1,038等を加算しない。最終Canonical補完も独立probe/raw/freshBundleで確認。
9. **Q9 残条件**：正式科学SOURCE/input/precision採択、新real接続SOURCE、live互換性/実correctnessのscope・coverage・予算/STOP・別実行承認、実authorityの外部条件が残る。driver/helperの人工許可は科学承認ではない。実科学/limited pilot/Full-N/production解除へ自動移行しない。
10. **Q10 禁止操作0**：core実G7/F3のimport/load/sampling、科学bits、新科学計算/保存科学値再解析、科学採択、実authorization/grant/claim、native、旧claim再利用、STOP解除、real correctness/pilot/Full-Nはすべて0。許可後のdriver/helper人工import/関数実行だけを実施。修正前helper1attemptはload前拒否、core6への試行0。旧994member/291pathを保全、他worktree/旧raw/grant/ledger/marker/STOPへの書込み0。

[機械判定](verification.json)、[公開照合](publication_receipt.json)、[範囲修正履歴](import_scope_finding.md)、[次の担当](pending_and_required_decisions.md)。
''')
    # These documents remain valid; update their count and resolved scope notes.
    for name in ('canonical_full_context_acceptance.md','result_failure_and_provenance_acceptance.md','semantic_mutation_acceptance.md'):
        text=(I.HERE/name).read_text().replace('197 probe','197 Codec/Result probe')
        text+='\n利用者がdriver/helperの人工用途を明示許可した後、214独立probe、最終SOURCEの132 Producer tests/107差分回帰、fresh Bundle2件を完了した。scope不一致は解決済み。\n'
        md(name,text)
    print(json.dumps(dict(independent_checks=224,PASS=224,Producer_unit=102,Producer_integration=30,final_SOURCE_regression=107,saved_cases=22,fresh_cases=2,mutations=20,status=current['status'])))

if __name__=='__main__':main()
