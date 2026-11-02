"""Build the evidence report from independently produced observations."""
from collections import Counter
import json
from pathlib import Path
import independent as I

SOURCE='b7ad344401b10395c49e4cfa806909282859bab5'
HEAD='80b7bf61fe0a735d4031bb719c4b3e788611c3ba'
BRANCH='track-b-g6-w2c-b3-real-adapter-independent-server-20261020-r2'
DOC='docs/research/track_b_g6_w2c_b3_real_adapter_independent_server_20261020'
PRODUCER='https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/'+HEAD

def read(name):return json.loads((I.HERE/name).read_text())
def save(name,value):(I.HERE/name).write_text(json.dumps(value,indent=2)+'\n')
def md(name,value):(I.HERE/name).write_text(value.strip()+'\n')

def main():
    probes=read('independent_probe_verification.json');authority=read('independent_authority_verification.json')
    cases=read('artificial_22_case_verification.json')['cases'];mutations=read('independent_mutation_results.json')['mutations']
    assert all(x['pass_'] for x in probes['results']) and authority['FAIL']==0
    classifications=dict(Counter(x['status'] for x in cases))
    assert classifications==dict(TECHNICAL_FAILURE=13,INCOMPLETE_EVIDENCE=7,VERIFIED_ARTIFICIAL_BUNDLE_COMPLETE=2)
    assert len(mutations)==20 and all(x['independent_adapter_status']=='REJECTED' and x['storage_only_status']=='VERIFIED_ARTIFICIAL_BUNDLE_COMPLETE' for x in mutations)
    n256=next(x for x in cases if x['case']=='normal_n256')
    assert n256['dispatch']['trials']==n256['dispatch']['joined']==16896
    for group,file in [('codec','independent_codec_verification.json'),('result','independent_result_verification.json')]:
        rows=[r for r in probes['results'] if r['group']==group]
        save(file,dict(PASS=len(rows),FAIL=0,source=SOURCE,subject_process='candidate_probe.py',oracle='independent.py',
                       shared_candidate_validator_in_oracle=False,checks=rows))
    save('independent_bitstream_verification.json',dict(
        saved_artificial_trials_replayed=16896,domain_count=8,width_observed=160,
        prefix_S_P1_A_B_phase_pass=True,Mode_in_global_identity=True,Mode_in_seed=False,
        block_counter='uint64 big endian',hash='SHA256(UTF8 seed || counter)',
        replay_oracle='independent.py:bit_draw uses a bit-string construction',
        dynamic_candidate_boundaries='NOT_EXECUTED_STATIC_ONLY_DEPENDENCY',
        boundary_widths_pending=[1,160,256,257,512,513],
        artificial_only=True,scientific_namespace_bit_generation=0,actual_G7_variable_consumption_equivalence=False))
    save('independent_dispatch_verification.json',dict(normal_n256=n256,normal_n16=next(x for x in cases if x['case']=='normal_n16'),
         new_producer_spawn_execution=0,published_spawn_worker_evidence_independently_verified=True,
         historical_worker_process_attestation=False,hash_only_original_documents_acquired=False))
    save('verification.json',dict(schema='b3_real_adapter_independent_server_acceptance_v1',
        status='INDEPENDENT_TECHNICAL_ACCEPTANCE_ON_HOLD',acceptance_complete=False,
        independent_base='5c1e13fa74b3c5e4f3166babed785274844655d0',candidate_public_HEAD=HEAD,
        measured_SOURCE=SOURCE,artifact_identity='024d8812e97c5d541391533680fdbb3aee4cf6d8e51f020d3302a00767af42cb',
        independent_new_checks=dict(codec=49,result_context_angle_audit=148,fake_authority=10,total=207,PASS=207,FAIL=0),
        saved_semantic_mutations=dict(total=20,storage_only_accepted=20,independent_rejected=20),
        published_artificial_cases=dict(total=22,classifications=classifications),n256=n256['dispatch'],
        source_pins=89,input_pins=10,archive_members=994,publication_content_files=82,
        science_static_modules=8,science_static_entries=201,preserved_old_source_tests=291,
        blocked_independent_dynamic_checks=17,Producer_132_tests_independently_rerun=0,old_107_tests_independently_rerun=0,
        old_independent_1038_added_to_current_count=False,
        unresolved=['Static-only scope in user section 3E conflicts with eager artificial-runtime imports of g6_w2c_cost_preparation; old regressions also import g6_w2c_b3_driver'],
        core_science_runtime_bug_established=False,physical_host_independence=False,AI_subagents=0,
        prohibited_operations=dict(real_G7_F3_modules_loaded=0,all_user_static_only_modules_loaded=0,
            scientific_sampling=0,scientific_bitstream_generation=0,new_scientific_computation=0,
            saved_scientific_numerical_reanalysis=0,scientific_source_adoption=0,
            real_authorization_grant_claim_issued_or_consumed=0,native_quantum_build_compile_synthesis=0,
            old_claim_reuse=0,STOP_release=0,real_correctness=0,limited_pilot=0,Full_N=0),
        incidental_helper_import_intercepted_before_load=1,
        content_adopted=False,run_ready=False,contract_grants_execution=False,science_release=None,
        scientific_adoption=None,artificial_only=True,scientific_GO=False,mandatory_STOP=True,
        next_actors=['USER_AND_GPT']))
    md('README.md',f'''
# Track B W2c-B3：サーバー内独立技術受入

**受入保留。人工証拠の整合検証は成立したが、静的限定moduleと人工runtimeの依存の不一致が残る。Mandatory STOP。**

対象[公開HEAD]({PRODUCER})は`{HEAD}`、測定SOURCEは`{SOURCE}`、artifact identityは`024d8812e97c5d541391533680fdbb3aee4cf6d8e51f020d3302a00767af42cb`。
独立基点は`5c1e13fa74b3c5e4f3166babed785274844655d0`。既存同名worktreeを変更せず、新branch `{BRANCH}` を作成した。

- 独立追加check **207 PASS**：Codec49、Result/Context/Angle/Audit148、fake authority10。
- 公開人工22 case：正常2、技術失敗13、証拠不完全7。全件独立分類が一致。
- n256：16,896 trial完了・join一致。full document1,536件、hash-only15,360件。
- 保存semantic mutation20件：旧独立Storage checkerでは20件受理、新独立Adapter checkerでは20件拒否。
- SOURCE89、input10、科学8 module/201 entry、archive994 member、公開content82 file、旧291 SOURCE/testのbytes/hashを照合。
- Bits/gateの動的追加17件、Producer132件および旧107件の再実行は未実施。旧1,038件も今回に加算しない。

[Q1〜Q10](handoff.md)、[機械判定](verification.json)、[受入保留の最小原因](import_scope_finding.md)、[次の判断](pending_and_required_decisions.md)。
実科学処理・科学bits・実authority・科学採択・native・旧STOP解除は0。

SOURCEの実装整合性、保存人工証拠の整合性、今回の実行許可との一致を分けて評価した。
保存rawからの成立だけで、未実行のdynamic境界や実科学互換性をPASSにしない。

## 独立性と再現

このCodexセッション自身が実装者と別の独立worktreeで検証した。AI sub-agentは0、物理ホストは同じ。
`independent.py`は候補Codec/Result/Fixture/Offlineをimportしない。既存の独立Storage V2 consumerだけを再利用する。
`candidate_probe.py`は試験対象を別processで呼び出すsubjectであり、期待値は`run_probes.py`と`independent.py`が生成する。

`audit_source.py`はGit blobとarchiveを独自復元する。`run_audit.py`は22 case/20 mutationを8計算workerで検証する。
`run_probes.py`は許可されたCodec/Resultだけを実行し、`authority_probe.py`は新FAKE_ONLY scope2件だけを使う。
`static_audit.py`は残る依存、input/runtime、旧SOURCE、原証拠の保全を静的に確認する。
保存根拠は今回のreportと固定Producer commitのarchiveに分け、既公開rawを重複収録しない。
一時runtimeは`/tmp/b3-independent-server-r2-20261020/`。再現は新しい領域へ固定blobを展開し、既存scope/rawを上書きしない。
''')
    md('independent_source_verification.md',f'''
# 固定SOURCEと原証拠

GitHub originからHEAD `{HEAD}` と測定SOURCE `{SOURCE}` をexact fetchした。
content commitは`9bbf87f47688340c1d52899b0a5c614c2d8932ef`、最終HEADはreceipt-only後続であり、測定SOURCEと混同しない。
独自SHA256で89 pinsとcontent receipt82 fileを実Git bytesへ照合し、HEADと測定SOURCEの対象src/testsが等しいことを確認した。
Archiveは7 part、26,597,824 bytes、SHA256 `cf3485b46a55b0cfddf78469ea38eff0f3b387b2e8204ce21942634e302725bb`。
994 memberの存在、重複なし、path安全性、bytes/SHA256を照合して復元し、検証後にも全memberを再照合した。
manifest自身のSHA、全part、全memberは[source_inventory.json](source_inventory.json)に保存。

科学SOURCE `76ca3c4089b201d55da0fede08e1f77733ac38a5`の8 module/201 entryはbytes/ASTだけで照合した。
10 input pinsは内容の数値解析をせずbytes/hashで確認した。parent/worker宣言runtimeのexecutable/library bytesは現在のhostと一致。
保存当時のprocessがそのbytesを使ったという外部attestationや別物理hostの独立性までは証明しない。

API測定SOURCEの指示SHAに不一致があり、原inventoryの`78c571005fe766c61b32dc867e006531099ff608`を使用した。
元Bundleの測定SOURCEは`deade78a19f5b23859d16816d3a7bfb37765dd80`で一致。
[参照解決](fixed_reference_resolution.json)、[input/runtime/291旧path保全](pins_and_preservation_verification.json)。
''')
    md('codec_and_sidecar_acceptance.md','''
# Codecとsidecar

独立exact algebraとliteral golden値を使用し、候補validatorをoracleへimportせず49 check PASS。
None、bool、signed/large int（4096 bit）、既約Fraction、tuple/list、nested map、heterogeneous hashable key、unicodeの受理subsetで原document/digest/sourceとdriver投影が一致。
Bool/int、非既約・不正denominator、signed/noncanonical hex、duplicate/equal key、reserved/unknown tag、missing/extra sidecar、wrong digest/source、unsupported float、投影key collisionを拒否した。

Driver投影はtuple/listをarray、keyをstrへ写すので単独ではlosslessではない。
F3原document/digestを含むsidecarを保持した場合に限り元の型・値を復元できる。
旧F3が寛容にdecodeできる非canonical文書全域との互換性は主張しない。
DriverDocumentより前のtuple/key型も復元不能。実live G7/F3 documentとの互換性は未確認。
詳細：[独立Codec結果](independent_codec_verification.json)、[入力](probe_inputs.json)、[subject出力](probe_outputs.json)。
''')
    md('result_failure_and_provenance_acceptance.md','''
# Result、Failure、provenance

Result/Context/Angle/Audit合計148独立check PASS。EVENT/ACCEPTED、ZERO/RAW_NONREDUCED、ZERO/COIN_REJECT、IDENTITY、EVENT_QUERY、typed failureを区別した。
word/raw/reversed circuit word、child、phase、exact proposal×weight=coefficient、provider count、fingerprint/identity、元Evidenceを人工入力へ照合。
Full Noneはbranch/trace/provenanceなしでは拒否し、Canonical NoneもZEROへ変換しない。
IDENTITY/EVENT_QUERYはdriver sampling trialへ変換されず、typed failureはterminalのまま。
failure原document/digest、category/reason/stage/input/detailsが保持され、保存caseのF01へSTOP/非random-zero/未完了trialが伝播することを別途確認した。

この検証は人工structural条件であり、実G7のlaw、child分布、係数法則を採択していない。
詳細：[独立Result結果](independent_result_verification.json)、[22 case](artificial_22_case_verification.json)、[raw判定](raw_audit/)。
''')
    md('canonical_full_context_acceptance.md','''
# Canonical / Full contextと最終差分

Canonical3方式はgroup_index/raw_word/child、driver event digest、元Legacy input/policy/preflight contextを保持する。
FullはF3-shaped session source/input/policy/preflight/sigma/parent/child contextを保持する。
原angle documentをnormalized一時documentで上書きしていないことを、独立に生成した原bytes/digestと保存rawから確認した。
Canonical driver Audit Sealは元text/digest、identity/state、completed_calls_beforeを保持し、X→Yを通した順序もraw replayで照合。
Seal9追加checkを含む197 probeと全Layer B rawで、最終Canonical補完を重点回帰した。

Canonical正入力をFull F3.prepareへ接続していない。`CANONICAL_PREFLIGHT_NOT_IMPLEMENTED`は科学SOURCEのAST/文字列を静的確認し、実行していない。
Exact Fraction intervalの代表値/radius/precision、same-event requeryを確認したが、実atan enclosureの数学的精度は証明していない。
''')
    md('seed_and_bits_acceptance.md','''
# 人工seed / bits / consumption

独自bit-string SHA256 oracleで公開n256の16,896 trialをreplayした。
UTF8 seed、uint64 big-endian counter、SHA256 block、high160 bits、96 discarded bits、累積consumptionと各trialのdraw境界が一致。
Canonical2 draw、Full3/4/5 drawはsynthetic fixtureの仕様。実degree依存draw、conditional law、reject率、fresh IIDや実消費同値は未確認。

Modeはglobal IDへ含みseedへ入らず、S/P1・A/B・phase/passのprefixが一致した。arm/X-Yは8 domainで分離。
生成したのは`ARTIFICIAL_REAL_ADAPTER_V1/arm/axis`だけ。科学namespaceは文字列参照のみ。
候補Bitsのwidth1/160/256/257/512/513やfailure-position動的再実行は、静的限定helper依存のため未実施。
保存rawの160-bit成立と、未実行の境界試験を混同しない。
[機械結果](independent_bitstream_verification.json)、[未実行check](blocked_dynamic_checks.json)。
''')
    md('global_bundle_dispatch_acceptance.md','''
# Global Bundleへの保存dispatch証拠

独立oracleが原F3-shaped document、driver projection、Result、audit、Angle/query、bitsからdispatchを再構成した。
rawのAdapter source/artifact、factory、runtime/candidate/manifest binding、worker PID、task、Mode入りglobal ID/offsetを、RESERVE/ENTER/COMPLETE、物理state、journal、receipt/delivery/callerへ結合した。
Adapter名の記載だけを接続証拠にしていない。

n256はplanned/reserved/started/completedすべて16,896、unknown/cancelled0、全16,896 join。
EVENT14,728・ZERO_R1,020・ZERO_C1,148は人工fixture件数。
full document1,536とhash-only15,360を区別し、後者の未保存原documentを取得済みと表示しない。
hash-onlyの一致は固定人工SOURCEに基づく独立再構成と保存hashの一致であり、保存されなかった元bytesの直接取得ではない。

今回新たなProducer spawnを実行した件数は0。公開spawn workerが残したartifactの整合性は確認したが、動的境界のfresh実行と過去processのattestationは保留。
[結果](independent_dispatch_verification.json)、[22全case](artificial_22_case_verification.json)。
''')
    md('semantic_mutation_acceptance.md','''
# Semantic mutation

保存済み20 overlayの正常基点normal_n16は独立に成立した。
各overlayを専用領域に適用し、Producerのmutationラベルを判定に使わず、元rawとの差分field pathを独立parserで記録した。
hash-chain、worker report、final/delivery/callerまで再封印された20件は旧独立Storage V2 checkerを通るが、新独立Adapter oracleでは20件すべて拒否した。
Codec child/tag/original/driver digest、classification/fingerprint、factory/runtime/candidate/global Mode、seed/bit value/width/discard、angle/queryを含む。
Oracleは候補Codec/Result validatorを共用していない。

追加negativeはCodec49/Result148 probeに含めた。追加の保存overlay件数は0で、既存20件とtest数を合算しない。
core科学の新しい不具合は確定していない。新たな指摘は人工import closureと利用者の静的限定scopeの不一致である。
[field差分と判定](independent_mutation_results.json)、[scope不一致](import_scope_finding.md)。
''')
    md('storage_stop_and_authority_acceptance.md','''
# Storage / STOP / authority

前回公開の独立Storage V2 consumerを再利用し、RAM intentと物理state、trusted trace prefix、journal、単一call/result/log budget、UNKNOWN非返却、STOP後ENTER禁止、inflight drain、S失敗後P1禁止、P1失敗時S保持、receipt/marker/close/delivery/callerを検査した。
22 caseは正常2、技術失敗13、証拠不完全7に一致し、missing evidenceを補完しなかった。
旧checkerの範囲は保存構造・状態遷移・会計であり、新Codec/Result意味論は新oracleが別に検査する。

固定FakeStoreだけを新FAKE_ONLY scope2件で実行。race32 attempt/1 winner、double claim/SPENT拒否、ACK unknown後SPENT保持、binding mismatch拒否、STOP後再利用拒否の10 check PASS。
既存scope/claimを再利用せず、実authority/store/grantに触れていない。原DBはauthority_evidenceへ保存した。

closed flagsはfalse/null/true指定に一致し、実production_entry/real factory/bitsは静的に拒否定義を確認。
親/workerのdynamic gate再実行はstatic-only helper依存のため未確認。外部trusted store所有・鍵/失効・実atomicity/ACK運用・power durabilityは未確立。
[authority](independent_authority_verification.json)、[保全](pins_and_preservation_verification.json)。
''')
    md('regression_and_source_delta.md','''
# 最終SOURCEと旧回帰

旧107 PASSのSOURCEは`8d5135ee77e6cd06d8b2ed8aa0b9ee50a41a0e36`。
最終`b7ad3444...`との差分は7 path、144 insertion/4 deletionで、新AdapterのCanonical Seal/angle contextとそのtestsが中心。
旧4 test fileと291旧SOURCE/test bytesは不変。変更経路は独立197 probe（Codec49/Result148）と保存Layer B全件replayで重点検証した。

旧107件を最終SOURCEで再実行したとは主張しない。再実行数0。
Producer102 unit/30 integrationも今回の再実行数0。これらはtop-level依存で静的限定helperやdriverをimportするため、実行しなかった。
Bits/gate17 planned checkも未実行。重要境界の省略をPASSへ振り替えず、受入は保留。
旧独立1,038、Producer132、旧107、今回207独立check、20 mutation、22 caseを別集計する。
[差分](final_source_delta.patch)、[機械集計](regression_summary.json)、[理由](import_scope_finding.md)。
''')
    md('producer_consumer_discrepancies.md','''
# Producer / 独立consumerの不一致と限界

1. API測定SOURCE：指示文のSHAは原inventoryと不一致。公開HEADと原inventoryの`78c571005fe766c61b32dc867e006531099ff608`を採用し、exact fetchした。
2. 静的限定scope：利用者§3Eは8 moduleの実行を許可しないが、Producerの科学guardはcore6 moduleを対象とし、人工runtimeはcost preparation helperへ依存する。旧回帰はdriverにも依存。scopeの独断縮小はせず保留。
3. 独立checker開発：初回subject processは前述helperのload前拒否でtest開始前に失敗。以後は既知依存を再試行せずCodec/Result subjectを分離し、197 PASSと17未実行に分けた。この失敗をPASSに置換していない。
4. Producerの保存人工22 case分類と20 mutationの結論は、独立parser/oracleでも一致した。

物理host独立性、保存当時processのattestation、実科学law/live型/角度保証、実authorityは主張しない。
scope不一致を実G7/F3 sampling実行の証拠とは表示しない。固定Producer・科学SOURCE・既存STOPは変更していない。
''')
    md('pending_and_required_decisions.md','''
# 次の担当と判断事項

**ユーザー・GPT**：今回の§3Eの静的限定を維持するか、`g6_w2c_cost_preparation`と旧回帰用`g6_w2c_b3_driver`の人工用途のみを明示的に許可するか判断する。
代替は別versioned Producerによる依存分離。現SOURCEの科学gateをboolean変更で開く案ではない。

**Codex（追加指示後）**：選択されたscopeと固定SOURCEで、保留17動的check、Producer132件、適切な旧107差分回帰、親/worker gateを実行し、今回の保存受入証拠と結合する。現在は未実行。

**ユーザー・GPT（技術受入完了後）**：科学SOURCE/input/precisionの正式採択、新real接続SOURCE、実correctnessのscope/coverage/予算/STOP・新authorization、authority外部条件を判断する。
実G7/F3 live互換性、実variable consumption/law、数学的angle/precision、科学費用、limited pilot/Full-N、native、production解除は自動承認しない。
GPT詳細科学レビューの開始、科学SOURCE採択、実科学計算の実行承認は別事項。

**Codex（今回）**：Evidence Packet公開・remote/bytes照合後mandatory STOP。許可範囲の変更を回答なしで仮定しない。
''')
    md('handoff.md',f'''
# Q1〜Q10：独立技術受入結果

**総合判定：INDEPENDENT_TECHNICAL_ACCEPTANCE_ON_HOLD。未解決scope境界を残したため受入完了とはしない。Mandatory STOP。**

1. **Q1 固定照合**：公開HEAD `{HEAD}`、測定SOURCE `{SOURCE}`、content `9bbf87f47688340c1d52899b0a5c614c2d8932ef`、artifactを分離。SOURCE89/input10、科学8 module/201 entry、content82 file、archive994 memberのbytes/SHA一致。API測定SHAだけ指示と原inventoryに不一致があり、原資料の`78c571005fe766c61b32dc867e006531099ff608`を採用した。
2. **Q2 独立性**：別session/worktree、独自exact parser/canonicalization/hash/Result oracle。候補validatorはoracleにimportしない。既存独立Storage V2 consumerは保存検証だけに再利用。subjectは別process、物理host独立性はない。
3. **Q3 Codec**：49独立check PASS。exact signed/large int・Fraction・tuple/list・nested/heterogeneous map等の受理subsetでsidecar付き復元が成立。driver単独はtuple/key情報を失う。非canonical、collision、reserved/unsupported型を拒否し、実live互換性は未確認。
4. **Q4 Result等**：148独立check PASS。EVENT/ZERO_R/ZERO_C/IDENTITY/EVENT_QUERY/typed failure、Canonical/Full context、Angle/query/Sealを区別。None単独をZEROにしない。原failureがF01 terminal STOPへ保持。人工interval整合を数学的atan保証へ昇格しない。
5. **Q5 Bits**：公開16,896人工trialの独自SHA256-counter replay・width160/discard96/consumption、8 domain、S/P1/A/B/phase/pass prefix一致。Modeはglobal IDへ入りseedから除外。候補の境界幅/failure-position動的試験はhelperの静的限定と衝突し17 planned check未実施。実科学bits0。
6. **Q6 dispatch**：保存rawからfactory/runtime/PID/global offset→RESERVE/ENTER/COMPLETE/physical state→receipt/delivery/callerを独立に結合。n256全16,896完了一致、full1,536/hash-only15,360。新Producer spawn実行0、未保存原document直接取得は主張しない。
7. **Q7 mutation**：20保存overlayすべて旧独立Storage checkerを通り、新独立Adapter oracleは20件拒否。変更fieldは元rawとの差分から記録。新指摘は人工runtimeの静的限定helper依存であり、core科学lawの欠陥とは確定していない。
8. **Q8 cases/tests/regression**：22ケースは正常2/技術失敗13/証拠不完全7。今回新独立207 check（49+148+fake authority10）PASS。Producer132・旧107・旧1,038を加算しない。旧107は旧SOURCEの報告値、今回再実行0。最終Canonical補完は独立probe/raw replayで重点確認したが、import制限により総合受入を保留。
9. **Q9 残条件**：ユーザー・GPTによる静的限定helper/driverの人工用途の明示許可、または別Producer依存分離を先に判断。後続Codexが保留動的gate/回帰を完了。その後も正式科学SOURCE/input/precision採択、新real接続SOURCE、live互換性/correctness scope・coverage・新実行承認、実authority外部条件が必要。pilot/Full-N/production解除へ自動進行しない。
10. **Q10 禁止操作**：実G7/F3 module load/sampling・科学bit生成・新科学計算/科学値再解析・科学採択・実authorization/grant/claim・native・旧claim再利用・STOP解除・real correctness/pilot/Full-Nはすべて0。初回subjectのhelper import1 attemptは独立guardがload前に遮断した。helper実行も0。原証拠994 member/旧291 SOURCE-testを再照合し、他worktreeや旧grant/ledger/marker/STOPへ書込み0。

SOURCEとEvidence： [検証JSON](verification.json)、[scope最小再現](import_scope_finding.md)、[次の担当](pending_and_required_decisions.md)。
公開commitとremote/公開bytes照合は`publication_receipt.json`およびreceipt-only後続の最終chatの固定SHAへ記録する。
''')
    print(json.dumps(dict(status='ON_HOLD',independent_checks=207,cases=22,mutations=20)))

if __name__=='__main__':main()
