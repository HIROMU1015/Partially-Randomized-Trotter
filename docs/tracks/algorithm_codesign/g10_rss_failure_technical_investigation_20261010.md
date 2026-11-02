# Track B G10：RSS上限超過の技術調査（2026-10-10）

**結論：全resultの再帰複製と、一括JSON化によるchunk list/joinが、最も有力なメモリ増幅要因である。**
保存JSONだけの独立I/O検算で、同じbytes・SHA256を生成する一括方式のpeak RSSは508.90625 MiB、
逐次方式は253.25 MiBだった。元G10のheapは再現していないため、元の例外行や修正後のG10成功は未確定。
最も妥当な対応は、科学計算を維持したままI/Oと参照寿命を限定修正するためのsource preparationを別途reviewすること。
本調査では修正・再実行・cap変更を行わず、**mandatory STOP**を維持する。

## 1. 正本・認可・調査境界

- result R：`e429c99d77b3222c5cca62750b2d111f87e4cb50`、execution branch `track-b-g10-one-shot-execution-20261010`。
- frozen science S：`05c5ef23fce775a822ab5686f5da2f0d77675864`、authorization-only A：`f5cd0755424d1b11e2249cc115518d84fb8bb8d3`。
- 調査branch：`track-b-g10-rss-cause-audit-20261010`、Rから作成。remote execution branch=Rを照合。
- [原引継ぎ](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e429c99d77b3222c5cca62750b2d111f87e4cb50/docs/tracks/algorithm_codesign/g10_results_and_gpt_handoff_20261010.md)、
  [固定contract](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/05c5ef23fce775a822ab5686f5da2f0d77675864/artifacts/track_b_g10_degree_preparation/2026-10-10/contract_v1.json)、
  [原結果inventory](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e429c99d77b3222c5cca62750b2d111f87e4cb50/artifacts/track_b_g10_degree_result/2026-10-10/v1/evidence_manifest_v1.json)。

対象は既知developmentの同p/x・3-system-qubit providerで各mの有限P_mを扱ったG10。
本調査は保存17行を費用比較・科学的勝敗・研究GOへ使用しない。
本番runner、generator、matrix、native lowering、synthesis、sampler、budget、lower、LPを呼ばない。
sourceの読み取り、保存metadata、stdlib I/O検算だけ。既存result・audit・marker・STOP・sourceはそのまま保持する。

## 2. 確認済み事実と発生区間

| 確認事項 | 根拠・意味 |
| --- | --- |
| 直接の停止理由 | `MemoryError: SP-1 RSS cap hit; no retry`。共通guardの明示raiseと一致。実行stageはG10 |
| RSS cap / guard peak | 512 MiB / 552,496 KiB = 539.546875 MiB |
| 最終科学分類 | `G10_TECHNICAL_INCONCLUSIVE`、prefix使用不可、retry0 |
| 保存状態 | 全17行・全17 lower field・protected_after1241/違反0がある |
| native acquisition | new27 / reuse19。new27 guard記録のpeakは215,088–215,600 KiB |
| raw result / cache | 66,842,493 bytes / 46 keys・41,705 bytes。cache byte cap違反ではない |
| output byte cap | 134,217,728。今回の原JSONは上限内 |
| process | exit0、外部wall18.6660838802 s。technical例外処理と保存が完了したという意味 |
| 計測欠落 | traceback、allocation trace、各完了stage前後のRSS、fallback後のwhole-process peakなし |

G10 runnerの[source末尾](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/05c5ef23fce775a822ab5686f5da2f0d77675864/scripts/tracks/algorithm_codesign/g10_degree_matched_native.py#L165)は、
row set/binding cap確認 → 各row lower追加（171–173）→ protected_after（174–176）→
final check（177）→ resource/status complete設定（178–180）→ JSON化（181）→
UTF8 byte cap（182–183）→ final check（184）の順。
全lowerとprotected_afterが残るため、通常のrow取得途中の失敗ではなく、**late completion区間**まで到達したことは確認できる。
ただしprotected_checkの最後とperiodic alarm、177のcheck、181/182の作業、184のcheckを区別する記録はない。
intervals/bytesの保存だけから181に断定しない。

classical stage wall合計11.5227791560 s、new key合成の記録合計1.3433990628 s、
failure guard wall14.6129226200 s。差分にはguard/hash/下界/I/O等が入り、正確な時系列の復元には使えない。
exceptionの文字列は[共通guard](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/05c5ef23fce775a822ab5686f5da2f0d77675864/src/trottertracks/algorithm_codesign/synthesis_placement/wrapper_launch.py#L115)
120–121と一致する。実allocator exhaustion、OS OOM kill、AS cap hitを実証する記録ではない。

## 3. 保持構造と追加allocation

| 処理・保持 | 静的に確認した構造 | 原因との対応 |
| --- | --- | --- |
| `old_raw` / `old` | G9 raw11,341,870 bytesとdecoded全11行をwith末尾まで保持（runner63–68） | 再利用後も不要部分が残る。補助的な重複 |
| m5 anchor | `deepcopy(saved)`と`deepcopy(budget)`（g10_comparison44/53） | 元old graphと新6行が同時に残る。出力identity維持が必要 |
| `pending` | 新11行分のevent listを保持し、row bindingsも同じeventを参照 | listの追加はあるが、event全体が二重deepcopyされたわけではない |
| row result | event・native_ir・price・診断・予算を全17行保持 | 最大の出力母体。保存IR gate records582,770 |
| `serial(result)` | dict/list/tupleを全て新containerへ再帰複製、Fractionをstr化（g10_saved119–126） | result graphと全変換graphが同時に生存 |
| `json.dumps` | 固定CPython3.10 encoderは`list(iterencode(...))`して`''.join(chunks)` | 多数の小stringとpointer array、join先が同時に生存 |
| newline / UTF8 byte check | newline追加と`raw.encode()`。byte checkは全JSONをbytes化 | 追加copyになり得る。検算では最大peakはdumps中で、newlineはpeakを更新しなかった |
| synthesis cache | 小さいsequence/identity46 keys、JSON約41KB | 小型。単独で今回の数百MiB増幅を説明しない |
| fixed matrix cache | `@lru_cache(None)`（g9_matrix32）。保存IRの非R gate distinct20 | n4 ndarray dataは計81,920 bytes相当。matrixは新規評価せず静的sizeのみ |
| protected hash | 各fileをread_bytesでhash（g7_launch23）。最大ledger file14,865,179 bytes | 完了前に追加bufferを作る可能性。今回の正確なpeak寄与は未計測 |

`pending`を消しても`result.rows.events`がeventを参照するため、event本体は消えない。
`old`を消す際も再利用19 cache recordsの参照はresultから保持する必要がある。
loop localの`events`/`es`/`ce`/`g`/`bs`等が残る点も点検対象。
8×8/16×16 temporary matricesはrow return値に保持せず診断floatだけ残す。
numpy/BLAS/mpmath/synthesizerのnative heap・allocator retentionは原プロセスの追加基底として未計測。
`del`や`gc.collect()`がRSSを必ず戻すとは限らない。

## 4. 保存JSONだけの非科学的I/O検算

[診断script](../../../scripts/tracks/algorithm_codesign/audit_g10_rss_saved_json.py)を固定Python3.10.12で、事前に定めたcensus/materialized/streamの3独立processに一回ずつ適用した。
全processはRSS512 MiB、AS1536 MiB、wall/CPU60sに限定。これは別のtechnical workloadの上限で、
G10のcontract/capsは変更していない。science import/runner/marker/synthesis/matrix/samplingは0。
materializedでは固定g10_savedの**pure serial関数ASTだけ**を抽出し、他のsource関数を実行しない。
streamはdecoded保存treeからiterencodeし、各UTF8 chunkをhashへ送り、原JSONを書き換えない。

| 保存JSON I/O phase | current RSS MiB | process peak RSS MiB |
| --- | ---: | ---: |
| materialized：decoded treeのみ | 126.15625 | 253.5（load時の一時peak） |
| materialized：serial全container複製を保持 | 203.40625 | 253.5 |
| materialized：dumps返却後 | 280.3984375 | **508.90625** |
| materialized：UTF8 bytesも保持 | 343.8984375 | 508.90625 |
| stream：encode/hash完了 | 125.765625 | **253.25**（load peakから増加なし） |

両方式でbyte count66,842,493、SHA256
`b62695c19964a5a121b965c39142427bfa8048efad9bce05f3221494bb14dfe1` に一致した。
どちらもdiagnostic cap内で完了しており、**この検算で512 MiB超過そのものを再現したとは言わない**。
保存treeは約100.20 MiBのunique-object shallow-size合計、1,005,466 objects、list626,378。
encoder chunks4,478,270、pointer slotsだけで最低35,826,160 bytes（64bit）。
chunk getsizeofのoccurrence合計286,277,722 bytesは共有stringの重複を含み得るため、unique allocation/RSSの厳密値ではない。
CPython encoder source hashは`06b881b824f71e95d72af4ab865de4c35553e791b6d959a125caac61401cc350`。

重要な限定：保存JSONは既にFraction→str、tuple→listであり、原runのFraction/tuple・aliasing・
library/native heap・allocator状態を復元していない。上記はI/O経路の増幅とbytes保存を確認する技術証拠で、
G10全体の新resource mapや成功証明ではない。
**confidence**：guard由来の直接停止=確認済み、I/O増幅=検算確認済み、原181での超過=最有力だが未確定、
stream修正後のG10が512 MiB内で完了=未確認。

[raw profile/receipts](../../../artifacts/track_b_g10_rss_cause_audit/2026-10-10/v1/technical_findings_v1.json)と
[synthetic serialization checks](../../../artifacts/track_b_g10_rss_cause_audit/2026-10-10/v1/serialization_semantics_checks_v1.json)を保存。
Fraction/tuple/unicodeの人工2fixtureはlazy Fraction-only encoderでbytes一致、非finite/unsupported値の拒否も一致。
一方bool等の非string dict keyでは旧serialとの違いを確認したため、**string key schemaの確認なしに一般置換しない**。
人工fixtureでの小型検算は、live G10 graph全体の同値性を証明しない。

## 5. guard・例外保存の残る問題

共通guardは`ru_maxrss`（過去のhigh-water mark）を使用し、0.2s周期SIGALRMと明示checkで例外を出す。
瞬間的に上限を超えた後にfreeしてもpeakは戻らず、late checkで停止し得る。
periodic Python signalはallocationそのものをhard-blockする機構ではなく、処理境界/周期で検知するためovershootがあり得る。
OS `RLIMIT_AS=1536 MiB`はvirtual address上限で、512 MiBのresident hard capではない。

`with guard`がexceptionでunwindすると、`__exit__`はtimerとCPU/AS limitsを元へ戻す。
runner185–192のfallbackは**guard scopeの外**で同じ全serial/dumps/encodeを繰り返す。
成功・失敗双方のfile write/STOP write（193–199）もscope外。
したがって539.546875 MiBはfallback完了後の最終peakではなく、例外処理直前側のsnapshotである。
今回はfallback保存に成功したが、再度のallocation failure/partial output/未保存STOPのリスクが残る。
これは確認済み実装上の監視範囲であり、今回のfallbackが実際にcapをどれだけ超えたかは分からない。

## 6. 修正候補・validation・承認

| 候補 | 区分・期待効果 | 残るリスク・必要validation | 必要承認 |
| --- | --- | --- | --- |
| Fraction-only lazy JSON encoder + iterencode UTF8 chunk writer | **1：成功内容/科学意味論を維持するI/O修正候補**。全serial copy、chunk list/join、全encodeを回避 | 全live-type schema/string keys/tuple/Fraction/order/indent/newline/allow_nan=False、saved bytes一致、bounded UTF8 chunksとoutput countを検証。実heap512成功は未保証 | 今回未認可。GPTの限定source-preparation採択後、source review |
| old_raw/old/pending等の参照寿命短縮 | **1：不要保持の削減**。G9全graph・copy等を生存期間短縮 | cache参照とm5全event/original budget identity、production/reference分離、deterministic orderを確認。GCだけではhigh-waterの回復不可 | 別source変更scope承認とfocused review |
| row単位spoolしてrelease | **1：同じ成功fieldsを保持する設計なら可能**。最大resident母体を小さくする | later lower/row-set validationが同じ情報で成立、all17 completion、reference traversal order、partialの扱い、byte cap/temporary disk会計。広い変更 | streaming+cleanupで不足する技術証拠がある場合だけ別review |
| stage名・current/peak RSS・traceback、stream全終端までguard、exclusive partial/final publication | **2：実行・保存契約の明文化が必要**。次失敗の帰属/証拠保存を改善 | I/O/flush後のcompletion、partialを科学利用禁止、旧marker不変。compact emergency receiptのbounded予算とSTOP生成を事前固定 | 新contract/source/別authorizationで明示承認 |
| byteを捨てずhashをincremental化 | **1：technical hashingのbuffer削減候補** | hash/append-only prefix identity同値。今回は最大約14MBの補助問題で主因と断定しない | 必要なら独立最小変更、shared helper変更は別提案 |
| compact JSON whitespace変更 | **1：内容が同じなら科学意味論は維持可能** | chunk list問題が残る。bytes identityが変わる。第一選択にはしない | source/output identity review |
| RSS cap拡大 | **2：資源契約変更** | 不要保持/unguarded fallbackを残す。512内の構成可否を閉じない | 今回禁止、解決策として採用しない |
| event/IR省略、Fraction丸め、matrix/precision/budget/policy削減、旧17行を科学結果へ昇格 | **3：出力・科学条件/誤差会計へ影響** | 本G10と同じ比較を保証しない | 今回禁止、技術修正として扱わない |

第一候補は**typed streaming + 必要範囲のlifetime cleanup + 事前固定したfailure/finalization方式**。
`json.dump(serial(result), f)`だけでは全serial graphの複製が残る。
`json.dumps(..., default=str)`だけではchunk list/joinが残り、未対応typeも黙ってstr化し得る。
成功JSON内容・error会計・科学処理順序を維持し、guardを外して大きなfallbackを完了させる方式を解決とは呼ばない。
今回は候補の検討と人工I/O検算までであり、production修正は未実施。

## 7. 再実行に必要な段階と安全性

1. GPTへ本調査を渡し、I/O/lifetime/failure-reportだけの限定source preparationを行う価値とscopeを決める。
2. 別preparation branchでsource S2を作る。旧S/A/R・全marker・STOPを新protected ledgerへ収録する。
   scientific identity（p/x/provider/m/arms/precision/seed/semantic guard/alpha/beta/budgets/bounds）と
   RSS512/AS1536/time/key/shot/output capの不変を機械的に比較する。path/manifest/failure reportingの契約差分は明示。
3. science0で、人工typed fixtures・保存IO・output-cap/IO-failure/guard-failure/partial-final扱いのfocused validation。
   fullsuiteや新scienceを先行実行しない。library基底/real-type shapeを反映した非科学的memory stress scopeも事前固定する。
   streaming検算だけで本番の成功を保証しない。
4. 新contractにfresh result/marker directory、protected hashes、tiny failure receiptの上限・終端保証・prefix利用禁止を固定。
   total/peak/current memoryのstage log、flush/close/hash検証までのguard、atomic final publication順序をsource reviewで確認する。
   途中fileのstatus fieldだけでcompleteと扱わない。上限超過後のemergency記録手順もboundedで明文化し、
   old markerを解除・再利用せず、新source直接子authorization-only A2と明示one-shot指示を別に得る。
5. その後に限り、新しい一回を行える。失敗してもretry0、protected旧証拠保持、technical分類/小型receipt/partial identity/STOPを保存。
   原v1 prefixを新runの完成行やcache rescueとして採用することは、本提案に含めない。

現時点では**次の限定修正scopeについてGPTの追加判断が必要**。
Codexは採択済みscopeの実装・focused verification・provenanceを担当できるが、
この技術調査指示だけでsource/contractを変更したり、S2/A2を実行authorizationとして作成したりしない。
原因が完全には特定できなくても、根拠のあるmemory増幅経路と残余不確実性は整理できた。
G10科学分類・研究方針・方式採択は変わらない。

[調査evidence inventory](../../../artifacts/track_b_g10_rss_cause_audit/2026-10-10/v1/evidence_manifest_v1.json)と
[source/旧証拠provenance](../../../artifacts/track_b_g10_rss_cause_audit/2026-10-10/v1/provenance_audit_v1.json)を参照。
[materialized profile](../../../artifacts/track_b_g10_rss_cause_audit/2026-10-10/v1/materialized_profile_v1.jsonl)、
[stream profile](../../../artifacts/track_b_g10_rss_cause_audit/2026-10-10/v1/stream_profile_v1.jsonl)、
[census](../../../artifacts/track_b_g10_rss_cause_audit/2026-10-10/v1/census_profile_v1.jsonl)は新しい技術監査記録で、原G10のlogではない。

**mandatory STOP。再実行・新しい科学的検証・G11進行なし。**
