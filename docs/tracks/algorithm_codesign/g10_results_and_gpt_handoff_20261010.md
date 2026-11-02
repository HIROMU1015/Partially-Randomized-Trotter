# G10 one-shot：技術的停止とGPT引継ぎ（2026-10-10）

最終分類は **`G10_TECHNICAL_INCONCLUSIVE`**。固定RSS上限512 MiBを超え、
`MemoryError: SP-1 RSS cap hit; no retry` で停止した。guard記録のpeak RSSは
552,496 KiB（539.546875 MiB）。runnerは一回のみ、retry=0、marker消費済み、**mandatory STOP**。

17行と下界fieldが保存されていても、固定contractはtechnical failure時のprefixによる科学判断を禁止する。
`prefix_rows_usable_for_final_research_decision=false` を維持し、方式の勝敗、次数による反転、
general generatorの価値、IS分離、新規性をこのrunから判定しない。原G9 v2の成功分類も変更していない。

## 固定sourceと実行provenance

| 項目 | 固定identity |
| --- | --- |
| science source S | `05c5ef23fce775a822ab5686f5da2f0d77675864` |
| execution HEAD / authorization-only A | `f5cd0755424d1b11e2249cc115518d84fb8bb8d3`（唯一のparent=S） |
| execution branch | `track-b-g10-one-shot-execution-20261010` |
| contract SHA256 | `71ae2310a08f9d111faab76f27e6de277189ab132370d8996a0e2420c9510818` |
| authorization SHA256 | `2ea068af935b14a3c78f6c536fd4817f736f84be0460bd44bce541a8c48067e7` |
| marker SHA256 | `1ddfa9ed9804cf044d68da2da5a8607d3a32e3bf63ecdc0c18a7f2f9b99d83ce` |
| final-review receipt commit | `3b490c2e8f5f4b71a40bce381b4f32dbe939c39c`（実行HEADとして使用せず） |

[固定source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/05c5ef23fce775a822ab5686f5da2f0d77675864)、
[固定contract](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/05c5ef23fce775a822ab5686f5da2f0d77675864/artifacts/track_b_g10_degree_preparation/2026-10-10/contract_v1.json)、
[authorization](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f5cd0755424d1b11e2249cc115518d84fb8bb8d3/artifacts/track_b_g10_degree_preparation/2026-10-10/authorization.json)、
[明示指示のreceipt](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f5cd0755424d1b11e2249cc115518d84fb8bb8d3/docs/tracks/algorithm_codesign/g10_execution_authorization_receipt.md)、
[採用GPT source review](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3b490c2e8f5f4b71a40bce381b4f32dbe939c39c/docs/research/track_b_G10_source_final_review_20261010.md)。
Aの変更はauthorization JSONとreceiptの二pathのみ。remote=A、clean、critical113、protected1241、
runtime identity、fresh markerを測定前に確認した。記録は
[preflight receipt](../../../artifacts/track_b_g10_degree_result/2026-10-10/v1/preflight_receipt_v1.json)に固定。

実行commandは固定isolated runtimeの `python -B scripts/tracks/algorithm_codesign/g10_degree_matched_native.py --source-commit 05c5ef23fce775a822ab5686f5da2f0d77675864`。
`PYTHONPATH=src`、`OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`。
Python3.10.12、固定pygridsynth2.0.0、mpmath1.3.0、numpy2.2.6。
runtime checkの保存field `synthesizer_calls=0` は**launch検査時点**の値であり、本runの新規合成は27件。
実行結果・科学source・contract・authorization・旧marker・旧STOPは変更していない。
この文書を含む結果commitはAの子として公開し、そのfull SHAを利用者へ報告する。

## 実行対象と保存された技術記録

G9既知developmentのsynthetic 3-system-qubit provider、`p=(1/5,3/10,1/2)`、`x=5/7`。
`R_P(theta)=exp(-i theta P/2)`、`Q0=Z0`、`V1=R_XX01(pi/4)`、
`V2=R_XX12(pi/4) R_ZZ01(pi/4)`（右作用先）、`Qi=Vi†ZiVi`。
各m内で同じfull first operator moment `P_m(-i x sum p_i Q_i)`を比較する契約。
m3/m7を取得、m5は保存G9 direct6行を34-axis共通policyへ再会計。
異なるmの同じexponential accuracy、molecular geometry/basis/DF rank/split、PR/QPE最終資源は対象外。
明示cheap Pauli I1 accessであり、DFでのI0取得優位や独立held-outの証拠ではない。

| 保存項目 | 件数・状態 |
| --- | --- |
| registered rows / axes | 17 / 34（新native11、saved m5再会計6） |
| degree別保存行 | m3=5、m5=6、m7=6 |
| 保存event bindings | 10,936（m3=102、m5=945、m7=9,889、cap12,000） |
| synthesis cache | 46 keys：new27、saved reuse19、41,705 bytes |
| primitive guard | 保存46 keysのstrict error PASS、`up_to_phase=false`、epsilon=10^-6 |
| policy-lower field | 全17行に保存。停止後の再評価・比較・分類なし |
| fixed interface diagnostic | 9 arms × 64 = 576 trials。実量子shot/trajectoryではない |
| CTS certificate field | m3/m7で保存 |
| protected history | 実行前後1241 path、違反0 |

保存cacheはsequence SHA、T/T†、1Q/global-W、strict error boundを含み、
[小型identity一覧](../../../artifacts/track_b_g10_degree_result/2026-10-10/v1/synthesis_identity_inventory_v1.json)から参照できる。
全sequence・native IR・operator診断・費用/予算fieldは[原結果](../../../artifacts/track_b_g10_degree_result/2026-10-10/v1/result_v1.json)に保持する。
費用・予算の保存整合を確認しても、technical prefixを有効なresource mapへ昇格させない。

## 資源停止の記録と判明範囲

| 項目 | 記録 |
| --- | --- |
| guard wall / CPU | 14.6129226200 s / 14.612332 s |
| guard peak RSS | 552,496 KiB = 539.546875 MiB（cap512 MiB） |
| 外部process wall | 18.6660838802 s |
| raw result bytes | 66,842,493（cap134,217,728） |
| process exit | 0：exception処理・結果保存完了を示す。科学完了の意味はない |
| synthesis retry / whole-run retry | 0 / 0 |
| 実量子shots・trajectory・GPU・DF・分子/NPZ・LP | すべて0 |

guard snapshotはfailure出力のserialize前であり、whole-process最終peak RSSは別計測していない。
固定sourceではprotected_after確認の後にfinal guard、通常JSON化、再guardがあり、
exception handlerでtechnical分類とfallback出力を保存する。全17行・下界field・protected_afterがあるため、
遅い完了段階でのRSS failureは確認できる。ただしtracebackやallocation traceは保存されておらず、
**exact throw位置・大きいallocationの原因は未確定**。final guard/serialization/periodic alarmは候補に留める。
エラー文の`SP-1`は共通guard由来で、実行stageはG10。
[最終source review](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3b490c2e8f5f4b71a40bce381b4f32dbe939c39c/docs/research/track_b_G10_source_final_review_20261010.md)
§11もmemory/serialization/failure handlerの残余リスクを指摘していた。

## 保存値・provenance監査

[保存値auditor](../../../scripts/tracks/algorithm_codesign/audit_g10_saved_outputs.py)はstdlibだけを使い、保存hash、sequence文字列、
event `a=q*w`、literal saved IRのT/CX/1Q再計数、保存expected/accepted/hard会計を照合した。
10,936 bindings、46 cache keys、945 m5 bindingsと旧G9 event/budget identity、原結果/marker/STOP、
critical113/protected1241を確認し、[audit receipt](../../../artifacts/track_b_g10_degree_result/2026-10-10/v1/saved_output_audit_v1.json)は
`SAVED_OUTPUT_IDENTITY_AND_ACCOUNTING_PASS`（45,605整合条件）。
これはscience再実行、独立再現、45,605件の科学testsではない。matrix、synthesis、sampler、
shot budget、proposal lowerの再評価は行っていない。
source preparationの41 focused testsは既存証拠として参照し、この実行後に追加science test/full suiteは実行しない。
GPT source reviewの49 selfchecksもreview側の報告であり、ここで再現・合算していない。

[provenance audit](../../../artifacts/track_b_g10_degree_result/2026-10-10/v1/provenance_audit_v1.json)はfull S prefixを含むcritical113と
既存1241 path、contract/auth/marker/result/STOPの不変性を確認する。
追加は契約済み9つのindexへの追記、保存値auditor、結果・監査・引継ぎ資料だけ。
Track A・shared API・solver/caps/synthesis設定は変更していない。
[evidence inventory](../../../artifacts/track_b_g10_degree_result/2026-10-10/v1/evidence_manifest_v1.json)、
[小型技術summary](../../../artifacts/track_b_g10_degree_result/2026-10-10/v1/technical_summary_v1.json)、
[process receipt](../../../artifacts/track_b_g10_degree_result/2026-10-10/v1/process_receipt_v1.json)、
[STOP](../../../artifacts/track_b_g10_degree_result/2026-10-10/v1/STOP.json)がGPT向けの入口。
原結果は約66.84 MBなので、最初は小型summaryとauditを読む。

## GPTへ戻す未決事項と停止状態

G10の予定された次数比較はtechnical inconclusiveであり、prefixのpositive/negativeどちらも採用しない。
旧G9 v2原証拠とG10以前のproof/claim boundaryはそれぞれの固定commitに残る。
次に限定的な技術対処を行う情報価値があるか、どのscope/認証/one-shot契約が必要かはGPT側の判断へ返す。
Codexはcaps変更、source修正、再実行、m9/新p/x/provider、追加synthesis、IS探索、DF/QPE接続へ進まない。

**runs=1 / retries=0 / mandatory STOP / next_science_authorized=false**。
markerは保持し、この報告の公開後も追加科学処理は行わない。
