# DF correction E/T：保存方式 v5 の限定修正と実行承認待ち

2026-10-12 JST。判定：**`READY_FOR_SEPARATE_E_T_V5_ONE_SHOT_AUTHORIZATION`**。
技術準備・登録外検証・保存値のI/O検算を完了した。登録E/Tは実行せず、新authorization-only commitも作成していない。
追加GPT詳細レビューを要求する新たな科学条件・誤差保証・位相・比較対象の問題は検出していない。
次の境界は、固定した新sourceへの**別途の明示one-shot実行承認**である。mandatory STOPを維持する。

## 来歴と旧実行の扱い

- 旧science source：`016e02b486cd4b182b654af18743ef1ef50fe7e6`。
- 旧authorization-only直接子：`111199c45a9c37ac81821512ea8a44e6093f50cc`。
- 旧結果commit：`df973bbd76947d211043bebf0e9fd2bc34ae0367`。
- この準備の親となる公開確認commit：`a70af62798849ba1add7491375fe4825acdcaaa0`。
- 準備branch：`track-b-df-correction-et-storage-v5-preparation-20261012`。
- 新source S5と公開確認P5は、準備artifactの`publication_receipt.json`に完全SHAと取得検証を記録する。
  P5は公開確認用であり、将来の実行HEADに使用するauthorization-only直接子ではない。

旧実行の分類は`TECHNICAL_INCONCLUSIVE`、理由は`CapFailure: Output/log cap at metadata boundary`。
[旧結果・技術監査](df_correction_et_v4_results_and_gpt_handoff_20261012.md)を保持する。
旧marker SHA256は`446d97e0f1744285ed21fbfff6e349b3f06c65ee0959af0f605dff9bb0c19cbb`。
旧284ファイル、67,637,141 bytesを照合した。旧source manifestの363非indexファイルは完全一致し、
6個のindexはこの新branch上で親commit内容を保持した追記のみである。旧実行worktreeは変更していない。
旧結果のprefixを科学的勝敗へ転用せず、旧STOP・consumed markerを解除しない。

## 原因と今回の対処

確認済みなのは出力guardによる停止、保存総量、旧serializerが大きなraw observation全体を
複数の状態で再保存する構造、およびprefix/finalの同じPauli・native情報の重複である。
旧r6 A controlled observationの公開済みlogical JSONは32,718,729 bytes、未参照の途中generationは
7,869,608 bytesある。最初のguard検出時点の正確なallocation traceやディスク総量は記録されていない。
したがって、これらの保存圧迫は証拠に対応する対処対象だが、唯一の実行失敗要因だったとは断定しない。

新namespace [`df_correction_et_storage_v5`](../../../src/trottertracks/algorithm_codesign/df_correction_et_storage_v5/README.md)
で、以下の保存方式に変更した。旧serializer・共有API・旧sourceは変更していない。

1. 大きい`recursive_payload`、`recursive_definition_index`、Pauli係数表、全B event会計、
   input array、各native rowを内容SHA256で識別するimmutable compact JSONへ分離する。
2. 状態ごとに小さいtyped snapshotを保存し、logical recordのroot pointerをatomicに更新する。
   previous snapshotの鎖を残し、以前の状態や途中失敗のファイルを削除しない。
3. prefix/finalも同じimmutable objectを参照する。状態更新で大きい既存generationを複製しない。
4. readerはsnapshot・blob・chunk・履歴参照の完全なhash graphを照合した後、元のlogical JSONを復元する。
   元と同じ`json default=str, indent=2`＋末尾newlineのbytes・SHA256を照合する。
5. native結果とT inventoryのbindingには、logical bytes/hashに加えroot descriptorのbytes/hashを記録する。
   値が同じでも保存後の履歴差替えを検出する。inventory→job→worker→result→会計の既存bindingは保持する。

全physical review fileは512 KiB以下、chunkは256 KiB以下でUTF-8を保持する。
既存logical key・値・順序・自動生成名・matrix bytes・符号・位相・誤差記録を削除または正規化しない。
名前に依存する旧raw fingerprint policyも変更していない。
cross-worker publicationはPOSIX flockで直列化し、quantum algorithmやtask順・seedを変更しない。
旧artifactへの書込みはlock作成前から拒否する。

## 科学契約・guardとの同一性

[`contract_v5.json`](../../../artifacts/track_b_df_correction_et_storage_preparation/2026-10-12/v5/contract_v5.json)
はv4から`version`、`status`、`base_commit`、追加`storage_amendment`だけを変更した。
A/B、CD、confidence、error、digital、moments、E、compiler、T、cost、parallelism、caps、
numerical certification、runtime、layered acceptanceの全既存sectionは同一である。
native evaluation・saved-data・T accounting moduleはv4とbyte同一。
worker・coordinator・acceptanceのASTはstorage/guard import、namespace、contract path置換後に同一である。
guard本体も旧guardとbyte同一で、relative readerだけ新codecへ接続した。

| 制約 | v5でも維持する値 |
| --- | --- |
| fixture | 保存済みE-r3/r6、rank2×3、seed7101107、生成順不変 |
| local task | uncontrolled／MSB-controlled、theta1/32、kappa4、u1/128 |
| acceptance | strict phase、numeric2e-7、synthesis8e-7、full1e-6 |
| compiler | Qiskit1.3.0、rz/sx/x/cx、opt0、seed20261011 |
| native | 最大22 compile、2workers、threads1、CPU450s、wall600s、aggregate RSS2560 MiB |
| T | pygridsynth2.0.0、dps100、up_to_phase=false、gate precision1/40000000000、最大64keys |
| synthesis | 2workers、CPU900s、wall1200s、aggregate RSS1536 MiB、per-key wall30s/CPU20s |
| total | CPU1800s、wall2400s、aggregate RSS3072 MiB |
| output | 64 MiB、log4 MiB、review512 KiB、metadata reserve256 KiB |
| STOP | runs1/retry0は別authorization後だけ、全結果mandatory STOP |

storage writerは既存active output予算66,846,720 bytesを共有lock内で事前確認する。
外側SupervisorのCPU/RSS/output/log監視と終端metadata reserveを保持する。
pollによるRSS overshootの可能性やkernel CPU保護の既存仕様を変えない。
本契約は以前に明示採択されたwall上限を継承し、今後の新設計の一般的なno-wall defaultと区別する。

CD6taskは保存済みsubstageの参照のみ。CDと別fixtureのE費用を掛け合わせた実測PR総費用にしない。
全T coverageがない場合は既定のT_INCONCLUSIVEを保持する。
G10／RA-D0、分子、NPZ、GPU、sampling、LPへ進まない。

## 実際に行った検証

[検証記録](../../../artifacts/track_b_df_correction_et_storage_preparation/2026-10-12/v5/focused_validation_receipt.json)
に全attemptのstdout・JUnit identityを記録した。最終は**164 PASS、3.58秒**。
new storage23、v5 integration53、既存v2 regression62、既存integrity v3 regression26。
登録外1/2mode semantic control、mock worker/coordinator、tamper・中断・cap・one-shot拒否を使用し、full suiteは実行していない。

最初は73 PASS/1 FAIL（74件）。synthetic 600k文字列が大きなsnapshotに残る問題を検出し、
大きい文字列のblob factoringを追加した。次は74 PASS、追加regression込み164 PASS。
最後に旧legacy directoryへ拒否前のlockを作らない保護を追加し、164 PASSを確認した。
全失敗・修正経緯を新資料として残し、失敗を科学的結果として扱わない。

| I/O検算 | logical identity | physical保存量 | peak RSS | wall / CPU |
| --- | --- | ---: | ---: | ---: |
| 保存済み15recordのread-only replay | 全recordの元bytes/SHA完全一致 | 24,050,826 bytes | 163.0 MiB | 29.351 / 28.998 s |
| 33 MiB literal payloadのsynthetic control | 復元・状態更新・prefix/final共有PASS | 37,602,070 bytes | 118.516 MiB | 3.078 / 2.941 s |

両方の最大観測disk使用量はactive cap以下で、全physical fileが512 KiB以下だった。
syntheticの6状態更新増分は7,158 bytes、prefix→final増分は1,714 bytes。
保存値replayは元値を計算・再採点せず、別`/tmp`scratchで旧JSONのparse・保存・復元だけを実施した。
旧284ファイルは前後で一致した。artifactには小さい検算receiptだけを保存し、大きいscratch copyを追加していない。
これらのI/O receiptは最後のlegacy拒否precheck修正前に取得したが、その修正はfresh v5保存経路・bytesに影響しない。
最終focused testsはその拒否修正を含む。

[runtime照合](../../../artifacts/track_b_df_correction_et_storage_preparation/2026-10-12/v5/runtime_preflight.json)
は固定numeric/synthesis interpreterのbinary・package version・source treeをread-onlyで確認しPASS。
backend synthesis、登録input matrix、circuit、native compileは呼ばない。
[静的provenance監査](../../../artifacts/track_b_df_correction_et_storage_preparation/2026-10-12/v5/static_scope_audit.json)
でcontract差分、AST同一性、旧source/result/marker、pending authorization拒否、新result namespace未作成を確認した。

## 未検証事項と実行境界

登録r6 controlledのnative完了、全B native・T stageの資源使用、64keysの実coverage、
full runでの保存総量は未測定である。今回のreplayは既に保存されたprefixとliteral controlであり、
未知の登録結果がcap内に収まることやA/B資源優位を保証しない。
全source validationはlocal technical evidenceで、immutable CI・外部再現・科学的GOではない。

将来実行する場合は新source S5から専用branch/worktreeを作り、**S5のauthorization-only直接子A5**を作る。
許可pathは新preparationの`authorization.json`と任意
`docs/tracks/algorithm_codesign/df_correction_et_execution_authorization_v5.md`のみ。
別の明示指示、runs1/retry0/mandatory STOP、S5・contract・manifest identityを固定し、A5をpushして
remote SHA一致／clean／新result namespace未存在を確認する。

runnerは`scripts/tracks/algorithm_codesign/run_df_correction_et_resumption_v5.py`。
新result namespaceは`artifacts/track_b_df_correction_et_resumption_result/2026-10-12/v2/`であり、
旧v1 namespace・marker・authorizationを使用しない。現pending authorizationをrunnerは拒否する。
failure時もretry・cap変更・partial勝敗判定を行わず、外側process statusとterminal／storage failureを照合する。

科学的意味論を変更する必要が生じればGPT側へ戻す。現時点でその問題はなく、Codex技術準備を閉じる。
**明示実行承認が来るまでmandatory STOP。登録E/T、T合成、authorization-only commit作成は0。**
