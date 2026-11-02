# Track A H6：保存integrals DF診断一回の結果・GPTレビュー索引 v1

2026-10-10 JST。ユーザーの明示承認により、固定sourceと保存integralsからDF診断を**一回だけ**実行した。
原statusは`H6_DF_DIAGNOSTIC_RECORDED`、[保存bytes監査](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic_execution_v1/2026-10-10/saved_diagnostic_audit_v1.json)は`SAVED_DIAGNOSTIC_BYTES_PASS`、必要記録の欠測0。
**このstatusは診断記録の完了であり、Hermitization政策PASS・H6入力受理・H6 pilot GOではない。**
H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、mandatory STOP。独立科学判断はGPTへ戻す。

## 実行identityと固定条件

- [診断準備](track_a_h6_df_hermitization_diagnostic_preparation_v1.md) / [認可文書](track_a_h6_df_diagnostic_execution_authorization_v1.md)。
- 新diagnostic source：`ff24de4bc410234472a416186b773fc7875ae373`、[source freeze](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic_preparation/2026-10-10/source_freeze_v1.json) science187件・validation2件。
- 新grant公開commit：`a99e6ba28d0cf26c9d63d131c2d0f620e2880c5f`。execution identity：`h6_df_diagnostic_20261010_launch_v1`。
- raw/結果保存commit：`c4bf9de3af1580dc54ad055f8e3153477aecf6b1`。
- [GitHub再取得・保存bytes再監査記録](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic_execution_v1/2026-10-10/remote_verification_v1.json)：この結果commitのraw25件・source187件、旧source183件/旧raw20件と既存4075 pathの保全を確認。数値診断やDFは再実行していない。
- [grant](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic_execution_v1/2026-10-10/authorization_v1.json) SHA-256：`5249805956780614c2421994a701caaff300ab05cfe50ecfd9431b5d290ec4e4`。
- [seal](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic_preparation/2026-10-10/sealed_diagnostic_preparation_v1.json) digest：`59ff0f13202e25731ad63cb0c47bc5f9600aff9f671b412999279508b3631bd0`。source/input/env/CPU/capsを変更していない。
- [実行前remote照合](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic_execution_v1/2026-10-10/prelaunch_remote_verification_v1.json)と[exclusive起動意図](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic_execution_v1/2026-10-10/launch_attempt_v1.json)を保存した。
- 元結果：`df3b1f694ceb72a198ab6e3e89706b239e56e1da`、元source：`67312f3195aede26e8ba4f5727d89c236772f82e`。旧runのraw分解は未保存/未回収のまま。

linear H6 / 1.00 Å / STO-3G、12 spin orbitals。入力は[旧保存integrals](../../artifacts/resource_applicability/track_a_ax2b_h6_input_generation/2026-10-10/launch_v1/integrals.npz)だけ。
input SHA-256：`edd0a618f86011757cacae481eff44dc637c11a3f64c55b0cbfb7ffbe637e51d`。
明示kwargsは`{"truncation_threshold":1e-8}`のみ。implicit defaults final_rank=None / spin_basis=Trueは追加引数ではない。
returned order・coefficient cutoff0を保持。fragment削除、rank fallback、許容1e-10の変更なし。
PF prefix/delta/signal/state/sector referenceの評価は含まない。原12-mode配列をdecodeして一回decomposerに渡した。

## 一次artifactと監査

| 資料 | 役割 |
|---|---|
| [raw_decomposition.npz](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic/2026-10-10/launch_v1/raw_decomposition.npz) / [receipt](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic/2026-10-10/launch_v1/raw_decomposition_receipt.json) | 全19 lambda、19 raw g、one-body correction、truncation scalarをreturned dtype/valueで保存。検査/要約より先に保存 |
| [diagnostic_summary.json](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic/2026-10-10/launch_v1/diagnostic_summary.json) | 全19fragmentのnorm/偏差/hash、index15、全違反index、weighted係数差、表現整合性量 |
| [hypothetical_hermitization.npz](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic/2026-10-10/launch_v1/hypothetical_hermitization.npz) / [receipt](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic/2026-10-10/launch_v1/hypothetical_receipt.json) | gH/補正one-bodyの仮projection、chemist tensor、normal-order/antisym係数。診断専用・入力に採用しない |
| [input decode](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic/2026-10-10/launch_v1/input_decode_receipt.json) / [call](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic/2026-10-10/launch_v1/decomposition_call.json) | input/data SHA、完全な明示kwargs、call_count1、新旧execution区別 |
| [runtime環境](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic/2026-10-10/launch_v1/runtime_environment.json) | Python/packages/helper/config SHA、NumPy BLAS config、thread1、actual affinity[2] |
| [frozen manifest](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic/2026-10-10/launch_v1/frozen_diagnostic.json) / [original grant bytes](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic/2026-10-10/launch_v1/authorization_source.json) / [worker claim](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic/2026-10-10/launch_v1/worker_claim.json) | 結果を新source/plan/input/env/grant SHAに結合 |
| [parent terminal](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic/2026-10-10/launch_v1/terminal_status.json) / [worker terminal](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic/2026-10-10/launch_v1/worker_terminal.json) / [最終progress](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic/2026-10-10/launch_v1/progress_0004.json) | 原status、calls、raw/summary保存、wall/RSS観測、mandatory STOP |
| [保存bytes監査](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic_execution_v1/2026-10-10/saved_diagnostic_audit_v1.json) / [別execution inventory](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic_execution_v1/2026-10-10/execution_evidence_inventory_v1.json) | raw25ファイルのhash/layout/schema、fragment前後hash・lambda row、input/call/grant/terminal一致。数値要約を再計算していない |
| [保全監査](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic_execution_v1/2026-10-10/post_execution_preservation_audit_v1.json) | 旧source183・旧raw20、新science187・入力/env・既存dirty/untracked/root文書の保全 |

全raw保存25ファイル・1000580 bytes。欠測0。OpenFermion1.6.1 / NumPy1.26.4 / SciPy1.14.1 / Python3.11.0rc1。
source/外部helperのbytesは新診断sealへ結合。旧runの外部helper bytesを遡及認定したものではない。
新rawを旧失敗runで実際に返されたraw bytesと同一とは主張しない。

## 保存済み要約の観測値（新run限定）

actual rankは19、returned truncation valueは`2.693610667847679e-09`。
元検査量`||gH-g||F`が許容`1e-10`を超えるindexは**15,16,17,18**（0始まり）。全19fragmentを保存し、削除していない。

| index | lambda | ‖gH−g‖F | ‖g−g†‖F | weighted one-body projection差 | weighted antisym two-body projection差 |
|---:|---:|---:|---:|---:|---:|
| 15 | 1.16312136059e-07 | 1.10091250612e-09 | 2.20182501225e-09 | 7.65851368563e-17 | 1.65945159642e-16 |
| 16 | 5.51800590827e-08 | 2.38468248263e-09 | 4.76936496527e-09 | 8.8348728283e-17 | 1.65568182328e-16 |
| 17 | 5.3294894363e-08 | 2.00518768872e-09 | 4.01037537744e-09 | 4.42862239237e-17 | 1.3973916101e-16 |
| 18 | 1.49852308056e-10 | 9.6680082562e-07 | 1.93360165124e-06 | 4.44095301168e-17 | 1.82701844578e-16 |

index15の`||g||F`は`1.4142135623730947`、relative changeは`7.784626985724718e-10`。
raw hashは`1010a5dc2da1a4dc15636ddbc316c3ebc90d9e1881b430d513dbf569153384d8`、hypothetical post hashは`3e3f6a566435d60c2ead34a168398b8f7029b1fce460e686348f38d5423b6d1d`。
lambda、gの非Hermiticity、weighted係数への影響は別量として記録した。
表のweighted差は係数のFrobenius normであり、Hamiltonian作用素norm・signal error・総uの認定ではない。

| 表現整合性の保存量 | 新runの観測値 |
|---|---:|
| chemist transpose asymmetry l1 | 1.14951451136e-14 |
| chemist plain-square再構成差F | 5.40655173904e-11 |
| correctionとsource reorderingの差F | 0 |
| raw normal-order one-body残差F | 3.88060528487e-11 |
| raw antisym two-body残差F | 7.14553041181e-11 |
| hypothetical g projectionによるone-body係数差F | 3.90131214561e-16 |
| hypothetical g projectionによるantisym two-body係数差F | 7.8836359067e-16 |

正規順序の符号/antisym定義、conjugationを追加しない平方、chemist転置/spin抽出は[固定仕様](track_a_h6_df_hermitization_diagnostic_preparation_v1.md)を参照。
これらは新しい保存済み要約から転記した観測値。ここで再fit/再分解/再最適化していない。
残差の原因・許容性・DF/Hermitization政策の適否は未判断。

## 古典費用・実行範囲・残る未確定事項

parent wallは`2.107514850795269`秒。最終保存境界のworker ru_maxrssは`211460096` bytes。
上限はphase60/300/120秒・total480秒・AS8GiB・output32MiB・log64KiB、CPU ID2/worker1/BLAS1。
DF attempted1/returned1、fragment診断19、retry/resumeなし。新分子生成・state/solver/pilot/signal/sampling/compileは起動していない。
古典wall/RSSと量子回路資源を混同しない。本診断は量子資源評価を行っていない。

旧runのactual rank/raw/lambda/切断値/補正/fragment偏差は依然未保存/未回収で、新runから過去の数値を補わない。
DFの根本原因、representationの適切性、truncationと数値誤差の区別、weighted量の科学的な意味、政策変更の可否はGPTの独立レビュー事項。
H6入力受理false、all_original_hermitization_checks_satisfied=false。
N/G null・u未認定・UNDETERMINED。grantは一回消費済み、追加診断/変更/H6 pilotの自動認可なし。
raw/監査/source identityをGitHubへ公開・再取得確認した。上記remote記録とこの索引追記だけを後続commitへ公開し、mandatory STOPを維持する。
