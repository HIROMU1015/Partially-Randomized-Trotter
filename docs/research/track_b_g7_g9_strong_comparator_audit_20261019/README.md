# G6/G7/G9：強い既存対照との比較可能性監査

2026-10-19 JST。状態：**`STATIC_COMPARABILITY_AUDIT_COMPLETE_WITH_CONTRACT_GAPS`**。

**既存G9 v2だけで、登録P5のGeneral Full・Closed P5・matched CTSのmatched-native比較を引用できる。** その条件ではClosed P5がT-primary、CTSがCXで有利という既存結果を保つ。G7の4方式・helper/query-modelの結果だけから、強い対照を含む一般資源優位を主張することはできない。G7とG9のNやnative総量を直接横断比較していない。

## 成果物

| 資料 | 内容 |
|---|---|
| [G7/G9比較条件対応表](comparison_conditions.md) | finite operator、law、bias/confidence、provider/direct/helper、費用・accessの一致と差 |
| [強い対照の既存成果・限界](existing_strong_baselines.md) | 元報告の表示表・既存affine判定を引用。再会計・再ランキングなし |
| [不足証拠・最小追加検証案](evidence_gaps_and_minimal_plan.md) | 引用で閉じる範囲、必要時だけ行うpolicy/費用接続、STOP・承認 |
| [Decision Handoff](handoff.md) | B3/C1完了後に主張できる範囲と次の担当 |
| [固定証拠索引](evidence_index.json) / [SOURCE位置索引](source_inventory.json) | fixed commit/path/bytes/SHA256、読解範囲、AST位置 |
| [比較条件・引用索引](comparison_inventory.json) | 条件の区分と既存判定へのJSON pointer。新しい科学判定ではない |
| [管理検査](verification.json) / [公開照合](publication_receipt.json) | リンク・保全・公開bytes/SHA256。数値correctnessの再検証ではない |

## 固定入口と証拠境界

- [G7結果][G7R]：`a689694080f4b7600fe67cf77d841d1cbbf04503`。科学SOURCEは`ab2549f41b3546fb3940342a2162dd9ee93699c4`。
- [G9 v2結果][G9R]：`c95736fd2990f5ef6dd1cb5866421fd78bb28687`。科学SOURCE Sは`0ef2b92738750a3c0d187743dd4b7c9b927db802`、実行authorization-only Aは`5ed597d8163af002d315a9e2b9eee0e9f14edce2`。
- [W2c-A][A]：`864097c1285f1f06399040cf0f7c0f42334d9252`。本監査の独立branchはこのcommitを基点とし、他branchをmerge/rebaseしない。
- [共通指示2026-10-17][COMMON]：実ローカル原本と固定公開版の同一bytes、§8.4を確認。過去の全文読解は同一hashで再利用した。

両者のP5 anchorはp=(1/5,3/10,1/2)、x=5/7、m=5。同じfinite **first operator moment**が対象で、exact exponentialやQPE全体の精度ではない。G9には明示3-system-qubit synthetic providerがある。分子/geometry/basis/DF rank/split/δ窓の新評価はない。known development・I1 contextであり、held-out/独立再現/分子代表性ではない。

G7のP5入力とG9のClosed P5方式は別概念。Closed P5は実装済みm=5 fast pathであり、一般mの形式P5＋tailを実装済みとする根拠にしない。General FullとClosed P5は同familyの理想ensembleを実現し、finite-bit lawやbudgetは異なる。

## 実施したこと・していないこと

実施は固定資料・SOURCEの読解、AST/text/hash・リンク・Git保全の管理検査、文書・metadata公開のみ。既存の表示表・保存affine判定・既存監査PASSを**引用**し、差・比・N・期待値を新計算していない。G9の25 checksや数式・native検証を再実行していない。

**0件**：科学SOURCE変更/import/実行、新SOURCE/test/契約/authorization/grant/marker/claim、sampling/profiling/列挙/保存科学値再解析/再ランキング、回路build/compile/synthesis、非自明証明、先行研究の再調査、重要科学レビュー、研究方針採択。

Codex AのB3/C1作業領域、SOURCE/schema/契約/provider、live job/index/lock/outputにはアクセスせず、完了を待っていない。B3/C1の完成状態や新実費結果を確認済みとはしない。専用checkout内の**新規監査文書・metadataだけ**を変更し、親worktree・旧結果・消費済みclaim・既存STOP/CLOSED/HOLDを保全する。CPU32・available RAM約49 GiB・/tmp空き約376 GiBを観測し、独立read/hash/public read-backを効率化した。共有設定・venv/cache・他processは変更しない。

公開・保全確認後は**mandatory STOP**。次はユーザーが既存G9の限定比較を採用範囲として十分とするか、特定の追加費用判断が必要かを決める。B3/C1契約・新科学条件・実行承認を本監査が代行しない。

<!-- FIXED_REFERENCES -->

[A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/864097c1285f1f06399040cf0f7c0f42334d9252/docs/research/track_b_g6_w2c_total_cost_conditions_20261018/README.md
[COMMON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7eff8a33bef4ee1f29698ab10c6da46988dc8db5/docs/research/qc_relative_form_finite_step_audit_20261017/inputs/research_project_instructions_unified_2026-10-17.md
[G7R]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a689694080f4b7600fe67cf77d841d1cbbf04503/docs/tracks/algorithm_codesign/g7_results_and_gpt_handoff_20261010.md
[G9R]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c95736fd2990f5ef6dd1cb5866421fd78bb28687/docs/tracks/algorithm_codesign/g9_v2_results_and_gpt_handoff_20261010.md
