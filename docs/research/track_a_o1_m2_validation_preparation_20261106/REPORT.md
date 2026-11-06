# Track A（O1）M2：保存core・完全残差の技術調査と最小検証準備

2026-11-06 JST。**新fitなしで参照できる保存coreを4件特定し、元入力とともに公開取得・bytes一致を確認した。M2の完全残差接続、元H状態での性能、全RPE費用は未検証であり、科学実行の準備完了とはしない。**

## 成果物と作業環境

| 成果物 | 内容 |
|---|---|
| [M2_EXISTING_CORE_MAP.md](M2_EXISTING_CORE_MAP.md) | 通常CDF7、PR-aware update005/064、RC2比較端点の固定SOURCE・結果commit・保存NPZ・元H対応 |
| [M2_RESIDUAL_CONNECTION.md](M2_RESIDUAL_CONNECTION.md) | 完全残差の符号・一体/scalar・型の違い、既存D/RTE/native/状態/会計の接続不足、最小比較とSTOP条件 |
| 本REPORT.md | 到達点・準備状態・次の判断 |

SERVER 2（GPUA100-gleap02）。専用branch `codex/track-a-o1-m2-validation-preparation-20261106`、同名末尾の独立worktreeで作業した。M1・main・既存研究branch・他Trackの作業領域は変更していない。

基点は[O1 `aeb9d1847f9d500f2389db12b384bac6087117a9`][O1]。必須の4文書を継承し、[3環境統合 `3571477093441ac4852166fe829f8ccc5db5ec4f`][I]の台帳・再利用profileと[文献調査 `e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8`][S]を再利用した。新しい全棚卸し、科学的優先順位や独立新規性の判断はしていない。適用AGENTS/PROJECT_MAP・研究概要と正式共通運用指示2026-10-25の許可・保全・再利用・STOP規約に従う。

## 確認できた保存入力と技術事項

| 候補・元入力 | 固定取得commit | 主要な確認 |
|---|---|---|
| 通常CDF7 `fit/final.npz` | `794875af24cb954985ac5e43fc799421f5d8736a` | J/U header `(7,4,4)`、fit科学SOURCE `c13d6f82a22d497b3334eaaa40d8430a5010f70f` |
| update005 `update005_candidate.npz` | `aced1647b81e8d28091058cdc7b6e3e1ee54896a` | J/U header `(8,4,4)`、科学SOURCE `980b514c81eb415fe053f558d7d3c0b8cfb5b9b4` |
| update064 `terminal.npz` | `ea54b9b8ba42ff993d803ec6f5c57e4e622fdee6` | J/U header `(8,4,4)`。同commitの`update064_candidate.npz`とbyte同一、physical metadataのcandidate hashも一致。科学SOURCE `025d495166311a58ec9c24c43f07ed6e40f65f42` |
| RC2比較端点 `rc2/final.npz` | `f6bb336b41cddface5647c6b1acf23c330f32637` | J/U header `(8,4,4)`。元fit SOURCE `e2f7dfd44357510d54ab9c6c9c5032e0072ab376`と後続出力復旧を区別 |
| 元h/g/constant `prepare/input.npz` | `aced1647b81e8d28091058cdc7b6e3e1ee54896a` | header `(4,4)`／`(4,4,4,4)`／scalar、既存input hashとbytes識別。保存直鎖H4・1Å・STO-3Gの入力 |

以上5 fileをGitHubからbase64取得し、local固定Git blobとの内容一致を確認した。配列を数値decodeせず、NPY headerのshape/dtypeとfile bytesだけを扱った。入力の科学的再評価、DF/PF/RTE再計算、性能再ランキングではない。固定path・各file hash・元Hの積分規約と関連stateは[core map](M2_EXISTING_CORE_MAP.md)に明記した。

原g補正κ・核scalarを共有するgeneral signed-J coreの既存再構成がある。旧LD2 tailはcandidate内のframe残部であり、**元H−全coreという完全残差とは異なる**。また保存tensor residualはg_fit−g_originalなので、Rの符号としてそのまま使えない。一体補正と電子identityの対応も必要である。

general-K/native、StrongBuilder、RTE lawとnormalization、Pauli collector、既存RPE会計の部品は再利用候補になる。一方、signed Jを標準square DFへ型変換するだけでは足りず、全coreのDと任意の完全残差をつなぐ入力・phase・referenceの境界が不足する。新builder/mapper/会計framework一式を作り直す必要があるとは判断していない。

原SOURCE980…の下流はcandidate Hのgroundを使用していた。保存physical ground fileも同結果commitに存在するが、旧candidate信号・shot・価格を元H groundのM2結果へ流用できない。保存physical数値stateは将来の診断reference候補であり、算法にE₀/gap/古典vectorを無料で与える条件ではない。

## 最小科学検証の準備状態

| 項目 | 現在の状態 |
|---|---|
| 保存coreと元入力の取得 | **準備済み**。新fitを前提にしない |
| core正式選択 | **未決定**。4候補の情報価値と接続作業をcore mapに整理。性能順位は付けていない |
| 完全残差の生成・同一H確認 | **未実施**。符号、一体/scalar、full-Fock/sector対応の最小adapterと検査が必要 |
| 元H state上のPF/finite mean/半径 | **未検証**。旧own-DF ground結果は代用しない |
| full-wrapper価格・shot・全RPE成功/費用 | **未資格**。旧r32/r128/K0、旧CA・phase/failure政策や長q価格をM2へ自動採用しない |
| 科学実行・新実装 | **未承認・未開始**。完全な実行契約を無理に作成していない |

最小比較の対象は、ユーザーが選んだ保存core **1件**でもよい。最初に元H/stateと全core＋完全残差の対応を閉じ、その後必要なPF/RTE/半径、native full価格、全round N/Gを確認する。新fit・全rank sweep・新cost-aware optimizerを追加する計画ではない。

## 強baselineと残る科学的問い

UWC、collected Pauli、適用可能なDF/RC-DF、既知factorization-residual PR、同candidate強Dへ、同じ元H情報・core候補・error/価格情報と合法control/fusion・PF最適化機会を与える。近似core-only対照も元Hへの誤差・state・成功条件の転送を支払う。別の近似H ground-energy taskを比較して済ませない。[既存比較条件][prior]

未解決なのは、圧縮で軽くなるD費用より、完全残差λ_R・PF誤差・normalization・shot・basis/control・古典前処理が高くならないかという点である。保存DF予算や旧nominal支持だけでは答えられない。RZ/CX/T/depthは指標別に扱い、局所normや単回価格を最終目的にしない。

研究Cの新cost-aware fitに対する既存NO-GO、旧SOURCE/結果/STOP、singleton等の一時終了は維持する。M2という名称で停止済み研究を再開していない。将来、適合coreがなく新fit不可欠と判明した場合は、その理由を報告してSTOPする。

## 次の担当・公開・mandatory STOP

- **ユーザー**：保存coreを1件選ぶか、どの比較に情報価値を認めるか判断する。元H/state reference、既存条件から採る時間・精度/成功確率・主資源指標を選ぶ。必要なら最小adapter実装と残差生成・科学検証の範囲を別途明示許可する。重要科学レビューの開始許可は科学実行許可と独立。
- **Codex（新しい依頼後のみ）**：選ばれた対象に限り、既存部品の最小接続と必要な確認を行う。新fit・新selector・全候補探索を自動追加しない。

公開対象はM2専用directoryの本3文書のみ。文書参照・固定path・変更範囲を技術検査し、専用branchのremote commit SHA、3文書の取得可能性・内容一致を確認して停止する。公開SHAと照合結果は最終報告に示す。

**Mandatory STOP：本3成果物の公開・取得照合後に停止。** 新科学計算、配列数値再解析、state/residual生成、sampling/RNG replay、shot再計算、回路build/compile/synthesis、fit、コード実装、重要科学レビューは実施していない。次の科学作業を自動開始しない。

[O1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/aeb9d1847f9d500f2389db12b384bac6087117a9/docs/research/o1_hamiltonian_partition_preflight_20261106
[I]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105
[S]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106
[prior]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106/PRIOR_ART_COMPARISON.md
