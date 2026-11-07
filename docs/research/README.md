# 研究計画資料

複数ブランチにまたがる成果と後続記録は[研究結果への統一索引](RESEARCH_RESULTS_INDEX.md)から辿る。
以下の計画・停止履歴は、このcheckoutに保存された時点の記述として読む。

このディレクトリには、DF部分ランダム化・有限RTE・RPE compiled cost研究の目的、背景、未解決点、解析手順および数値評価計画をまとめる。

実装、runner、テスト、成果物を含むリポジトリ全体の区分は
[`../../PROJECT_MAP.md`](../../PROJECT_MAP.md)を参照する。

初めてこの研究を確認する場合や、Codexを使って発表・共有資料を作る場合は、まず
[研究概要・現状](研究概要・現状.md)を読む。この一冊で現在の研究段階、採用済みの
方針、主要な検証結果、未決定事項および証拠の読み順を確認できる。

## 文書の役割

| 種類 | 正本とする内容 |
|---|---|
| [研究概要・現状](研究概要・現状.md) | 現在地と資料作成用の統合要約 |
| 下記の主資料4本 | 研究目的、理論的位置付け、解析方法、評価計画 |
| `docs/*.md`の検証資料 | 個別検証の方法、条件、数値結果、限界 |
| `VALIDATION_STATUS.md`とmanifest | 外部再現性、証拠status、利用禁止結果 |
| `artifacts/` | machine-readableな数値結果とprovenance |
| 研究ノート | 判断の経緯。現行仕様の正本ではない |

主資料4本だけで研究設計を追えるようにし、補足Q&Aは式の直感、記号の違いおよび
具体例を確認するために用いる。

## 推奨する閲覧順序

### 主資料

| 順序 | 資料 | 説明 |
|---:|---|---|
| 1 | [研究目的・研究課題](研究目的・研究課題.md) | 研究目的、現段階の範囲、最終目的関数およびResearch Questionsを示す。 |
| 2 | [先行研究と未解決点](先行研究と未解決点.md) | PR論文の理論・resource modelと、本研究で扱う有限・compiled-cost上の未解決点を整理する。 |
| 3 | [研究方法・解析手順](研究方法・解析手順.md) | 記号、入力、探索変数、誤差・shot・期待compiled costの依存関係と最適化手順を定める。 |
| 4 | [数値実験・評価計画](数値実験・評価計画.md) | 現行実装で評価できる範囲、固定条件、実験手順、比較方法および判定基準を定める。 |

### 補足資料

| 順序 | 資料 | 説明 |
|---:|---|---|
| 1 | [研究目的・研究課題 補足Q&A](研究目的・研究課題_補足QA.md) | normalization、attenuation、位相誤差換算の直感を補う。 |
| 2 | [先行研究と未解決点 補足Q&A](先行研究と未解決点_補足QA.md) | PR論文の各量の意味と有限RTEへの対応を具体化する。 |
| 3 | [研究方法・解析手順 補足Q&A](研究方法・解析手順_補足QA.md) | 外側・内側最適化、信号半径および数値例を詳しく説明する。 |

## 研究ノート

日付ごとの実装方針、判断理由、検証結果および次の課題は、
[研究ノート](研究ノート/README.md)に時系列で記録する。研究ノートは変更の
経緯を残すための資料であり、現行仕様は上記の主資料、検証可能性と保証statusは
`VALIDATION_STATUS.md`およびmachine-readable artifactを正本とする。

## 実装・検証資料

現行実装の保証範囲、検証状況およびAPI規約は、研究資料の閲覧順序とは分けて次を参照する。

- [検証状況](../../VALIDATION_STATUS.md)
- [有限RTE規約](../rte_conventions.md)
- [finite-RTE信号近似の小規模検証](../finite_rte_signal_validation.md)
- [RPE短roundの信号・shot・回路cost接続検証](../rpe_round_cost_connection_validation.md)
- [RPE短roundの仮想Hadamard測定・失敗確率検証](../rpe_hadamard_failure_validation.md)
- [$q=8$ Hadamard 1 shot cost proxy・resource接続検証](../rpe_hadamard_proxy_resource_validation.md)
- [RPE位相誤差・失敗確率配分の感度検証](../rpe_allocation_sensitivity_validation.md)
- [RPE限定4段の集計・失敗確率検証](../rpe_four_round_accounting_validation.md)
- [RPE 4段の物理信号・分枝復元検証](../rpe_four_round_phase_validation.md)
- [目標精度からのRPE round範囲・固定設定診断](../rpe_target_round_horizon_validation.md)
- [RPEのdelta候補とround別有限RTE schedule検証](../rpe_delta_round_schedule_validation.md)
- [delta scheduleの中央RTE compiled-cost proxy検証](../rpe_delta_compiled_cost_validation.md)
- [ランダム回路compiled-cost加法モデルのpilot検証](../random_circuit_cost_validation.md)
- [RTE境界補正cost modelのpilot検証](../rte_boundary_cost_validation.md)
- [RTE境界補正のfragment層別・高統計検証](../rte_boundary_pair_validation.md)
- [階層compiled-cost modelの拡張holdout検証](../hierarchical_cost_validation.md)
- [RTE connected-cluster運用cost推定の独立holdout検証](../rte_connected_cluster_cost_validation.md)
- [ランダムRTE回路compiled-cost近似検証の統合結果](../rte_compiled_cost_validation_summary.md)
- [PF誤差surrogate・CPU摂動・QPE分枝のholdout検証](../pf_delta_validation.md)
- [H-chain系サイズ・実行可能delta窓におけるPF係数検証](../pf_c_system_size_validation.md)
- [RPE resource accounting](../rpe_resource_accounting.md)
- [RTE一次資料の版管理](../rte_source_versions.md)

## main集約の版差と原SOURCE（2026-11-03）

[索引の統合・SOURCEの版差・保存原版](main_collection_versions_20261103.md)から、同名ファイルの各公開版とmainの配置を確認できる。
過去のSOURCE・test・結果は元snapshotの条件で読む。科学実行と元の研究testの再実行は行っていない。

## 2026-11-07 新A CT-3A：全support strict fallbackの静的調査

[REPORT](pr_reference_ct3a_20261107/REPORT.md)、[候補比較](pr_reference_ct3a_20261107/FALLBACK_POLICY_CANDIDATES.md)、[停止性・実出力T上界](pr_reference_ct3a_20261107/POLICY_TERMINATION_AND_BOUNDS.md)、[次判断の不足条件](pr_reference_ct3a_20261107/CT3B_MINIMAL_SCOPE.md)。strict SU(2)生成集合とideal SK十分定数を記号的整理したが、初期net coverage・有限精度保証が不足しTHEORETICAL_ONLY。READY候補なし、全support実価格未資格。新実装・net/word生成・科学計算・価格評価0、旧SOURCE/結果/STOP不変更、mandatory STOP。次担当GPT／ユーザー。

## 2026-11-07 新A CT-4A：大きいqの有限T比較preflight

[PREFLIGHT](pr_reference_ct4a_20261107/PREFLIGHT.md)、[SCOPE_PROPOSAL](pr_reference_ct4a_20261107/SCOPE_PROPOSAL.md)。保存q128/r1・rare/nonrare各j0・B′/Reference×X/Yの8構成、一様epsilon1e-11を提案。q128 whole QPY/完全literal key表は未保存で、承認後のnative準備compile→実key固定→missing合成が必要。今回は静的確認のみ、新合成/compile/科学数値再解析/T価格集計0、旧SOURCE/結果/STOP不変更。CT-4A完了でSTOP、CT-4B未承認・未開始。次担当ユーザー。

## 2026-11-07 新A CT-4B：q128の有限strict T比較

[REPORT](pr_reference_ct4b_20261107/REPORT.md)、[SCOPE](pr_reference_ct4b_20261107/SCOPE.md)、[SOURCE/入力/raw/STOP](../../artifacts/pr_reference_ct4/2026-11-07/README.md)。H4/STO-3G/DF12・Partial S2/LD3/h=.02/K2・保存q128/r1の2 trajectory・X/Y・epsilon1e-11でReferenceの実TはB′比1.865933%/1.881674%減。299 word/25 local/保存199shot条件PASS。有限結果であり全support/18round総T/最小T/T-depthへ一般化しない。旧結果/STOP保全、mandatory STOP。次担当GPT／ユーザー。

## 2026-11-07 新A CT-5B：構造条件と必要価格境界の静的調査

[REPORT](pr_reference_ct5b_20261107/REPORT.md)、[構造保証](pr_reference_ct5b_20261107/STRUCTURAL_CERTIFICATES.md)、[有限key](pr_reference_ct5b_20261107/FINITE_KEY_AND_UNKNOWN_PRICE.md)、[条件式](pr_reference_ct5b_20261107/CONDITIONAL_T_ADVANTAGE.md)、[次判断](pr_reference_ct5b_20261107/NEXT_SCOPE.md)。D/frameのSOURCE構造とcompiler contextを分離し、共通frame価格の記号的取消し、中央4 key、product/event/vacuumの境界を整理。native価格移送は条件付き、全support policy/T優越は未資格。新数値再解析/合成/compile/価格0、旧SOURCE/結果/STOP不変更、mandatory STOP。次担当GPT／ユーザー。
