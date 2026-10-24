# 研究計画資料

[C1 raw CAR frame縮約修正](research_c_c1_frame_repair_20261025.md)：旧rawの起動／読込順に依存する人工並列不一致を再現し、explicit matmulへ変更。人工63件＋29 subtests・別プロセス64 frame比較PASS。[未承認引継ぎ](research_c_c1_frame_repair_handoff_unapproved_20261025.json)、証拠 `artifacts/research_c_c1_frame_repair/2026-10-25/`。H4追加解析・fit0、科学的資格は未完了、旧STOP保全・mandatory STOP。

[C1 Stage 1互換性修正版の科学実行](research_c_c1_stage1_compat_execution_20261025.md)：新しい1 fitはrank8で収束、full256 frame0同値性FAILでSTOP。κ/J/U・baseはPASS、CDF/XDFの物理sector誤差は未測定。科学call107/147、retry/resume0、証拠 `artifacts/research_c_c1_stage1_compat_science/2026-10-25/`。旧SOURCE／run／STOPを保全し、追加科学処理・Stage2/3へ進まない。

[C1 Stage 1：ffsim／SciPy互換性修正](research_c_c1_stage1_compatibility_20261025.md)：既存2D logmを順に呼ぶ専用bridge、venv起動修正、人工59件＋29 subtests PASS。[新未承認契約](research_c_c1_stage1_compatibility_contract_unapproved_20261025.json)とSOURCE／lockは `artifacts/research_c_c1_stage1_compatibility/2026-10-25/`。実分子科学call0、旧結果／STOP保全、別の1 fit明示承認待ち・mandatory STOP。

このディレクトリには、DF部分ランダム化・有限RTE・RPE compiled cost研究の目的、背景、未解決点、解析手順および数値評価計画をまとめる。

実装、runner、テスト、成果物を含むリポジトリ全体の区分は
[`../../PROJECT_MAP.md`](../../PROJECT_MAP.md)を参照する。

初めてこの研究を確認する場合や、Codexを使って発表・共有資料を作る場合は、まず
[研究概要・現状](研究概要・現状.md)を読む。この一冊で現在の研究段階、採用済みの
方針、主要な検証結果、未決定事項および証拠の読み順を確認できる。

## 文書の役割

研究Cの[C4終了・成果引継ぎ](research_c_c4_closeout_handoff_20261024.md)はユーザー採択済みの一時終了と限定結論・B2等の資産・再開条件の正本。[C1圧縮DF接口調査](research_c_compressed_df_interface_survey_20261024.md)は外部実装・一般対角fragment・誤差/位相・既存SOURCEの再利用と未承認の次案を整理する。C1方式の採択・新実装・科学実行は含まない。

Gaussian singletonの[改善候補設計](pr_gaussian_singleton_synthesis_design_20261024.md)は、support1／support2を区別した選択部分空間Givens案と、既存helperの再利用・同値性条件・未承認の48-wrapper対照案。静的調査のみで、削減量未測定・候補未実装。

限定F2の[保存case寄与分析](pr_algorithm_improvement_phase2d_f2_attribution_20261024.md)は、8分類表のP・条件付き費用・寄与とexact全体照合をまとめる。SP／PS差の記述分解、限界、次の改善候補を明示し、新build／compileは行っていない。

限定F2の[回路資源差の原因調査](pr_algorithm_improvement_phase2d_f2_resource_causes_20261024.md)は、固定SOURCEと保存caseの静的照合による技術メモ。Gaussian往復・共有条件・K0の成分選択と、未測定の寄与を区別する。新しい数値集計・build／compileは行っていない。

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
