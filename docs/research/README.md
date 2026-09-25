# 研究計画資料

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
| [事前検証カタログ実施証拠索引](prevalidation_catalog_evidence_map.md) | 実施したカタログID・work packageとGitHub上の証拠の対応 |
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
| 4 | [P-A joint synthesis先行研究監査](pa_joint_synthesis_prior_art_audit.md) | P-A v1の検索範囲、closest prior art、限定novelty statement、v2との境界を記録する。 |
| 5 | [P-A blind transfer事前登録](pa_joint_synthesis_blind_validation_preregistration.md) | 実行前にH5 physical transferとH4 compiler transferの条件・gateを固定した記録。 |
| 6 | [P-A blind transfer検証結果](../research_direction_joint_synthesis_blind_validation.md) | 両stratumの完了、固定gate、結果、判断、scopeを記録する。 |
| 7 | [P-A v1形式化・機構監査](pa_joint_synthesis_v1_formalization.md) | 目的関数、DP、計算量、同値性条件と、完成済み証拠が一run一segment・order 0へ退化している制約を記録する。 |
| 8 | [P-A非退化mechanism validation事前登録](pa_joint_synthesis_mechanism_validation_preregistration.md) | 明示的一区間baseline、forced support変化、order 2 stream、固定gateと停止規則を計算前に記録する。 |
| 9 | [P-A非退化mechanism validation結果](../research_direction_joint_synthesis_mechanism_validation.md) | 30/30 taskで分割・plan差・追加RZ改善が0だった結果と、P-A停止・P-C復帰判断を記録する。 |
| 10 | [P-C geometry tracking・breakdown事前登録](pc_geometry_tracking_breakdown_preregistration.md) | 8 geometry、2 policy、blind region、4 pair、7 gate、固定停止規則を計算前に記録する。 |
| 11 | [P-C geometry tracking・breakdown結果](../research_direction_geometry_tracking_breakdown.md) | 16/16 task、prefix変更0、stretch予測破れ、current H4 family停止を記録する。 |
| 12 | [P-D energy・tail Pareto事前登録](pd_energy_tail_pareto_preregistration.md) | 固定5公式、`L_D=3` development、`L_D=4` blind、7 gateと停止規則を記録する。 |
| 13 | [P-D energy・tail Pareto結果](../research_direction_energy_tail_pareto.md) | energy-onlyとtail-awareのblind逆転、条件付き候補判断、後続の現実化gateを記録する。 |
| 14 | [P-D現実化Go/No-Go事前登録](pd_realization_go_no_go_preregistration.md) | 負時間D1、内部`H_D` D2、fresh `L_D=5` D3、固定閾値、終了後の停止規則を記録する。 |
| 15 | [P-D現実化Go/No-Go結果](../research_direction_pd_realization.md) | D1--D3全通過、正式主研究候補化、未評価範囲、研究再設計の停止点を記録する。 |
| 16 | [P-D主研究契約](pd_primary_research_contract.md) | S0の主RQ、固定H4 scope、B0/B1a/B1b/B2/B4、Case A--D、S1後の強制停止を固定する。 |
| 17 | [P-D既知baseline](pd_prior_art_and_baselines.md) | 既知のabsolute-tail-time modelとS1で新たに問うfinite/internal/construction差を分離する。 |
| 18 | [P-D S1事前登録](pd_s1_fair_comparison_preregistration.md) | 共通時間・位相予算、nested/native、regret、K4 trigger、境界規則を結果前に固定する。 |
| 19 | [P-D S1結果](../research_direction_pd_fair_comparison.md) | B1b/B2/B4一致、Case C/D不成立、B1a上限依存によるCase B＋undetermined停止を記録する。 |

## 研究ノート

日付ごとの実装方針、判断理由、検証結果および次の課題は、
[研究ノート](研究ノート/README.md)に時系列で記録する。研究ノートは変更の
経緯を残すための資料であり、現行仕様は上記の主資料、検証可能性と保証statusは
`VALIDATION_STATUS.md`およびmachine-readable artifactを正本とする。

## 実装・検証資料

現行実装の保証範囲、検証状況およびAPI規約は、研究資料の閲覧順序とは分けて次を参照する。

- [検証状況](../../VALIDATION_STATUS.md)
- [事前検証カタログ実施証拠索引](prevalidation_catalog_evidence_map.md)
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
