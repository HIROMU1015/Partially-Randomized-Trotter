# 研究計画資料

このディレクトリには、DF部分ランダム化・有限RTE・RPE compiled cost研究の目的、背景、未解決点、解析手順および数値評価計画をまとめる。

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
