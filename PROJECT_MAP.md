# プロジェクト案内

最終更新：2026-09-25

このファイルは、人またはGPTがリポジトリ全体を読むときの入口である。研究内容の正本、
実装、検証コード、結果データ、発表資料を区別し、古い研究経路を現在の結論として読まない
ための案内をまとめる。

## 最初に読む順序

1. [`docs/research/研究概要・現状.md`](docs/research/研究概要・現状.md)
   研究目的、採用中の前提、現在の実装・検証段階、主要結果、未解決事項の最新要約。
2. [`docs/research/研究目的・研究課題.md`](docs/research/研究目的・研究課題.md) と
   [`docs/research/研究方法・解析手順.md`](docs/research/研究方法・解析手順.md)
   研究設計と解析手順の正本。
3. [`VALIDATION_STATUS.md`](VALIDATION_STATUS.md)
   各結果を再利用できるか、欠落データや再計算が必要かを示す状態表。
4. [`artifacts/validation_manifest.json`](artifacts/validation_manifest.json)
   検証結果、生成コード、テスト、成果物を結ぶ機械可読な証拠台帳。
5. [`docs/research/prevalidation_catalog_evidence_map.md`](docs/research/prevalidation_catalog_evidence_map.md)
   事前検証カタログのうち、実施したID・work packageと文書、artifact、runner、testの対応表。
6. 数値を使う場合だけ、対応する `docs/*_validation.md` と `artifacts/` のJSONを確認する。

`docs/research/研究ノート/` は意思決定の時系列記録であり、現在の仕様ではない。過去の暫定値が
後日のノートや研究概要で変更されている場合は、最新の研究概要と規範文書を優先する。

## 現在の研究段階

現在の中心課題は、DF Hamiltonianを決定論部分とランダム部分へ分けたpartial-$S_2$について、
有限RTE、RPEの信号半径・測定回数、1 shot当たりのコンパイル後回路コストを接続することである。
PF係数、有限RTE、ランダム回路コスト、短いRPE段の接続は限定条件で検証を進めている。
既存設定`CA/10`を暫定目標にすると$q_{\max}=32768$が必要で、従来の固定
$\delta=0.1,r=4,K=2$は長roundへ単純外挿できないことを確認した。その後、H4の
実行済み$\delta$窓でround別$(r_m,K_m)$を再探索し、$\delta=0.01,0.0125,0.02$に
行列検査を通るscheduleを構成した。さらに、短時間幅0.02--0.000390625で局所回路指標が
変わらないことと、イベント列長8、16、32への移送を検査し、$r\leq16$では1--3イベント、
$r=32$では4イベント補正を用いる中央RTEブロックcost proxyを接続した。この限定proxyでは
$\delta=0.02$が全6指標で最小となった。

2026-09-21に、広い検証backlogを一括実行せず、研究方向を選ぶGate S1を先に置く方針を採用し、
最初の`WP00 -> WP02 -> WP01-S`を実行した。H4 rank-12固定snapshotへPF入力を結び直し、
CA/CA/10/CA/100のround horizonを監査した。CA/10は既存3 schedule・56点行列検査を再利用できるが、
CA/100は既存3つの$\delta$がすべて経験的PF予算を超える。CA/10の条件付き比較では
$L_D=0$を拡張した解析的成分作用数proxyでscreen outし、$L_D=12$の点推定は$L_D=3$より
21.0%低かった。
ただし保守的な長$q$移送scenarioでは区間が重なるため、結論は未決定である。続くWP04では、
両候補に公平な$\beta$・$\alpha$再配分を与えると決定論endpointの点推定差は4.96%へ縮み、
5%・25%区間がともに重なることを確認した。共通の主要因は$\beta$、次いで$\alpha$再配分で、
$L_D=3$のcompiled-cost整合round schedule単独の利得は固定schedule比1.93%だった。WP03では
$C_D$、論文D6、支配固有位相係数だけを差し替えた18条件の選択が全て
$L_D=12,\delta=0.02$で変わらず、係数選択も区間重なりを解消しなかった。Gate S1では
これらを統合し、区間判定を「同点」でなく`undetermined`、最大の残存不確かさをfull controlled
interrogationの回路scope・構造とした。T4とT7を主軸、T1/T2/T5/T6を限定継続、T3を保留とし、
次の一件をWP06-aとした。WP06-aではsupport限定Gaussian completionが単一Z/ZZ eventのRZを
39--62%減らした一方、異なるsupportの長さ3列ではfull basis共有より15.0%増え、事前の5% triggerが
発火した。control・relative-phaseの現行方針は同値性を通過した。WP06-bでは独立trainingから、同一
元basisのsingleton runだけsupport限定へ置換するpolicyを固定した。未使用列長3, 6でRZ -10.67%、
CX -8.13%、total depth -2.25%、最大operator残差$1.34\times10^{-15}$だった。中央RTE差だけを既存
proxyの$q$ slopeへ加えたbridgeでは$L_D=3/12$の点順位が反転した。続くWP05-aでは選択policyを
complete controlled partial-$S_2$／Hadamard wrapperへ接続し、$q=1,2$較正から未使用$q=4$をRZ最大
2.29%で予測した。中央additive bridgeのfull-wrapper RZ残差も最大2.63%で5%基準を通過した。
固定WP04条件の点順位は$L_D=3$となったが区間は重なった。WP05-bでは$q=8$と比較対照
$\delta=0.01$へ拡張し、$\delta=0.02,r=32,q=8$の初回5%逸脱を独立32 trajectoryで再検証した。
再検証では選択policyのRZ誤差0.52%、全metric最大0.54%、full basisのRZ誤差0.83%となり、
5%基準を通過した。続くWP01-D/C07では$\alpha$・shot数を候補ごとに再最適化し、点推定で
$L_D=3$が$L_D=12$より13.92%低かった。5% local model区間は僅かに分離した一方、25%移送区間は
重なるため、頑健な方向判断は未確定である。G08で後半3 roundへのcost集中を確認し、M08の
$q=16,32$直接holdoutはselected RZ 2.466%、観測RZ最大3.286%で通過した。これらの実測幅による
再集計ではlocal区間が分離するが、直接domainは$q\leq32$で25%移送区間は重なるため、頑健判定は
変わらず、現比較は最終的な科学的優位性評価ではない。
2026-09-23のM06/L08では、同一trajectoryをoptimization level 2で再compileした。q=16,32 proxyは5%基準を通過したが、固定plan focused再集計の点推定差は8.42%、区間分離上限は1.881%となり、実測selected RZ discrepancy 2.340%で区間が重なった。従ってcompilerをまたぐlocal分離は未確立で、頑健判定は`undetermined_under_compiler_and_transfer_sensitivity`である。
続くN07/P03では不確かさをsampling、model bias、compiler、長q移送、状態準備、外部移送に分離し、状態準備をRZ相当/shotのパラメータとして再集計した。`L_D=3`は2,376 shot多く、共通準備costは常に点推定利得を縮める。点推定break-evenはopt1で約9,905万、opt2 focusedで約4,707万RZ相当/shotだが、opt2 focusedはP=0ですでに区間が重なり、compiler-robustな区間優位性は確立しない。WP11では11個のartifactをT1--T7へ統合し、T4/T7を主軸、T1を範囲変更、T3を保留とした。次の一件は`L_D=3`のopt2未測定`r=1,2,4,8,16`を埋めるall-r coherent opt2再最適化であり、外部instance pilotは棄却せずその後へ延期する。
固定条件、数値、成果物は
[`research_direction_prevalidation.md`](docs/research_direction_prevalidation.md)と
[`research_direction_ablation.md`](docs/research_direction_ablation.md)、
[`research_direction_pf_sensitivity.md`](docs/research_direction_pf_sensitivity.md)、
[`research_direction_gate_s1.md`](docs/research_direction_gate_s1.md)、
[`research_direction_structure_pilot.md`](docs/research_direction_structure_pilot.md)、
[`research_direction_sequence_policy.md`](docs/research_direction_sequence_policy.md)、
[`research_direction_full_scope.md`](docs/research_direction_full_scope.md)、
[`research_direction_full_scope_extension.md`](docs/research_direction_full_scope_extension.md)、
[`research_direction_decision_cost.md`](docs/research_direction_decision_cost.md)、
[`research_direction_late_round_proxy.md`](docs/research_direction_late_round_proxy.md)、
[`research_direction_compiler_transfer.md`](docs/research_direction_compiler_transfer.md)、
[`research_direction_uncertainty_break_even.md`](docs/research_direction_uncertainty_break_even.md)、
[`research_direction_wp11_synthesis.md`](docs/research_direction_wp11_synthesis.md)に記録する。

最終的な全RPE段の総コスト最適化と、決定論PFに対する最終的な優位性評価はまだ行っていない。
最新の到達点と次の検証は、必ず
[`研究概要・現状.md`](docs/research/研究概要・現状.md)で確認する。

## ディレクトリの役割

| 場所 | 役割 | 読み方 |
|---|---|---|
| `src/trotterlib/` | ライブラリ本体 | 現行実装と検証ロジック。分類は[`src/trotterlib/README.md`](src/trotterlib/README.md) |
| `scripts/` | 実行入口 | 検証runner、batch、診断、旧経路。分類は[`scripts/README.md`](scripts/README.md) |
| `tests/` | 自動テスト | API・数値恒等式・成果物schemaの回帰検査。科学的結論そのものではない |
| `docs/research/` | 研究方針の正本 | 概要、目的、方法、評価計画、研究ノート |
| `docs/` | 実装・検証の説明 | 各検証の条件、結果、限界、実装規約。索引は[`docs/README.md`](docs/README.md) |
| `artifacts/` | 計算結果と入力snapshot | JSON等の証拠、キャッシュ、途中状態。利用規則は[`artifacts/README.md`](artifacts/README.md) |
| [`partial_randomized_trotter_prevalidation_catalog.md`](partial_randomized_trotter_prevalidation_catalog.md) | 研究方向を選ぶための事前検証backlog | 実行層・Gate S1・条件付きbranchの索引。実施済み範囲との対応は[`prevalidation_catalog_evidence_map.md`](docs/research/prevalidation_catalog_evidence_map.md)を参照 |
| `BentoSlide構成案*.md` | 発表資料生成用の指示 | 研究の正本ではない。位置づけは[`docs/presentations/README.md`](docs/presentations/README.md) |
| ルートのPDF | 参考論文または発表資料 | 種別は[`docs/references/README.md`](docs/references/README.md)で確認 |

## 実装と検証の関係

```text
研究方針・条件
  docs/research/
        ↓
ライブラリ実装
  src/trotterlib/
        ↓
実行入口                 回帰テスト
  scripts/run_*.py   ↔    tests/test_*.py
        ↓
結果・入力snapshot
  artifacts/
        ↓
結果の説明と状態
  docs/*_validation.md
  VALIDATION_STATUS.md
  artifacts/validation_manifest.json
```

`scripts/run_*.py` はメインロジックの置き場所ではなく、引数を受け取り
`src/trotterlib/` の処理を呼び出して成果物を保存する入口である。検証名と同名の
ライブラリ、runner、test、文書、artifactを一組として読む。

## 共有サーバー向けexecution infrastructure

長時間検証を独立taskへ分割して安全に実行・再開する経路は、
`src/trotterlib/parallel_validation_executor.py`、`scripts/run_parallel_validation_batch.py`、
`tests/test_parallel_validation_executor.py`、
[`docs/server_parallel_validation_execution.md`](docs/server_parallel_validation_execution.md)を一組として読む。
これは実装・運用基盤であり、新しい科学的検証結果や最終総cost評価ではない。

WP11が選択したM06-F計算経路は、`src/trotterlib/research_direction_full_opt2.py`、
`src/trotterlib/research_direction_full_opt2_completion.py`、
`src/trotterlib/research_direction_full_opt2_extension_analysis.py`、
`scripts/run_research_direction_full_opt2_compute.py`、
`scripts/run_research_direction_full_opt2_analysis.py`、
`scripts/run_research_direction_full_opt2_completion.py`、
`scripts/run_research_direction_full_opt2_extension_analysis.py`、
`tests/test_research_direction_full_opt2.py`、
`tests/test_research_direction_full_opt2_completion.py`、
`tests/test_research_direction_full_opt2_extension_analysis.py`、
[`docs/research_direction_full_opt2.md`](docs/research_direction_full_opt2.md)を一組として読む。
初期36 taskとfresh-32拡張15 taskは51/51で完了し、両gate通過後のcoherent再最適化も完了した。
compute resultは`artifacts/research_direction_full_opt2/2026-09-24/`、最終監査・解析は
`artifacts/research_direction_full_opt2/2026-09-25/`に置く。

## ファイルの状態区分

### 現行の正本

- `docs/research/研究概要・現状.md`
- `docs/research/研究目的・研究課題.md`
- `docs/research/研究方法・解析手順.md`
- `docs/research/数値実験・評価計画.md`
- `VALIDATION_STATUS.md`
- `artifacts/validation_manifest.json`

### 現行実装・検証

- `src/trotterlib/` のDF、RTE、RPE、compiled-cost関連モジュール
- 対応する `scripts/run_*.py`、`tests/test_*.py`、`docs/*_validation.md`
- manifestに登録され、statusと限界が明示されたartifact

### 歴史的・補助的資料

- `abe_trotter_project.ipynb`：旧来の高次PF解析ノートブック
- `Partial Randomized Study Protocol.md`：初期計画と意思決定の履歴
- `codex-inst.md`：過去の実装依頼メモ
- `README_partial_randomized_pf.md`：旧screeningを含む実装経路の説明
- `docs/main_audit_20260801.md`：特定時点の監査記録

これらは削除していないが、現在の研究方針や最新結果を確定する根拠には使わない。

## GPTが回答・資料作成するときの確認事項

- 事前検証カタログの実施状況は、[`prevalidation_catalog_evidence_map.md`](docs/research/prevalidation_catalog_evidence_map.md)から専用文書・artifact・testまで追跡する。
- 「理論上の関係」「ローカル検証済み」「実装のみ」「未検証」「最終結論」を分ける。
- 数値には、分子、距離、basis、DF rank、$L_D$、時間幅、Taylor cutoff、検証範囲を添える。
- `C_use`を厳密上界と呼ばない。実行したdelta窓上の経験的包絡である。
- H4/H6の係数をH12へ外挿しない。
- PFの決定論的biasとQPE/RPEの統計誤差を混同しない。
- dirty worktreeの結果やローカルテストを、公開済み・CI固定済みの証拠と表現しない。
- スライド構成案や研究ノートの単独記述より、研究概要、検証文書、manifestを優先する。

## 保守規則

研究上の決定や検証結果が変わった場合は、次を同じ変更で更新する。

1. `docs/research/研究概要・現状.md`
2. 対応する研究方針文書または検証文書
3. 当日の研究ノート
4. 証拠構成が変わる場合は `artifacts/validation_manifest.json`

新しい検証を追加するときは、可能な限り同じ語幹で
`src/trotterlib/`、`scripts/`、`tests/`、`docs/`、`artifacts/`を対応させる。
