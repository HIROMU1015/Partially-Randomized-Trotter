# プロジェクト案内

## 研究B outer/RTE固定22行の実行準備（2026-10-25）

[固定契約・SOURCE・人工119件・単一command](docs/research/research_b_outer_rte_22row_execution_ready_20261025.md)。SOURCE `26e8734aff3969dc8077cb058826d12394be28e1`、入口 `scripts/run_b2d_outer_rte22.py`、接続 `b2d_evaluator/outer_rte_execution.py`、人工test `tests/management/test_b2d_outer_rte22.py`。候補別r/K、22行・1239 dispatch・274 key・RTE各96・110 expectations。Read-only preflight PASS、科学計算未実行。旧16行の結果・STOPを保全し、別途明示許可後だけ実行する。

## 研究B outer固定16行の科学実行完了（2026-10-25）

[結果・残差・SOURCE・STOP](docs/research/research_b_outer_splitting_stage1_execution_20261025.md)。SOURCE b85dd3284、全gate先行、16行・906 dispatch・490 key・80 expectationsを完了、retry0。原本9ファイル・CSV・process logは `artifacts/research_b_outer_splitting/20261025/`。Corrected signal改善を限定観測したが、raw改善・総資源優位は認定しない。追加科学計算へ進まずmandatory STOP。

## 研究B outer 16行最終実行準備（2026-10-25）

[固定SOURCE・未承認科学契約・実行command](docs/research/research_b_outer_splitting_stage1_execution_ready_20261025.md)。SOURCE候補 `b85dd328423de5b1159f11939835f04db90f3187`、入口 `scripts/run_b2d_outer_stage1.py`、接続 `b2d_evaluator/outer_execution.py`、人工test `tests/management/test_b2d_outer_execution.py`。16行・906 dispatch・490 key・80 expectationを登録、105人工件PASS。科学処理未実行、明示許可後だけ実行する。以下の準備履歴・旧Stage 1結果・STOPは保全する。

## 研究B outer splitting準備（2026-10-25）

[技術報告・未承認科学契約案](docs/research/research_b_outer_splitting_stage1_preparation_20261025.md)、`src/trottertracks/b2d_evaluator/{outer,outer_backend}.py`、`tests/management/test_b2d_outer_preparation.py`を一組として読む。4方式のstage／heterogeneous RTE接続と人工91件PASS。科学処理未実行、新outer専用launcherなし。旧Stage 1 SOURCE・結果・scope・STOPは保全する。

最終更新：2026-09-20

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
5. 数値を使う場合だけ、対応する `docs/*_validation.md` と `artifacts/` のJSONを確認する。

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
$\delta=0.02$が全6指標で最小となった。現在は$\delta=0.02$と比較対照0.01について、
制御付きpartial-$S_2$反復とHadamard 1 shot costへ接続する段階である。

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


## PR混在辞書 F1の保存B入力adapter（準備のみ）

[Phase 2D-F1-PRE](docs/research/pr_algorithm_improvement_phase2d_f1_preparation_20261023/README.md)に、専用環境、既存prototypeとの接続、runner／管理test／将来artifact path、実行承認境界をまとめた。科学的検証は未実行。既存SOURCE・入力・結果を保全した独立branchの技術準備である。


## 2026-10-24：PR混在辞書F1のone-shot部分結果

[docs/research/pr_algorithm_improvement_phase2d_f1_execution_20261024/README.md](docs/research/pr_algorithm_improvement_phase2d_f1_execution_20261024/README.md)に、固定SOURCE・runner・管理testとartifactの対応、metadata TypeErrorによるSTOP、実行済みPASSと未完了を記録した。科学SOURCEはF1-PREのまま、retry／resume・F2なし。要約とhash台帳は `artifacts/pr_algorithm_improvement_phase2d_f1/2026-10-24/`。


## PR混在辞書F1：修正版の登録検証完了（2026-10-24）

[新結果とSTOP](docs/research/pr_algorithm_improvement_phase2d_f1_rerun_20261024/README.md)。既存saved runner／adapter／reference／pricingを再利用し、SOURCE `8192530` と保存B入力で16familyを全件完了。全4armの登録正しさ検査とcatalog価格-DP照合はPASS、未資格intervalはfallbackのみ。認証score／shot／資源優位性は未評価。artifactは `artifacts/pr_algorithm_improvement_phase2d_f1/2026-10-24-r1/`。旧停止記録は保全し、F2へ進まない。


## PR混在辞書 F2実行前準備（2026-10-24）

[準備・SOURCE再利用・case・統計資格・契約・STOP](docs/research/pr_algorithm_improvement_phase2d_f2_preparation_20261024/README.md)。入口 `scripts/run_pr_dictionary_f2_saved.py`、SOURCE `trottertracks/pr_dictionary_validation/f2_*.py`、管理test `tests/management/test_pr_dictionary_f2_preparation.py`。planned outputは `artifacts/pr_algorithm_improvement_phase2d_f2/<execution-date>/<fresh-run-id>/`。finite-target主比較を採択し、mock/管理確認のみ完了。統計条件とinterval資格は未解決で、科学実行・新grantは0件。


## PR混在辞書 F2統計資格確認（2026-10-24）

[資格確認と実行前STOP](docs/research/pr_algorithm_improvement_phase2d_f2_qualification_20261024/README.md)。既存interval／log／ceilの技術test `tests/management/test_pr_dictionary_f2_scalar_intervals.py` は5件PASS。科学SOURCEはF2-PREのまま、共通finite targetに対する有限ビット誤差予算γの保証と統計条件の採択が未解決で、F2科学実行・新grantは0件。機械可読記録は同資料の `qualification.json`。入口は既存 `scripts/run_pr_dictionary_f2_saved.py` を維持する。


## PR混在辞書 F2限定資源計測（2026-10-24）

[限定契約・再利用SOURCE・停止条件](docs/research/pr_algorithm_improvement_phase2d_f2_resources_20261024/README.md)。入口 `scripts/run_pr_dictionary_f2_resources_saved.py`、接続 `trottertracks/pr_dictionary_validation/f2_resources.py`、管理test `tests/management/test_pr_dictionary_f2_resources.py`。出力は `artifacts/pr_algorithm_improvement_phase2d_f2_resources/`。旧統計資格付き契約を維持し、認証Nを生成しない。

限定F2の[実行結果・部分資源・STOP](docs/research/pr_algorithm_improvement_phase2d_f2_resources_20261024/EXECUTION_REPORT.md)：raw全件PASS、compiled operator3件FAIL、mean／4arm期待資源は未完了。SOURCE `9b1e103` と一回の実行原本を保全し、retryなし。


## 限定F2・位相保持修正と新規実行（2026-10-24）

[修正・実行契約・停止条件](docs/research/pr_algorithm_improvement_phase2d_f2_resources_rerun_20261024/README.md)。既存runnerとf2_resources／f2_executionを再利用し、gates.compiled_wrapperの公式definition展開を修正。回帰testは `tests/management/test_pr_dictionary_compile_phase.py`、planned artifactは `artifacts/pr_algorithm_improvement_phase2d_f2_resources/2026-10-24-r1/`。

[修正版F2の全件結果・資源表・STOP](docs/research/pr_algorithm_improvement_phase2d_f2_resources_rerun_20261024/EXECUTION_REPORT.md)：SOURCE `9468c70`、raw／compiled各13,676・reference27,356・mean4arm・weighted集計を完了。1-shotの指標順はPP<SP<PS<SS、認証Nは対象外。旧失敗は保持し、追加実験へ進まない。

## 2026-10-24：保存F2回路資源の寄与分析

[結果と分類定義](docs/research/pr_algorithm_improvement_phase2d_f2_attribution_20261024.md)、`src/trottertracks/pr_dictionary_validation/f2_attribution.py`、`scripts/run_pr_dictionary_f2_attribution_saved.py`、`tests/management/test_pr_dictionary_f2_attribution.py`、`artifacts/pr_algorithm_improvement_phase2d_f2_attribution/2026-10-24/`を一組として読む。既存exact集計を再利用する保存データの再集計だけで、回路build／compile・sampling・law生成・N計算はしない。主7指標＋measureの公開期待値へexact照合し、因果効果の認定や次の実験には進まない。

## Rectangular Givens候補Aの限定実装・PS48準備（2026-10-24）

[条件・実行境界](docs/research/pr_rectangular_givens_preparation_20261024.md)、`src/trottertracks/pr_dictionary_validation/rectangular.py`・`rectangular_preflight.py`・`rectangular_execution.py`、`scripts/run_pr_rectangular_givens_saved.py`、`tests/management/test_pr_rectangular_givens_preparation.py`、`artifacts/pr_rectangular_givens_preparation/2026-10-24/`を一組として読む。実装・metadata/mock確認のみ。保存PS48の新科学実行は別承認・新grantが必要で、今回build／compileは0件。

## Rectangular Givens PS48科学結果（2026-10-24）

[実行結果・資源差・STOP](docs/research/pr_rectangular_givens_ps48_execution_20261024.md)。固定SOURCE `770e53a` の既存runner／builder／reference target／compilerを再利用し、raw／compiled各48がPASS。結果・CSV・provenance・保存結果集計utilityは `artifacts/pr_rectangular_givens/2026-10-24/`。費用差はcase依存で、全F2への一律展開や追加実験は未実施。

## Gaussian singleton構造選択の静的設計調査（2026-10-24）

[設計メモ・限界・独立検証案・STOP](docs/research/pr_rectangular_selection_design_20261024.md)。PS48結果と既存SOURCEを読むだけで、数値再集計・recipe生成・選択器実装・build／compileはしない。一般的な正の選択条件は未確立で、一旦終了の技術提案はユーザー／GPTの採否待ち。既存の科学的PASS・結果・承認状態は保持する。

## B2-D Input package・preflight準備（2026-10-25）

[技術報告](docs/research/research_b_b2d_input_preflight_technical_report_20261025.md)と[未採択の第一段階契約案](docs/research/research_b_b2d_stage1_science_execution_contract_proposal_20261025.md)。実装は `src/trottertracks/b2d_evaluator/{input_package,preflight}.py`、人工testは `tests/management/test_b2d_input_preflight.py`。science runner未起動、将来outputは `artifacts/research_b_b2d_stage1/<date>/<fresh-run-id>/`。全call scope・実運用承認接続は未確定、科学実行・新grantなし。

## B2-D Stage 1最終実行準備（2026-10-25）

[最終提案](docs/research/research_b_b2d_stage1_final_execution_proposal_20261025.md)・[技術報告](docs/research/research_b_b2d_stage1_final_preparation_technical_report_20261025.md)・[scope JSON](docs/research/research_b_b2d_stage1_scope_20261025.json)。SOURCE候補 `97a449ad53c74361ffc5c4cffd6de5f3f0f0a72e`、実装 `src/trottertracks/b2d_evaluator/{scope,launcher,errors}.py`、入口 `scripts/run_b2d_stage1.py`、人工test `tests/management/test_b2d_stage1_launcher.py`。境界scope／実績照合・失敗保存はmock確認済み、production authorityは未接続で実行不可。既存46＋追加16＝62人工test PASS、科学計算・grant操作なし。予定artifactは従来どおり `artifacts/research_b_b2d_stage1/<date>/<fresh-run-id>/`、今回未生成。

## B2-D Stage 1一括準備完了（2026-10-25）

[最終契約・ユーザー承認操作](docs/research/b2d_stage1_execution_ready_20261025/CONTRACT.md)・[技術報告](docs/research/b2d_stage1_execution_ready_20261025/TECHNICAL_REPORT.md)。科学SOURCE候補 `348e08cbc44fb0a81cfc4b6932914f7b146a5b9c`、手動承認接続・実環境binding・published列独立gateを追加。入口は `scripts/{prepare_b2d_stage1,issue_b2d_stage1_approval,run_b2d_stage1}.py`、実装は `src/trottertracks/b2d_evaluator/`、人工testは `tests/management/test_b2d_execution_ready.py`。既存62＋追加12＝74件PASS。実行にはユーザーの別途明示承認と管理者所有receipt発行が必要。科学計算・正式grant操作なし、予定artifactは従来の専用namespaceで未生成。


## B2-D Stage 1通常ユーザー承認方式（2026-10-25）

最新の[未承認契約・通常権限の具体command](docs/research/b2d_stage1_user_approval_20261025/CONTRACT.md)と[技術報告](docs/research/b2d_stage1_user_approval_20261025/TECHNICAL_REPORT.md)。新科学SOURCE `91f45aeb50275bb6d22050a304d0a9b1ff025d70`、通常ユーザーのprivate手動記録へ限定修正し、root／外部APIを要求しない。同じOSアカウントの本人識別は運用規則であり独立認証ではない。入口 `scripts/{prepare_b2d_stage1,issue_b2d_stage1_approval,run_b2d_stage1}.py`、実装 `src/trottertracks/b2d_evaluator/`、人工test `tests/management/test_b2d_execution_ready.py`。既存74件を適合更新＋追加6件＝80件PASS。科学条件・scope・環境は不変、正式grant／claim／科学実行なし。予定artifactは固定checkout内 `artifacts/research_b_b2d_stage1/20261025/user-approval-once/`、今回未生成。旧root方式は上記過去段階の履歴として保全する。


## B2-D Stage 1承認管理撤去・実行簡素化（2026-10-25）

最新の[科学実行契約・単一command・技術報告](docs/research/b2d_stage1_simple_execution_20261025.md)。新科学SOURCE `54008c697e686a42ec48941d25219d523bafd454`。入口は `scripts/run_b2d_stage1.py --execute --source-commit … --output …`。承認packet／grant／receipt・発行／準備scriptを撤去し、科学SOURCE・input identity・scope・correctness・fresh output・失敗保存・STOPを維持した。実装 `src/trottertracks/b2d_evaluator/`、人工test `tests/management/test_b2d_simple_execution.py` と既存5ファイル、計70件PASS。科学計算は未実行。予定artifact `artifacts/research_b_b2d_stage1/20261025/simple-once/` は未生成。旧方式は過去の未採択提案として保全し、今後のB2-D実行要件にはしない。


## B2-D Stage 1限定科学検証の完了（2026-10-25）

[実行結果・技術復旧・比較表・STOP](docs/research/b2d_stage1_execution_20261025.md)。科学SOURCE758a05ed、入力／A4数学／published列gate全PASS、36行・941 dispatch・448 key完了。A4のA0／D-S2からのfull signal改善は観測したが、A3との差は丸め誤差水準でINCONCLUSIVE。原本2attempt・全比較・環境・CSV・保存値集計は `artifacts/research_b_b2d_stage1/20261025/`。ローカル結果でCI／外部再現ではない。追加科学計算・Stage2へ進まずmandatory STOP。
