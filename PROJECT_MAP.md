# プロジェクト案内

## C1 Stage2限定科学実行：native接続PASS（2026-10-25）

[結果・全gate・native費用・STOP](docs/research/research_c_c1_stage2_native_execution_20261025.md)。公開SOURCE `a977a4390668bc107360c8ff05f43d851dc81dc9` を無変更で使用し、局所40／raw full4／actual adjoint4／compiled full4／X/Y8組の60主検査とunitarity52件が全PASS。P-H4・CDF rank8／XDF rank10、T1/10 ordered S2・P0 profile。科学call243/243、fit／optimizer／retry／resume0、transpile12・QPY68。CDF fullのRZ12118／CX7352／depth12724、XDF16659／9520／16215。物理DF精度が異なり同精度総資源優位とはしない。証拠 `artifacts/research_c_c1_stage2_science/2026-10-25/`、Stage3未承認・mandatory STOP。下記の準備時点の未承認表示は履歴。

## C1 Stage2 native接続の技術準備（2026-10-25）

[技術報告・再利用・人工検証](docs/research/research_c_c1_stage2_native_preparation_20261025.md)、[未承認科学契約](docs/research/research_c_c1_stage2_native_contract_unapproved_20261025.json)。ユーザーがCDF rank8を限定対象に採用。新SOURCE `c1_stage2_native.py`／`c1_stage2_validation.py`／`c1_stage2_execution.py`、entry `scripts/run_research_c_c1_stage2_native.py`、test `tests/management/test_research_c_c1_stage2_native.py`。既存Gaussian・中央RZ/RZZ・phase・controlled／X/Y／compilerを接続し、人工80件＋29 subtests PASS。SOURCE・52 pins・証拠 `artifacts/research_c_c1_stage2_technical/2026-10-25/`。時間／S2／compiler／allowance／最大243 unitsは未採択の推奨案。実H4 decode・実回路build/compile・科学call0、別科学承認待ち・mandatory STOP。同精度総資源優位の主張なし。

## C1 保存fit評価専用科学実行：全gate PASS（2026-10-25）

[実行結果・全gate・誤差・STOP](docs/research/research_c_c1_stage1_eval_only_execution_20261025.md)。公開SOURCE `bec41ac7dc7e034bd664817acfdcfed858ddadd0` の既存評価専用entryを無変更で実行した。P-H4・sector36/full256、保存C-DF rank8／XDF rank10で37 gate PASS、以前不一致のframe0も今回PASS。物理sector誤差はCDF `7.666027761392501e-5`／XDF `3.3751558354542566e-15` Hartreeの別測定。科学call141/141、新fit／optimizer／retry／resume0。原本 `artifacts/research_c_c1_stage1_eval_only_science/2026-10-25/`、人工19件PASS。旧FAIL保全、Stage2適格性は別判断、mandatory STOP。下記の準備時点の未承認表示は履歴。

## C1 保存fit評価専用経路の準備（2026-10-25）

[技術報告](docs/research/research_c_c1_stage1_eval_only_preparation_20261025.md)、[未承認科学契約](docs/research/research_c_c1_stage1_eval_only_contract_unapproved_20261025.json)。新entry `scripts/run_research_c_c1_stage1_eval_only.py`／`src/trottertracks/c1_stage1_eval_only.py` は旧fitのSOURCE/input/exportと新評価SOURCEを独立に保持し、新fit0で既存CAR/K/XDF検証を再利用する。新test `tests/management/test_research_c_c1_stage1_eval_only.py`、人工82件＋29 subtests PASS。証拠・SOURCE `artifacts/research_c_c1_stage1_eval_only_technical/2026-10-25/`、call上限141、retry/resume0。実H4 decode・評価・fit0、旧科学的STOP保全、別明示承認待ち・mandatory STOP。

## C1 raw CAR frame修正・人工検証（2026-10-25）

[修正報告](docs/research/research_c_c1_frame_repair_20261025.md)、[未承認引継ぎ](docs/research/research_c_c1_frame_repair_handoff_unapproved_20261025.json)。`src/trottertracks/c1_stage1.py` のraw縮約をexplicit batched matmulへ変更し、一般K／物理モデル／solver／allowanceは維持。新test `tests/management/test_research_c_c1_frame_repair.py` と既存回帰63件＋29 subtests PASS、別プロセス64人工frame比較もPASS。証拠・SOURCE `artifacts/research_c_c1_frame_repair/2026-10-25/`。分子exportの数値再解析・新fitは0。旧科学的FAILとの因果帰属・H4資格は未確認、旧STOP保全・mandatory STOP。現行one-fit runnerは評価専用入口ではない。

## C1 Stage 1互換性修正版の科学的STOP（2026-10-25）

[実行結果・gate・部分証拠](docs/research/research_c_c1_stage1_compat_execution_20261025.md)。公開SOURCE `da2edb465d5ae6432439fbd1df8e57512ee59c11`、新approval／fresh outputで1 fit収束・実R8。κ/J/Uとfull baseはPASS、full256 frame0のraw CAR／K同値性がFAILしmandatory STOP。物理CDF/XDF sector norm・sector adapterは未到達。科学call107/147、retry/resume0。原本は `artifacts/research_c_c1_stage1_compat_science/2026-10-25/`、既存runner `scripts/run_research_c_c1_stage1.py` とSOURCEは無変更。追加計算・Stage2/3なし。

## C1 Stage 1互換性修正・人工検証（2026-10-25）

[修正報告](docs/research/research_c_c1_stage1_compatibility_20261025.md)、[新未承認契約](docs/research/research_c_c1_stage1_compatibility_contract_unapproved_20261025.json)。専用bridge `src/trottertracks/c1_ffsim_compat.py` を既存Stage1 runnerへ接続し、venv Python symlinkを保持。新test `tests/management/test_research_c_c1_ffsim_compat.py` と既存回帰59件＋29 subtests PASS。証拠・SOURCE・同版依存lockは `artifacts/research_c_c1_stage1_compatibility/2026-10-25/`。実ffsim人工初期化はoptimizerを遮断して確認、分子科学call0。旧run01/run02/STOP保全、新しい1 fitは別明示承認が必要。mandatory STOP。

## 研究C・C1 Stage 1科学実行の技術的STOP（2026-10-25）

[結果・固定SOURCE・部分証拠・STOP](docs/research/research_c_c1_stage1_execution_20261025.md)。固定 `c1_stage1.py` / `c1_stage1_execution.py` と `scripts/run_research_c_c1_stage1.py` を使用。artifactは `artifacts/research_c_c1_stage1_science/2026-10-25/`。1 fitはffsim/SciPy batch logm非互換でoptimizer前に停止、科学call2/147、CDF/XDF誤差とadapter同値性は未測定。追加fit・Stage2/3なし、mandatory STOP。

## 研究C・C1 Stage 1技術準備（2026-10-25）

[実装・人工検証・未承認契約](docs/research/research_c_c1_stage1_technical_preparation_20261025.md)。libraryは `src/trottertracks/c1_stage1.py` と `c1_stage1_execution.py`、入口は `scripts/run_research_c_c1_stage1.py`、人工testsは `tests/management/test_research_c_c1_stage1*.py`、証拠と独立環境lockは `artifacts/research_c_c1_stage1_technical/2026-10-24/`。既存ffsim/一般K/CAR/sector演算を接続、人工45件＋29 subtests PASS。科学runは未承認、allowance採択と新明示承認が必要。

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

## Gaussian singleton独立研究の一時終了・次研究への引継ぎ（2026-10-24、現在の採択状態）

[採択記録と引継ぎ](docs/research/pr_gaussian_singleton_closeout_handoff_20261024.md)。ユーザーが一時終了を正式採択した。上記の採否待ちは当時の履歴で、科学的PASS・負結果・未確立事項は変更しない。既存Track BのC4／Parent allocation／G9／C1と先行研究整理を再利用し、辞書・law・実回路費用・統計負担の共同検討へ論点を渡す。具体的機構の採択・新実装・科学実行は含めずmandatory STOP。


## 研究C・C4：Q13限定科学検証（2026-10-24）

[実行契約・履歴・結果](docs/research/research_c_rustiq_execution_20261024.md)。SOURCEは `src/trottertracks/pr_dictionary_validation/rustiq_adapter.py` と `rustiq_execution.py`、入口は `scripts/run_research_c_rustiq_limited.py`、管理testは `tests/management/test_research_c_rustiq_*.py`、artifactは `artifacts/research_c_rustiq_limited/2026-10-24/`。保存F1の4case・B0/B1/R1だけを対象とし、旧SOURCE・grant・STOPを保持する。


## 研究C・C4：B2角度結合baseline限定科学比較（2026-10-24）

[結果・固定SOURCE・契約・証拠](docs/research/research_c_angle_fusion_b2_execution_20261024.md)。既存 `rustiq_adapter.py`／`rustiq_execution.py` と `scripts/run_research_c_rustiq_limited.py --scope C4_B2_SAVED_COMPARISON_V1` を使用。testは `tests/management/test_research_c_rustiq_*.py`、artifactは `artifacts/research_c_rustiq_angle_fusion/2026-10-24/attempt01/`。全登録gateがPASSし、全8wrapper・全8compiled指標でB2=保存R1。単純融合がこのfixtureの改善を再現した。旧route再実行・新reference・Rustiq/R2・retryは0。独立研究を区切るかはユーザー／研究C／統括の次判断であり未採択、mandatory STOP。

## 研究C：C4一時終了・C1接口調査（2026-10-24、現在の判断）

[C4終了・成果引継ぎ](docs/research/research_c_c4_closeout_handoff_20261024.md)でユーザー採択済みの一時終了を追記した。上記の未採択状態は当時の履歴。B2等のSOURCE・検証資産・旧科学結果・STOPは維持する。
[C1圧縮DFの静的接口調査](docs/research/research_c_compressed_df_interface_survey_20261024.md)は、X-DF／通常C-DF／RC-DF／SCDF、外部実装と既存DF下流の再利用、general diagonal・scalar/sector/phase、強いbaselineと別承認の最小案を整理する。C1の方式未採択、新実装・科学計算なし、公開照合後mandatory STOP。
