# プロジェクト案内

A1 Q1/Q2-Lの[実行SOURCE・identity不一致STOP](docs/research/o2_a1_q1_q2l_20261106/REPORT.md)（2026-11-06）：
許可された固定H4 m0の一回取得はCDF評価前に停止。実CDF/ν/biasは未取得、再試行なし。library・runner・人工test・原STOPへ同報告から辿れる。

A1-0の[技術接続状況・後日の限定実行scope](docs/research/o2_a1_0_connections_20261106/REPORT.md)（2026-11-06）：
実CDF/固定controlとselected-only full価格への薄い接続を人工入力で検査する。科学runner・実入力評価・新しい科学結果・実行許可ではない。

O2-S2固定pair-table限定評価（2026-11-06）：[v2六cell結果・SOURCE・raw](docs/research/o2_s2_multi_event_validation_v2_20261106/REPORT.md)。
許可された転送修正再実行で六cell完了。T主指標は元law P0/Bが最小で、qB/Bによる追加改善なし。v1途中停止結果は保全。
実装は `src/trottertracks/algorithm_codesign/s2_multi_event.py`、runnerは
`scripts/tracks/algorithm_codesign/run_s2_multi_event_validation.py`、人工testは
`tests/tracks/algorithm_codesign/test_s2_multi_event.py`。限定評価・公開後はmandatory STOP。

**研究結果への入口（2026-11-02）：[統一索引](docs/research/RESEARCH_RESULTS_INDEX.md)。**
複数ブランチの総括・後続結果・SOURCE・raw・検証状態を固定commitで辿れる。
以下の研究記述は、このcheckoutに保存された時点の履歴として読む。

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

## 旧snapshotに記録された研究段階（2026-09-20）

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

## 既存研究成果のmain収録（2026-11-03）

[収録済み成果と原SOURCE](docs/research/main_collection_20261103.md)から、main内のG6報告・保存結果・STOP・再現用SOURCEを辿れる。
これは過去の証拠とSOURCEの収録であり、新しい科学結果・実行許可・共通ライブラリ採択ではない。

第2回は[公開snapshot間で衝突のない追加ファイルと原データ所在](docs/research/main_collection_20261103.md#第2回競合のない追加ファイルの収録)を収録。既存pathの別版と枝全体のmergeは保留し、元の科学SOURCE・結果・STOPを区別して保存する。

2026-11-03のmain集約で追加したルートの旧計画・`REPORT.md`・`references/` 等は、
当時の研究別記録として[出典対応](docs/research/main_collection_20261103_additional_files.tsv)から辿る。
現行の統合仕様・実行許可は、この入口で示す優先順位と各研究の既存契約に従う。

## 索引・SOURCEの版差（2026-11-03）

[同名37 pathの整理結果](docs/research/main_collection_versions_20261103.md)から、統合した7索引、原内容で配置したSOURCE、全88保存版を辿る。
旧snapshotの研究記述を、全研究系列の現在の状態へ読み替えない。

## P1・研究C SOURCEの組（2026-11-03）

[選定したSOURCEと依存版・検査範囲](docs/integration/main_p1_c_price_cohorts_20261103.md)を参照する。P1は人工・mock検査79件PASS、研究Cはpassive importと静的互換を確認した。
共有coreの変更前版は保存済み。過去の科学SOURCE・入力・結果・契約・STOPは元snapshotを使う。

## サーバー照合からの成果収録（2026-11-03）

[収録ファイル・保存版と確認範囲](docs/research/main_server_results_collection_20261103.md)から、過去の公開成果と元SOURCEを辿る。
既存mainの科学ファイルは上書きせず、研究test関連の原版は実行contextと分けて保存する。新科学計算は行っていない。

## ローカルに残っていた文書・SOURCEの収録（2026-11-03）

[原稿・受領レビュー・研究ノート・SOURCEと保存別版の対応](docs/research/main_local_results_collection_20261103.md)から辿る。元の科学SOURCE・status・STOPは維持する。


A1 v2の[payload保存修正・限定診断とSTOP](docs/research/o2_a1_q1_q2l_20261106/REPORT_V2.md)（2026-11-06）：旧target同値は未閉鎖、CDF以降は未実行。元runnerの`--diagnose`、完全payload、人工29件と原v1保全を記録する。


A1の[旧opt2来歴・完全payload・環境の限定静的調査](docs/research/o2_a1_opt2_provenance_20261106/REPORT.md)（2026-11-06）：記録SOURCEの回収先と未回収payload、元checkpoint抜粋、stdlib-only collector。科学再実行・library変更は0件、STOP維持。


A1の[新finite-target科学契約設計案](docs/research/o2_a1_new_finite_target_20261106/REPORT.md)（2026-11-06）：actual-Q/selected full複素signalの定義、Raw/2層/5層/CV、公平なconfidence/全費用と後日の科学call。設計のみ、未採択・科学実行0件。


A1新targetの[N1限定実行](docs/research/o2_a1_new_target_n1_20261106/EXECUTION.md)：既存DF/CDF/CV/policyを再利用するmodule・runner・人工test、SOURCE/環境lock。N1のみ、N2/N3は起動しない。


A1新targetの[N1結果・完全payload・correctness](docs/research/o2_a1_new_target_n1_20261106/REPORT.md)（2026-11-06）：同じ一回の準備からCDF/Q/νと層質量・basis対応を取得。N1成功後STOP、N2/N3・測定・価格/費用は未実行。


A1の[N2限定SOURCE・state/engine検証](docs/research/o2_a1_new_target_n2_20261106/EXECUTION.md)：N1保存payloadを再利用するmodule/runner/人工test、state inventoryと環境lock。N3なし、公開後STOP。


A1の[N2結果・state/engine保証範囲・STOP](docs/research/o2_a1_new_target_n2_20261106/REPORT.md)：一意ground/state誤差を確認。総b_engineはUNKNOWN、N3未準備。SOURCE/payload/人工test/保全へ辿れる。


A1の[保存engine接続・一様誤差評価plan・人工correctness](docs/research/o2_a1_engine_connection_20261106/REPORT.md)：最小adapter/test、接続/誤差計画/次回の限定科学scope。実上界未取得、N1/N2再計算・N3なし、公開後STOP。


A1の[一回のengine数値認証・runner仕様](docs/research/o2_a1_engine_qualification_20261106/EXECUTION.md)：固定kernel、厳密state上界による最終会計、旧SOURCE保全とSTOP。


A1の[一回の総engine誤差認証・全gate証明・STOP](docs/research/o2_a1_engine_qualification_20261106/REPORT.md)：固定exact engine/理想モデルの条件付き上界取得。N3・費用評価は未実行。


A1 N3の[技術接続・科学条件草案](docs/research/o2_a1_n3_connection_20261106/REPORT.md)：`a1_n3_connection.py`、人工専用runner/test、保存証拠への入口。人工46件PASS、実科学0・未採択・公開後STOP。


A1 N3の[採択済み一campaign・SOURCE/実行仕様](docs/research/o2_a1_n3_science_20261106/EXECUTION.md)：追加driver/test、固定exact engine・実call境界・保存とmandatory STOP。

A1 N3の[一campaign中断結果](docs/research/o2_a1_n3_science_20261106/REPORT.md)：取得済397 bit/価格、未完了77 trial、精度/CV比較未判定、journal保全とmandatory STOP。

A1 N3の[中断診断・再発防止・復旧2案](docs/research/o2_a1_n3_interruption_20261106/REPORT.md)：journal専用runner `scripts/diagnostics/a1_n3_interruption_audit.py`、`a1_n3_interruption_diagnostics`/`a1_n3_lifecycle_observer`（src/trottertracks/algorithm_codesign）、人工test `tests/test_a1_n3_interruption_diagnostics.py`。直接原因UNKNOWN、新科学0、mandatory STOP。

A1 N3の[持続manager接続と案B草案](docs/research/o2_a1_n3_manager_preparation_20261106/REPORT.md)：`a1_n3_cron_manager.py`/`a1_n3_managed_runtime.py`（src/trottertracks/algorithm_codesign）、人工入口 `scripts/diagnostics/run_a1_n3_manager_qualification.py`、unit `tests/test_a1_n3_managed_runtime.py`。旧数値SOURCE不変、科学0。


A1 N3案Bの[新しい一campaignの実行仕様](docs/research/o2_a1_n3_B_science_20261106/REPORT.md)（2026-11-06）：seed20261107、旧397件を分離、旧数値SOURCEとqualified managerを再利用する薄い入口。SOURCE公開照合後に一回開始し、成功・中断でmandatory STOP。


2026-11-06 A1 N3案B追記：5f2eac41…の一回cron投入はentryのsrc import不備で科学開始前STOP。新規stream/shot/compileは0、trigger撤去済み。入口修正・最小cron環境testを含む人工37件PASS、再投入なし。[結果・失敗SOURCE・次回条件](docs/research/o2_a1_n3_B_science_20261106/REPORT.md)。
