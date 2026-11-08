# プロジェクト案内

**Track C DFの3分子比較（2026-11-08）：[Full PF・Truncated PF・改良PR](docs/research/track_c_df_truncation_20261108/REPORT.md)。** 固定元H/oracle ground/18round/compiler Aの有限探索でH2はPR、H4/H6はTruncatedが低RZ。新しい科学計算は完了、mandatory STOP。旧記録は保存。

**Track C（O3）全研究の総括（2026-11-08）：[D内部BCH・共同isometry・対角融合と強Full比較](docs/research/track_c_o3_research_summary_20261108/TRACK_C_O3_RESEARCH_SUMMARY.md)。** 新規科学計算を伴わない保存成果の統合。旧研究CのHamiltonian表現系列と区別する。

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

## Track C H4 pilot（2026-11-06）

[3候補の実行・compiler資格不一致によるSTOP](docs/research/track_c_h4_pilots_20261106/REPORT.md)。新adapterは `src/trottertracks/track_c_h4*.py`、runner・人工test・rawへの入口も同報告を参照。全RPE総G・勝者は未資格。

## 2026-11-06 Track C compiler A：限定correctness

[実装・結果・未資格事項](docs/research/track_c_compiler_a_correctness_20261106/REPORT.md)。Linear H4 1Å/STO-3G/DF12、LD3/12、h=.02・q1。人工44ケースとX/Yの全512次元operator・強baseline・負時間検証は数値PASS。control前のscalarを保持し、既にcontrolledなjoint全体の共通global phaseだけを許容する。18round native RPE・総Gは未資格。新価格/T合成0件、旧STOP保持、限定検証でmandatory STOP。

## 2026-11-06 Track C compiler A：18round誤差会計

[局所区間上界・全support・18round会計](docs/research/track_c_compiler_a_18round_qualification_20261106/REPORT.md)。Linear H4 1Å/STO-3G/DF12、S2 LD3/Full Y4 LD12、h=.02。128 basis contextと1,302局所変換を192-bit区間評価し、全36条件で保存meanモデル下の正半径・統計予算が残る。SOURCE→mean、旧数値disk、alias保証が未完のため全RPEはUNQUALIFIED。価格/T合成0件、旧STOP保全、mandatory STOP。

## 2026-11-06 Track C 保証検証A・C-N価格B

[結果・SOURCE・raw](docs/research/track_c_guarantee_cn_price_20261106/REPORT.md)。Linear H4 1Å/STO-3G/DF12、S2 LD3/Full Y4 LD12、h=.02。Aは固定保存行列・理想gate・独立bitsモデルで全18roundの十分条件成立（shot8598/6914）。元H表現の無条件包含は未資格。Bは参照Aで39セル2307標本を完了し、S2の全2304paired標本でfull/強baselineのRZ/CX/depth/sizeが一致。q8最大点誤差6.09%、長q総G・勝者はUNDETERMINED。旧run保全、T合成0、mandatory STOP。

## Track C2-A限定pilot（2026-11-06）

[84信号/60短価格セル。3round診断はS2、PF交差未観測。長q期待価格と新構成保証は未資格、全18roundへ進めずSTOP。](docs/research/track_c2_a_pilot_20261106/REPORT.md)。SOURCE、検査、raw、STOPは報告から辿る。旧STOPは維持。

## Track C混合公式（2026-11-08）

[固定H4のBCH・全18round信号・paired価格検証](docs/research/track_c_mixed_formula_20261108/REPORT.md)。半刻み対照に対する資源改善は未観測、native全18round保証・絶対Gは未資格、mandatory STOP。新SOURCEは`track_c_mixed_*.py`、初期入口は`run_track_c_mixed_formula.py`、保存結果の読取検査は`check_track_c_mixed_formula.py`。

## Track C D内部PF（2026-11-08）

[固定H4のBCH設計・有限RTE・18round native資源](docs/research/track_c_d_internal_20261108/REPORT.md)。実験SOURCEは`src/trottertracks/track_c_d_*.py`、人工testは`test_track_c_d_internal.py`、保存結果検査は`check_track_c_d_internal.py`。PR内改善あり・強Fullより高価、mandatory STOP。

## Track C資源ボトルネック（2026-11-08）

[全18round分解・追加改善・独立holdout](docs/research/track_c_bottleneck_20261108/REPORT.md)。singleton部分QR・step/resolution/α配分を検証し、改良PR出発点比RZ46.52%減。ただし同じα最適化後の強FullよりRZ1.96倍・CX1.27倍。実装`track_c_bottleneck*.py`、人工test`test_track_c_bottleneck.py`、読取検査`check_track_c_bottleneck.py`。固定保存H4モデルの条件付き結果、mandatory STOP。

## Track C共同isometry（2026-11-08）

[構成・Fock/native/X/Y・全18round・独立holdout](docs/research/track_c_joint_isometry_20261108/REPORT.md)。実装`track_c_joint_*.py`、人工test`test_track_c_joint_isometry.py`、保存値集計`summarize_track_c_joint_isometry.py`、読取検査`check_track_c_joint_isometry.py`。paired総RZ3.355%・CX1.789%減、serial depth上界0.696%増。強Fullがなお低コスト、mandatory STOP。

## Track C対角gate融合（2026-11-08）

[既知control・run融合・共通Full比較](docs/research/track_c_diagonal_fusion_20261108/REPORT.md)。実装`track_c_diagonal_*.py`、人工test`test_track_c_diagonal_fusion.py`、保存値集計`summarize_track_c_diagonal_fusion.py`、読取checker`check_track_c_diagonal_fusion.py`。RTE-run固有の追加効果は既知term基準で全PRのRZ約0.135%・CX約1.67%。同じ合成でFullも改善し差は縮まらず、mandatory STOP。

## Track C Closed P3統合（2026-11-08）

[同じ有限P3・weighted/native資格・全18round独立比較](docs/research/track_c_closed_p3_20261108/REPORT.md)。総RZ約1.965%減、強Full PFはなお低コスト。新SOURCEは`track_c_closed_p3*.py`、旧SOURCE/STOPは保全。

## Track C Closed P3：h/resolution/α共同選択（2026-11-08）

[既存menu・限定局所拡張・weighted保証・独立全18round比較](docs/research/track_c_closed_p3_joint_opt_20261108/REPORT.md)。Sourceは`track_c_closed_p3_joint_opt*.py`と`track_c_closed_p3_joint_extend.py`。最終値は`artifacts/track_c_closed_p3_joint_opt/2026-11-08/extended/FINAL.json`。旧SOURCE・raw・STOPを保全。
