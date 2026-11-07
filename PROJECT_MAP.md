# プロジェクト案内

[C8-1保存q64の数値診断](docs/research/track_d_c8_1_q64_20261108/NUMERICAL_REPAIR_PLAN.md)：利用者承認による数値修正・再検証。旧SOURCE/失敗/STOPを保持し、保存sample00の4 P0回路を既存checker/同閾値で再postcheck、例外前のscalar/observableを保全する。sampling/回路生成/compile0、残7は未解放。

[Track D C8-1数値preflight不成立](docs/research/track_d_c8_1_q64_20261108/REPORT.md)：Linear H4/1Å/STO-3G/Nα=Nβ=2・元H/number-CDF8/元順LD2/S2/s1/r32/K0/h=.1、q64/cut32、元H厳密ground仮定・準備費除外。全8 event/512 occurrence/16384 eventsを固定し、sample00 raw1 PASS・旧P0 compile4完了をSOURCE8a6bbdacへ承認済み引継ぎ。両方式X/Y全4 postcheckがCAR-observable Frobenius2e−10でFAILしSTOP。残7/追加28 compile未開始、価格統計/モデル診断未評価。精度緩和/再実行/fit/N/全RPE更新0、旧SOURCE/STOP・非公開原本保持。

[C8-1 CPU予約修正・残task引継ぎ](docs/research/track_d_c8_1_q64_20261108/ACCELERATED_COMPLETION_PLAN.md)：利用者承認のSOURCE切替。旧6b1d613dの完了済みraw1・compile4を保全し、compile予約1 CPUとraw/postへの残CPU配分へ修正。sampling追加0、残7raw/最大28新P0、sample00全post gate維持、固定精度/モデル/32総上限は不変。

[Track D B/C全13-round条件付き評価](docs/research/track_d_bc_total_20261108/REPORT.md)：同Linear H4/1Å/STO-3G/Nα=Nβ=2・元H/number-CDF8/元順LD2/S2/s1/r32/K0/h=.1、元H厳密ground・準備費除外。保存十分N14738と固定A/F/Sで全13-round条件付き資源会計を一回実施。主AはReferenceでCX11.17%・total depth54.65%・RZ1.12%減、幅9→17。約99.5%が長q proxy、全体不確かさ/actual未資格。新sampling/compile/fit/N0、SOURCEb09b7e7a、公開照合後mandatory STOP。

[Track D B6b＋B/C統合準備](docs/research/track_d_b6b_bc_integration_20261108/REPORT.md)：Referenceのideal coherenceを任意入力へ導出し、whole-V-ray条件とDF半path/最終真空条件を区別。SOURCE13114b2b、人工/回帰24件と人工P0 16 wrapper postcheck PASS。B4-2保存Nの条件付き共通利用・13-round価格coverage・薄い会計adapterを整理。actual N/長q期待価格は未資格、実H4の新計算/価格再解析/N算定/全RPE数値会計0。公開照合後mandatory STOP。

[Track D C7-3結果](docs/research/track_d_c7_3_q32_20261107/RESULT.md)：同H4/1Å/STO-3G/Nα=Nβ=2・number-CDF8/元順LD2/S2/s1/r32/K0/h=.1、独立q32/cut16/8 path。sampling/raw SOURCE35bfd、承認済み高速化compile/post SOURCE94e94。旧raw8再利用・新raw/sampling0、全32 P0/post・pairedχ8 PASS。CX差は全8本で−6960、RZ/size/各depthも有限全8本で減、qubits9→17。一次12量は探索的幅内、別population-line6量幅外。統計/全law/q≥64/N/全RPE未資格、旧STOP/raw非公開を維持、公開照合後mandatory STOP。

[C7-3承認済み高速化引継ぎ](docs/research/track_d_c7_3_q32_20261107/ACCELERATED_COMPLETION_PLAN.md)：旧SOURCE35bfdのraw8 PASSを再利用し、旧controller STOP/集約未完を保全。未着手32 P0/各postだけを分離並列化し、exact局所cache・公式Aer double小演算子検証を選択。sampling0、旧新合算32compile上限、公開後mandatory STOP。

[Track D C7-3実行前PLAN](docs/research/track_d_c7_3_q32_20261107/PLAN.md)：同H4/CDF8/LD2/S2/s1/r32/K0/h=.1、独立q32/cut16/n8。凍結C6aモデル・新seedをsampling前固定し、raw PASS後最大32 P0 wrapper。構造/DAG特徴とpaired価格の探索的検証、旧SOURCE/STOP保全、公開照合後mandatory STOP。

[Track D C7-2](docs/research/track_d_c7_2_cx_structure_20261107/REPORT.md)／[q32未採択案](docs/research/track_d_c7_2_cx_structure_20261107/C7_3_EXECUTION_DRAFT.md)：実H4の新計算/価格再解析0、固定54人工P0・raw98項目/post54/observable等36項目・回帰12件PASS。局所非加法/単純lane式へ反例、208−224qの全path bridgeは未取得。旧SOURCE/STOP保全、q32未実行、公開照合後mandatory STOP。

[Track D C7-1](docs/research/track_d_c7_1_q16_20261107/RESULT.md)：同H4/1Å/STO-3G/Nα=Nβ=2・number-CDF8/元順LD2/S2/s1/r32/K0/h=.1、独立q16/cut8/16 path。SOURCE462f6bff、raw16・post64・pairedχ16 PASS。CX差は全観測pathで−3376、RZ/size/各depthも有限全16本で減、幅9→17。一次12量は探索的幅内、別population-line4量幅外。統計/全law/q≥32/N/全RPE未資格、raw非公開、mandatory STOP。

[Track D C7-1固定PLAN](docs/research/track_d_c7_1_q16_20261107/PLAN.md)：同H4/CDF8/LD2/S2/s1/r32/K0/h=.1・独立q16/cut8/16 path。凍結C6a予測・新seedをsampling前公開、raw PASS後だけ最大64 P0 wrapper。探索的paired価格/CX整数残差、raw非公開、完了公開後mandatory STOP。

[Track D C7-0静的構造解析](docs/research/track_d_c7_0_structural_preparation_20261107/REPORT.md)／[次段階条件案](docs/research/track_d_c7_0_structural_preparation_20261107/C7_1_EXECUTION_DRAFT.md)：C6a/C6b保存価格とSOURCEからCX・RZ/size・lane/depth、校正不確かさと長q移送条件を整理。CX一次式は全path未資格。新計算/計数/fit/テスト/compileなし、q16案は未採択・未実行、公開照合後mandatory STOP。

[Track D C6b](docs/research/track_d_c6b_q8_20261107/RESULT.md)：同H4/1Å/STO-3G/Nα=Nβ=2・number-CDF8/元順LD2/S2/s1/r32/K0/h=.1/q8/cut4、独立domainの32 path。元H厳密ground仮定・準備費除外。SOURCE44a58b9fへ承認済み限定引継ぎ、raw32・postcheck128・pairedχ32 PASS。一次frozen条件付き12量は探索的幅内、別population-line診断4量は幅外。統計未資格、全law/長q/N/全RPE未評価、raw非公開、mandatory STOP。

[C6b高速化・限定引継ぎ](docs/research/track_d_c6b_q8_20261107/ACCELERATED_COMPLETION_PLAN.md)：旧SOURCE/raw16・compile/postcheck48を保全し、新SOURCEで未着手raw16/未compile80だけ。sampling0、旧新合算128上限、同じq8/cohort/精度/モデル、完了公開後mandatory STOP。

[Track D C6b独立q8](docs/research/track_d_c6b_q8_20261107/PLAN.md)：fresh32 whole-path/cut4、凍結C6aモデル・paired12量の探索的診断。各path raw PASS後最大128 P0 wrapper。旧SOURCE/STOP保全、raw非公開、長q/全RPE未資格、公開後mandatory STOP。

[Track D C6b-0静的準備](docs/research/track_d_c6b0_preparation_20261107/REPORT.md)：paired12量と絶対24量、旧20 floorの所在、CX構造の証明義務、追加校正/独立q8の最小案と未採択契約。新計算・再解析・実装0、公開照合後mandatory STOP。

[C6a結果](docs/research/track_d_c6a_paired_price_20261106/RESULT.md)：fresh q2/q4各16、raw32/P0 postcheck128/pairedχ PASS。有限pilot統計/共分散・C6b提案、rawローカル保全、全RPE未評価、mandatory STOP。

[Track D C6a](docs/research/track_d_c6a_paired_price_20261106/PLAN.md)：C5公開基点。canonical fresh q2/q4各16 path、fixed half-step cut、paired価格pilot。raw PASS後最大128 P0 wrapper、q8 holdout/全RPE対象外、公開後mandatory STOP。

[C5結果](docs/research/track_d_c5_q2_20261106/RESULT.md)：共通request入口・人工16件・固定q2 raw/4postcheck PASS。strong手動境界融合/位相集約を統一し、P0後7価格は旧C2と一致。有限fixture限定、C6未実施、mandatory STOP。

[Track D C5固定q2](docs/research/track_d_c5_q2_20261106/PLAN.md)：C3の演算を共通request入口へ最小抽出し、保存00→01/cut1のstrong境界融合とReferenceを比較。人工検証・raw PASS後だけ最大4 P0 wrapper、公開後mandatory STOP。C6 sampling/fit/holdoutは対象外。

[Track D C4価格モデル準備](docs/research/track_d_c4_price_model_preparation_20261106/REPORT.md)：保存q1/2/4/8の条件差、whole-pathとfenced kernel会計の境界、全law期待価格・共分散・長q holdoutの不足と最小実行案。静的調査のみ、科学実行・実装変更なし、公開後mandatory STOP。

[C3 q4/q8補完結果](docs/research/track_d_c3_q4_q8_20261106/COMPLETION_RESULT.md)：raw両q・全8 postcheck・最終paired/χ照合PASS。強いstep境界融合を維持して両fixtureでCX/depth減、qubits+8。中断後は保存回路の最後の検証のみ、新compile0。固定fixture限定・mandatory STOP。

[C3保存回路の中断補完PLAN](docs/research/track_d_c3_q4_q8_20261106/COMPLETION_PLAN.md)：run02の8 compile・7 postcheckを保持し、最後のq8 Reference Y postcheckと最終照合のみを別出力で実施。回路再生成・再compileなし。

[C3 run02再実行PLAN](docs/research/track_d_c3_q4_q8_20261106/RUN02_PLAN.md)：利用者承認によりcontract参照先を修正。同じq4/q8 fixture/cut/精度で再実行し、run01のSOURCE・失敗・STOPを保持する。

[Track D C3結果](docs/research/track_d_c3_q4_q8_20261106/RESULT.md)：contract参照先の実装誤りでq4/q8ともraw構築前STOP。科学的correctness・価格未評価、compile/retry/resume0。SOURCE・失敗・STOPを保全し、修正／再実行は新たな判断へ戻す。

[Track D C3固定q4/q8](docs/research/track_d_c3_q4_q8_20261106/PLAN.md)：ascending保存fixture・complete半path cutを固定し、合法なstep境界融合を含むstrong対照でraw correctness、PASS時だけ最大8 P0 wrapper。公開後mandatory STOP。

[C2固定q2結果](docs/research/track_d_c2_q2_compile_20261106/RESULT.md)：4 wrapper compile/postcheck PASS。ReferenceはCX4100→3860・depth6096→3014、RZ/qubits増。固定fixture限定、mandatory STOP。

[Track D C2固定q2 compile](docs/research/track_d_c2_q2_compile_20261106/PLAN.md)：C1の保存rawを使う2方式×X/Yの4 wrapper限定。P0・cut・phase固定、公開後mandatory STOP。

[B1/C1結果](docs/research/track_d_b1_c1_validation_20261106/RESULT.md)：元H向け13-round条件付きoperator十分条件成立、固定q2 raw correctness PASS。compile・新sampling・N評価0、B2/C2未実施、mandatory STOP。

[Track D B1/C1固定検証](docs/research/track_d_b1_c1_validation_20261106/PLAN.md)：保存13-roundの元H向けoperator誤差接続と、sample00→01・q2のwhole-path raw correctnessだけ。B2/C2・compile・新samplingは対象外。

[Track D Stage B・Cの技術準備](docs/research/track_d_bc_static_preparation_20261106/REPORT.md)：元H向け信号・統計とwhole-round Reference価格の再利用条件・不足・最小検証案。静的調査のみ、科学計算・実装0、公開後mandatory STOP。

[Stage A結果](docs/research/track_d_h4_paired32_reference_20261106/RESULT.md)：保存32本すべてでCX/depth減、RZ増、qubits9→17。32 precompile／128 postcheck PASS、有限cohort限定、mandatory STOP。

Track D Stage Aの[保存32 trajectory paired検証](docs/research/track_d_h4_paired32_reference_20261106/PLAN.md)は同CDF8 endpoint・event16後cut・2方式×X/Yに限定する。全32 precompile PASS時のみ最大128 wrapper compile、公開後mandatory STOP。B/C段階は対象外。

Track Dの実H4移送検証は[固定word02/cutと実行範囲](docs/research/track_d_h4_reference_validation_20261106/PLAN.md)／[結果](docs/research/track_d_h4_reference_validation_20261106/RESULT.md)を参照する。旧SOURCEを別treeから再利用し、通常のtestや旧研究runnerから実行しない。

Track Dの限定人工検証は[固定scopeとSOURCE](docs/research/track_d_minimal_reference_validation_20261106/PLAN.md)を参照する。分子・全RPE検証とは別で、明示runnerだけが実行入口である。

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
