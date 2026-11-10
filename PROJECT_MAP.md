# プロジェクト案内

## 2026-11-10 RPE価格成果の再利用調査

[技術調査報告](docs/research/rpe_price_reuse_survey_20261110/REPORT.md)と[固定SOURCE索引](docs/research/rpe_price_reuse_survey_20261110/SOURCE_INDEX.md)。Dのq別直接価格・A/F/S・q64完了、DF10/19有限時間とDF12 V3/R1の条件差、保存N固定の最小評価案を整理。既存branchを継続し、新科学計算・数値再解析0。公開照合後mandatory STOP。

## 2026-11-09 PR＋RPE総費用の横断調査

[REPORT](docs/research/pr_rpe_cross_track_20261109/REPORT.md)から、[既存成果・重複](docs/research/pr_rpe_cross_track_20261109/EXISTING_RESULTS_AND_OVERLAP.md)、[改善候補](docs/research/pr_rpe_cross_track_20261109/IMPROVEMENT_CANDIDATES.md)、[最小検証案](docs/research/pr_rpe_cross_track_20261109/MINIMUM_VALIDATION_PLANS.md)へ辿る。同目的の既存横断調査branchを継続し、Closed P3共同選択と後続C/D/O2成果を静的確認。D C1・q64の完了を反映し、新計算・再解析・実装0、固定成果/STOP不変更。公開照合後mandatory STOP、次判断はユーザー／GPT。

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

## Track C2-B限定pilot（2026-11-06）

[固定Full Y4/LD12/h=.02の18round。共役block合成配分は登録非一様policy比で総T 0.5470%減。強baseline下界には勝てず優位性未資格、STOP。](docs/research/track_c2_b_pilot_20261106/REPORT.md)。SOURCE、検査、raw、STOPは報告から辿る。旧STOPは維持。

## Track C2-B強baseline（2026-11-06）

[数値上の総T削減0.514072%が残るが、全alpha下界とのgapが残り最適性未確定。登録72配分・保存7603 recordだけで検証しmandatory STOP。](docs/research/track_c2_b_strong_baseline_20261106/REPORT.md)。新合成・compile・候補追加なし。

## 新A reference干渉限定pilot（2026-11-06）

[成立条件・強baseline・短round資源差と全RPE未確定](docs/research/pr_reference_interference_pilot_20261106/REPORT.md)。SOURCEは `src/trottertracks/pr_reference_interference_*.py`、runnerは `scripts/run_pr_reference_interference_pilot.py`。A1/A2の検査関数・raw・入力・STOPは同報告から辿る。旧C/C2保留・STOPは維持。

## 2026-11-07 新A：長round移送・全support資格

[REPORT](docs/research/pr_reference_transfer_qualification_20261106/REPORT.md)。固定H4/S2/LD3/h=.02/K2・compiler A・算術モデルで、再算出shotを含む18round総CX/RZのreference上界が登録baseline下界を下回る。SOURCEは `src/trottertracks/pr_reference_transfer_*.py`、検査は `adapter_checks` / `qualify` / `validate_rz_paths`。depth/size/T・別制御分解への優位は未資格。mandatory STOP、旧結果・STOP保全。

## 2026-11-07 新A：強baseline B′限定検証

[REPORT](docs/research/pr_reference_bprime_20261107/REPORT.md)。固定H4/S2/LD3/h=.02/K2・compiler A・算術モデルで、referenceは4-CX B′に対し期待総CXを少なくとも4.681782%削減。一方、期待総RZは少なくとも0.150597%増加。B′ shot8598、reference8606。depth/size/Tは未資格、mandatory STOP。旧結果・STOP保全。 SOURCEは `src/trottertracks/pr_reference_bprime.py` / `pr_reference_bprime_accounting.py`。検査・実行関数、raw、構造補遺は同報告を参照。

## 2026-11-07 新A V3：回転分類・18round再会計

[REPORT](docs/research/pr_reference_rotation_v3_20261107/REPORT.md)、[規約](docs/research/pr_reference_rotation_v3_20261107/CLASSIFICATION_RULES.md)、[SOURCE・結果](artifacts/pr_reference_rotation_v3/2026-11-07/README.md)。固定H4/STO-3G/DF12・Partial S2 LD3 h=.02 K2・compiler A。確定任意角部分はB′比2.818200%減だが、未確定角度を含む全任意角数の改善は未資格。SOURCE型と実native型を分離。2165 local/384 cohort/14 QPY/15人工検査。旧CX/RZ/shot会計を維持、新compile/T合成0、V5未開始、mandatory STOP。新SOURCE `src/trottertracks/pr_reference_rotation_v3.py` の明示関数を使用。

## 2026-11-07 新A V5：全18round native DAG depth

[REPORT](docs/research/pr_reference_depth_v5_20261107/REPORT.md)、[導出](docs/research/pr_reference_depth_v5_20261107/DERIVATION.md)、[SOURCE・結果](artifacts/pr_reference_depth_v5/2026-11-07/README.md)。固定H4/STO-3G/DF12・Partial S2 LD3 h=.02 K2・compiler A・canonical finite-bit lawで、referenceの期待総depth上界がB′下界を下回り、≥40.995952%削減。max-plus frontierと独立2 laneの期待max上界。追加8qubit、RZ増加、総T未取得を維持。新SOURCE `src/trottertracks/pr_reference_depth_v5.py`、人工13検査・保存256価格・8 QPY照合。科学対象の新build/compile0、旧STOP保全、mandatory STOP。

## 2026-11-07 新A V5：レビュー指摘2点の限定補強

[REPORT](docs/research/pr_reference_depth_v5_reinforcement_20261107/REPORT.md)、[SOURCE・結果](artifacts/pr_reference_depth_v5_reinforcement/2026-11-07/README.md)。5 inverse D×4 contextを4 batch compileし、正逆の行列は異なるが全spanが旧上界以下。B′144 D/frame record（下界対象19×4を含む）はX-freeで、CX→X相殺後も非X鎖とδを保持。A/B PASS、18round再計算0、旧≥40.995952%保証を固定モデル下で維持。新SOURCE `src/trottertracks/pr_reference_depth_v5_reinforcement.py`、人工3検査。独立レビュー本文は未提供・未確認、著者技術検証。旧SOURCE/結果/STOP保全、mandatory STOP。

## 2026-11-07 新A：Clifford+T評価の静的再利用調査

[REPORT](docs/research/pr_reference_clifford_t_survey_20261107/REPORT.md)、[REUSE_APPLICABILITY](docs/research/pr_reference_clifford_t_survey_20261107/REUSE_APPLICABILITY.md)。strict RZ engine・actual adjoint・raw disk会計の再利用条件を整理。全support位相key、precision、期待価格、T-depthは追加資格が必要で、総T/T-depthは未取得のまま。有理期待Tへ整数ceil/gcd下界を直接移さない。新規合成・価格再集計・回路/数値検証0、旧SOURCE/結果/STOP不変更、mandatory STOP。次担当GPT／ユーザー。

## 2026-11-07 新A CT-0：key・位相・precision資格

[REPORT](docs/research/pr_reference_ct0_20261107/REPORT.md)、[coverage](docs/research/pr_reference_ct0_20261107/KEY_COVERAGE.md)、[phase/precision](docs/research/pr_reference_ct0_20261107/PHASE_PRECISION_FEASIBILITY.md)。保存numeric positive keyはB′400中153、reference384中154にrecordあり。全support位相/誤差域は有限表現、word価格は未資格。正のRPE追加誤差予算あり、実word込みはCONDITIONALLY_FEASIBLE。新規合成/compile/T価格/precision最適化/総T0、mandatory STOP。SOURCE `src/trottertracks/pr_reference_ct0.py`、入力・人工9検査・結果は[成果物](artifacts/pr_reference_ct0/2026-11-07/README.md)。

## 2026-11-07 新A CT-1：有限strict合成correctness

[REPORT](docs/research/pr_reference_ct1_20261107/REPORT.md)、[SCOPE](docs/research/pr_reference_ct1_20261107/SCOPE.md)、[SOURCE/入力/raw](artifacts/pr_reference_ct1/2026-11-07/README.md)。固定q1/r1・保存2 trajectory・X/Y、epsilon1e-9。97 word再利用＋171新合成、268 word/21 local/8 whole検査PASS。q1保存192/axisの予算条件成立、全support CT law/位相価格・18round総Tは未資格／未取得。旧SOURCE/結果/STOP保全、mandatory STOP。

## 2026-11-07 新A CT-2：全support strict合成価格の静的調査

[REPORT](docs/research/pr_reference_ct2_20261107/REPORT.md)、[理論/実装境界](docs/research/pr_reference_ct2_20261107/STRICT_SYNTHESIS_BOUNDS.md)、[全support適用条件](docs/research/pr_reference_ct2_20261107/FULL_SUPPORT_PRICE_APPLICABILITY.md)。理論式は前提付きTHEORETICAL_ONLY、固定pygridsynth2.0.0の全入力取得/価格はUNQUALIFIED。受理後word長制限を全域価格にしない。新科学計算/合成/価格評価0、旧SOURCE/結果/STOP不変更、mandatory STOP。次担当GPT／ユーザー。

## 2026-11-07 新A CT-3A：全support strict fallbackの静的調査

[REPORT](docs/research/pr_reference_ct3a_20261107/REPORT.md)、[候補比較](docs/research/pr_reference_ct3a_20261107/FALLBACK_POLICY_CANDIDATES.md)、[停止性・実出力T上界](docs/research/pr_reference_ct3a_20261107/POLICY_TERMINATION_AND_BOUNDS.md)、[次判断の不足条件](docs/research/pr_reference_ct3a_20261107/CT3B_MINIMAL_SCOPE.md)。strict SU(2)生成集合とideal SK十分定数を記号的整理したが、初期net coverage・有限精度保証が不足しTHEORETICAL_ONLY。READY候補なし、全support実価格未資格。新実装・net/word生成・科学計算・価格評価0、旧SOURCE/結果/STOP不変更、mandatory STOP。次担当GPT／ユーザー。

## 2026-11-07 新A CT-4A：大きいqの有限T比較preflight

[PREFLIGHT](docs/research/pr_reference_ct4a_20261107/PREFLIGHT.md)、[SCOPE_PROPOSAL](docs/research/pr_reference_ct4a_20261107/SCOPE_PROPOSAL.md)。保存q128/r1・rare/nonrare各j0・B′/Reference×X/Yの8構成、一様epsilon1e-11を提案。q128 whole QPY/完全literal key表は未保存で、承認後のnative準備compile→実key固定→missing合成が必要。今回は静的確認のみ、新合成/compile/科学数値再解析/T価格集計0、旧SOURCE/結果/STOP不変更。CT-4A完了でSTOP、CT-4B未承認・未開始。次担当ユーザー。

## 2026-11-07 新A CT-4B：q128の有限strict T比較

[REPORT](docs/research/pr_reference_ct4b_20261107/REPORT.md)、[SCOPE](docs/research/pr_reference_ct4b_20261107/SCOPE.md)、[SOURCE/入力/raw/STOP](artifacts/pr_reference_ct4/2026-11-07/README.md)。固定H4/STO-3G/DF12・Partial S2/LD3/h=.02/K2・保存q128/r1・rare/nonrare各j0・X/Y・一様epsilon1e-11で、Referenceの1shot実TはB′比1.865933%/1.881674%減。全support期待T/18round総Tは未資格/未取得。8 native再compile、299 word（111再利用/188新規）、25 local/保存199shot条件PASS。SOURCE `src/trottertracks/pr_reference_ct4.py`、入口 `scripts/run_pr_reference_ct4.py`。旧SOURCE/結果/STOP保全、mandatory STOP。次担当GPT／ユーザー。

## 2026-11-07 新A CT-5B：全supportの構造条件と必要価格境界

[REPORT](docs/research/pr_reference_ct5b_20261107/REPORT.md)、[構造保証](docs/research/pr_reference_ct5b_20261107/STRUCTURAL_CERTIFICATES.md)、[有限key](docs/research/pr_reference_ct5b_20261107/FINITE_KEY_AND_UNKNOWN_PRICE.md)、[条件式](docs/research/pr_reference_ct5b_20261107/CONDITIONAL_T_ADVANTAGE.md)、[次判断](docs/research/pr_reference_ct5b_20261107/NEXT_SCOPE.md)。q128/r1・epsilon1e-11の静的調査。D SOURCE固定とframe pair共通性を導出し、native価格への移送は固定compiler/context条件付き。中央4 keyは保存価格、共通frame/event b2とvacuumの取得資格が残る。差に不要な数値価格と実行に必要なwordを分離し、pointwise/期待条件を未評価式で記載。新科学計算/合成/価格/compile0、旧証拠不変更、[mandatory STOP](artifacts/pr_reference_ct5b/2026-11-07/STOP.json)。次担当GPT／ユーザー。
