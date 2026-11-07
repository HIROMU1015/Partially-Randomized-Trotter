# 研究計画資料

複数ブランチにまたがる成果と後続記録は[研究結果への統一索引](RESEARCH_RESULTS_INDEX.md)から辿る。
以下の計画・停止履歴は、このcheckoutに保存された時点の記述として読む。

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

## main集約の版差と原SOURCE（2026-11-03）

[索引の統合・SOURCEの版差・保存原版](main_collection_versions_20261103.md)から、同名ファイルの各公開版とmainの配置を確認できる。
過去のSOURCE・test・結果は元snapshotの条件で読む。科学実行と元の研究testの再実行は行っていない。


A1の[新finite-target科学契約設計案](o2_a1_new_finite_target_20261106/REPORT.md)（2026-11-06）：actual-Q/selected full複素signalの定義、Raw/2層/5層/CV、公平なconfidence/全費用と後日の科学call。設計のみ、未採択・科学実行0件。


2026-11-07 A1対B1-OL：[採択済み752-shot予備検証](../../docs/research/o2_a1_n3_pilot_science_20261107/REPORT.md)。保存H4/1 Å/STO-3G/DF12/L_D3/m0/q1/δ.02/r1/K2/atol0、X/Y、seed20261108。既存pilot driverとqualified managerを新activationで接続。人工テスト後SOURCE公開照合して一campaignだけ実行し、完了/中断でmandatory STOP。精度・期待費用保証は対象外。


2026-11-07 A1対B1-OL予備検証：科学SOURCE a139d7dからconsume-once managerで一回起動したが、資源判定で最初のtrial前に正規STOP。取得0/752、Born/compile/RZ0、bundle decode1。比較は未評価、科学retry0。停止原本と全会計を[結果報告](../../docs/research/o2_a1_n3_pilot_science_20261107/REPORT.md)へ保全。次attemptは空き資源確認と別途明示承認が必要。C1凍結維持。


2026-11-07 A1/B1-OL：[752-shot pilotのCPU判定改善とattempt02](../../docs/research/o2_a1_n3_pilot_admission_20261107/REPORT.md)。前attemptの0件STOPを保全し、同じ科学条件・seed/streamで一回再投入する。実入力前にCPU tickを二窓観測し、最大30分の有限資源待機を行う。科学的STOP条件・exact kernel・C1凍結を維持。


2026-11-07 A1/B1-OL：[752-shot予備検証attempt02の完了結果](../../docs/research/o2_a1_n3_pilot_admission_20261107/REPORT.md)。保存H4/1 Å/STO-3G/legacy DF12/L_D3/m0/q1/δ.02/r1/K2/atol0、X/Y、actual Q/保証ground/exact engineを維持。752 bit/価格を完備、A1 2,009,673 RZ、B1-OL 1,622,442 RZ、全3,632,115 RZ。両軸zero-variance保留規則により全体INCONCLUSIVE。人工52 PASS、旧SOURCE/STOPとC1凍結を保全、公開照合後mandatory STOP。本検証の精度・期待費用保証ではない。


2026-11-07：[層内control・層別＋CVの未採択設計](../../docs/research/intrastratum_cv_design_20261107/REPORT.md)。固定理論8d8a290/752結果3c8e55a/N1 law744e1a0を静的確認。rotation fragment群、signed support、product label一致を候補化し、一層一特徴・224-shot paired診断を提案。追加測定なしのmetadata特徴とexact条件付き平均の式を整理したが、相関・分散/費用改善は未検証。実装・新科学計算・保存結果再解析・test実行0、正式採択なし、C1凍結・旧STOP保全、公開照合後mandatory STOP。


2026-11-07：[SCV最小実装・224-shot準備](scv224_preparation_20261107/REPORT.md)。rotation fragment群indicator/条件付き平均/h1 scalar fit/freeze/全5層paired評価を独立SOURCEへ追加。新規人工34＋再利用72＝106件PASS。実H4群・平均/moment・科学RNG/Born/compile/測定・保存結果数値再解析0。224 shots/seed20261109/streamは未採択案。旧SOURCE/結果/STOP・C1凍結を保全し、公開照合後mandatory STOP。


2026-11-07：[O2-SCV224採択済み一campaign](o2_scv224_science_20261107/REPORT.md)。保存H4/1 Å/STO-3G/legacy DF12/L_D3/m0/q1/δ.02/r1/K2/atol0、X/Y、N1 actual Q/N2保証state/exact Fraction engine。群・平均一回、train64＋共通fresh評価160＝最大224 physical shots、seed20261109。旧SOURCEを変更せず薄いactivation、人工118件PASS。公開SOURCE照合後一回開始し、完了/中断でmandatory STOP。


2026-11-07：[O2-SCV224完了結果](o2_scv224_science_20261107/REPORT.md)。保存H4/1 Å/STO-3G/DF12/L_D3/m0/q1/δ.02/r1/K2/atol0、X/Y。群S={df-fragment-3}、224 bit/価格完備、rank2/range合格。X分散点指標は実質不変、Yは約0.02471%低下。学習込み実現RZ1,079,241（train306,990、評価772,251）で、学習費回収の明確な兆候なし。精度/期待費用保証・正式GO/NO-GOなし。約24.26分、全process正常終了、旧SOURCE/結果/STOPとC1凍結を保全しmandatory STOP。


## O2-SCV 2pilotの統合資料（2026-11-07）

[固定224/448結果の統合・要因整理・次方向提案](o2_scv_two_pilot_synthesis_20261107/REPORT.md)。既報値引用とSOURCE静的調査のみ。研究方針の採択や科学的status更新ではない。

2026-11-08：[O2 A2/J1の限定技術調査](o2_a2_j1_mechanism_20261108/O2_NEXT_ALGORITHM_REPORT.md)。PR target liftと成功flag会計、return価格接口を整理。実装・科学計算0、旧R1判定を維持。

2026-11-08：[A2の物理的LCU測定アクセス](o2_a2_executable_access_20261108/REPORT.md)。PREP/SELECT、success/failure、全attempt decoderとnoise床を明示。旧STOPとJ1/R1を維持。

2026-11-08：[O2研究成果の最新版統合](o2_research_results_integration_20261108/O2_RESEARCH_SUMMARY.md)。原SOURCEと報告収録commit、39の主要stage/資料入口、改善機構と現在のSTOP/保留を整理。科学レビューや再検証ではない。

2026-11-08：[O2新規候補の設計・選別](o2_new_algorithm_candidate_exploration_20261108/O2_NEW_ALGORITHM_CANDIDATES.md)、[比較](o2_new_algorithm_candidate_exploration_20261108/O2_CANDIDATE_COMPARISON.md)、[次の有限技術確認案](o2_new_algorithm_candidate_exploration_20261108/O2_NEXT_RESEARCH_PROPOSAL.md)。共通basisのlabel平均を測定へ移す案を先に提案。科学実行/新規性の採択は保留。

2026-11-08：[O2案Mの適格性・測定instrument・人工証拠](o2_m_instrument_technical_validation_20261108/REPORT.md)。Product/Rotation双方の人工平均保存、M2式、既知groupingとの関係を確認。科学的GO/実H4資格は未取得。

2026-11-08：[O2案Mの保存header接続と次の一回metadata pass草案](o2_m_n0_header_binding_preparation_20261108/REPORT.md)。既存Mの群判定を再利用し、回路buildなしの8-wire literalに対応。技術準備のみ、実N1の科学集計は未実行。

2026-11-08：[案M n=0 metadata-only実行結果](o2_m_n0_metadata_result_20261108/REPORT.md)。正232 labelが全て単独群、非自明群0、direct適用はliteral比較でUNKNOWN。一回/retry0でSTOP、測定や別slotへ進めない。

2026-11-08：[O2新機構・案Eの設計](o2_new_mechanism_exploration_20261108/NEW_MECHANISM_CANDIDATES.md)、[技術選別](o2_new_mechanism_exploration_20261108/CANDIDATE_TECHNICAL_ASSESSMENT.md)、[次の技術判別案](o2_new_mechanism_exploration_20261108/NEXT_RESEARCH_PROPOSAL.md)。失敗flagからsuffixを省く全attempt推定。M/Dの救済や新科学実行ではない。

2026-11-08：[O2案Eの限定技術検証](o2_e_early_abort_technical_validation_20261108/REPORT.md)。TECHNICALLY_VALID_KNOWN_LCU_APPLICATION。全attempt平均/M2/flag zeroを人工確認、独自の資源利益は未認定。

2026-11-08：[O2実装前の同値性・strong baseline選別](o2_distinct_mechanism_screening_20261108/REPORT.md)。3方向の操作/式・既知法への還元・summary取得のUNKNOWNを整理。新実装/計算/科学的採択なし。
