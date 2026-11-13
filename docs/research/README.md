# 研究計画資料

[H4 Full PF H_eff適用準備](h4_full_pf_heff_preparation_20261113/REPORT.md)：決定論PFの主誤差評価にH_effシフトを採用。S2/Δ.1・m0/m12の[有限案](h4_full_pf_heff_preparation_20261113/SCIENCE_PLAN.md)と[B HANDOFF](h4_full_pf_heff_preparation_20261113/HANDOFF.md)を具体化。人工新4＋旧15PASS、実科学/保存再解析0、幅探索へ進まず公開後STOP。

[旧2f0時点・採用前：PF有効Hamiltonian技術基盤](pf_effective_hamiltonian_20261113/REPORT.md)：Full/Truncated/head/PR exact-tailのunitary診断とfinite-K corrected/raw対象外を分離。minimum shiftとRPE biasを区別し、新15＋旧8人工回帰PASS。実H4/H6・保存signal再解析・幅選択/N/G0、手法未採択、公開後STOP。

[PR-LD1/LD2全round signal結果](pr_rpe_first_signal_results_20261113/REPORT.md)：同保存H4 DF10・S2/Δ.1/K2・元exactκ/h.1/M12で26候補が全数値gate通過。B canonical全meanとD52軸metadataを供給、新reference/旧mean再解析/retry/N/MC/C/G0。初回二候補の最適性は未認定、公開後STOP。

[初回PR全RPE signal menu（承認待ち）](pr_rpe_first_signal_menu_20261113/REPORT.md)：保存descriptorの隣接prefix1/2、同exactκ/外側条件、B437ba4acdのfixed-A V2で26metadata rowを具体化。保存input/reference再利用、旧meanを新backendへ再ラベルせず、新科学/保存再解析0でSTOP。

[PR全round signal pipeline](pr_rpe_signal_pipeline_20261113/REPORT.md)：uniform PR S2/M5・K2/4/6の数値provider、直接expm・有限RTE・独立raw/corrected・power/cacheを接続。17技術test PASS、実科学0。BのN/G・価格評価は重複実装せず、公開照合後STOP。

[PRの全RPE資源探索設計](pr_rpe_search_design_20261113/REPORT.md)：Track Aの旧順位を使わず全scheduleで同RMSEのN×Cを比較する設計。B/価格接続、direct-head PF・finite-K独立meanの再利用差分、cheap優先順位と厳密除外の区別、未採択2schedule pilot案。17技術test PASS、実科学0、公開後STOP。

[H4 PF長時間の二経路診断](pf_power_stability_h4_20261113/REPORT.md)：代表F2-COARSE/m12の旧spectral・direct expm、計2meanを評価。directは数値gate通過、物理biasや全RPE資格は別。PF/B0Iのdirect expm採用による残15セル再開を提案し、未承認の追加計算はせずSTOP。

[PF power安定性の人工調査](pf_power_stability_artificial_20261113/REPORT.md)：再構成だけでは見えないGram defectとpower増幅を人工入力で確認。直接expmを次の限定検証候補とし、実H4の採用は保留。17技術test PASS、実科学・保存再解析0、旧STOPを保持。

[H4・4方式screen run02の部分結果](h4_four_method_signal_screen_results_20261113/run02/REPORT.md)：同一SOURCEの再実行でFull PF4セルを取得、6数値gate失敗・9未開始・1空例外未解決。新reference0、保存40不変。全13round候補の判断材料は未完了で、次は数値安定性の技術調査案。

[H4・4方式screenの入力STOP](h4_four_method_signal_screen_results_20261113/REPORT.md)：固定20セルbatchはcontext importで停止、新mean/normalizer/reference0。保存40・SOURCE・PLANを保持。数値による候補判断は未取得、再実行はユーザー判断後。

[H4・4方式signal screen準備](h4_four_method_signal_screen_preparation_20261113/REPORT.md)：B51500の暫定11案、新20セル・保存40セルとreference再利用を固定。17人工・回帰test PASS、実signal/N/RMSE/C/G・保存再解析0。科学実行と正式候補採択は別段階。

[S3から全energy RMSEへの接続調査](rpe_s3_global_rmse_connection_20261113/REPORT.md)：固定S3の条件付き点RMSEと全population riskを区別し、target包含・全事象loss・raw数値/law証拠を整理。直接riskと既定Minkowski成分の経路を分け、PF/finite-K二重計上を防ぐ。不足表/B HANDOFF/有限案を作成、既存8技術回帰PASS、新科学・保存再解析0。十分N/global boundは未資格、公開照合後STOP。

[RPE decoder R3高速化・B S3接続](rpe_decoder_r3_20261113/REPORT.md)：独立R3 decoderの4候補branch選択を実装し、元scalar/vector各tie規則とBのcount/RNG/loss/CIを保全。新12＋旧20の32技術test PASS。人工M12/512trialで旧vector比96.3倍、B batch接続664.6倍、実H4/S3/保存再解析/C/G0。旧vector exact-tieのみ互換fallback、公開照合後STOP。

[H4 PR全13roundの物理誤差契約](h4_pr_physical_error_contract_20261113/REPORT.md)：保存S0/m1/S1＋S2の全signal・bias・allowanceをSOURCE付きで転記。raw↔X/Y・PF/finite-K/state/lawとMinkowskiの適用条件をBへ整理。新8＋既存13の21技術テストPASS、新科学・保存再解析0、十分N/global RMSEは未資格/null。次は別承認のB ideal-law decoder統計、公開照合後STOP。

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

2026-11-08：[O5：Controlled/Reference選択と改善候補](track_d_o5_algorithm_design_20261108/REPORT.md)。既知mixedとの重複、descriptor-onlyとoracleの区別、cut/接口融合の条件、O2/Track Dのdomain差を整理。Aの安価な技術選別を優先提案、正式採択/新科学計算なし。

## Track D O5：L/R cut候補Bの人工検証

[実装と結果](track_d_cut_artificial_20261108/REPORT.md)／[全7資源比較](track_d_cut_artificial_20261108/ARTIFICIAL_RESOURCE_COMPARISON.md)／[未採択H4案](track_d_cut_artificial_20261108/NEXT_H4_VALIDATION_DRAFT.md)。既存builderとP0を再利用、人工不均衡例だけ追加depth減・均衡例0、実H4未実行。

[O5 cut実H4移送preflight](track_d_cut_h4_q4_20261108/REPORT.md)／[入力・再利用状況](track_d_cut_h4_q4_20261108/INPUT_AND_REUSE.md)：C6a q4_sample00原本未取得でcut前STOP。selector変更/実科学実行なし。

[O5 q4移送完了](track_d_cut_h4_q4_completion_20261108/REPORT.md)／[実装・再利用](track_d_cut_h4_q4_completion_20261108/IMPLEMENTATION_AND_REUSE.md)：q4_sample00で中央2、追加depth減0。前回入力STOPを消さず、原本回収後の結果を別runへ保存。
