# 4方式の保存証拠matrix

2026-11-13 JST。保存データの存在・物理identityの一致・RMSE資格・価格資格を別欄にした。数値の新しい適格判定・rankingは行っていない。**現時点で4方式の実G比較可能な行はない。**

## 共通SOURCE・claim

| SOURCE | 今回確認した内容 |
|---|---|
| [B2084791 S3](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2084791b1b09cff65e07fa8566bb734e1e6bff8e/docs/research/rpe_s3_final_plan_20261113/run01/REPORT.md) | 固定nominal13round、条件付きproxy・IIDの標本診断。PRIMARY/RESULTを参照し、全trialは開かず再解析なし |
| [A97146481 RMSE調査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/97146481f7b05e9418d8fc016f133c7b78d93d08/docs/research/rpe_s3_global_rmse_connection_20261113/REPORT.md) | 全事象loss、population risk、target包含、raw/law移送と二重会計防止。既存NEXT_SCIENCE_PLANを後続の基礎とする |
| Bbf515と現B2084791 | controller/assess/cost/select、evidence.plan、full_baseline_v2_requests、固定price/provider、N/C受入。今回の必要機能は実装済み |
| [D e7dca receipt](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e7dca5f7301c8633b17a59b6a7161331303aef2d/docs/research/d_full_pf_full_h_v2_final_receipts_20261113/REPORT.md) | 2nd/4th/M5、全H pure RTE物理request、保存136候補/73直接価格、receipt追跡済みの分類を利用 |
| [C9d95構造](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9d95d3ea544e1fa1d0e4710c1ca24d8c5776469f/docs/research/h4_structural_schedule_results_20261113/REPORT.md) | 全13round/26軸E[S]、旧4値copy＋新22値。全identity/lawとB正本を照合 |
| [D54b64新lowering](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54b64ef72bf507be25970f2d3362d3be091321c3/docs/research/d_p1_efficient_czz_20261113/REPORT.md) | P1_COMPLETE/4 path/8 compile、SOURCE20fe1194、新旧lowering、有限development cohort。P2/fit/全law価格へ昇格しない |

共通bindingは[metadata](EVIDENCE_METADATA.json)の保存値。H hash=`c1a6c4281f11bba90df2fd18a998bdec96bed00379116537d22faf548c525ea0`、state=`9bcc0b0ecf44440c4411bcb76d652d78ff0595f036144580660c56c47c78e80b`、snapshot=`12219b7d14dc2a275853314269e5e8628005ed72b42d16bc4d3831c8ea515c96`。binding内T=.8は元入力参照であり、全roundの回路時間へ代用しない。

追加で確認した旧研究は、B2084791に保存された`track_a_h246_pf_common_final_20261022.md`（H4 rank10、T.8、3公式、B0/B0I/Fullの共通格子）、`track_a_h46_m5_det_cost_run01_results_20261020.md`（M5の直接wrapper価格と旧適格領域）、[Full-H T=2研究](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7a8bc7f132e166390c943e6b32c92cc6abe1baf4/artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/REPORT.md)。旧報告の候補とSOURCEを設計の手掛かりに使い、旧signal精度の費用順位やGを今回のRMSE条件へ再計算・採用していない。

## 方式別

| 項目 | PR | Full PF | Truncated PF | Full-H pure RTE |
|---|---|---|---|---|
| 物理H | one-body＋DF prefix1が決定論、tail全保持 | one-body＋全10fragment、切断なし | one-body＋prefix7等。B0とtail-identity補正B0Iは別 | 全非scalar成分をRTE化。one-body固有basis Zを含み、決定論block0、scalar一回 |
| 現在のschedule | canonical587d…、S2/.1/K2、共通exactκから13roundのR2…8160 | 保存T∈{.8,1.2,1.6,2}、q/formula別。13roundを持つ正式比較menuは未採択 | B0 p7/S2/.1は全13物理候補を旧研究で保持。B0I p7/Y4/.4は短roundの保存セル | T=.8 q16/R16/K4の専用full-H候補と、T=2 q8/R448/K2等の旧有限研究。今回13roundは新しい提案layout |
| signal/reference/bias | A169d/bbaに全13raw/corrected/reference/exact-tail/bias/logB。数値allowanceは経験的 | D136中signal EXACT134/CONDITIONAL2、reference値136。独立pairはT=2の38のみ。H4の本menuではm3/m4セルに保存signal | B0全13 raw/referenceを利用可能（m3はPF common carrierへ戻る）。B0Iの短round raw/referenceも存在。切断＋PFのsigned biasを元Hに対し評価する必要 | 専用prepare_full_h/full_oracle/evolve_batchで有限mean/logB。旧T2にはK0/2/4/6の保存格子あり。13roundの新meanは不足 |
| N・decoder/RMSE | B S3経験的条件付きRMSEあり。全事象loss未定義、主population CI/十分N/global bound null | 保存Nは旧signal精度/HP型。全RPEのRMSE型N・branch/loss/risk契約不足 | 保存旧HP型Nは全13あるが新RMSEへ採用不可。保持HだけのPASSで元H資格を作らない | 旧signal精度N・cohort費用は全RPE RMSE Nではない。新layoutの全体loss/normalization/N不足 |
| compiled full-wrapper価格 | 旧C1/長時間cohortとD新P1の有限pathは存在するが別domain/lowering。固定canonicalの適格全lawE[C]なし | D receipt X/Y EXACT73、欠測63（H4/H6合計136）。同formula/T/q/prefix/H/profile一致時だけ再利用候補 | PF common・M5・長時間のB0/B0I直接価格あり。全13/同recipeの不足は保存record単位で確認 | light T.8 n9、T2探索n9・別確認n33等の有限cohortあり。旧qRのmean等価性を価格へ移さない |
| 構造価格 | C全26 E[S]と数値enclosure、IDEAL_FINITE_K_SOURCE_LAW。compiled Cはnull | 直接compiled証拠を優先。PR用Sを代用しない | 同左 | Full-Hの別preparation/lawへPRのSを移植しない |
| SOURCE/価格lowering | D新P1はsampler law開発cohort。旧価格と別ID | 旧native/sourceのまま。同compiler hashでも新ZZ経路の適用証拠にはならない | B0/B0Iのscalar・retained H・native profileも区別 | 旧controlled ZZと新loweringで別scenario、全H preparation/one-bodyを照合 |
| 現在の比較資格 | 条件付きpilotのみ | 保存物理/価格の技術一致のみ | 保存物理/旧Nのみ | 保存有限Tの条件付き研究のみ |

DF B3はprefix0でもone-bodyを決定論的に保持するため、Full-H pure RTEではない。Pauli版L_D0は別表現の旧SOURCEとして保存し、今回同H/stateのDF比較へ無検証で混ぜない。

## Full PF最小menuへの保存照合

以下はe7dcaの既存classificationの転記。m3はT=.8、m4はT=1.6。幅.1ではq8/16、幅.4ではq2/4。全12セルのsignalはEXACT_METADATA_MATCH。これは保存scopeでの存在・一致で、全RPE RMSE PASSではない。

| Full PF公式 | 幅 | m3のX/Y直接価格 | m4のX/Y直接価格 | その他11round |
|---|---:|---|---|---|
| 2nd | .1 | EXACT | EXACT | 当該136表の保存grid外 |
| 2nd | .4 | EXACT | 欠測 | 同上 |
| 標準Yoshida4 | .1 | EXACT | 欠測 | 同上 |
| 標準Yoshida4 | .4 | EXACT | EXACT | 同上 |
| M5 best | .1 | EXACT | 欠測 | 同上 |
| M5 best | .4 | EXACT | 欠測 | 同上 |

physical candidate ID・signal carrier・軸statusはEVIDENCE_METADATAの`full_pf_inventory.six_proposed_lanes`。既存73件を再監査せず、正本receiptを参照する。EXACTは旧回路のfull-wrapper値を同定した意味で、新V2への全条件適格性・新lowering・長qモデル資格を意味しない。欠測を0又はq比例価格で埋めない。

## 再利用する関数・テスト

| 既存実装 | 利用方法・不足 |
|---|---|
| `pr_rpe_rmse_controller.generate/assess/cost/select`、`pr_rpe_rmse_evidence.plan` | 候補identity/4方式/全schedule資格/不足セル。実cost/selectは今回呼ばない。科学契約未確定のmenuは採択済み入力にしない |
| `full_baseline_v2_requests.full_pf_spec/full_h_rte_spec/generate/requests/validate_requests` | 2nd/Y4/M5、全H preparation、κなし明示layoutを再利用。A/C同一physical/source。Bのnative physical hook未実装は外部request表で補えるため全面再実装不要 |
| D e7dcaの`full_pf_saved_inventory.for_packet`、`full_pf_price_receipts` | 保存signal/価格carrierとV2へのmetadata結合。receipt照合結果を再利用し、元値の再計数・再compileなし |
| `h46_long_time_signal.context/evolve_batch`、`h46_m5_plan.weights/schedule` | 全H・B0・B0Iの既存作用と係数。実分子へ再適用するのは別の科学scope |
| `h46_full_h_pure_rte.prepare_full_h/full_oracle`、T2の`h46_full_h_t20_opt.evaluate` | one-body込み全H、scalar抽出、finite-K作用。s=qRによるmean共有は元SOURCEの条件付きで、価格同一性なし |
| `rpe_decoder_statistics.prepare/simulate_prepared/summarize`、R3 hook | 供給mean/Nを評価する既存統計器。必要なshot政策・loss/population精度はA/Bの契約で与える。元S3を勝手に再解析しない |
| `pf_pr_rpe_pipeline`、既存price lookup、`compiled_residual` | N/C identity/law/cohort/profile/domain・単位を照合。E[S]＋Dはlaw一致/距離と資格が必要。P1をmodelとして登録しない |

関連テスト40件は[REGRESSION_TESTS](REGRESSION_TESTS.txt)。8次PFの生成・接続・科学検証は対象外。既存資料にある8次ラベルを変更・削除していない。
